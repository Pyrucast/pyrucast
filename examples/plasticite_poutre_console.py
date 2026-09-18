"""Elasto-plastic cantilever beam — a hand-rolled Newton loop on top of pyrucast.

Python version of `examples/plasticite_poutre_console.rs` (same building blocks,
same algorithm). It favours readability; for the **parallelism bench**, prefer
the Rust example (pure, with no interpreter overhead).

Physics
-------
2-D plane-stress continuum, small strains. Perfect von Mises plasticity (J2
radial return, no hardening): the equivalent stress is capped at `sigma_y`. Beam
clamped at the left end (`u_x = u_y = 0`), sheared downwards on the right face.
The load is raised by increments; beyond first yield a plastic zone develops
near the clamped end and the deflection departs from the linear response.

What pyrucast does vs. what the example does
--------------------------------------------
pyrucast knows NOTHING about Newton. It provides the pointwise operators:

- `stiffness`: the **elastic** stiffness `K` (iteration operator);
- `deformation`: the strain `ε = ½(∇u + ∇uᵀ)` at the Gauss points;
- `integrate_behavior` (Cast3m `COMP`): the law at the point — radial return,
  which yields `σ` and the updated plastic state (`VAR0` → `VAR1`);
- `internal_forces` (Cast3m `BSIG`): the internal forces `∫ Bᵀ σ dΩ`;
- `solve`: the linear solve (sparse LU, cached factorisation);
- **field arithmetic** (`+ - *`), the **union** (`|`) and `restrict_like`
  (reprojection of a field onto the support/components of another), which
  replace every nodal loop: `residual = f_ext - f_int`, `u = u + du`;
- an `Evolution` with field values for the **loading history**: the load of each
  step is interpolated at the pseudo-time (`load_evo.interpolate(t)`).

The example assembles its own Newton loop: residual `r = F_ext − F_int`,
increment `δu = K⁻¹ r`, `u ← u + δu`, and the carry-over of the internal state
from one step to the next. This is a **modified Newton** (constant operator =
elastic `K`): `K` is assembled and factorised once and for all.

Run with ::

    maturin develop --release
    python examples/plasticite_poutre_console.py

Environment variables: `PYRUCAST_NX`, `PYRUCAST_NY` (cells along the length /
through the height), `PYRUCAST_NSTEPS` (load steps), `PYRUCAST_PMAX` (final load).
"""

import os

import pyrucast


def _plastic_diagnostics(state):
    """(p_max, number of yielded Gauss points) — `p > 0` marks a point.

    Without a loop: `p_max` through `max`, the count by masking the `p`
    component into 0/1 (band "> 1e-12") then summing it."""
    p_max = state.max("p")
    masked = state.mask(gt=1e-12, components=["p"])
    n_plastic = round(masked.sum("p"))
    return p_max, n_plastic


def main():
    # ── Parameters (steel material, geometry, loading) ──────────────────────
    young, nu, sigma_y = 210_000.0, 0.3, 250.0
    length, height = 10.0, 1.0
    nx = int(os.environ.get("PYRUCAST_NX", 24))
    ny = int(os.environ.get("PYRUCAST_NY", 6))
    nsteps = int(os.environ.get("PYRUCAST_NSTEPS", 10))
    p_max_load = float(os.environ.get("PYRUCAST_PMAX", 5.0))

    print(
        f"Plastic cantilever beam: {nx}×{ny} QUA4  (L={length}, H={height}), "
        f"E={young}, ν={nu}, σy={sigma_y}"
    )
    print(f"Loading: 0 → {p_max_load} in {nsteps} steps (modified Newton, elastic K)\n")

    # ── Mesh: left/right SEG2 edges, then a sweep into QUA4 ─────────────────
    c = pyrucast.Coords(2)
    pt_a = c.add_node([0.0, 0.0])
    pt_b = c.add_node([0.0, height])
    pt_c = c.add_node([length, 0.0])
    pt_d = c.add_node([length, height])
    left_edge = pyrucast.mesh.line(pt_a, pt_b, ny)
    right_edge = pyrucast.mesh.line(pt_c, pt_d, ny)
    mesh = pyrucast.mesh.sweep(left_edge, right_edge, nx)
    fes = pyrucast.FiniteElementSpace(mesh)

    # Tip node (mid-height) and POI1 mesh of the FREE nodes (X > 0) — target
    # support for the residual norm on the free DOFs only.
    tip = mesh.nearest_node([length, height / 2.0])
    coords_field = pyrucast.node_field.positions(mesh, ["X"])
    free_mesh = pyrucast.mesh.select(coords_field, ge=length / nx / 2.0)
    imposed_mesh = pyrucast.mesh.to_poi1(left_edge)
    multiplier = pyrucast.mesh.translate(imposed_mesh, [0.0, 0.0])

    # ── Model: plasticity (plane stress) + clamped end (Dirichlet) ──────────
    model = pyrucast.model.plasticity_perfect(fes, "plane_stress")
    model = model | pyrucast.model.dirichlet(model, "u_x", imposed_mesh, multiplier)
    model = model | pyrucast.model.dirichlet(model, "u_y", imposed_mesh, multiplier)

    # ── Reference load: unit shear (density −1) on the right face, as consistent
    #    nodal forces. It is a term of the model: it joins the model, and its
    #    density joins the material. ──────────────────────────────────────────
    right_fes = pyrucast.FiniteElementSpace(right_edge)
    model = model | pyrucast.model.flux(right_fes, model, "f_y")
    materials = pyrucast.element_field.material_field(
        model, [("E", young), ("nu", nu), ("sigma_y", sigma_y), ("phi_f_y", -1.0)]
    )
    load_unit = pyrucast.node_field.external_forces(model, materials)

    # ELASTIC stiffness: iteration operator of the modified Newton. Assembled
    # once; `solve` caches the factorisation and reuses it.
    k = pyrucast.matrix.stiffness(model, materials)

    # ── Loading history: an Evolution with FIELD values, tabulated against the
    #    pseudo-time t ∈ [0, 1]. Two keyframes of the nodal force field — zero at
    #    t=0, complete (`p_max · unit_load`) at t=1 — on the SAME support. The
    #    load of each step is read by linear interpolation. ────────────────────
    zero_frame = load_unit * 0.0
    full_frame = load_unit * p_max_load
    load_evo = pyrucast.Evolution(
        [(0.0, zero_frame), (1.0, full_frame)], out_of_range="clamp"
    )

    # ── Simulation state (persistent across the steps) ──────────────────────
    u = pyrucast.NodeField(
        mesh, ["u_x", "u_y"]
    )  # accumulated displacement, zero at first
    # The state at rest: `None` at the first step, where A is the reference
    # configuration. The operator materialises it itself, with **every** component
    # the law reads back afterwards — σ(A), ε(A), the internal state — where a
    # hand-written list forgets some.
    state = None

    # ── Loop over the load steps ────────────────────────────────────────────
    # Modified Newton (operator = elastic K): linear convergence, hence slow on
    # the plastic branch. High iteration cap, relative residual 1e-6.
    max_newton = 200
    print(
        f"{'step':>4} {'P':>8} {'iter':>6} {'deflection u_y':>14} {'p_max':>14} {'n_plast':>8}"
    )

    prev_defl = 0.0
    any_plasticity = False

    for step in range(1, nsteps + 1):
        # Pseudo-time of the step ∈ ]0, 1]; the external load follows from it by
        # interpolation of the Evolution (nodal force field of the step).
        t = step / nsteps
        load_p = p_max_load * t  # nominal shear at the tip (for the display)
        load_scaled = load_evo.interpolate(t)
        # Norm of the step load (relative scale of the residual): xᵀx of the field.
        ext_norm = pyrucast.measure.xtx(load_scaled) ** 0.5
        tol = 1e-6 * ext_norm + 1e-12

        iters = 0
        last_out = None
        res_norm = float("inf")

        for _ in range(max_newton):
            # ε(u) → behaviour input (ε | VAR0) → σ, VAR1 (COMP).
            strain = pyrucast.element_field.deformation(u, fes)
            out = pyrucast.element_field.integrate_behavior(
                model, strain, materials, prev=state
            )
            # Internal forces F_int = ∫ Bᵀ σ dΩ (BSIG).
            f_int = pyrucast.node_field.internal_forces(model, out, u, materials)

            # Residual r = F_ext − F_int and its norm on the **free** DOFs, with
            # no nodal loop at all — everything through the operators and the
            # primitives:
            # - `f_ext` = external load of the step reprojected onto the support
            #   AND the components of `f_int` (`restrict_like`): `f_x` (=0), `f_y`;
            # - `residual = f_ext − f_int` through the `-` operator;
            # - the norm is read on the free nodes only: `residual` `restrict`ed
            #   to `free_mesh` then `xtx` (the clamped nodes carry the reaction).
            f_ext = pyrucast.node_field.restrict_like(load_scaled, f_int)
            residual = f_ext - f_int
            res_norm = (
                pyrucast.measure.xtx(pyrucast.node_field.restrict(residual, free_mesh))
                ** 0.5
            )
            last_out = out

            if res_norm <= tol:
                break
            # δu = K⁻¹ r (elastic K, cached factorisation). δu carries the primal
            # AND the dual DOFs (multipliers). Its support already coincides with
            # that of u (same hidden POI1 companion from `to_poi1`, shared by
            # `solve` and `NodeField(mesh)`); `restrict_like` only serves to filter
            # out the dual components — otherwise `u + δu` would copy the
            # multipliers into u by union. Then u ← u + δu.
            du = pyrucast.solver.solve(k, residual)
            u = u + pyrucast.node_field.restrict_like(du, u)
            iters += 1

        converged = res_norm <= tol

        # State commit: VAR0 ← VAR1. The converged behaviour output carries, on
        # top of the plastic state (`eps_p_*`, `p`), the stresses (`sig_*`); it is
        # carried over as it stands as the new VAR0. The law reads its inputs by
        # name, so the extra components are ignored.
        state = last_out

        # Diagnostics of the step.
        p_max_val, n_plastic = _plastic_diagnostics(state)
        defl = u.value(tip, "u_y")
        any_plasticity = any_plasticity or n_plastic > 0
        flag = "" if converged else "  (residual left over)"
        print(
            f"{step:>4} {load_p:>8.3f} {iters:>6} {defl:>14.6e} {p_max_val:>14.6e} {n_plastic:>8}{flag}"
        )

        # The deflection grows (in absolute value, downwards) with the load.
        assert abs(defl) >= abs(prev_defl) - 1e-9, (
            f"non-monotonic deflection at step {step}"
        )
        prev_defl = defl

    # Beyond first yield, a plastic zone must appear.
    p_first_yield = sigma_y * (height * height / 6.0) / length
    if p_max_load > p_first_yield:
        assert any_plasticity, (
            f"P_max={p_max_load} exceeds first yield "
            f"(≈{p_first_yield:.2f}) but no plastic point was detected"
        )
        print(
            f"\nOK: plasticity developed (P_max={p_max_load} > P_elastic≈{p_first_yield:.2f})."
        )
    else:
        print(
            f"\nOK: the response stayed elastic (P_max={p_max_load} ≤ ≈{p_first_yield:.2f})."
        )


if __name__ == "__main__":
    main()
