"""Poutre de Timoshenko — console exacte dès un seul élément.

Physique
--------
Poutre déformable en cisaillement. Cinématique : courbure `κ = θ'`, distorsion
`γ = w' - θ`. Efforts : moment `M = E·I·θ'`, effort tranchant
`V = G·A_s·(w' - θ)`. Équilibre : `dV/dx + q = 0`, `dM/dx - V = 0`.

The assembled element is the **exact solution** of these two equations on a
span free of distributed loads — the closed form parameterized by
`Φ = 12·E·I/(G·A_s·L²)`. Its shape functions therefore depend on the material,
which no finite element space can tabulate: the space declares
`MODEL_EMBEDDED`, that is, the formulation owns its interpolation.

Problème
--------
Clamped cantilever (`w = θ = 0`), transverse load `P` at the free end.
Analytical solution `w = P·L³/(3·E·I) + P·L/(G·A_s)` — both compliances,
cisaillement, **en série**.

The element being exact at the nodes, **one** is enough: refining changes
nothing, which this script checks. (The previous version was linear with
under-integrated shear; it converged towards that value instead of reaching
it, and this example showed its convergence.)

Lancement ::

    maturin develop --features extension-module
    python examples/timoshenko.py
"""

import pyrucast

E, I, G, A_S, L, P = 1.0, 1.0, 30.0, 1.0, 1.0, 1.0


def _clamp(target, node, var):
    imposed = pyrucast.mesh.poi1_from_nodes([node])
    multiplier = pyrucast.mesh.barycenter(imposed)
    return pyrucast.model.dirichlet(target, var, imposed, multiplier)


def tip_deflection(n_elems: int) -> float:
    c = pyrucast.Coords(1)
    base = c.add_node([0.0])
    tip = c.add_node([L])
    mesh = pyrucast.mesh.line(base, tip, n_elems)  # console 1-D (`line`)
    # The basis belongs to the formulation, not to the space: it depends on `Φ`,
    # hence on the material, and is computed cell by cell.
    fes = pyrucast.FiniteElementSpace(mesh, interpolation="MODEL_EMBEDDED")

    model = pyrucast.model.timoshenko(fes)
    model = model | _clamp(model, base, "w")
    model = model | _clamp(model, base, "theta")

    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("I", I), ("G", G), ("A_s", A_S)]
    )

    load = pyrucast.mesh.poi1_from_nodes([tip])
    rhs = pyrucast.NodeField(load, ["f_w"])
    rhs[0].set_value(tip, "f_w", P)

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)
    return solution.value(tip, "w")


def main() -> None:
    analytical = P * L**3 / (3.0 * E * I) + P * L / (G * A_S)
    print(f"{'N':>4} {'w_tip':>12} {'err. rel.':>12}")
    for n in (1, 2, 5, 10, 40):
        w = tip_deflection(n)
        print(f"{n:4d} {w:12.6f} {abs(w - analytical) / analytical:12.2e}")
    print(f"\nanalytique  = {analytical:.6f}  (P·L³/3EI + P·L/GA_s)")

    # Exact at the nodes: one element already gives the answer, and refining does
    # not improve it — there is nothing to improve.
    one = tip_deflection(1)
    assert abs(one - analytical) < 1e-12 * analytical, one
    assert abs(tip_deflection(40) - one) < 1e-12 * analytical
    print("OK: exact at the nodes from one element on, refining changes nothing.")


if __name__ == "__main__":
    main()
