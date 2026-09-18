"""2-D thermal conduction + convection — compared with the analytical solution.

Problème
--------
On the unit square [0, 1]² (a structured QUA4 grid):

  * **left** edge (x = 0): a distributed **heat source** (flux of
    Neumann, densité ``Q``) ;
  * **right** edge (x = 1): a **convective exchange** (Robin / film)
    ``q·n = h·(T - T_ext)`` with a fluid at ``T_ext``;
  * bords **haut/bas** : aucune condition ⇒ condition naturelle (flux nul,
    bords *isolés*).

No Dirichlet is needed: the film term ``h ∫ N_i N_j dΓ`` anchors the
temperature (pure-Neumann conduction is singular) and makes the matrix
definite. The lateral edges being insulated, the field does not depend on ``y`` ::

    T(x) = T_ext + Q/h + (Q/k) * (1 - x)

all the injected flux ``Q`` leaving by convection at x = 1
(``h·(T(1) - T_ext) = Q``).

Mise en donnée de la convection
-------------------------------
Le modèle ``model.boundary_transfer`` fournit la **matrice de film** (variables
``T``/``q`` shared with the conduction ⇒ a direct coupling). The external part
``h·T_ext·∫N_i dΓ`` is a **right-hand side**, built with the same operator
its ambient, as the source carries its density — no normal is required.

This is the Python equivalent of the integration test ``tests/thermal_convection.rs``.

Lancement ::

    maturin develop --features extension-module
    python examples/thermal_convection_2d.py
"""

import pyrucast

# ── Problem data ────────────────────────────────────────────────────────────
K = 2.0  # conductivité
Q = 10.0  # flux density injected on the left edge
H = 5.0  # exchange (film) coefficient on the right edge
T_EXT = 20.0  # the fluid's ambient temperature
N = 4  # N×N éléments QUA4


def main() -> None:
    h = 1.0 / N

    def idx(i: int, j: int) -> int:
        return j * (N + 1) + i

    # ── Mesh: an N×N grid of QUA4 on [0,1]², by sweeping two lines ───────────
    c = pyrucast.Coords(2)
    bottom = pyrucast.mesh.line(c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0]), N)
    top = pyrucast.mesh.line(c.add_node([0.0, 1.0]), c.add_node([1.0, 1.0]), N)
    mesh = pyrucast.mesh.sweep(bottom, top, N)

    grid = [None] * ((N + 1) * (N + 1))
    for cy in range(N):
        for cx in range(N):
            cell = cy * N + cx
            grid[idx(cx, cy)] = mesh.node(0, cell, 0)
            grid[idx(cx + 1, cy)] = mesh.node(0, cell, 1)
            grid[idx(cx + 1, cy + 1)] = mesh.node(0, cell, 2)
            grid[idx(cx, cy + 1)] = mesh.node(0, cell, 3)
    fes = pyrucast.FiniteElementSpace(mesh)

    # ── Modèle : conduction (volume) + convection (bord droit x = 1) ─────────
    right_edge = pyrucast.Mesh(c, "SEG2")
    for j in range(N):
        right_edge.unit().add_cell([grid[idx(N, j)], grid[idx(N, j + 1)]])
    right_fes = pyrucast.FiniteElementSpace(right_edge)

    # The film is built against the conduction it cools: that conduction assembles
    # "T" and "q", and gives it its thermal kind.
    conduction = pyrucast.model.heat_conduction(fes)
    model = conduction | pyrucast.model.boundary_transfer(
        right_fes, conduction, [("T", "q")]
    )

    # ── Chargement ───────────────────────────────────────────────────────────
    # Source: uniform flux (density Q) on the left edge. It is a term of the model
    # like any other.
    left_edge = pyrucast.Mesh(c, "SEG2")
    for j in range(N):
        left_edge.unit().add_cell([grid[idx(0, j)], grid[idx(0, j + 1)]])
    left_fes = pyrucast.FiniteElementSpace(left_edge)
    model = model | pyrucast.model.flux(left_fes, model, "q")

    # Material: k for the conduction, h and the ambient for the convection, the
    # density for the source (each sub-model takes what it requires).
    materials = pyrucast.element_field.material_field(
        model, [("k", K), ("h_T", H), ("a_ext_T", T_EXT), ("phi_q", Q)]
    )

    # Both given terms — the source and the convection's external part h·T_ext —
    # belong to the model, which returns them together.

    rhs = pyrucast.node_field.external_forces(model, materials)

    # ── Assembly + solve (K made definite by the film term) ──────────────────
    K_mat = pyrucast.matrix.stiffness(model, materials)
    solution = pyrucast.solver.solve(K_mat, rhs)

    # ── Compared with the analytical T(x) = T_ext + Q/h + (Q/k)(1 - x), ∀ y ──
    tol = 1e-9
    max_err = 0.0
    for j in range(N + 1):
        for i in range(N + 1):
            x = i * h
            expected = T_EXT + Q / H + (Q / K) * (1.0 - x)
            got = solution.value(grid[idx(i, j)], "T")
            max_err = max(max_err, abs(got - expected))
            assert abs(got - expected) < tol, f"({x},{j * h}): {got} != {expected}"

    print(f"{'x':>6} {'T_calc':>12} {'T_exact':>12}")
    for i in range(N + 1):
        x = i * h
        got = solution.value(grid[idx(i, 0)], "T")
        print(f"{x:6.3f} {got:12.6f} {T_EXT + Q / H + (Q / K) * (1.0 - x):12.6f}")
    print(f"\nmax error over the whole grid = {max_err:.2e}")

    # Balance: all the flux leaves by convection ⇒ T(x=1) = T_ext + Q/h.
    t_right = solution.value(grid[idx(N, 0)], "T")
    print(f"T(x=1) = {t_right:.6f}  (attendu {T_EXT + Q / H})")
    assert abs(t_right - (T_EXT + Q / H)) < tol

    print("\nOK: field independent of y and matching the analytical solution.")


if __name__ == "__main__":
    main()
