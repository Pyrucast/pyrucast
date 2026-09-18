"""2-D thermal conduction — a heated square, compared with the analytical solution.

Problème
--------
On the unit square [0, 1]² (a structured QUA4 grid):

  * **left** edge (x = 0): a distributed **heat source** (flux of
    Neumann, total ``Q``) ;
  * **right** edge (x = 1): an **imposed temperature** ``T = 20`` (Dirichlet);
  * bords **haut/bas** : aucune condition ⇒ condition naturelle (flux nul,
    bords *isolés*).

The lateral edges being insulated, the field does not depend on ``y``: the
redonne le profil de la ligne ::

    u(x) = 20 + (Q / k) * (1 - x)

and the total reaction (the sum of the multipliers on the imposed edge) equals
flux injecté ``Q``.

Mise en donnée d'un flux réparti
--------------------------------
There is no border flux operator: a distributed source applies as **consistent
nodal loads**. For a uniform flux on linear elements, an interior node of the
edge receives ``Q*h`` and a corner ``Q*h/2`` (their
somme vaut ``Q``).

This is the Python equivalent of the integration test ``tests/thermal_square.rs``.

Lancement ::

    maturin develop --features extension-module
    python examples/thermal_square_2d.py
"""

import pyrucast

# ── Problem data ────────────────────────────────────────────────────────────
K = 1.0  # conductivité
Q = 10.0  # TOTAL heat flux injected on the left edge
T_IMPOSED = 20.0  # imposed temperature on the right edge
N = 4  # N×N éléments QUA4


def main() -> None:
    h = 1.0 / N

    def idx(i: int, j: int) -> int:
        return j * (N + 1) + i

    # ── Mesh: an N×N grid of QUA4 on [0,1]², by sweeping two lines ───────────
    # SEG2 (bottom → top) with the `sweep` mesher (Cast3m "regle").
    c = pyrucast.Coords(2)
    bottom = pyrucast.mesh.line(c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0]), N)
    top = pyrucast.mesh.line(c.add_node([0.0, 1.0]), c.add_node([1.0, 1.0]), N)
    mesh = pyrucast.mesh.sweep(bottom, top, N)

    # Nodes laid out by idx(i, j) (i along x, j along y) by reading the
    # connectivité QUA4 : maille (cy, cx) = cy*N + cx, nœuds locaux 0..3.
    grid = [None] * ((N + 1) * (N + 1))
    for cy in range(N):
        for cx in range(N):
            cell = cy * N + cx
            grid[idx(cx, cy)] = mesh.node(0, cell, 0)
            grid[idx(cx + 1, cy)] = mesh.node(0, cell, 1)
            grid[idx(cx + 1, cy + 1)] = mesh.node(0, cell, 2)
            grid[idx(cx, cy + 1)] = mesh.node(0, cell, 3)
    fes = pyrucast.FiniteElementSpace(mesh)

    # ── Dirichlet T = 20 on the right edge (x = 1) ───────────────────────────
    right_nodes = [grid[idx(N, j)] for j in range(N + 1)]
    imposed = pyrucast.mesh.poi1_from_nodes(right_nodes)
    multiplier = pyrucast.mesh.barycenter(imposed)
    mults = [multiplier.node(0, j, 0) for j in range(N + 1)]
    cible = pyrucast.model.heat_conduction(fes)

    model = cible | pyrucast.model.dirichlet(cible, "T", imposed, multiplier)

    # ── Matériau : k uniforme (Dirichlet ignoré automatiquement) ─────────────

    # ── Chargement ───────────────────────────────────────────────────────────
    # Source: uniform flux (density Q) on the left edge, turned into consistent
    # nodal loads by the `flux` operator (Cast3m FLUX) — no more Q*h / Q*h/2
    # distribution by hand. The edge is a SEG2 mesh built on the grid's nodes
    # (integrated as a line).
    left_edge = pyrucast.Mesh(c, "SEG2")
    for j in range(N):
        left_edge.unit().add_cell([grid[idx(0, j)], grid[idx(0, j + 1)]])
    left_fes = pyrucast.FiniteElementSpace(left_edge)
    model = model | pyrucast.model.flux(left_fes, model, "q")
    materials = pyrucast.element_field.material_field(model, [("k", K), ("phi_q", Q)])
    source = pyrucast.node_field.external_forces(model, materials)

    # Imposed value T = 20 at the multiplier nodes' "imposed_T" slot.
    imposed_mesh = pyrucast.mesh.poi1_from_nodes(mults)
    imposed_load = pyrucast.NodeField(imposed_mesh, ["imposed_T"])
    for m in mults:
        imposed_load[0].set_value(m, "imposed_T", T_IMPOSED)

    # Loading = the edge's flux + the imposed values (union of the zones).
    rhs = source | imposed_load

    # ── Assemblage + résolution ──────────────────────────────────────────────
    K_mat = pyrucast.matrix.stiffness(model, materials)
    solution = pyrucast.solver.solve(K_mat, rhs)

    # ── Compared with the analytical u(x) = 20 + (Q/k)(1 - x), ∀ y ───────────
    tol = 1e-9
    max_err = 0.0
    for j in range(N + 1):
        for i in range(N + 1):
            x = i * h
            expected = T_IMPOSED + (Q / K) * (1.0 - x)
            got = solution.value(grid[idx(i, j)], "T")
            max_err = max(max_err, abs(got - expected))
            assert abs(got - expected) < tol, f"({x},{j * h}): {got} != {expected}"

    # Profil le long de x (constant en y) :
    print(f"{'x':>6} {'T_calc':>12} {'T_exact':>12}")
    for i in range(N + 1):
        x = i * h
        got = solution.value(grid[idx(i, 0)], "T")
        print(f"{x:6.3f} {got:12.6f} {T_IMPOSED + (Q / K) * (1.0 - x):12.6f}")
    print(f"\nmax error over the whole grid = {max_err:.2e}")

    # La réaction totale équilibre le flux injecté : Σλ = Q.
    reaction = sum(solution.value(m, "lambda_T") for m in mults)
    print(f"réaction totale Σλ = {reaction:.6f}  (attendu {Q})")
    assert abs(reaction - Q) < tol

    print("\nOK: field independent of y and matching the analytical solution.")


if __name__ == "__main__":
    main()
