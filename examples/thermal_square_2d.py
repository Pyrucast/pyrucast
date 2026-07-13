"""Conduction thermique 2-D — carré chauffé, comparé à l'analytique.

Problème
--------
Sur le carré unité [0, 1]² (grille structurée de QUA4) :

  * bord **gauche** (x = 0) : une **source de chaleur** répartie (flux de
    Neumann, total ``Q``) ;
  * bord **droit** (x = 1) : une **température imposée** ``T = 20`` (Dirichlet) ;
  * bords **haut/bas** : aucune condition ⇒ condition naturelle (flux nul,
    bords *isolés*).

Les bords latéraux étant isolés, le champ ne dépend pas de ``y`` : le carré
redonne le profil de la ligne ::

    u(x) = 20 + (Q / k) * (1 - x)

et la réaction totale (somme des multiplicateurs sur le bord imposé) vaut le
flux injecté ``Q``.

Mise en donnée d'un flux réparti
--------------------------------
Il n'y a pas d'opérateur de flux de bord : une source répartie s'applique comme
des **charges nodales cohérentes**. Pour un flux uniforme sur des éléments
linéaires, un nœud intérieur du bord reçoit ``Q*h`` et un coin ``Q*h/2`` (leur
somme vaut ``Q``).

C'est l'équivalent Python du test d'intégration ``tests/thermal_square.rs``.

Lancement ::

    maturin develop --features extension-module
    python examples/thermal_square_2d.py
"""

import pyrucast

# ── Données du problème ──────────────────────────────────────────────────────
K = 1.0  # conductivité
Q = 10.0  # flux de chaleur TOTAL injecté sur le bord gauche
T_IMPOSED = 20.0  # température imposée sur le bord droit
N = 4  # N×N éléments QUA4


def main() -> None:
    h = 1.0 / N

    def idx(i: int, j: int) -> int:
        return j * (N + 1) + i

    # ── Maillage : grille N×N de QUA4 sur [0,1]², par balayage de deux lignes ─
    # SEG2 (bas → haut) avec le mailleur `sweep_qua4` (Cast3m « regle »).
    c = pyrucast.Coords(2)
    bottom = pyrucast.mesher.line_seg2(c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0]), N)
    top = pyrucast.mesher.line_seg2(c.add_node([0.0, 1.0]), c.add_node([1.0, 1.0]), N)
    mesh = pyrucast.mesher.sweep_qua4(bottom, top, N)

    # Nœuds rangés par idx(i, j) (i selon x, j selon y) en relisant la
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

    # ── Dirichlet T = 20 sur le bord droit (x = 1) ───────────────────────────
    right_nodes = [grid[idx(N, j)] for j in range(N + 1)]
    imposed = pyrucast.mesher.poi1_from_nodes(right_nodes)
    multiplier = pyrucast.mesher.barycenter(imposed)
    mults = [multiplier.node(0, j, 0) for j in range(N + 1)]
    model = pyrucast.Model.heat_conduction(fes) | pyrucast.Model.dirichlet(
        "T", "q", imposed, multiplier
    )

    # ── Matériau : k uniforme (Dirichlet ignoré automatiquement) ─────────────
    materials = pyrucast.build.material_field(model, [("k", K)])

    # ── Chargement ───────────────────────────────────────────────────────────
    # Source : flux uniforme (densité Q) sur le bord gauche, transformé en
    # charges nodales cohérentes par l'opérateur `flux` (Cast3m FLUX) — plus de
    # répartition Q*h / Q*h/2 à la main. Le bord est un maillage SEG2 bâti sur
    # les nœuds de la grille (intégré comme une ligne).
    left_edge = pyrucast.Mesh(c, "SEG2")
    for j in range(N):
        left_edge.unit().add_cell([grid[idx(0, j)], grid[idx(0, j + 1)]])
    left_fes = pyrucast.FiniteElementSpace(left_edge)
    source = pyrucast.assemble.flux(left_fes[0], Q, "q")

    # Valeur imposée T = 20 au slot "imposed_T" des nœuds-multiplicateurs.
    imposed_mesh = pyrucast.mesher.poi1_from_nodes(mults)
    imposed_load = pyrucast.NodeField(imposed_mesh, ["imposed_T"])
    for m in mults:
        imposed_load[0].set_value(m, "imposed_T", T_IMPOSED)

    # Chargement = flux du bord + valeurs imposées (union des zones).
    rhs = source | imposed_load

    # ── Assemblage + résolution ──────────────────────────────────────────────
    K_mat = pyrucast.assemble.stiffness(model, materials)
    solution = pyrucast.solver.solve(K_mat, rhs)

    # ── Comparaison à l'analytique u(x) = 20 + (Q/k)(1 - x), ∀ y ─────────────
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
    print(f"\nerreur max sur toute la grille = {max_err:.2e}")

    # La réaction totale équilibre le flux injecté : Σλ = Q.
    reaction = sum(solution.value(m, "lambda_T") for m in mults)
    print(f"réaction totale Σλ = {reaction:.6f}  (attendu {Q})")
    assert abs(reaction - Q) < tol

    print("\nOK : champ indépendant de y et conforme à la solution analytique.")


if __name__ == "__main__":
    main()
