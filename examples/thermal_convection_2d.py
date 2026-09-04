"""Conduction + convection thermique 2-D — comparé à l'analytique.

Problème
--------
Sur le carré unité [0, 1]² (grille structurée de QUA4) :

  * bord **gauche** (x = 0) : une **source de chaleur** répartie (flux de
    Neumann, densité ``Q``) ;
  * bord **droit** (x = 1) : un **échange par convection** (Robin / film)
    ``q·n = h·(T - T_ext)`` avec un fluide à ``T_ext`` ;
  * bords **haut/bas** : aucune condition ⇒ condition naturelle (flux nul,
    bords *isolés*).

Aucun Dirichlet n'est nécessaire : le terme de film ``h ∫ N_i N_j dΓ`` ancre la
température (la conduction pure-Neumann est singulière) et rend la matrice
définie. Les bords latéraux étant isolés, le champ ne dépend pas de ``y`` ::

    T(x) = T_ext + Q/h + (Q/k) * (1 - x)

tout le flux injecté ``Q`` ressortant par convection en x = 1
(``h·(T(1) - T_ext) = Q``).

Mise en donnée de la convection
-------------------------------
Le modèle ``model.boundary_transfer`` fournit la **matrice de film** (variables
``T``/``q`` partagées avec la conduction ⇒ couplage direct). La part externe
``h·T_ext·∫N_i dΓ`` est un **second membre**, bâti avec le même opérateur
son ambiant, comme la source porte sa densité — aucune normale n'est requise.

C'est l'équivalent Python du test d'intégration ``tests/thermal_convection.rs``.

Lancement ::

    maturin develop --features extension-module
    python examples/thermal_convection_2d.py
"""

import pyrucast

# ── Données du problème ──────────────────────────────────────────────────────
K = 2.0  # conductivité
Q = 10.0  # densité de flux injectée sur le bord gauche
H = 5.0  # coefficient d'échange (film) sur le bord droit
T_EXT = 20.0  # température ambiante du fluide
N = 4  # N×N éléments QUA4


def main() -> None:
    h = 1.0 / N

    def idx(i: int, j: int) -> int:
        return j * (N + 1) + i

    # ── Maillage : grille N×N de QUA4 sur [0,1]², par balayage de deux lignes ─
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

    model = pyrucast.model.heat_conduction(fes) | pyrucast.model.boundary_transfer(
        right_fes, [("T", "q")], "thermal"
    )

    # ── Chargement ───────────────────────────────────────────────────────────
    # Source : flux uniforme (densité Q) sur le bord gauche. C'est un terme du
    # modèle comme un autre.
    left_edge = pyrucast.Mesh(c, "SEG2")
    for j in range(N):
        left_edge.unit().add_cell([grid[idx(0, j)], grid[idx(0, j + 1)]])
    left_fes = pyrucast.FiniteElementSpace(left_edge)
    model = model | pyrucast.model.flux(left_fes, "q", "thermal")

    # Matériau : k pour la conduction, h et l'ambiant pour la convection, la
    # densité pour la source (chaque sous-modèle prélève ce qu'il requiert).
    materials = pyrucast.element_field.material_field(
        model, [("k", K), ("h_T", H), ("a_ext_T", T_EXT), ("phi_q", Q)]
    )

    # Les deux termes donnés — la source et la part externe h·T_ext de la
    # convection — appartiennent au modèle, qui les rend ensemble.

    rhs = pyrucast.node_field.external_forces(model, materials)

    # ── Assemblage + résolution (K rendue définie par le terme de film) ──────
    K_mat = pyrucast.matrix.stiffness(model, materials)
    solution = pyrucast.solver.solve(K_mat, rhs)

    # ── Comparaison à l'analytique T(x) = T_ext + Q/h + (Q/k)(1 - x), ∀ y ────
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
    print(f"\nerreur max sur toute la grille = {max_err:.2e}")

    # Bilan : tout le flux ressort par convection ⇒ T(x=1) = T_ext + Q/h.
    t_right = solution.value(grid[idx(N, 0)], "T")
    print(f"T(x=1) = {t_right:.6f}  (attendu {T_EXT + Q / H})")
    assert abs(t_right - (T_EXT + Q / H)) < tol

    print("\nOK : champ indépendant de y et conforme à la solution analytique.")


if __name__ == "__main__":
    main()
