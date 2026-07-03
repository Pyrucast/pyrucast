"""Contrainte multi-points (MPC) — relation linéaire par multiplicateurs de Lagrange.

Physique
--------
Conduction 1-D `-u'' = 0` sur `[0, 1]` (barre SEG2, `k = 1`), dont la solution
analytique est linéaire. Les MPC imposent une relation `Σ aₖ·u(nœudₖ, varₖ) = g`
sur le **même** système augmenté que Dirichlet — c'en est la généralisation à
plusieurs termes (Dirichlet = relation à un seul terme, coefficient 1).

Mise en donnée
--------------
Chaque terme est un tuple `(maillage POI1, variable, dual, coefficient)`. Les
maillages sont appariés élément-par-élément : la relation `r` relie la `r`-ème
cellule de chaque terme-maillage au `r`-ème nœud multiplicateur. Le dual se
trouve avec `model.dual_of(variable)`. Le second membre `g` est écrit par
l'utilisateur dans le champ de charge, à la composante `mpc_rhs` du nœud
multiplicateur (défaut `g = 0`).

Ici : Dirichlet `T(0) = 0` + MPC `1·T(1) − 1·T(0) = 1`. La relation impose donc
`T(1) = 1`, et la conduction sans source complète en `u(x) = x`.

Lancement ::

    python examples/mpc_periodicite.py
"""

import pyrucast

N_ELEMS = 4
H = 1.0 / N_ELEMS


def main():
    c = pyrucast.Coords(1)
    nodes = [c.add_node([i * H]) for i in range(N_ELEMS + 1)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(N_ELEMS):
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    fes = pyrucast.FiniteElementSpace(mesh)
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", 1.0)

    base = pyrucast.Model.heat_conduction(fes)
    dual = base.dual_of("T")  # "q"

    # Dirichlet T(0) = 0.
    imposed0 = pyrucast.poi1_from_nodes([nodes[0]])
    mult0 = pyrucast.barycenter(imposed0)
    dirichlet = pyrucast.Model.dirichlet("T", dual, imposed0, mult0)

    # MPC 1·T(node_last) − 1·T(node_0) = 1.
    mesh_last = pyrucast.poi1_from_nodes([nodes[-1]])
    mesh_first = pyrucast.poi1_from_nodes([nodes[0]])
    mult_mpc = pyrucast.barycenter(mesh_last)
    mpc = pyrucast.Model.mpc(
        [(mesh_last, "T", dual, 1.0), (mesh_first, "T", dual, -1.0)],
        mult_mpc,
    )

    model = base | dirichlet | mpc

    # Charge : valeur imposée de Dirichlet + second membre g de la MPC. Le helper
    # `constraint_rhs` construit chaque second membre à partir d'un nœud désignant
    # la relation : le nœud contraint pour Dirichlet, un nœud-terme pour la MPC.
    # Il retrouve seul le nœud multiplicateur et la composante imposée
    # (`imposed_T`, `mpc_rhs`). On fusionne les deux avec `|`.
    rhs = dirichlet.constraint_rhs([(nodes[0], 0.0)]) | mpc.constraint_rhs(
        [(nodes[-1], 1.0)]
    )

    solution = pyrucast.solve(pyrucast.stiffness(model, materials), rhs)

    print("x      T(x)   attendu")
    for i, node in enumerate(nodes):
        print(f"{i * H:.3f}  {solution.value(node, 'T'):.4f}  {i * H:.3f}")
    lam = solution.value(mult_mpc.node(0, 0, 0), "lambda_mpc")
    print(f"\nmultiplicateur MPC (réaction) : {lam:+.4f}")


if __name__ == "__main__":
    main()
