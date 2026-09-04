"""Contrainte multi-points (MPC) — imposition par condensation (élimination maître/esclave).

Physique
--------
Conduction 1-D `-u'' = 0` sur `[0, 1]` (barre SEG2, `k = 1`). On impose les
mêmes contraintes que la voie Lagrange, mais par **élimination** : chaque
relation `Σ aₖ·u(nœudₖ, varₖ) = g` élimine un degré de liberté *esclave*
(`u_s = (g − Σ_{k≠s} aₖ·u_k)/a_s`). Le système résolu est **réduit** (pas de
degré multiplicateur) et défini, `K̂ û = f̂` avec `K̂ = Tᵀ K T`, puis prolongé
`u = T·û + u₀`.

Deux voies pour un même résultat
--------------------------------
`solve(K, rhs)` résout le système augmenté par multiplicateurs de Lagrange ;
`solve_eliminate(K, model, rhs)` condense les contraintes. Les deux produisent le
**même** champ. L'élimination récupère en prime la *réaction* (équivalent du
multiplicateur) dans la ligne duale de chaque esclave.

Mise en donnée
--------------
Ici, un cas **non chaîné** (esclaves disjoints, le périmètre v1) : Dirichlet
`T(node0) = 0` + MPC `2·T(node4) − 1·T(node2) = 1.5`. La relation à deux termes
injecte des réactions aux deux nœuds, donc le champ minimise l'énergie sous
contrainte (il n'est pas simplement `u = x`).

Lancement ::

    python examples/mpc_condensation.py
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

    base = pyrucast.model.heat_conduction(fes)
    dual = base.dual_of("T")  # "q"

    # Dirichlet T(node0) = 0 (esclave node0).
    imposed0 = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    mult0 = pyrucast.mesh.barycenter(imposed0)
    dirichlet = pyrucast.model.dirichlet(base, "T", imposed0, mult0)

    # MPC 2·T(node4) − 1·T(node2) = 1.5 (esclave node4, maître node2 — disjoints).
    mesh4 = pyrucast.mesh.poi1_from_nodes([nodes[4]])
    mesh2 = pyrucast.mesh.poi1_from_nodes([nodes[2]])
    mult_mpc = pyrucast.mesh.barycenter(mesh4)
    mpc = pyrucast.model.mpc(
        [(mesh4, "T", dual, 2.0), (mesh2, "T", dual, -1.0)],
        mult_mpc,
    )

    model = base | dirichlet | mpc

    # Charge : second membres via le helper `constraint_rhs`, fusionnés avec `|`.
    rhs = dirichlet.constraint_rhs([(nodes[0], 0.0)]) | mpc.constraint_rhs(
        [(nodes[4], 1.5)]
    )

    k = pyrucast.matrix.stiffness(model, materials)
    lagrange = pyrucast.solver.solve(k, rhs)
    elimination = pyrucast.solver.solve_eliminate(k, model, rhs)

    print("x      T (Lagrange)  T (élimination)")
    for i, node in enumerate(nodes):
        a = lagrange.value(node, "T")
        b = elimination.value(node, "T")
        print(f"{i * H:.3f}  {a:12.6f}  {b:12.6f}")

    # La relation tient exactement sur le champ condensé.
    t2 = elimination.value(nodes[2], "T")
    t4 = elimination.value(nodes[4], "T")
    print(f"\nrelation : 2·T(node4) − T(node2) = {2.0 * t4 - t2:.4f}  (attendu 1.5)")

    # Réactions (équivalent des multiplicateurs), lues à la ligne duale des
    # esclaves aux nœuds contraints.
    print(f"réaction node0 (Dirichlet) : {elimination.value(nodes[0], dual):+.4f}")
    print(f"réaction node4 (MPC)       : {elimination.value(nodes[4], dual):+.4f}")


if __name__ == "__main__":
    main()
