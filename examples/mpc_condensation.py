"""Multi-point constraint (MPC) — imposed by condensation (master/slave elimination).

Physique
--------
1-D conduction `-u'' = 0` on `[0, 1]` (a SEG2 bar, `k = 1`). The same
constraints as the Lagrange path are imposed, but by **elimination**: each
relation `Σ aₖ·u(nœudₖ, varₖ) = g` élimine un degré de liberté *esclave*
(`u_s = (g − Σ_{k≠s} aₖ·u_k)/a_s`). The system solved is **reduced** (no
multiplier degree) and definite, `K̂ û = f̂` with `K̂ = Tᵀ K T`, then extended
`u = T·û + u₀`.

Two paths to the same result
--------------------------------
`solve(K, rhs)` solves the system augmented by Lagrange multipliers;
`solve_eliminate(K, model, rhs)` condenses the constraints. Both produce the
**same** field. The elimination recovers, as a bonus, the *reaction* (the
multiplier's equivalent) in each slave's dual row.

Mise en donnée
--------------
Ici, un cas **non chaîné** (esclaves disjoints, le périmètre v1) : Dirichlet
`T(node0) = 0` + MPC `2·T(node4) − 1·T(node2) = 1.5`. The two-term relation
injects reactions at both nodes, so the field minimizes the energy under
constraint (it is not simply `u = x`).

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
        base,
        [(mesh4, "T", 2.0), (mesh2, "T", -1.0)],
        mult_mpc,
    )

    model = base | dirichlet | mpc

    # Load: right-hand sides through the `constraint_rhs` helper, merged with `|`.
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

    # The relation holds exactly on the condensed field.
    t2 = elimination.value(nodes[2], "T")
    t4 = elimination.value(nodes[4], "T")
    print(f"\nrelation : 2·T(node4) − T(node2) = {2.0 * t4 - t2:.4f}  (attendu 1.5)")

    # Reactions (the multipliers' equivalent), read at the slaves' dual row at the
    # constrained nodes.
    print(f"réaction node0 (Dirichlet) : {elimination.value(nodes[0], dual):+.4f}")
    print(f"réaction node4 (MPC)       : {elimination.value(nodes[4], dual):+.4f}")


if __name__ == "__main__":
    main()
