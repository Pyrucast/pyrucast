"""Multi-point constraint (MPC) — a linear relation through Lagrange multipliers.

Physique
--------
1-D conduction `-u'' = 0` on `[0, 1]` (a SEG2 bar, `k = 1`), whose analytical
solution is linear. MPCs impose a relation `Σ aₖ·u(nodeₖ, varₖ) = g` on the
**same** augmented system as Dirichlet — they are its generalization to
several terms (Dirichlet = a one-term relation, coefficient 1).

Mise en donnée
--------------
Every term is a tuple `(POI1 mesh, variable, coefficient)`, all on the one
target model, which gives each variable its dual. The meshes are paired cell by
cell: relation `r` ties the `r`-th cell of each term-mesh, and gets a
multiplier node of its own, minted by the constraint. The right-hand side `g`
is written by the user in the load field, at the `imposed` component of the
multiplier node (default `g = 0`).

Here: Dirichlet `T(0) = 0` + MPC `1·T(1) − 1·T(0) = 1`. The relation therefore
imposes `T(1) = 1`, and source-free conduction completes it into `u(x) = x`.

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

    base = pyrucast.model.heat_conduction(fes)

    # Dirichlet T(0) = 0.
    imposed0 = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    dirichlet = pyrucast.model.dirichlet(imposed0, base, "T")

    # MPC 1·T(node_last) − 1·T(node_0) = 1.
    mesh_last = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
    mesh_first = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    mpc = pyrucast.model.mpc(
        [(mesh_last, "T", 1.0), (mesh_first, "T", -1.0)],
        base,
    )

    model = base | dirichlet | mpc

    # Load: the Dirichlet imposed value + the MPC right-hand side g. The helper
    # `constraint_rhs` builds each right-hand side from a node designating the
    # relation: the constrained node for Dirichlet, a term node for the MPC.
    # It finds the multiplier node on its own and writes the `imposed`
    # component there. Both are merged with `|`.
    rhs = dirichlet.constraint_rhs([(nodes[0], 0.0)]) | mpc.constraint_rhs(
        [(nodes[-1], 1.0)]
    )

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)

    print("x      T(x)   attendu")
    for i, node in enumerate(nodes):
        print(f"{i * H:.3f}  {solution.value(node, 'T'):.4f}  {i * H:.3f}")
    lam = solution.value(mpc.multiplier_mesh().node(0, 0, 0), "lambda")
    print(f"\nmultiplicateur MPC (réaction) : {lam:+.4f}")


if __name__ == "__main__":
    main()
