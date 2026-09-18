"""Multi-point constraint (MPC) — a linear relation through Lagrange multipliers.

Physique
--------
1-D conduction `-u'' = 0` on `[0, 1]` (a SEG2 bar, `k = 1`), whose analytical
solution is linear. MPCs impose a relation `Σ aₖ·u(nodeₖ, varₖ) = g` on the
**same** augmented system as Dirichlet — they are its generalization to
several terms (Dirichlet = a one-term relation, coefficient 1).

Mise en donnée
--------------
Every term is a tuple `(POI1 mesh, variable, dual, coefficient)`. The meshes
are paired element by element: relation `r` ties the `r`-th cell of each
term-mesh to the `r`-th multiplier node. The dual is found with
`model.dual_of(variable)`. The right-hand side `g` is written by the user in
the load field, at the `mpc_rhs` component of the
multiplicateur (défaut `g = 0`).

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
    dual = base.dual_of("T")  # "q"

    # Dirichlet T(0) = 0.
    imposed0 = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    mult0 = pyrucast.mesh.barycenter(imposed0)
    dirichlet = pyrucast.model.dirichlet(base, "T", imposed0, mult0)

    # MPC 1·T(node_last) − 1·T(node_0) = 1.
    mesh_last = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
    mesh_first = pyrucast.mesh.poi1_from_nodes([nodes[0]])
    mult_mpc = pyrucast.mesh.barycenter(mesh_last)
    mpc = pyrucast.model.mpc(
        base,
        [(mesh_last, "T", 1.0), (mesh_first, "T", -1.0)],
        mult_mpc,
    )

    model = base | dirichlet | mpc

    # Charge : valeur imposée de Dirichlet + second membre g de la MPC. Le helper
    # `constraint_rhs` builds each right-hand side from a node designating the
    # relation: the constrained node for Dirichlet, a term node for the MPC.
    # It finds the multiplier node and the imposed component on its own
    # (`imposed_T`, `mpc_rhs`). Both are merged with `|`.
    rhs = dirichlet.constraint_rhs([(nodes[0], 0.0)]) | mpc.constraint_rhs(
        [(nodes[-1], 1.0)]
    )

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)

    print("x      T(x)   attendu")
    for i, node in enumerate(nodes):
        print(f"{i * H:.3f}  {solution.value(node, 'T'):.4f}  {i * H:.3f}")
    lam = solution.value(mult_mpc.node(0, 0, 0), "lambda_mpc")
    print(f"\nmultiplicateur MPC (réaction) : {lam:+.4f}")


if __name__ == "__main__":
    main()
