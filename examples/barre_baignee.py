"""A bar "immersed" in a volume: an embedded constraint.

A HEX8 cube in thermal conduction, its eight corners fixed to a linear field
`T(x) = 1 + 2x + 3y + 4z` — which the HEX8's trilinear interpolation reproduces
exactly inside. A node immersed at the cube's core is tied to the host by a
`model.embedded` constraint: its solved temperature equals the host's
interpolation at the same point, without the two meshes sharing a node.

Lancer : `python examples/barre_baignee.py` (après `maturin develop`).
"""

import pyrucast

CORNERS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
]


def field(c):
    return 1.0 + 2.0 * c[0] + 3.0 * c[1] + 4.0 * c[2]


def main():
    c = pyrucast.Coords(dim=3)
    corner_nodes = [c.add_node(x) for x in CORNERS]

    # Hôte HEX8 en conduction thermique (k = 1).
    host = pyrucast.Mesh(c, "HEX8")
    host.unit().add_cell(corner_nodes)
    fes = pyrucast.FiniteElementSpace(host)
    base = pyrucast.model.heat_conduction(fes)

    # Corners fixed to the linear field (Dirichlet).
    corner_mesh = pyrucast.mesh.poi1_from_nodes(corner_nodes)
    corner_mult = pyrucast.mesh.barycenter(corner_mesh)
    dirichlet = pyrucast.model.dirichlet(base, "T", corner_mesh, corner_mult)

    # Node immersed at the cube's core, tied to the host.
    p = c.add_node([0.3, 0.6, 0.2])
    bar = pyrucast.mesh.poi1_from_nodes([p])
    embedded = pyrucast.model.embedded(base, bar, host, ["T"])

    model = base | dirichlet | embedded
    materials = pyrucast.element_field.material_field(model, [("k", 1.0)])

    # Loading: the field's value at each corner (Dirichlet); g = 0 at the
    # immergé (liaison rigide, le défaut).
    rhs = dirichlet.constraint_rhs(
        [(n, field(x)) for n, x in zip(corner_nodes, CORNERS)]
    )
    rhs = rhs | embedded.constraint_rhs([(p, 0.0)])

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)

    got = solution.value(p, "T")
    expected = field([0.3, 0.6, 0.2])  # 1 + 0.6 + 1.8 + 0.8 = 4.2
    print(f"T(nœud immergé) = {got:.6f}  (attendu {expected:.6f})")
    assert abs(got - expected) < 1e-9
    print("OK : le nœud immergé suit l'interpolation trilinéaire de l'hôte.")


if __name__ == "__main__":
    main()
