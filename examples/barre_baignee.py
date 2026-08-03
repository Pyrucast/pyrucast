"""Barre « baignée » dans un volume : contrainte embedded.

Un cube HEX8 en conduction thermique, ses huit coins fixés à un champ linéaire
`T(x) = 1 + 2x + 3y + 4z` — que l'interpolation trilinéaire du HEX8 reproduit
exactement à l'intérieur. Un nœud immergé au cœur du cube est lié à l'hôte par
une contrainte `Model.embedded` : sa température résolue égale l'interpolation de
l'hôte au même point, sans que les deux maillages partagent de nœud.

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
    base = pyrucast.Model.heat_conduction(fes)

    # Coins fixés au champ linéaire (Dirichlet).
    corner_mesh = pyrucast.Mesh.poi1_from_nodes(corner_nodes)
    corner_mult = pyrucast.mesh.barycenter(corner_mesh)
    dirichlet = pyrucast.Model.dirichlet("T", "q", corner_mesh, corner_mult)

    # Nœud immergé au cœur du cube, lié à l'hôte.
    p = c.add_node([0.3, 0.6, 0.2])
    bar = pyrucast.Mesh.poi1_from_nodes([p])
    embedded = pyrucast.Model.embedded(bar, host, [("T", "q")])

    model = base | dirichlet | embedded
    materials = pyrucast.element_field.material_field(model, [("k", 1.0)])

    # Chargement : valeur du champ à chaque coin (Dirichlet) ; g = 0 au nœud
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
