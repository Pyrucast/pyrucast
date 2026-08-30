"""Barre « baignée » dans un volume élastique : contrainte embedded vectorielle.

Le cas qui motive le baignage : un nœud immergé suit les **déplacements** de
l'interpolation volumique, en x, y **et** z. Un cube HEX8 en traction uniaxiale
(élasticité 3-D) a le champ linéaire `u_x = (S/E)x`, `u_y = −(νS/E)y`,
`u_z = −(νS/E)z` ; un nœud immergé au cœur, lié à l'hôte par
`model.embedded(..., [("u_x","f_x"), ("u_y","f_y"), ("u_z","f_z")])`, retrouve ce
champ à sa position — sans que la barre et le volume partagent de nœud.

Lancer : `python examples/barre_baignee_elastique.py` (après `maturin develop`).
"""

import pyrucast

E = 210.0  # module d'Young
NU = 0.3  # coefficient de Poisson
S = 2.0  # traction sur la face x = 1

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


def main():
    c = pyrucast.Coords(dim=3)
    nodes = [c.add_node(x) for x in CORNERS]

    # Hôte : cube HEX8 en élasticité 3-D.
    host = pyrucast.Mesh(c, "HEX8")
    host.unit().add_cell(nodes)
    fes = pyrucast.FiniteElementSpace(host)
    model = pyrucast.model.elasticity(fes, "full_3d")

    # Appuis de symétrie sur les trois faces passant par l'origine.
    def clamp(ids, var, dual):
        picked = [nodes[i] for i in ids]
        imposed = pyrucast.mesh.poi1_from_nodes(picked)
        mult = pyrucast.mesh.barycenter(imposed)
        return pyrucast.model.dirichlet(var, dual, imposed, mult)

    model = model | clamp([0, 3, 4, 7], "u_x", "f_x")  # face x = 0
    model = model | clamp([0, 1, 4, 5], "u_y", "f_y")  # face y = 0
    model = model | clamp([0, 1, 2, 3], "u_z", "f_z")  # face z = 0

    # Nœud immergé au cœur du cube, lié en u_x/u_y/u_z (liaison rigide, g = 0).
    pc = [0.4, 0.7, 0.2]
    p = c.add_node(pc)
    bar = pyrucast.mesh.poi1_from_nodes([p])
    embedded = pyrucast.model.embedded(
        bar,
        host,
        [("u_x", "f_x"), ("u_y", "f_y"), ("u_z", "f_z")],
    )
    model = model | embedded

    materials = pyrucast.element_field.material_field(model, [("E", E), ("nu", NU)])

    # Traction S sur la face x = 1 (QUA4 [1, 2, 6, 5]) → charges nodales cohérentes.
    face = pyrucast.Mesh(c, "QUA4")
    face.unit().add_cell([nodes[1], nodes[2], nodes[6], nodes[5]])
    face_fes = pyrucast.FiniteElementSpace(face)
    rhs = pyrucast.node_field.flux(face_fes, S, "f_x")

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)

    ux = solution.value(p, "u_x")
    uy = solution.value(p, "u_y")
    uz = solution.value(p, "u_z")
    exp = (S / E * pc[0], -NU * S / E * pc[1], -NU * S / E * pc[2])
    print(f"u(nœud immergé) = ({ux:.6e}, {uy:.6e}, {uz:.6e})")
    print(f"attendu         = ({exp[0]:.6e}, {exp[1]:.6e}, {exp[2]:.6e})")
    assert abs(ux - exp[0]) < 1e-9
    assert abs(uy - exp[1]) < 1e-9
    assert abs(uz - exp[2]) < 1e-9
    print("OK : le nœud immergé suit le champ de déplacement volumique (x, y, z).")


if __name__ == "__main__":
    main()
