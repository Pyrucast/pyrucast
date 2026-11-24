"""vibr10 — response of a point mass to a rotating unbalance, on a modal basis.

Problem: a point of mass M1 = 40 kg held by a stiffness K1 = 4e5 N/m along y
and along z is excited by an unbalance m·d = 1e-4 kg·m turning at
Ω = 19.0985 rad/s. The point describes an ellipse — a circle here, the system
being isotropic — of semi-axis A = m·d·Ω² / (K1 − Ω²·M1) on y and −A on z.

Modal basis: two modes written by hand, as the `.dgibi` file does: the shapes
u_y = 1 and u_z = 1, of eigenvalue K1/M1 and generalized mass M1, each on a
node of its own. The mass, the stiffness and the unbalance force are projected
on them, the projected system is solved at Ω, and the response recombined.
Without damping, the two parts of the rotating force do not couple: the part in
phase with the sine gives the response the quantities read, the other one its
imaginary part.

Running::

    python validation/12_dynamics/vibr10/run.py
"""

import pyrucast as pc

M1, K1 = 40.0, 4e5
UNBALANCE = 1e-4  # m·d, kg·m
OMEGA = 19.0985  # rad/s


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(3)
    p1 = coords.add_node([1.0, 0.0, 0.0])
    point = pc.mesh.poi1_from_nodes([p1])

    def diagonal(value: float) -> pc.Matrix:
        matrix = pc.Matrix.block(
            point, point, ["f_y", "f_z"], ["u_y", "u_z"], symmetry="full"
        )
        for dual, primal in (("f_y", "u_y"), ("f_z", "u_z")):
            matrix[0].add_entry(p1, dual, p1, primal, value)
        matrix.assemble()
        return matrix

    mass, stiffness = diagonal(M1), diagonal(K1)

    # The two modes, each on a node of its own, sharing their support.
    nodes = [coords.add_node([0.0, 0.0, 0.0]) for _ in range(2)]
    support = pc.mesh.poi1_from_nodes(nodes)
    shapes = [
        pc.NodeField(point, [("u_y", 1.0), ("u_z", 0.0)]),
        pc.NodeField(point, [("u_y", 0.0), ("u_z", 1.0)]),
    ]
    modes = [
        {
            "node": node,
            "support": support,
            "eigenvalue": K1 / M1,
            "shape": shape,
            "pairs": [("u_y", "f_y"), ("u_z", "f_z")],
            "generalized_mass": M1,
        }
        for node, shape in zip(nodes, shapes)
    ]

    force = UNBALANCE * OMEGA**2
    unbalance = pc.NodeField(point, [("f_y", force), ("f_z", -force)])
    k_star = pc.matrix.project_on_basis(stiffness, modes)
    m_star = pc.matrix.project_on_basis(mass, modes)
    f_star = pc.node_field.project_on_basis(unbalance, modes)
    q = pc.solver.solve(k_star - m_star * OMEGA**2, f_star)
    u = pc.node_field.recombine(q, modes)
    return {"y_axis": u.value(p1, "u_y"), "z_axis": u.value(p1, "u_z")}


if __name__ == "__main__":
    print(main("."))
