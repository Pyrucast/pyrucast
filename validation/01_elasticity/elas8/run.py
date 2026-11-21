"""elas8 — elliptic membrane under an outward pressure (NAFEMS LE1), plane stress.

Problem: a quarter of the membrane between the ellipses x²/2² + y²/1² = 1 and
x²/3.25² + y²/2.75² = 1, symmetric about x = 0 and y = 0, is pulled by a
uniform outward pressure of 10 MPa on its outer edge. Reference:
σ_yy = 92.7 MPa at the point D (2, 0).

The QUA8 mesh is the one of the Cast3M deck, node for node: 6 elements along
each ellipse (a node at every 15° of the ellipse's parameter, the mid-nodes
on the ellipse), 4 across, the straight sides cut at 1.583 on x = 0 and at
2.417 on y = 0. The corner nodes inside are placed by a transfinite (Coons)
interpolation of the boundary nodes, the inner mid-nodes at the middle of
their edge. `pyrucast.mesh.transfinite` would put the mid-nodes of the
elliptic edges on the chords, so the nodes are laid out here.

σ_yy at D is read in the only cell holding D, from that cell's Gauss points.

Running::

    python validation/01_elasticity/elas8/run.py
"""

import math
import os

import pyrucast as pc

E, NU = 210e9, 0.3  # Pa, —
PRESSURE = 10e6  # outward pressure on the outer edge, Pa
A_IN, B_IN = 2.0, 1.0  # semi-axes of the inner ellipse, m
A_OUT, B_OUT = 3.25, 2.75  # semi-axes of the outer ellipse, m
N_ARC = 6  # elements along each ellipse
N_SIDE = 4  # elements across, along each straight side
Y_SIDE = [1.0, 1.583, 2.75]  # side x = 0: two lines of 2 elements
X_SIDE = [2.0, 2.417, 3.25]  # side y = 0: two lines of 2 elements


def main(out: str | None = None) -> dict[str, float]:
    # Node grid: i along the ellipses from x = 0 (i = 0) to y = 0 (i = NI),
    # j across from the inner ellipse (j = 0) to the outer one (j = NJ);
    # even indices are corners, odd ones mid-nodes.
    ni, nj = 2 * N_ARC, 2 * N_SIDE
    inner, outer, left, right = [], [], [], []
    for i in range(ni + 1):
        phi = math.pi / 2.0 * (1.0 - i / ni)
        inner.append((A_IN * math.cos(phi), B_IN * math.sin(phi)))
        outer.append((A_OUT * math.cos(phi), B_OUT * math.sin(phi)))
    for j in range(nj + 1):
        half, k = divmod(j, N_SIDE)  # which straight line, and where on it
        if half == 2:
            half, k = 1, N_SIDE
        t = k / N_SIDE
        left.append((0.0, Y_SIDE[half] + t * (Y_SIDE[half + 1] - Y_SIDE[half])))
        right.append((X_SIDE[half] + t * (X_SIDE[half + 1] - X_SIDE[half]), 0.0))

    position = {}
    for i in range(0, ni + 1, 2):
        for j in range(0, nj + 1, 2):
            u, v = i / ni, j / nj
            point = []
            for d in range(2):
                coons = (
                    (1 - v) * inner[i][d]
                    + v * outer[i][d]
                    + (1 - u) * left[j][d]
                    + u * right[j][d]
                    - (1 - u) * (1 - v) * inner[0][d]
                    - u * (1 - v) * inner[ni][d]
                    - (1 - u) * v * outer[0][d]
                    - u * v * outer[ni][d]
                )
                point.append(coons)
            position[i, j] = point
    for i in range(ni + 1):
        for j in range(nj + 1):
            if (i % 2, j % 2) == (0, 0) or (i % 2, j % 2) == (1, 1):
                continue
            if j == 0:
                position[i, j] = list(inner[i])
            elif j == nj:
                position[i, j] = list(outer[i])
            elif i == 0:
                position[i, j] = list(left[j])
            elif i == ni:
                position[i, j] = list(right[j])
            elif i % 2 == 1:
                a, b = position[i - 1, j], position[i + 1, j]
                position[i, j] = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
            else:
                a, b = position[i, j - 1], position[i, j + 1]
                position[i, j] = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]

    coords = pc.Coords(2)
    node = {}
    for key in sorted(position):
        node[key] = coords.add_node(position[key])
    membrane = pc.Mesh(coords, "QUA8")
    for i in range(0, ni, 2):
        for j in range(0, nj, 2):
            membrane.unit().add_cell(
                [
                    node[i, j],
                    node[i + 2, j],
                    node[i + 2, j + 2],
                    node[i, j + 2],
                    node[i + 1, j],
                    node[i + 2, j + 1],
                    node[i + 1, j + 2],
                    node[i, j + 1],
                ]
            )
    # Outer edge, travelled counter-clockwise: from y = 0 (i = NI) to x = 0.
    outer_edge = pc.Mesh(coords, "SEG3")
    for i in range(ni, 0, -2):
        outer_edge.unit().add_cell([node[i, nj], node[i - 2, nj], node[i - 1, nj]])
    side_x0 = pc.mesh.poi1_from_nodes([node[0, j] for j in range(nj + 1)])
    side_y0 = pc.mesh.poi1_from_nodes([node[ni, j] for j in range(nj + 1)])
    point_d = node[ni, 0]

    fes = pc.FiniteElementSpace(membrane)
    elasticity = pc.model.elasticity(fes, "plane_stress")
    model = (
        elasticity
        | pc.model.dirichlet(side_x0, elasticity, "u_x")
        | pc.model.dirichlet(side_y0, elasticity, "u_y")
    )
    # An outward pressure on a counter-clockwise contour: φ_n = +p.
    traction = pc.model.flux(pc.FiniteElementSpace(outer_edge), elasticity, "n")
    model = model | traction
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("phi_n", PRESSURE)]
    )
    forces = pc.node_field.external_forces(model, materials)
    u = pc.solver.solve(pc.matrix.stiffness(model, materials), forces)

    strain = pc.element_field.strain(model, u, materials)
    stress = pc.element_field.elastic_stress(model, strain, materials)
    at_d = pc.node_field.project(stress, membrane, pc.mesh.poi1_from_nodes([point_d]))
    syy_d = at_d.value(point_d, "sigma_yy") / 1e6
    uy_b = u.value(node[0, nj], "u_y")

    if out:
        membrane.plot(title="elas8 — QUA8 mesh", save=os.path.join(out, "mesh.svg"))
        membrane.plot(
            title="elas8 — σ_yy (MPa)",
            field=stress * 1e-6,
            component="sigma_yy",
            cmap="viridis",
            save=os.path.join(out, "sigma_yy.svg"),
        )
    return {"syy_D": syy_d, "uy_B": uy_b}


if __name__ == "__main__":
    print(main("."))
