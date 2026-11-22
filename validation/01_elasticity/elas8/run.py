"""elas8 — elliptic membrane under an outward pressure (NAFEMS LE1), plane stress.

Problem: a quarter of the membrane between the ellipses x²/2² + y²/1² = 1 and
x²/3.25² + y²/2.75² = 1, symmetric about x = 0 and y = 0, is pulled by a
uniform outward pressure of 10 MPa on its outer edge. Reference:
σ_yy = 92.7 MPa at the point D (2, 0).

The QUA8 mesh has the `.dgibi` file's boundary lines: 6 elements along each ellipse
(a node at every 15° of the ellipse's parameter), 4 across, the straight
sides cut at 1.583 on x = 0 and at 2.417 on y = 0, filled by `transfinite`
as the `.dgibi` file's mesher does. pyrucast has no ellipse, nor the projection of a
circle onto a plane that the `.dgibi` file uses to draw one: the nodes of the two
elliptic lines are placed here. `transfinite` puts the QUA8 mid-nodes on the
chords; the `.dgibi` file's are on the ellipses. The QUA4 patch and the outer line
are made quadratic together, so that they share their mid-nodes.

σ_yy at D is what `to_nodes` gives there: D belongs to a single cell, whose
Gauss values are fitted over its nodes.

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
Y_CUT, X_CUT = 1.583, 2.417  # where the straight sides change their line


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(2)
    # The two elliptic lines, counter-clockwise around the membrane: the outer
    # one from y = 0 to x = 0, the inner one from x = 0 to y = 0.
    outer_nodes, inner_nodes = [], []
    for k in range(N_ARC + 1):
        phi = math.pi / 2.0 * k / N_ARC
        outer_nodes.append(
            coords.add_node([A_OUT * math.cos(phi), B_OUT * math.sin(phi)])
        )
        inner_nodes.append(
            coords.add_node([A_IN * math.sin(phi), B_IN * math.cos(phi)])
        )
    outer = pc.Mesh(coords, "SEG2")
    inner = pc.Mesh(coords, "SEG2")
    for k in range(N_ARC):
        outer.unit().add_cell([outer_nodes[k], outer_nodes[k + 1]])
        inner.unit().add_cell([inner_nodes[k], inner_nodes[k + 1]])
    pc_, pb = outer_nodes[0], outer_nodes[-1]
    pa, pd = inner_nodes[0], inner_nodes[-1]
    cut_y0 = coords.add_node([X_CUT, 0.0])
    cut_x0 = coords.add_node([0.0, Y_CUT])
    side_y0 = pc.mesh.consolidate(
        pc.mesh.line(pd, cut_y0, 2) | pc.mesh.line(cut_y0, pc_, 2)
    )
    side_x0 = pc.mesh.consolidate(
        pc.mesh.line(pb, cut_x0, 2) | pc.mesh.line(cut_x0, pa, 2)
    )
    patch = pc.mesh.transfinite(side_y0, outer, side_x0, inner)
    quadratic = pc.mesh.to_quadratic(patch | outer)
    membrane, outer_edge = pc.Mesh.empty(), pc.Mesh.empty()
    membrane.add_sub(quadratic[0])
    outer_edge.add_sub(quadratic[1])

    fes = pc.FiniteElementSpace(membrane)
    elasticity = pc.model.elasticity(fes, "plane_stress")
    model = (
        elasticity
        | pc.model.dirichlet(
            pc.mesh.points_on_line(membrane, [0.0, 0.0], [0.0, 1.0]), elasticity, "u_x"
        )
        | pc.model.dirichlet(
            pc.mesh.points_on_line(membrane, [0.0, 0.0], [1.0, 0.0]), elasticity, "u_y"
        )
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
    at_d = pc.node_field.to_nodes(stress)

    if out:
        membrane.plot(title="elas8 — QUA8 mesh", save=os.path.join(out, "mesh.svg"))
        membrane.plot(
            title="elas8 — σ_yy (MPa)",
            field=stress * 1e-6,
            component="sigma_yy",
            cmap="viridis",
            save=os.path.join(out, "sigma_yy.svg"),
        )
    return {
        "syy_D": at_d.value(pd, "sigma_yy") / 1e6,
        "uy_B": u.value(pb, "u_y"),
    }


if __name__ == "__main__":
    print(main("."))
