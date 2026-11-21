"""ther2 — thermal boundary conditions on a thick cylinder, axisymmetric, QUA4.

Three steady conduction problems on a thick-walled cylinder, each with an
imposed temperature on the inner face:

1. an imposed flux on the outer face (r from 0.0515 to 0.2015, k = 130,
   T_i = 0.57, flux 0.72 entering);
2. a volume heat source Q = 20 (r from 6 to 16, k = 1, T_i = 100, zero
   flux on the outer face);
3. a convection h = 5 towards T_ext = 125 on the outer face (r from 6 to 16,
   k = 8, T_i = 100).

References: the analytical temperatures at the outer corner P3, evaluated by
the Cast3M deck.

Meshes: case 1 is the deck's row of 10 QUA4. Cases 2 and 3 are free meshes in
the deck, from boundary lines of 8, 6, 9, 4 and 8, 5, 9, 6 elements; here
pyrucast paves the same boundary lines, so the inside differs from Cast3M's
and case 2's odd boundary leaves one triangle.

Running::

    python validation/14_thermal/ther2/run.py
"""

import os

import pyrucast as pc

# Case 1: imposed flux.
R_IN_1, R_OUT_1, HEIGHT_1, N_1 = 0.0515, 0.2015, 0.5, 10
K_1, T_IN_1, FLUX_1 = 130.0, 0.57, 0.72
# Cases 2 and 3: the section r from 6 to 16, z from 0 to 4.
R_IN, R_OUT, HEIGHT = 6.0, 16.0, 4.0
K_2, T_IN_2, SOURCE_2 = 1.0, 100.0, 20.0
K_3, T_IN_3, H_3, T_EXT_3 = 8.0, 100.0, 5.0, 125.0


def main(out: str | None = None) -> dict[str, float]:
    # ── Case 1: imposed temperature inside, imposed flux outside ────────────
    coords = pc.Coords.axisymmetric()
    p1 = coords.add_node([R_IN_1, 0.0])
    p3 = coords.add_node([R_OUT_1, 0.0])
    row = pc.mesh.extrude(pc.mesh.line(p1, p3, N_1), [0.0, HEIGHT_1], 1)
    inner = pc.mesh.points_on_line(row, [R_IN_1, 0.0], [R_IN_1, HEIGHT_1])
    outer_edge = pc.mesh.elements_on(
        pc.mesh.border(row),
        pc.mesh.points_on_line(row, [R_OUT_1, 0.0], [R_OUT_1, HEIGHT_1]),
    )
    conduction = pc.model.heat_conduction(pc.FiniteElementSpace(row))
    blocked = pc.model.dirichlet(inner, conduction, "T")
    model = (
        conduction
        | blocked
        | pc.model.flux(pc.FiniteElementSpace(outer_edge), conduction, "q")
    )
    materials = pc.element_field.material_field(model, [("k", K_1), ("phi_q", FLUX_1)])
    imposed = pc.NodeField(blocked.multiplier_mesh(), ["imposed"])
    imposed[0].add_to_component("imposed", T_IN_1)
    rhs = pc.node_field.external_forces(model, materials) | imposed
    t1 = pc.solver.solve(pc.matrix.stiffness(model, materials), rhs)
    t_case_1 = t1.value(p3, "T")

    # ── Cases 2 and 3: the free meshes of the deck's boundary lines ─────────
    results = []
    for n_right, n_left in ((6, 4), (5, 6)):
        coords = pc.Coords.axisymmetric()
        a = coords.add_node([R_IN, 0.0])
        b = coords.add_node([R_OUT, 0.0])
        c = coords.add_node([R_OUT, HEIGHT])
        d = coords.add_node([R_IN, HEIGHT])
        right = pc.mesh.line(b, c, n_right)
        left = pc.mesh.line(d, a, n_left)
        contour = pc.mesh.consolidate(
            pc.mesh.line(a, b, 8) | right | pc.mesh.line(c, d, 9) | left
        )
        section = pc.mesh.pave_surface(contour, "QUA4")
        results.append((section, right, pc.mesh.consolidate(left.to_poi1()), c))

    section, _right, inner, p3 = results[0]
    fes = pc.FiniteElementSpace(section)
    conduction = pc.model.heat_conduction(fes)
    blocked = pc.model.dirichlet(inner, conduction, "T")
    model = conduction | blocked | pc.model.flux(fes, conduction, "q")
    materials = pc.element_field.material_field(
        model, [("k", K_2), ("phi_q", SOURCE_2)]
    )
    imposed = pc.NodeField(blocked.multiplier_mesh(), ["imposed"])
    imposed[0].add_to_component("imposed", T_IN_2)
    rhs = pc.node_field.external_forces(model, materials) | imposed
    t2 = pc.solver.solve(pc.matrix.stiffness(model, materials), rhs)
    t_case_2 = t2.value(p3, "T")

    section_3, right, inner, p3 = results[1]
    conduction = pc.model.heat_conduction(pc.FiniteElementSpace(section_3))
    blocked = pc.model.dirichlet(inner, conduction, "T")
    model = (
        conduction
        | blocked
        | pc.model.boundary_transfer(pc.FiniteElementSpace(right), conduction, ["T"])
    )
    materials = pc.element_field.material_field(
        model, [("k", K_3), ("h_T", H_3), ("a_ext_T", T_EXT_3)]
    )
    imposed = pc.NodeField(blocked.multiplier_mesh(), ["imposed"])
    imposed[0].add_to_component("imposed", T_IN_3)
    rhs = pc.node_field.external_forces(model, materials) | imposed
    t3 = pc.solver.solve(pc.matrix.stiffness(model, materials), rhs)
    t_case_3 = t3.value(p3, "T")

    if out:
        for name, mesh, field in (
            ("case_1", row, t1),
            ("case_2", section, t2),
            ("case_3", section_3, t3),
        ):
            mesh.plot(
                title=f"ther2 — {name.replace('_', ' ')}, T (°C)",
                field=field,
                component="T",
                cmap="viridis",
                save=os.path.join(out, f"T_{name}.svg"),
            )
    return {"T_1": t_case_1, "T_2": t_case_2, "T_3": t_case_3}


if __name__ == "__main__":
    print(main("."))
