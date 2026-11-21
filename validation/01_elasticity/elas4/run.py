"""elas4 — Z-section cantilever in torsion (NAFEMS LE5), Kirchhoff shell on QUA4.

Problem: a cantilever 10 m long whose cross-section is a Z (two flanges of
1 m joined by a web of 2 m), clamped at x = 0, is twisted at x = 10 by two
opposite forces of 600 kN spread over the flanges (1.2 MN·m). Reference: the
axial stress on the mid-surface at A (2.5, 0, 0) is σ_xx = −108 MPa.

The mesh is the Cast3M `.dgibi` file's: the section line (2 elements per segment)
extruded along x in 16 layers, 96 QUA4. Each cell's first edge runs along
the section line, so the local y axis of the shell is the beam's axis and
the axial stress on the mid-surface is N_yy / h, as the `.dgibi` file reads SMYY. A
lies on the edge of the first flange, between two cells; `project` reads it
in one of them, from that cell's Gauss points.

Running::

    python validation/01_elasticity/elas4/run.py
"""

import os

import pyrucast as pc

E, NU, THICKNESS = 210e9, 0.3, 0.1
LENGTH, N_LENGTH = 10.0, 16
# F_z at x = 10: −600 kN on the first flange, +600 kN on the second.
END_FORCES = [
    ((0.0, 0.0), -150e3),
    ((0.0, 0.5), -300e3),
    ((0.0, 1.0), -150e3),
    ((2.0, 1.0), 150e3),
    ((2.0, 1.5), 300e3),
    ((2.0, 2.0), 150e3),
]  # ((y, z), F_z)
X_A = 2.5


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(3)
    p1 = coords.add_node([0.0, 0.0, 0.0])
    p2 = coords.add_node([0.0, 0.0, 1.0])
    p3 = coords.add_node([0.0, 2.0, 1.0])
    p4 = coords.add_node([0.0, 2.0, 2.0])
    section = pc.mesh.consolidate(
        pc.mesh.line(p1, p2, 2) | pc.mesh.line(p2, p3, 2) | pc.mesh.line(p3, p4, 2)
    )
    surface = pc.mesh.extrude(section, [LENGTH, 0.0, 0.0], N_LENGTH)
    clamped = pc.mesh.points_on_plane(surface, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    loaded = [surface.nearest_node([LENGTH, y, z]) for (y, z), _ in END_FORCES]
    point_a = surface.nearest_node([X_A, 0.0, 0.0])
    tip = loaded[0]

    shell = pc.model.shell(pc.FiniteElementSpace(surface), "kirchhoff")
    model = shell
    for variable in ("u_x", "u_y", "u_z"):  # displacements only, as the `.dgibi` file
        model = model | pc.model.dirichlet(clamped, shell, variable)
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("h", THICKNESS)]
    )
    load = pc.NodeField(pc.mesh.poi1_from_nodes(loaded), ["f_z"])
    for node, (_, force) in zip(loaded, END_FORCES):
        load[0].set_value(node, "f_z", force)
    u = pc.solver.solve(pc.matrix.stiffness(model, materials), load)

    strain = pc.element_field.strain(model, u, materials)
    forces = pc.element_field.integrate_behavior(
        model, strain, materials, pc.element_field.initial_state(model, materials)
    )
    at_a = pc.node_field.project(forces, surface, pc.mesh.poi1_from_nodes([point_a]))

    if out:
        surface.plot(title="elas4 — QUA4 mesh", save=os.path.join(out, "mesh.svg"))
        surface.plot(
            title="elas4 — axial stress N_yy / h (MPa)",
            field=forces * (1e-6 / THICKNESS),
            component="N_yy",
            cmap="viridis",
            save=os.path.join(out, "sigma_axial.svg"),
        )
    return {
        "sxx_A": at_a.value(point_a, "N_yy") / THICKNESS / 1e6,
        "uz_tip": u.value(tip, "u_z"),
    }


if __name__ == "__main__":
    print(main("."))
