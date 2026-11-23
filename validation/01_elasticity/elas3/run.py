"""elas3 — pinched cylinder with free ends (NAFEMS ME7), DKT shell.

Problem: a thin cylinder of radius 4.953 and length 2 × 5.175, free at its
ends, is pinched at its mid-length by two opposite point loads. One eighth is
modelled: a quarter of the circumference over half the length, with symmetry
conditions on the cut edges. Reference: u_z = −0.452 under the load.

Mesh: the quarter arc (10 elements, every 9°) extruded along y in 7
layers straight into TRI3, the diagonal of the triangles alternating from one
quadrangle to the next along the arc, as in the `.dgibi` file, with one
difference: the `.dgibi` file lets its mesh density (0.75) place the layers,
slightly closer towards the loaded section, where `extrude` spaces them
evenly.

Running::

    python validation/01_elasticity/elas3/run.py
"""

import os

import pyrucast as pc

E, NU, THICKNESS = 10.5e6, 0.3125, 0.094
RADIUS = 4.953
N_ARC = 10
HALF_LENGTH, N_LAYERS = 5.175, 7
FORCE = -100.0  # F_z at the loaded point


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(3)
    centre = coords.add_node([0.0, 0.0, 0.0])
    p1 = coords.add_node([RADIUS, 0.0, 0.0])
    p2 = coords.add_node([0.0, 0.0, RADIUS])
    arc = pc.mesh.arc(p1, centre, p2, N_ARC)
    surface = pc.mesh.extrude(arc, [0.0, HALF_LENGTH, 0.0], N_LAYERS, "TRI3")
    p3 = surface.nearest_node([0.0, HALF_LENGTH, RADIUS])

    plane_z0 = pc.mesh.points_on_plane(surface, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    plane_x0 = pc.mesh.points_on_plane(surface, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    mid_length = pc.mesh.points_on_plane(
        surface, [0.0, HALF_LENGTH, 0.0], [0.0, 1.0, 0.0]
    )

    # The blocked nodes of each unknown on the symmetry planes z = 0
    # (u_z, r_x, r_y), x = 0 (u_x, r_y, r_z) and y = 5.175 (u_y, r_x, r_z).
    # A node on two of these sets is blocked once (`consolidate`).
    blocked = {
        "u_x": plane_x0,
        "u_y": mid_length,
        "u_z": plane_z0,
        "r_x": plane_z0 | mid_length,
        "r_y": plane_z0 | plane_x0,
        "r_z": plane_x0 | mid_length,
    }
    shell = pc.model.shell(pc.FiniteElementSpace(surface), "kirchhoff")
    model = shell
    for variable, nodes in blocked.items():
        model = model | pc.model.dirichlet(pc.mesh.consolidate(nodes), shell, variable)
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("h", THICKNESS)]
    )
    load = pc.NodeField(pc.mesh.poi1_from_nodes([p3]), ["f_z"])
    load[0].set_value(p3, "f_z", FORCE)
    u = pc.solver.solve(pc.matrix.stiffness(model, materials), load)

    if out:
        surface.plot(title="elas3 — DKT mesh", save=os.path.join(out, "mesh.svg"))
        surface.plot(
            title="elas3 — u_z",
            field=u,
            component="u_z",
            cmap="viridis",
            save=os.path.join(out, "u_z.svg"),
        )
    return {"uz_P3": u.value(p3, "u_z")}


if __name__ == "__main__":
    print(main("."))
