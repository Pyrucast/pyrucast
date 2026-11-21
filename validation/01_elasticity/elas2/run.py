"""elas2 — pinched cylinder with a clamped end (NAFEMS ME6), DKT shell.

Problem: a thin cylinder of radius 7, clamped at one end, is pinched at its
mid-length by two opposite point loads. One eighth is modelled: a quarter of
the circumference over half the length, with symmetry conditions on the cut
edges. Reference: u_z = −1.14 under the load, u_x = 0 at the clamped point P1.

The mesh is the Cast3M deck's, cell for cell: the quarter arc (6 elements,
every 15°) translated by 70 along y (30 layers), each quadrangle cut into two
triangles along a diagonal that alternates from one quadrangle to the next
along the arc, as the deck's translation in TRI3 does.

Running::

    python validation/01_elasticity/elas2/run.py
"""

import math
import os

import pyrucast as pc

E, NU, THICKNESS = 3e7, 0.3, 0.1
RADIUS, HALF_LENGTH = 7.0, 70.0
N_ARC, N_LENGTH = 6, 30
FORCE = -2500.0  # F_z at the loaded point, a quarter of the pinching load


def main(out: str | None = None) -> dict[str, float]:
    # Node (k, j): k-th point of the arc from (R, 0, 0) to (0, 0, R), j-th layer.
    coords = pc.Coords(3)
    node = {}
    for j in range(N_LENGTH + 1):
        for k in range(N_ARC + 1):
            theta = math.pi / 2.0 * k / N_ARC
            y = HALF_LENGTH * j / N_LENGTH
            node[k, j] = coords.add_node(
                [RADIUS * math.cos(theta), y, RADIUS * math.sin(theta)]
            )
    surface = pc.Mesh(coords, "TRI3")
    for j in range(N_LENGTH):
        for k in range(N_ARC):
            a0, a1 = node[k, j], node[k + 1, j]
            b0, b1 = node[k, j + 1], node[k + 1, j + 1]
            if k % 2 == 0:
                surface.unit().add_cell([a0, a1, b1])
                surface.unit().add_cell([a0, b1, b0])
            else:
                surface.unit().add_cell([a0, a1, b0])
                surface.unit().add_cell([a1, b1, b0])

    clamped = pc.mesh.poi1_from_nodes([node[k, 0] for k in range(N_ARC + 1)])
    plane_z0 = pc.mesh.poi1_from_nodes([node[0, j] for j in range(N_LENGTH + 1)])
    plane_x0 = pc.mesh.poi1_from_nodes([node[N_ARC, j] for j in range(N_LENGTH + 1)])
    mid_length = pc.mesh.poi1_from_nodes([node[k, N_LENGTH] for k in range(N_ARC + 1)])
    p1 = node[0, 0]
    p3 = node[N_ARC, N_LENGTH]

    # The blocked nodes of each unknown: clamped end, then the symmetry planes
    # z = 0 (u_z, r_x, r_y), x = 0 (u_x, r_y, r_z) and y = 70 (u_y, r_x, r_z).
    # A node on two of these sets is blocked once (`consolidate`).
    blocked = {
        "u_x": clamped | plane_x0,
        "u_y": clamped | mid_length,
        "u_z": clamped | plane_z0,
        "r_x": clamped | plane_z0 | mid_length,
        "r_y": clamped | plane_z0 | plane_x0,
        "r_z": clamped | plane_x0 | mid_length,
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
        surface.plot(title="elas2 — DKT mesh", save=os.path.join(out, "mesh.svg"))
        surface.plot(
            title="elas2 — u_z",
            field=u,
            component="u_z",
            cmap="viridis",
            save=os.path.join(out, "u_z.svg"),
        )
    return {"ux_P1": u.value(p1, "u_x"), "uz_P3": u.value(p3, "u_z")}


if __name__ == "__main__":
    print(main("."))
