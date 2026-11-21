"""elas3 — pinched cylinder with free ends (NAFEMS ME7), DKT shell.

Problem: a thin cylinder of radius 4.953 and length 2 × 5.175, free at its
ends, is pinched at its mid-length by two opposite point loads. One eighth is
modelled: a quarter of the circumference over half the length, with symmetry
conditions on the cut edges. Reference: u_z = −0.452 under the load.

The mesh is the Cast3M deck's, cell for cell: the quarter arc (10 elements,
every 9°) translated along y in 7 layers, each quadrangle cut into two
triangles along a diagonal that alternates along the arc. The deck lets its
mesh density (0.75) place the layers, slightly closer towards the loaded
section; `Y_LAYERS` are the positions it produces.

Running::

    python validation/01_elasticity/elas3/run.py
"""

import math
import os

import pyrucast as pc

E, NU, THICKNESS = 10.5e6, 0.3125, 0.094
RADIUS = 4.953
N_ARC = 10
Y_LAYERS = [
    0.0,
    0.7501353964340833,
    1.496624840688324,
    2.239486053520015,
    2.978736669556621,
    3.714394237714402,
    4.446476221615003,
    5.175,
]
FORCE = -100.0  # F_z at the loaded point


def main(out: str | None = None) -> dict[str, float]:
    # Node (k, j): k-th point of the arc from (R, 0, 0) to (0, 0, R), j-th layer.
    n_layers = len(Y_LAYERS) - 1
    coords = pc.Coords(3)
    node = {}
    for j in range(n_layers + 1):
        for k in range(N_ARC + 1):
            theta = math.pi / 2.0 * k / N_ARC
            node[k, j] = coords.add_node(
                [RADIUS * math.cos(theta), Y_LAYERS[j], RADIUS * math.sin(theta)]
            )
    surface = pc.Mesh(coords, "TRI3")
    for j in range(n_layers):
        for k in range(N_ARC):
            a0, a1 = node[k, j], node[k + 1, j]
            b0, b1 = node[k, j + 1], node[k + 1, j + 1]
            if k % 2 == 0:
                surface.unit().add_cell([a0, a1, b1])
                surface.unit().add_cell([a0, b1, b0])
            else:
                surface.unit().add_cell([a0, a1, b0])
                surface.unit().add_cell([a1, b1, b0])

    plane_z0 = pc.mesh.poi1_from_nodes([node[0, j] for j in range(n_layers + 1)])
    plane_x0 = pc.mesh.poi1_from_nodes([node[N_ARC, j] for j in range(n_layers + 1)])
    mid_length = pc.mesh.poi1_from_nodes([node[k, n_layers] for k in range(N_ARC + 1)])
    p3 = node[N_ARC, n_layers]

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
