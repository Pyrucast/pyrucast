"""plas2 — thick hollow sphere under internal pressure (NAFEMS MP2), axisymmetric.

Problem: a hollow sphere of inner radius 1 mm and outer radius 2 mm, of an
elastic, perfectly plastic material, is loaded by an internal pressure
P = 1e6·t Pa. At t = 100 (P = 100 MPa) it is elastic; at t = 358.9 it is
plastic through most of its wall. Reference: the radial displacement of the
inner point A, 0.4 µm and 2.83 µm.

Mesh: the deck's quarter of the meridian section, 14 × 16 QUA4: 14 elements
through the wall (5 from 1 to 1.1 mm, 9 from 1.1 to 2 mm), 16 along each arc
(every 5.625°). The nodes inside are placed by a transfinite (Coons)
interpolation of the boundary nodes, as the deck's mesher does.

Running::

    python validation/03_plasticity/plas2/run.py
"""

import math
import os

import pyrucast as pc

E, NU, SIGMA_Y = 2e11, 0.3, 300e6
R_IN, R_MID, R_OUT = 1e-3, 1.1e-3, 2e-3
N_THIN, N_THICK, N_ARC = 5, 9, 16
PRESSURE_RATE = 1e6  # Pa per unit time
TIMES = [0.0, 100.0, 358.9]


def main(out: str | None = None) -> dict[str, float]:
    # Radii across the wall, from the inner sphere to the outer one.
    radii = [R_IN + (R_MID - R_IN) * k / N_THIN for k in range(N_THIN)]
    radii += [R_MID + (R_OUT - R_MID) * k / N_THICK for k in range(N_THICK + 1)]
    ni, nj = len(radii) - 1, N_ARC
    # Boundary nodes: along z = 0 (j = 0), along the axis (j = NJ), on the
    # inner arc (i = 0) and on the outer arc (i = NI).
    bottom = [(r, 0.0) for r in radii]
    top = [(0.0, r) for r in radii]
    inner, outer = [], []
    for j in range(nj + 1):
        theta = math.pi / 2.0 * j / nj
        inner.append((R_IN * math.cos(theta), R_IN * math.sin(theta)))
        outer.append((R_OUT * math.cos(theta), R_OUT * math.sin(theta)))

    coords = pc.Coords.axisymmetric()
    node = {}
    for i in range(ni + 1):
        for j in range(nj + 1):
            u, v = i / ni, j / nj
            point = []
            for d in range(2):
                coons = (
                    (1 - v) * bottom[i][d]
                    + v * top[i][d]
                    + (1 - u) * inner[j][d]
                    + u * outer[j][d]
                    - (1 - u) * (1 - v) * bottom[0][d]
                    - u * (1 - v) * bottom[ni][d]
                    - (1 - u) * v * top[0][d]
                    - u * v * top[ni][d]
                )
                point.append(coons)
            node[i, j] = coords.add_node(point)
    sphere = pc.Mesh(coords, "QUA4")
    for i in range(ni):
        for j in range(nj):
            sphere.unit().add_cell(
                [node[i, j], node[i + 1, j], node[i + 1, j + 1], node[i, j + 1]]
            )
    # The inner arc, travelled counter-clockwise around the section: from the
    # axis down to A.
    inner_arc = pc.Mesh(coords, "SEG2")
    for j in range(nj, 0, -1):
        inner_arc.unit().add_cell([node[0, j], node[0, j - 1]])
    point_a = node[0, 0]

    fes = pc.FiniteElementSpace(sphere)
    plasticity = pc.model.plasticity_perfect(fes, "axisymmetric")
    model = (
        plasticity
        | pc.model.dirichlet(
            pc.mesh.poi1_from_nodes([node[i, 0] for i in range(ni + 1)]),
            plasticity,
            "u_y",
        )
        | pc.model.dirichlet(
            pc.mesh.poi1_from_nodes([node[i, nj] for i in range(ni + 1)]),
            plasticity,
            "u_x",
        )
    )
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("sigma_y", SIGMA_Y)]
    )
    # The internal pressure at t = 1, pushing on the wall: φ_n = −p.
    pressure = pc.model.flux(pc.FiniteElementSpace(inner_arc), plasticity, "n")
    unit_force = pc.node_field.external_forces(
        pressure, pc.element_field.material_field(pressure, [("phi_n", -PRESSURE_RATE)])
    )
    data = {
        "model": model,
        "times": TIMES,
        "materials": materials,
        "loads": pc.Evolution(
            [(0.0, unit_force * 0.0), (TIMES[-1], unit_force * TIMES[-1])]
        ),
    }
    pc.stepping.step_by_step(data)

    values = {}
    for k, record in enumerate(data["results"][1:], start=1):
        values[f"ur_A_{k}"] = record["displacement"].value(point_a, "u_x") * 1e6

    if out:
        sphere.plot(title="plas2 — QUA4 mesh", save=os.path.join(out, "mesh.svg"))
        sphere.plot(
            title="plas2 — cumulated plastic strain p at t = 358.9",
            field=data["results"][-1]["behaviour"],
            component="p",
            cmap="viridis",
            save=os.path.join(out, "p.svg"),
        )
    return values


if __name__ == "__main__":
    print(main("."))
