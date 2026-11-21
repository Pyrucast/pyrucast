"""plas2 — thick hollow sphere under internal pressure (NAFEMS MP2), axisymmetric.

Problem: a hollow sphere of inner radius 1 mm and outer radius 2 mm, of an
elastic, perfectly plastic material, is loaded by an internal pressure
P = 1e6·t Pa. At t = 100 (P = 100 MPa) it is elastic; at t = 358.9 it is
plastic through most of its wall. Reference: the radial displacement of the
inner point A, 0.4 µm and 2.83 µm.

Mesh: the `.dgibi` file's quarter of the meridian section, 14 × 16 QUA4: 14 elements
through the wall (5 from 1 to 1.1 mm, 9 from 1.1 to 2 mm), 16 along each arc
(every 5.625°), filled by `transfinite` as the `.dgibi` file's mesher does.

Running::

    python validation/03_plasticity/plas2/run.py
"""

import os

import pyrucast as pc

E, NU, SIGMA_Y = 2e11, 0.3, 300e6
R_IN, R_MID, R_OUT = 1e-3, 1.1e-3, 2e-3
N_THIN, N_THICK, N_ARC = 5, 9, 16
PRESSURE_RATE = 1e6  # Pa per unit time
TIMES = [0.0, 100.0, 358.9]


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords.axisymmetric()
    origin = coords.add_node([0.0, 0.0])
    a = coords.add_node([R_IN, 0.0])
    x = coords.add_node([R_MID, 0.0])
    b = coords.add_node([R_OUT, 0.0])
    c = coords.add_node([0.0, R_OUT])
    y = coords.add_node([0.0, R_MID])
    d = coords.add_node([0.0, R_IN])
    # The four sides of the section, counter-clockwise, as in the `.dgibi` file.
    bottom = pc.mesh.consolidate(
        pc.mesh.line(a, x, N_THIN) | pc.mesh.line(x, b, N_THICK)
    )
    outer = pc.mesh.arc(b, origin, c, N_ARC)
    axis = pc.mesh.consolidate(pc.mesh.line(c, y, N_THICK) | pc.mesh.line(y, d, N_THIN))
    inner = pc.mesh.arc(d, origin, a, N_ARC)
    sphere = pc.mesh.transfinite(bottom, outer, axis, inner)

    fes = pc.FiniteElementSpace(sphere)
    plasticity = pc.model.plasticity_perfect(fes, "axisymmetric")
    model = (
        plasticity
        | pc.model.dirichlet(pc.mesh.consolidate(bottom.to_poi1()), plasticity, "u_y")
        | pc.model.dirichlet(pc.mesh.consolidate(axis.to_poi1()), plasticity, "u_x")
    )
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("sigma_y", SIGMA_Y)]
    )
    # The internal pressure at t = 1, pushing on the wall: φ_n = −p.
    pressure = pc.model.flux(pc.FiniteElementSpace(inner), plasticity, "n")
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
        values[f"ur_A_{k}"] = record["displacement"].value(a, "u_x") * 1e6

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
