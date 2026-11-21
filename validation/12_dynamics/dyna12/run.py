"""dyna12 — transient response of a free launcher to an axial thrust, bars.

Problem: a free bar of length 140 along y (two bars of 70), section 1,
E = 1e4, ρ = 0.1, at rest, receives at its bottom end P1 a thrust F_y = 100,
raised in 0.0125, held until 1.0125 and released in 0.0125. It moves as a
rigid body and vibrates along its axis. Reference (Przemieniecki): the
displacement of P1 at t = 1.9125 and t = 2.0625.

The motion is integrated by Newmark's average acceleration, as the deck's
DYNAMIC procedure does, here with a step of 0.0125 that falls on the load's
corners and on the two instants read.

Running::

    python validation/12_dynamics/dyna12/run.py
"""

import os

import pyrucast as pc

E, AREA, RHO = 1e4, 1.0, 0.1
LENGTH, N = 140.0, 2
THRUST = 100.0
LOAD = [(0.0, 0.0), (0.0125, 1.0), (1.0125, 1.0), (1.025, 0.0), (3.0, 0.0)]
DT = 0.0125
TIMES = [DT * i for i in range(166)]  # up to 2.0625
READ_AT = (1.9125, 2.0625)


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(3)
    p1 = coords.add_node([0.0, 0.0, 0.0])
    p3 = coords.add_node([0.0, LENGTH, 0.0])
    launcher = pc.mesh.line(p1, p3, N)

    model = pc.model.truss(pc.FiniteElementSpace(launcher))
    materials = pc.element_field.material_field(
        model, [("E", E), ("A", AREA), ("rho", RHO)]
    )
    thrust = pc.NodeField(pc.mesh.poi1_from_nodes([p1]), ["f_y"])
    thrust[0].set_value(p1, "f_y", THRUST)
    data = {
        "model": model,
        "times": TIMES,
        "materials": materials,
        "loads": pc.Evolution([(t, thrust * f) for t, f in LOAD]),
        "dynamics": True,
    }
    pc.stepping.step_by_step(data)

    values = {}
    times, displacements = [], []
    for record in data["results"]:
        uy = record["displacement"].value(p1, "u_y")
        times.append(record["time"])
        displacements.append(uy)
        for k, t in enumerate(READ_AT, start=1):
            if abs(record["time"] - t) < 1e-9:
                values[f"uy_{k}"] = uy

    if out:
        pc.SubEvolution(list(zip(times, displacements))).plot(
            x_label="t",
            y_label="u_y(P1)",
            title="dyna12 — displacement of the bottom point",
            save=os.path.join(out, "u_y.svg"),
        )
    return values


if __name__ == "__main__":
    print(main("."))
