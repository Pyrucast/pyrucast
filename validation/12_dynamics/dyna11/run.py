"""dyna11 — thick cylinder under a sudden internal pressure, axisymmetric, QUA4.

Problem: a thick cylinder, r from 6 to 12, of height 8 held by u_z = 0 on
its two ends, receives at t = 5e-6 (raised from t = 4e-6) an internal
pressure of 10, then held. E = 30e6, ν = 0, ρ = 0.28 × 0.002588. Without
damping, the largest radial displacement of a point is twice its static
value. Reference: that ratio, 2, at three points of the mid-height section,
r = 6, 9 and 11; the static displacements are the deck's.

Mesh: the deck's regular grid of 6 × 8 QUA4. The motion is integrated by
Newmark's average acceleration every 1e-6 up to 2e-4, as the deck's DYNAMIC
procedure does.

Running::

    python validation/12_dynamics/dyna11/run.py
"""

import os

import pyrucast as pc

E, NU, RHO = 30e6, 0.0, 0.28 * 0.002588
R_IN, R_OUT, HEIGHT = 6.0, 12.0, 8.0
N_R, N_Z = 6, 8
PRESSURE = 10.0
LOAD = [(0.0, 0.0), (4e-6, 0.0), (5e-6, 1.0), (204e-6, 1.0)]
TIMES = [1e-6 * i for i in range(201)]
# The points read, at mid-height, and their static radial displacement.
POINTS = [((6.0, 4.0), 3.326e-6), ((9.0, 4.0), 2.773e-6), ((11.0, 4.0), 2.673e-6)]


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords.axisymmetric()
    p1 = coords.add_node([R_IN, 0.0])
    p2 = coords.add_node([R_OUT, 0.0])
    p3 = coords.add_node([R_OUT, HEIGHT])
    p4 = coords.add_node([R_IN, HEIGHT])
    bottom = pc.mesh.line(p1, p2, N_R)
    top = pc.mesh.line(p3, p4, N_R)
    inner = pc.mesh.line(p4, p1, N_Z)
    cylinder = pc.mesh.transfinite(bottom, pc.mesh.line(p2, p3, N_Z), top, inner)
    points = [cylinder.nearest_node(list(xy)) for xy, _ in POINTS]

    fes = pc.FiniteElementSpace(cylinder)
    elasticity = pc.model.elasticity(fes, "axisymmetric")
    model = elasticity | pc.model.dirichlet(
        pc.mesh.consolidate((bottom | top).to_poi1()), elasticity, "u_y"
    )
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("rho", RHO)]
    )
    # The internal pressure, pushing on the wall: φ_n = −p.
    pressure = pc.model.flux(pc.FiniteElementSpace(inner), elasticity, "n")
    force = pc.node_field.external_forces(
        pressure, pc.element_field.material_field(pressure, [("phi_n", -PRESSURE)])
    )
    data = {
        "model": model,
        "times": TIMES,
        "materials": materials,
        "loads": pc.Evolution([(t, force * f) for t, f in LOAD]),
        "dynamics": True,
    }
    pc.stepping.step_by_step(data)

    largest = [0.0, 0.0, 0.0]
    history = []
    for record in data["results"]:
        u = record["displacement"]
        for k, node in enumerate(points):
            largest[k] = max(largest[k], u.value(node, "u_x"))
        history.append((record["time"], u.value(points[0], "u_x")))

    if out:
        pc.SubEvolution(history).plot(
            x_label="t",
            y_label="u_r at r = 6",
            title="dyna11 — radial displacement of the inner point",
            save=os.path.join(out, "u_r.svg"),
        )
    values = {}
    for k, (_, static) in enumerate(POINTS, start=1):
        values[f"amp_{k}"] = largest[k - 1] / static
    return values


if __name__ == "__main__":
    print(main("."))
