"""plas1 — bending of a strip into the plastic range (NAFEMS MP1), plane stress.

Problem: a strip 100 mm high, in plane stress, of an elastic, perfectly
plastic material, is bent by rotating its end section, which stays straight:
u_x = ±1e-3·t at its corners y = ±50 mm, and linear relations keep the
section's other nodes on the line between them. Half of the strip is
modelled (12.5 mm wide, u_x = 0 on the symmetry edge x = 0). At t = 0.02875
the strip is still elastic; at t = 0.05 it is plastic over most of its
height. Reference: M1 = 805 kN·m and M2 = 1074 kN·m.

Mesh: the deck's, one QUA4 across and 8 along y. `mpc` takes one
coefficient per term for all its relations: the seven relations, whose
coefficients differ, are seven `mpc`. The moment is the deck's:
the moment of the support forces on the end section, 2·Σ y·F_x over its
upper half, in kN·m.

Running::

    python validation/03_plasticity/plas1/run.py
"""

import os

import pyrucast as pc

E, NU, SIGMA_Y = 2.1e11, 0.3, 483e6
WIDTH, HALF_HEIGHT = 12.5e-3, 50e-3
N = 8  # elements along the end section
ROTATION = 1e-3  # u_x at the corners per unit time
TIMES = [0.0, 0.02875, 0.05]


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(2)
    a = coords.add_node([WIDTH, HALF_HEIGHT])
    b = coords.add_node([WIDTH, -HALF_HEIGHT])
    strip = pc.mesh.extrude(pc.mesh.line(b, a, N), [-WIDTH, 0.0], 1)
    # The end section's nodes, from A (y = +50 mm) down to B (y = −50 mm).
    section = [
        strip.nearest_node([WIDTH, HALF_HEIGHT - 2.0 * HALF_HEIGHT * i / N])
        for i in range(N + 1)
    ]
    axis = pc.mesh.points_on_line(strip, [0.0, -HALF_HEIGHT], [0.0, HALF_HEIGHT])
    origin = strip.nearest_node([0.0, 0.0])

    fes = pc.FiniteElementSpace(strip)
    plasticity = pc.model.plasticity_perfect(fes, "plane_stress")
    pull_a = pc.model.dirichlet(pc.mesh.poi1_from_nodes([a]), plasticity, "u_x")
    push_b = pc.model.dirichlet(pc.mesh.poi1_from_nodes([b]), plasticity, "u_x")
    model = (
        plasticity
        | pc.model.dirichlet(axis, plasticity, "u_x")
        | pc.model.dirichlet(pc.mesh.poi1_from_nodes([origin]), plasticity, "u_y")
        | pull_a
        | push_b
    )
    # The section stays straight: 8·u(A_r) − (8 − r)·u(A) − r·u(B) = 0.
    for r in range(1, N):
        model = model | pc.model.mpc(
            [
                (pc.mesh.poi1_from_nodes([section[r]]), "u_x", float(N)),
                (pc.mesh.poi1_from_nodes([a]), "u_x", -float(N - r)),
                (pc.mesh.poi1_from_nodes([b]), "u_x", -float(r)),
            ],
            plasticity,
        )
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("sigma_y", SIGMA_Y)]
    )
    frames = []
    for t in (0.0, 1.0):
        imposed = (
            pc.NodeField(pull_a.multiplier_mesh(), ["imposed"]) + ROTATION * t
        ) | (pc.NodeField(push_b.multiplier_mesh(), ["imposed"]) - ROTATION * t)
        frames.append((t, imposed))
    data = {
        "model": model,
        "times": TIMES,
        "materials": materials,
        "loads": pc.Evolution(frames),
    }
    pc.stepping.step_by_step(data)

    moments = []
    for record in data["results"][1:]:
        reactions = record["reactions"]
        moment = 0.0
        for i in range(N // 2 + 1):  # the upper half, A down to the middle
            y = HALF_HEIGHT - 2.0 * HALF_HEIGHT * i / N
            moment += y * reactions.value(section[i], "f_x")
        moments.append(2.0 * moment * 1e-3)

    if out:
        strip.plot(title="plas1 — QUA4 mesh", save=os.path.join(out, "mesh.svg"))
    return {"M1": moments[0], "M2": moments[1]}


if __name__ == "__main__":
    print(main("."))
