"""plas4 — uniaxial strain cycle on an elastic, perfectly plastic block, HEX8.

Problem: a block of two HEX8 (x from 0 to 1, section 1 × 1), held so that
u_y = u_z = 0 everywhere and u_x = 0 on x = 0, receives u_x = ε₀·f(t)·x on
its faces x = 0.5 and x = 1, with ε₀ = 2.5e-5 and f = 0, 1, 2, 1, 0 at
t = 0, 1, 2, 3, 4: the strain goes up into the plastic range and back.
Reference: σ_xx, σ_yy and σ_zz at t = 1, 2, 3, 4, the closed-form response of
J2 perfect plasticity in uniaxial strain.

The stress is homogeneous: it is read at the first Gauss point of the first
cell, as in the deck.

Running::

    python validation/03_plasticity/plas4/run.py
"""

import os

import pyrucast as pc

E, NU, SIGMA_Y = 2.5e11, 0.25, 5e6
EPS0 = 2.5e-5
HISTORY = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 1.0), (4.0, 0.0)]  # (t, f)


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(3)
    square = pc.Mesh(coords, "QUA4")
    square.unit().add_cell(
        [
            coords.add_node([0.0, 0.0, 0.0]),
            coords.add_node([0.0, 1.0, 0.0]),
            coords.add_node([0.0, 1.0, 1.0]),
            coords.add_node([0.0, 0.0, 1.0]),
        ]
    )
    block = pc.mesh.extrude(square, [1.0, 0.0, 0.0], 2)
    face_0 = pc.mesh.points_on_plane(block, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    face_half = pc.mesh.points_on_plane(block, [0.5, 0.0, 0.0], [1.0, 0.0, 0.0])
    face_1 = pc.mesh.points_on_plane(block, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    every_node = pc.mesh.consolidate(pc.mesh.to_poi1(block))

    fes = pc.FiniteElementSpace(block)
    plasticity = pc.model.plasticity_perfect(fes, "full_3d")
    pull_half = pc.model.dirichlet(face_half, plasticity, "u_x")
    pull_end = pc.model.dirichlet(face_1, plasticity, "u_x")
    model = (
        plasticity
        | pc.model.dirichlet(face_0, plasticity, "u_x")
        | pull_half
        | pull_end
        | pc.model.dirichlet(every_node, plasticity, "u_y")
        | pc.model.dirichlet(every_node, plasticity, "u_z")
    )
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("sigma_y", SIGMA_Y)]
    )
    frames = []
    for t, f in HISTORY:
        imposed = (
            pc.NodeField(pull_half.multiplier_mesh(), ["imposed"]) + EPS0 * f * 0.5
        ) | (pc.NodeField(pull_end.multiplier_mesh(), ["imposed"]) + EPS0 * f)
        frames.append((t, imposed))
    data = {
        "model": model,
        "times": [t for t, _ in HISTORY],
        "materials": materials,
        "loads": pc.Evolution(frames),
    }
    pc.stepping.step_by_step(data)

    values = {}
    for record in data["results"]:
        k = round(record["time"])
        if k == 0:
            continue
        behaviour = record["behaviour"][0]
        for axis in ("xx", "yy", "zz"):
            values[f"s{axis}_{k}"] = behaviour.value(0, 0, f"sigma_{axis}")

    if out:
        block.plot(title="plas4 — HEX8 mesh", save=os.path.join(out, "mesh.svg"))
    return values


if __name__ == "__main__":
    print(main("."))
