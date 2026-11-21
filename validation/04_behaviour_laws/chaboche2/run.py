"""chaboche2 — tension-compression cycle on a Chaboche viscoplastic material.

Problem: one axisymmetric QUA8, [0, 1] × [0, 1] in (r, z), held by u_z = 0
on z = 0 and u_r = 0 on the axis, receives u_z = 0.002·f(t) on z = 1, with
f = 0, 1, 0, −1, 0, 2 at t = 0, 100, 200, 300, 400, 600 s. The material is
Chaboche's unified viscoplastic law: a Norton flow on the overstress
J(σ − X) − R − k, one Armstrong-Frederick back stress and a saturating
isotropic hardening. Reference: the axial stress of the same computation with
Cast3M's ONERA model and equivalent data, read at t = 100, 200, 300, 400 and
600 s.

The Cast3M data map to pyrucast's as SIGY → k, K → K, N → n, A·C → C_1,
C → gamma_1, B → b, Q → Q. The strain is homogeneous: the stress is read at
the first Gauss point, every 10 s as the deck computes. The deck's B-bar
formulation changes nothing on a homogeneous strain; pyrucast has none.

Running::

    python validation/04_behaviour_laws/chaboche2/run.py
"""

import os

import pyrucast as pc

E, NU = 200e9, 0.3
CHABOCHE = [
    ("k", 82e6),  # SIGY
    ("K", 151e6),
    ("n", 24.0),
    ("C_1", 162400e6),  # A·C
    ("gamma_1", 2800.0),  # C
    ("b", 8.0),
    ("Q", 142e6 - 82e6),
]
U_MAX = 0.002
HISTORY = [
    (0.0, 0.0),
    (100.0, 1.0),
    (200.0, 0.0),
    (300.0, -1.0),
    (400.0, 0.0),
    (600.0, 2.0),
]
TIMES = [10.0 * i for i in range(61)]
READ_AT = (100.0, 200.0, 300.0, 400.0, 600.0)


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords.axisymmetric()
    a = coords.add_node([0.0, 0.0])
    b = coords.add_node([1.0, 0.0])
    c = coords.add_node([1.0, 1.0])
    d = coords.add_node([0.0, 1.0])
    bottom = pc.mesh.line(a, b, 1, "SEG3")
    top = pc.mesh.line(d, c, 1, "SEG3")
    cylinder = pc.mesh.sweep(bottom, top, 1, "QUA8")

    fes = pc.FiniteElementSpace(cylinder)
    law = pc.model.viscoplasticity_chaboche(fes, "axisymmetric")
    pull = pc.model.dirichlet(pc.mesh.consolidate(top.to_poi1()), law, "u_y")
    model = (
        law
        | pc.model.dirichlet(pc.mesh.consolidate(bottom.to_poi1()), law, "u_y")
        | pc.model.dirichlet(
            pc.mesh.points_on_line(cylinder, [0.0, 0.0], [0.0, 1.0]), law, "u_x"
        )
        | pull
    )
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU)] + CHABOCHE
    )
    frames = []
    for t, f in HISTORY:
        frames.append(
            (t, pc.NodeField(pull.multiplier_mesh(), ["imposed"]) + U_MAX * f)
        )
    data = {
        "model": model,
        "times": TIMES,
        "materials": materials,
        "loads": pc.Evolution(frames),
    }
    pc.stepping.step_by_step(data)

    values = {}
    times, stresses = [], []
    for record in data["results"]:
        szz = record["behaviour"][0].value(0, 0, "sigma_yy") / 1e6
        times.append(record["time"])
        stresses.append(szz)
        for t in READ_AT:
            if abs(record["time"] - t) < 1e-9:
                values[f"szz_{t:.0f}"] = szz

    if out:
        pc.SubEvolution(list(zip(times, stresses))).plot(
            x_label="t (s)",
            y_label="σ_zz (MPa)",
            title="chaboche2 — axial stress",
            save=os.path.join(out, "sigma_zz.svg"),
        )
    return values


if __name__ == "__main__":
    print(main("."))
