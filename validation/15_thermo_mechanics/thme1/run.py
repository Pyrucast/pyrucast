"""thme1 — heat diffusion in a bar and its expansion, step by step, QUA8.

Problem: a bar of length 0.1 (height 0.01), at 20 °C, has its end x = 0.1
brought to 500 °C at t = 0 while its end x = 0 stays at 20 °C. In plane
strain, held along y everywhere and along x at x = 0, it expands freely
along x. Reference: the temperature and the displacement u_x at x = 0.08,
from the analytical solution T = 500 − 480·erf((0.1 − x)/(2√(a·t))) /
erf(0.1/(2√(a·t))), a = k/(ρ·c), and its integral times α.

Mesh: the `.dgibi` file's row of 10 QUA8. The heat problem is solved with the linear
θ-scheme (θ = 1) every 5 s up to 150 s, the mechanics at each instant with
the temperature reached, as the `.dgibi` file's step-by-step procedure does.

Running::

    python validation/15_thermo_mechanics/thme1/run.py
"""

import os

import pyrucast as pc

LENGTH, HEIGHT, N = 0.1, 0.01, 10
K, C, RHO = 14.6, 460.0, 7800.0
E, NU, ALPHA = 2e11, 1e-5, 1e-5
T_START, T_HOT = 20.0, 500.0
X_READ = 0.08
TIMES = [5.0 * i for i in range(31)]  # 0 to 150 s
READ_AT = (80.0, 100.0, 150.0)


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(2)
    pa = coords.add_node([0.0, 0.0])
    pb = coords.add_node([LENGTH, 0.0])
    pc_ = coords.add_node([LENGTH, HEIGHT])
    pd = coords.add_node([0.0, HEIGHT])
    bar = pc.mesh.sweep(
        pc.mesh.line(pa, pb, N, "SEG3"), pc.mesh.line(pd, pc_, N, "SEG3"), 1, "QUA8"
    )
    hot_end = pc.mesh.points_on_line(bar, [LENGTH, 0.0], [LENGTH, HEIGHT])
    cold_end = pc.mesh.points_on_line(bar, [0.0, 0.0], [0.0, HEIGHT])
    every_node = pc.mesh.consolidate(pc.mesh.to_poi1(bar))
    point = bar.nearest_node([X_READ, 0.0])

    fes = pc.FiniteElementSpace(bar)
    conduction = pc.model.heat_conduction(fes)
    solid = pc.model.elasticity(fes, "plane_strain")
    hot = pc.model.dirichlet(hot_end, conduction, "T")
    cold = pc.model.dirichlet(cold_end, conduction, "T")
    model = (
        conduction
        | solid
        | hot
        | cold
        | pc.model.dirichlet(every_node, solid, "u_y")
        | pc.model.dirichlet(cold_end, solid, "u_x")
    )
    materials = pc.element_field.material_field(
        model,
        [
            ("k", K),
            ("cp", C),
            ("rho", RHO),
            ("E", E),
            ("nu", NU),
            ("alpha", ALPHA),
        ],
    )
    hot_value = pc.NodeField(hot.multiplier_mesh(), ["imposed"]) + T_HOT
    cold_value = pc.NodeField(cold.multiplier_mesh(), ["imposed"]) + T_START
    data = {
        "model": model,
        "times": TIMES,
        "materials": materials,
        "loads": hot_value | cold_value,
        "initial": {"temperature": pc.NodeField(bar, ["T"]) + T_START},
        "t_ref": T_START,
        "heat_scheme": "linear",
    }
    pc.stepping.step_by_step(data)

    values = {}
    for record in data["results"]:
        for t in READ_AT:
            if abs(record["time"] - t) < 1e-9:
                values[f"T_{t:.0f}"] = record["temperature"].value(point, "T")
                values[f"u_{t:.0f}"] = record["displacement"].value(point, "u_x")

    if out:
        last = data["results"][-1]
        bar.plot(
            title="thme1 — T at t = 150 s (°C)",
            field=last["temperature"],
            component="T",
            cmap="viridis",
            save=os.path.join(out, "T_150.svg"),
        )
    return values


if __name__ == "__main__":
    print(main("."))
