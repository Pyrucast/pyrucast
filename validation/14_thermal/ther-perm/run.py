"""ther-perm — steady conduction with a conductivity linear in T, QUA4.

Problem: an infinite wall of thickness L = 5 mm, heated by a volume source
q = 1e7 W/m³, its face x = L held at T₀ = 100 °C and its face x = 0
insulated. The conductivity is k = k₀·(1 + β·(T − T₀)), k₀ = 2, β = 0.025.
With θ = (T − T₀)·2k₀/(q·L²) and s = L²·q·β/k₀, the exact profile is

    θ(x) = 2·(√(1 + s·(1 − (x/L)²)) − 1) / s.

Compared: the largest gap of θ to that profile over the nodes (the deck's
test), and T at the insulated face, whose exact value is
T₀ + (√(1 + s) − 1)/β.

The non-linear iterations are converged to a precision of 1e-8.

Mesh: one row of QUA4 of height L/10, the deck's: its cells grow from L/100
to L/10 along x, at the positions `X_NODES` the deck's mesh density gives.

Running::

    python validation/14_thermal/ther-perm/run.py
"""

import math
import os

import pyrucast as pc

L = 5e-3  # m
HEIGHT = L / 10.0
K0, BETA, T0 = 2.0, 0.025, 100.0
SOURCE = 1e7  # W/m³
# The deck's THERMIC procedure stops on its own criterion; at step_by_step's
# default precision (1e-4) the profile would stay 1e-4 from the converged one.
PRECISION = 1e-8
X_NODES = [
    0.0,
    5.021902808304401e-05,
    1.051657296518693e-04,
    1.652851729891834e-04,
    2.310643255860185e-04,
    3.030359985784089e-04,
    3.817831625175931e-04,
    4.679436694314784e-04,
    5.622154194260674e-04,
    6.653620136763161e-04,
    7.782189395956515e-04,
    9.017003382840204e-04,
    1.036806409070798e-03,
    1.184631511129359e-03,
    1.346373027786375e-03,
    1.523341065326770e-03,
    1.716969064854620e-03,
    1.928825413166102e-03,
    2.160626146682537e-03,
    2.414248851345410e-03,
    2.691747871062489e-03,
    2.995370947893483e-03,
    3.327577428760723e-03,
    3.691058186159234e-03,
    4.088757414223942e-03,
    4.523896476702174e-03,
    5.0e-03,
]


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(2)
    bottom = [coords.add_node([x, 0.0]) for x in X_NODES]
    top = [coords.add_node([x, HEIGHT]) for x in X_NODES]
    wall = pc.Mesh(coords, "QUA4")
    for i in range(len(X_NODES) - 1):
        wall.unit().add_cell([bottom[i], bottom[i + 1], top[i + 1], top[i]])
    held_face = pc.mesh.poi1_from_nodes([bottom[-1], top[-1]])

    fes = pc.FiniteElementSpace(wall)
    conduction = pc.model.heat_conduction(fes)
    blocked = pc.model.dirichlet(held_face, conduction, "T")
    model = conduction | blocked | pc.model.flux(fes, conduction, "q")
    # No heat capacity: the problem is steady. k is overridden by k(T).
    materials = pc.element_field.material_field(model, [("k", K0), ("phi_q", SOURCE)])
    conductivity = pc.Evolution(
        [
            (T0 - 100.0, K0 * (1.0 - 100.0 * BETA)),
            (T0 + 1000.0, K0 * (1.0 + 1000.0 * BETA)),
        ],
        abscissa_type="T",
        ordinate_type="k",
        out_of_range="extrapolate",
    )
    imposed = pc.NodeField(blocked.multiplier_mesh(), ["imposed"])
    imposed[0].add_to_component("imposed", T0)
    data = {
        "model": model,
        "times": [0.0, 1.0],
        "materials": materials,
        "material_laws": conductivity,
        "loads": imposed,
        "initial": {"temperature": pc.NodeField(wall, ["T"]) + T0},
        "precision": PRECISION,
    }
    pc.stepping.step_by_step(data)
    temperature = data["results"][-1]["temperature"]

    s = L * L * SOURCE * BETA / K0
    worst = 0.0
    for i, x in enumerate(X_NODES):
        theta = (temperature.value(top[i], "T") - T0) * 2.0 * K0 / (SOURCE * L * L)
        exact = 2.0 * (math.sqrt(1.0 + s * (1.0 - (x / L) ** 2)) - 1.0) / s
        worst = max(worst, abs(theta - exact))

    if out:
        wall.plot(
            title="ther-perm — T (°C)",
            field=temperature,
            component="T",
            cmap="viridis",
            save=os.path.join(out, "T.svg"),
        )
    return {"gap_max": worst, "T_hot": temperature.value(bottom[0], "T")}


if __name__ == "__main__":
    print(main("."))
