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

Mesh: one row of 26 QUA4 of height L/10, as many as the deck's. The deck's
cells grow from L/100 to L/10 along x; pyrucast's `line` has no such
progression, its cells are even.

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
N = 26  # cells along x, as many as the deck's


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(2)
    hot = coords.add_node([0.0, 0.0])
    held = coords.add_node([L, 0.0])
    wall = pc.mesh.extrude(pc.mesh.line(hot, held, N), [0.0, HEIGHT], 1)
    held_face = pc.mesh.points_on_line(wall, [L, 0.0], [L, HEIGHT])

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
    for i in range(N + 1):
        x = L * i / N
        node = wall.nearest_node([x, HEIGHT])
        theta = (temperature.value(node, "T") - T0) * 2.0 * K0 / (SOURCE * L * L)
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
    return {"gap_max": worst, "T_hot": temperature.value(hot, "T")}


if __name__ == "__main__":
    print(main("."))
