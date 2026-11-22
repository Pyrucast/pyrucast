"""plas9 — uniaxial pull of a softening Drucker-Prager block, HEX8.

Problem: a block 1.5 × 1.5 × 1 (one HEX8), held by u_x = 0 on x = 0,
u_y = 0 on y = 0 and u_z = 0 on z = 0, is pulled by u_x = 1.5e-4·t on
x = 1.5 up to t = 4, in steps of 0.1. The material is a Drucker-Prager law
with α = 0.441, β = 1.02, K = 2 and a negative hardening H = −10 000.
Reference: the exact uniaxial curve,

    σ = E·ε                                  while ε < K / ((α + β)·E),
    σ = (K + H·m·ε) / ((α + β) + H·m/E)      after, floored at 0,

with m = √(2α² + β²) / (α + β). Compared: max |σ − σ_exact| / K over the
computed times (the `.dgibi` file's test).

The `.dgibi` file adds, on the same cell, an elastic material of stiffness
2e-3 in parallel, which keeps the tangent invertible once the stress has
softened to zero. It is `model.elasticity` on a second FE space of the same
mesh: each law reads its own `E` on its own space.

Running::

    python validation/03_plasticity/plas9/run.py
"""

import math
import os

import pyrucast as pc

E, NU = 40000.0, 0.21
ALPHA, BETA, K, H = 0.441, 1.02, 2.0, -10000.0
LX, LY, LZ = 1.5, 1.5, 1.0
RATE = 1.5e-4  # imposed u_x per unit time
TIMES = [0.1 * i for i in range(41)]


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(3)
    corners = [[0.0, 0.0], [LX, 0.0], [LX, LY], [0.0, LY]]
    base = pc.Mesh(coords, "QUA4")
    base.unit().add_cell([coords.add_node(p + [0.0]) for p in corners])
    block = pc.mesh.extrude(base, [0.0, 0.0, LZ], 1)
    face_x0 = pc.mesh.points_on_plane(block, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    face_x1 = pc.mesh.points_on_plane(block, [LX, 0.0, 0.0], [1.0, 0.0, 0.0])
    face_y0 = pc.mesh.points_on_plane(block, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    face_z0 = pc.mesh.points_on_plane(block, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    corner = block.nearest_node([LX, LY, LZ])

    fes = pc.FiniteElementSpace(block)
    law = pc.model.drucker_prager(fes, "full_3d")
    parallel = pc.model.elasticity(pc.FiniteElementSpace(block), "full_3d")
    pull = pc.model.dirichlet(face_x1, law, "u_x")
    model = (
        law
        | parallel
        | pc.model.dirichlet(face_x0, law, "u_x")
        | pc.model.dirichlet(face_y0, law, "u_y")
        | pc.model.dirichlet(face_z0, law, "u_z")
        | pull
    )
    materials = pc.element_field.material_field(
        parallel, [("E", 2e-3), ("nu", 0.0)]
    ) | pc.element_field.material_field(
        law,
        [
            ("E", E),
            ("nu", NU),
            ("friction", ALPHA),
            ("beta", BETA),
            ("k", K),
            ("psi", ALPHA),
            ("delta", BETA),
            ("friction_ult", ALPHA),
            ("beta_ult", BETA),
            ("k_ult", K),
            ("H", H),
        ],
    )
    frames = []
    for t in (0.0, TIMES[-1]):
        frames.append((t, pc.NodeField(pull.multiplier_mesh(), ["imposed"]) + RATE * t))
    data = {
        "model": model,
        "times": TIMES,
        "materials": materials,
        "loads": pc.Evolution(frames),
    }
    pc.stepping.step_by_step(data)

    m = math.sqrt(2.0 * ALPHA**2 + BETA**2) / (ALPHA + BETA)
    eps_yield = K / ((ALPHA + BETA) * E)
    worst = 0.0
    strains, stresses, exacts = [], [], []
    for record in data["results"]:
        eps = record["displacement"].value(corner, "u_x") / LX
        sigma = record["behaviour"][0].value(0, 0, "sigma_xx")
        if eps < eps_yield:
            exact = E * eps
        else:
            exact = max((K + H * m * eps) / ((ALPHA + BETA) + H * m / E), 0.0)
        worst = max(worst, abs(sigma - exact) / K)
        strains.append(eps)
        stresses.append(sigma)
        exacts.append(exact)

    if out:
        curve = pc.SubEvolution(list(zip(strains, stresses)))
        curve.plot(
            x_label="ε_xx",
            y_label="σ_xx",
            title="plas9 — computed σ_xx(ε_xx)",
            save=os.path.join(out, "curve.svg"),
        )
    return {"curve_gap": worst}


if __name__ == "__main__":
    print(main("."))
