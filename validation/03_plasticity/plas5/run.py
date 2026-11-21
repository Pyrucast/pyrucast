"""plas5 — equibiaxial pull of a Drucker-Prager element, plane stress, QUA4.

Problem: one square QUA4 of side 1.5, held by u_x = 0 on x = 0 and u_y = 0
on y = 0, is pulled by u_x = u_y = 1.5e-4·t on its two other sides, up to
t = 2. The material is the `.dgibi` file's two-surface Drucker-Prager: an initial
surface α·I₁ + β·q = K and an ultimate one η·I₁ + μ·q = K_L, with H = 0.
Reference: the equibiaxial stress at the apex of the ultimate surface,
σ_xx = σ_yy = K_L / (2η + μ).

The Cast3M parameters map to pyrucast's as ALFA → friction, BETA → beta,
K → k, GAMM → psi, DELT → delta, ETA → friction_ult, MU → beta_ult,
KL → k_ult, H → H.

Running::

    python validation/03_plasticity/plas5/run.py
"""

import os

import pyrucast as pc

E, NU = 40000.0, 0.21
SIDE = 1.5
RATE = 1.5e-4  # imposed displacement per unit time
DRUCKER_PRAGER = [
    ("friction", 0.441),  # ALFA
    ("beta", 1.02),  # BETA
    ("k", 4.1),  # K
    ("psi", 0.441),  # GAMM
    ("delta", 1.02),  # DELT
    ("friction_ult", 0.441),  # ETA
    ("beta_ult", 1.02),  # MU
    ("k_ult", 3.56),  # KL
    ("H", 0.0),
]


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(2)
    a = coords.add_node([0.0, 0.0])
    b = coords.add_node([SIDE, 0.0])
    c = coords.add_node([SIDE, SIDE])
    d = coords.add_node([0.0, SIDE])
    square = pc.Mesh(coords, "QUA4")
    square.unit().add_cell([a, b, c, d])

    fes = pc.FiniteElementSpace(square)
    law = pc.model.drucker_prager(fes, "plane_stress")
    pull_x = pc.model.dirichlet(pc.mesh.poi1_from_nodes([b, c]), law, "u_x")
    pull_y = pc.model.dirichlet(pc.mesh.poi1_from_nodes([c, d]), law, "u_y")
    model = (
        law
        | pc.model.dirichlet(pc.mesh.poi1_from_nodes([a, d]), law, "u_x")
        | pc.model.dirichlet(pc.mesh.poi1_from_nodes([a, b]), law, "u_y")
        | pull_x
        | pull_y
    )
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU)] + DRUCKER_PRAGER
    )
    frames = []
    for t in (0.0, 2.0):
        imposed = (pc.NodeField(pull_x.multiplier_mesh(), ["imposed"]) + RATE * t) | (
            pc.NodeField(pull_y.multiplier_mesh(), ["imposed"]) + RATE * t
        )
        frames.append((t, imposed))
    data = {
        "model": model,
        "times": [0.0, 1.0, 2.0],
        "materials": materials,
        "loads": pc.Evolution(frames),
    }
    pc.stepping.step_by_step(data)
    behaviour = data["results"][-1]["behaviour"][0]

    if out:
        square.plot(title="plas5 — QUA4", save=os.path.join(out, "mesh.svg"))
    return {
        "sxx": behaviour.value(0, 0, "sigma_xx"),
        "syy": behaviour.value(0, 0, "sigma_yy"),
    }


if __name__ == "__main__":
    print(main("."))
