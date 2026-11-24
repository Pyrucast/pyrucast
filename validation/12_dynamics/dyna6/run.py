"""dyna6 — bending modes of a cantilever beam (Timoshenko's analytical values).

Problem: a steel beam of length 2 m, section 0.01 m², clamped at one end. Its
weak plane (I = 6.25e-6 m⁴) gives the two lowest modes, f₁ = 17.71 Hz and
f₂ = 110.98 Hz; the strong plane (I = 1e-3 m⁴), the extension and the torsion
come far above.

Mesh: the `.dgibi` file's line, through the node at 0.783·L where the second
mode has its node — 16 cells on the first 1.566 m and 4 on the last 0.434 m,
as its density of 0.1 places them. Bernoulli cells with their consistent mass,
rotary inertia included; the root clamped on its six unknowns.

Running::

    python validation/12_dynamics/dyna6/run.py
"""

import math

import pyrucast as pc

E, NU, RHO = 2e11, 0.3, 7800.0
LENGTH = 2.0
SECTION, I_Y, I_Z, TORSION = 0.01, 6.25e-6, 1e-3, 1e-2


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(3)
    q1 = coords.add_node([0.0, 0.0, 0.0])
    q2 = coords.add_node([0.783 * LENGTH, 0.0, 0.0])
    q3 = coords.add_node([LENGTH, 0.0, 0.0])
    line = pc.mesh.line(q1, q2, 16) | pc.mesh.line(q2, q3, 4)

    beam = pc.model.bernoulli(pc.FiniteElementSpace(line, interpolation="HERMITE3"))
    clamp = pc.mesh.poi1_from_nodes([q1])
    model = beam
    for variable in beam.primal_vars():
        model = model | pc.model.dirichlet(clamp, beam, variable)
    materials = pc.element_field.material_field(
        model,
        [
            ("E", E),
            ("A", SECTION),
            ("I_y", I_Y),
            ("I_z", I_Z),
            ("J", TORSION),
            ("G", E / (2.0 * (1.0 + NU))),
            ("rho", RHO),
        ],
    )
    stiffness = pc.matrix.stiffness(model, materials)
    mass = pc.matrix.mass(beam, materials)

    modes = pc.solver.eigen(stiffness, mass, 2)
    f1, f2 = (math.sqrt(m["eigenvalue"]) / (2.0 * math.pi) for m in modes)
    return {"f1": f1, "f2": f2}


if __name__ == "__main__":
    print(main("."))
