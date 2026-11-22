"""elas13 — pinned beam on an elastic foundation, Bernoulli beam on SEG2.

Problem: a beam along x from A (−L/2, 0) to B (L/2, 0), L = π·√10 / 2,
E = 2e11, A = 1000, I = 1e-4, pinned at A and B (u_x = u_y = 0), rests on a
Winkler foundation of stiffness K = 840 000 N/m per metre. It carries
−10 kN at the centre D, −5000 N/m along its length, and the end moments
M_z = −15 000 N·m at A and +15 000 N·m at B. Reference: the rotation and
the reaction at A, the deflection and the bending moment at D.

The `.dgibi` file turns the foundation into nodal springs K·L/(2·n) at both
ends of each of its n cells (`APPUI`), and the distributed load into nodal
forces the same way. Here the foundation is `model.boundary_transfer` on
the beam's line, `h_u_y = K`, whose stiffness `lump` gathers on the
diagonal: the same nodal springs. The loads are the same nodal forces.

Running::

    python validation/01_elasticity/elas13/run.py
"""

import math

import pyrucast as pc

E, AREA, INERTIA = 2e11, 1000.0, 1e-4
K = 840_000.0  # foundation stiffness, N/m per metre
LENGTH = math.pi * math.sqrt(10.0) / 2.0
N_CELLS = 8
F_D = -10_000.0  # point load at D
Q = -5000.0  # distributed load, N/m
M_END = 15_000.0  # end moments, −M_END at A, +M_END at B


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(2)
    a = coords.add_node([-LENGTH / 2.0, 0.0])
    b = coords.add_node([LENGTH / 2.0, 0.0])
    line = pc.mesh.line(a, b, N_CELLS)
    d = line.nearest_node([0.0, 0.0])

    beam = pc.model.bernoulli(pc.FiniteElementSpace(line, interpolation="HERMITE3"))
    foundation = pc.model.boundary_transfer(pc.FiniteElementSpace(line), beam, ["u_y"])
    pins = pc.mesh.poi1_from_nodes([a, b])
    supports = pc.model.dirichlet(pins, beam, "u_x") | pc.model.dirichlet(
        pins, beam, "u_y"
    )
    materials = pc.element_field.material_field(
        beam | foundation | supports,
        [("E", E), ("A", AREA), ("I", INERTIA), ("h_u_y", K), ("a_ext_u_y", 0.0)],
    )
    stiffness = (
        pc.matrix.stiffness(beam | supports, materials)
        + pc.matrix.stiffness(foundation, materials).lump()
    )

    # The `.dgibi` file's loads: Q·L/(2n) at both ends of each cell.
    nodal = Q * LENGTH / (2.0 * N_CELLS)
    load = pc.NodeField(line, ["f_y", "m_z"])
    for i in range(line.cell_count):
        for k in range(2):
            node = line.node(0, i, k)
            load[0].set_value(node, "f_y", load.value(node, "f_y") + nodal)
    load[0].set_value(d, "f_y", load.value(d, "f_y") + F_D)
    load[0].set_value(a, "m_z", -M_END)
    load[0].set_value(b, "m_z", M_END)

    u = pc.solver.solve(stiffness, load)
    reactions = -pc.node_field.internal_forces(
        supports, pc.element_field.initial_state(supports, materials), u, materials
    )
    strain = pc.element_field.strain(beam, u, materials)
    forces = pc.element_field.integrate_behavior(
        beam, strain, materials, pc.element_field.initial_state(beam, materials)
    )
    at_d = pc.node_field.project(forces, line, pc.mesh.poi1_from_nodes([d]))

    return {
        "rz_A": u.value(a, "r_z"),
        "fy_A": reactions.value(a, "f_y"),
        "uy_D": u.value(d, "u_y"),
        "mz_D": at_d.value(d, "M"),
    }


if __name__ == "__main__":
    print(main("."))
