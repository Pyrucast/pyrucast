"""elas4 — Z-section cantilever in torsion (NAFEMS LE5), Kirchhoff shell on QUA4.

Problem: a cantilever 10 m long whose cross-section is a Z (two flanges of
1 m joined by a web of 2 m), clamped at x = 0, is twisted at x = 10 by two
opposite forces of 600 kN spread over the flanges (1.2 MN·m). Reference: the
axial stress on the mid-surface at A (2.5, 0, 0) is σ_xx = −108 MPa.

The mesh is the Cast3M deck's: the section line (2 elements per segment)
translated along x in 16 layers, 96 QUA4. Each cell's first edge runs along
x, so the local x axis of the shell is the beam's axis and σ_xx = N_xx / h.
A lies on the edge of the first flange, between two cells; the stress is
read in each of them, from its own Gauss points.

Running::

    python validation/01_elasticity/elas4/run.py
"""

import os

import pyrucast as pc

E, NU, THICKNESS = 210e9, 0.3, 0.1
LENGTH, N_LENGTH = 10.0, 16
# The section line in the plane x = 0, 2 elements per segment.
SECTION = [
    (0.0, 0.0),
    (0.0, 0.5),
    (0.0, 1.0),
    (1.0, 1.0),
    (2.0, 1.0),
    (2.0, 1.5),
    (2.0, 2.0),
]  # (y, z)
# F_z at x = 10 on the nodes of the section line: −600 kN on the first
# flange, +600 kN on the second.
END_FORCES = {0: -150e3, 1: -300e3, 2: -150e3, 4: 150e3, 5: 300e3, 6: 150e3}
X_A = 2.5


def main(out: str | None = None) -> dict[str, float]:
    # Node (s, i): s-th point of the section line, i-th layer along x.
    coords = pc.Coords(3)
    node = {}
    for i in range(N_LENGTH + 1):
        for s, (y, z) in enumerate(SECTION):
            node[s, i] = coords.add_node([LENGTH * i / N_LENGTH, y, z])
    surface = pc.Mesh(coords, "QUA4")
    for i in range(N_LENGTH):
        for s in range(len(SECTION) - 1):
            surface.unit().add_cell(
                [node[s, i], node[s, i + 1], node[s + 1, i + 1], node[s + 1, i]]
            )
    clamped = pc.mesh.poi1_from_nodes([node[s, 0] for s in range(len(SECTION))])

    shell = pc.model.shell(pc.FiniteElementSpace(surface), "kirchhoff")
    model = shell
    for variable in ("u_x", "u_y", "u_z"):  # displacements only, as the deck
        model = model | pc.model.dirichlet(clamped, shell, variable)
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("h", THICKNESS)]
    )
    loaded = [node[s, N_LENGTH] for s in END_FORCES]
    load = pc.NodeField(pc.mesh.poi1_from_nodes(loaded), ["f_z"])
    for s, force in END_FORCES.items():
        load[0].set_value(node[s, N_LENGTH], "f_z", force)
    u = pc.solver.solve(pc.matrix.stiffness(model, materials), load)

    strain = pc.element_field.strain(model, u, materials)
    forces = pc.element_field.integrate_behavior(
        model, strain, materials, pc.element_field.initial_state(model, materials)
    )
    # Two points just inside the two cells that share A.
    before = coords.add_node([X_A - 1e-9, 0.0, 0.0])
    after = coords.add_node([X_A + 1e-9, 0.0, 0.0])
    at_a = pc.node_field.project(
        forces, surface, pc.mesh.poi1_from_nodes([before, after])
    )
    sxx_before = at_a.value(before, "N_xx") / THICKNESS / 1e6
    sxx_after = at_a.value(after, "N_xx") / THICKNESS / 1e6

    if out:
        surface.plot(title="elas4 — QUA4 mesh", save=os.path.join(out, "mesh.svg"))
        surface.plot(
            title="elas4 — N_xx / h (MPa)",
            field=forces * (1e-6 / THICKNESS),
            component="N_xx",
            cmap="viridis",
            save=os.path.join(out, "sigma_xx.svg"),
        )
    return {
        "sxx_A": sxx_before,
        "sxx_A_after": sxx_after,
        "uz_tip": u.value(node[0, N_LENGTH], "u_z"),
    }


if __name__ == "__main__":
    print(main("."))
