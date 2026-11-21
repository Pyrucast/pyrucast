"""joi41 — two stacked cubes, continuous and joined by a stiff joint, HEX8.

Problem: two cubes of side 2, one above the other (z from −4 to 0), E =
2.1e11, ν = 0.3, are pulled by 1e5 Pa on their upper face; on their base
z = −4, the corner (2, 2) is held, the edge x = 2 along x and z, the edge
y = 2 along y and z, the corner (0, 0) along z. They are computed twice:
as one continuous body, and as two bodies joined at z = −2 by a joint of
stiffness k_s = k_n = 4.2e15 per unit area. Reference: the second
computation gives the first one's displacement, within 1e-9.

The joint is `model.interface_transfer` between the two facing squares,
whose `h_u_x`, `h_u_y` (tangential, k_s) and `h_u_z` (normal, k_n) are its
stiffnesses: the joint plane is z = constant, so the global components are
its own.

Running::

    python validation/11_joint/joi41/run.py
"""

import os

import pyrucast as pc

E, NU = 2.1e11, 0.3
SIDE = 2.0
K_S, K_N = 4.2e15, 4.2e15
PULL = 1e5  # Pa, on the upper face


def square(coords, z):
    """A QUA4 square of side SIDE at height z, counter-clockwise seen from +z."""
    mesh = pc.Mesh(coords, "QUA4")
    mesh.unit().add_cell(
        [
            coords.add_node([0.0, 0.0, z]),
            coords.add_node([SIDE, 0.0, z]),
            coords.add_node([SIDE, SIDE, z]),
            coords.add_node([0.0, SIDE, z]),
        ]
    )
    return mesh


def solve(coords, cubes, joint, extra_model):
    """u for the cubes, held and pulled as in the `.dgibi` file."""
    fes = pc.FiniteElementSpace(cubes)
    elasticity = pc.model.elasticity(fes, "full_3d")
    corner = pc.mesh.poi1_from_nodes([cubes.nearest_node([SIDE, SIDE, -2.0 * SIDE])])
    edge_x = pc.mesh.points_on_line(
        cubes, [SIDE, 0.0, -2.0 * SIDE], [SIDE, SIDE, -2.0 * SIDE]
    )
    edge_y = pc.mesh.points_on_line(
        cubes, [SIDE, SIDE, -2.0 * SIDE], [0.0, SIDE, -2.0 * SIDE]
    )
    origin = pc.mesh.poi1_from_nodes([cubes.nearest_node([0.0, 0.0, -2.0 * SIDE])])
    # The blocked nodes of each unknown, each node once (`consolidate`).
    model = (
        elasticity
        | pc.model.dirichlet(pc.mesh.consolidate(corner | edge_x), elasticity, "u_x")
        | pc.model.dirichlet(pc.mesh.consolidate(corner | edge_y), elasticity, "u_y")
        | pc.model.dirichlet(
            pc.mesh.consolidate(corner | edge_x | edge_y | origin), elasticity, "u_z"
        )
    )
    if joint is not None:
        model = model | extra_model(elasticity)
    top = pc.mesh.elements_on(
        pc.mesh.skin(cubes),
        pc.mesh.points_on_plane(cubes, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    )
    # A pull on the outward-oriented upper face: φ_n = +p.
    pull = pc.model.flux(pc.FiniteElementSpace(top), elasticity, "n")
    model = model | pull
    materials = pc.element_field.material_field(
        model,
        [
            ("E", E),
            ("nu", NU),
            ("phi_n", PULL),
            ("h_u_x", K_S),
            ("h_u_y", K_S),
            ("h_u_z", K_N),
        ],
    )
    forces = pc.node_field.external_forces(model, materials)
    return pc.solver.solve(pc.matrix.stiffness(model, materials), forces)


def main(out: str | None = None) -> dict[str, float]:
    # ── One continuous body: two layers of HEX8 from z = −4 up to z = 0 ─────
    coords = pc.Coords(3)
    continuous = pc.mesh.extrude(square(coords, -2.0 * SIDE), [0.0, 0.0, 2.0 * SIDE], 2)
    u = solve(coords, continuous, None, None)
    uz_continuous = u.value(continuous.nearest_node([0.0, 0.0, 0.0]), "u_z")

    # ── Two bodies, each extruded from its own square at z = −2 ─────────────
    coords = pc.Coords(3)
    face_lower = square(coords, -SIDE)
    face_upper = square(coords, -SIDE)
    lower = pc.mesh.extrude(face_lower, [0.0, 0.0, -SIDE], 1)
    upper = pc.mesh.extrude(face_upper, [0.0, 0.0, SIDE], 1)
    cubes = lower | upper

    def joint(elasticity):
        return pc.model.interface_transfer(
            pc.FiniteElementSpace(face_upper),
            pc.FiniteElementSpace(face_lower),
            elasticity,
            ["u_x", "u_y", "u_z"],
        )

    u = solve(coords, cubes, True, joint)
    uz_joint = u.value(upper.nearest_node([0.0, 0.0, 0.0]), "u_z")

    if out:
        cubes.plot(title="joi41 — two cubes", save=os.path.join(out, "mesh.svg"))
    return {"uz_continuous": uz_continuous, "uz_joint": uz_joint}


if __name__ == "__main__":
    print(main("."))
