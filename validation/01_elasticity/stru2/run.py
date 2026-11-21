"""stru2 — clamped orthotropic square plate, plane stress, QUA4.

Problem: a square plate of side 1, centred at the origin, of an orthotropic
material whose first axis makes 30° with x, is clamped on its side x = −0.5
and pulled along x by a total force of 1e6 N spread evenly over its side
x = 0.5. Reference (comparison with six other codes): the displacements of
the corners A (0.5, 0.5), D (0.5, −0.5) and of the centre E.

Mesh: 10 × 10 QUA4, the `.dgibi` file's (a regular grid). The force spread evenly is
a uniform line load of 1e6 N/m, whose consistent nodal forces are the `.dgibi` file's.

Running::

    python validation/01_elasticity/stru2/run.py
"""

import math
import os

import pyrucast as pc

E1, E2, E3 = 2.5737e10, 7.377e9, 7.377e9  # Pa
NU12, NU23, NU13 = 0.31, 0.31, 0.31
G12 = 2.319e9  # Pa; G_13 and G_23 play no part in plane stress
ANGLE = 30.0  # first material axis, from x, degrees
FORCE = 1e6  # total force along x on the side x = 0.5, N
N = 10  # elements per side


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(2)
    pa = coords.add_node([0.5, 0.5])
    pb = coords.add_node([-0.5, 0.5])
    pc_ = coords.add_node([-0.5, -0.5])
    pd = coords.add_node([0.5, -0.5])
    top = pc.mesh.line(pa, pb, N)
    clamped_side = pc.mesh.line(pb, pc_, N)
    bottom = pc.mesh.line(pc_, pd, N)
    loaded_side = pc.mesh.line(pd, pa, N)
    plate = pc.mesh.transfinite(top, clamped_side, bottom, loaded_side)
    pe = plate.nearest_node([0.0, 0.0])

    fes = pc.FiniteElementSpace(plate)
    elasticity = pc.model.elasticity(fes, "plane_stress", symmetry="orthotropic")
    clamped = pc.mesh.consolidate(clamped_side.to_poi1())
    model = (
        elasticity
        | pc.model.dirichlet(clamped, elasticity, "u_x")
        | pc.model.dirichlet(clamped, elasticity, "u_y")
        | pc.model.flux(pc.FiniteElementSpace(loaded_side), elasticity, "f_x")
    )
    a = math.radians(ANGLE)
    materials = pc.element_field.material_field(
        model,
        [
            ("E_1", E1),
            ("E_2", E2),
            ("E_3", E3),
            ("nu_12", NU12),
            ("nu_13", NU13),
            ("nu_23", NU23),
            ("G_12", G12),
            ("G_13", G12),
            ("G_23", G12),
            ("V1X", math.cos(a)),
            ("V1Y", math.sin(a)),
            ("phi_f_x", FORCE / 1.0),  # per unit length of the side
        ],
    )
    forces = pc.node_field.external_forces(model, materials)
    u = pc.solver.solve(pc.matrix.stiffness(model, materials), forces)

    if out:
        plate.plot(title="stru2 — QUA4 mesh", save=os.path.join(out, "mesh.svg"))
        plate.plot(
            title="stru2 — u_x",
            field=u,
            component="u_x",
            cmap="viridis",
            save=os.path.join(out, "u_x.svg"),
        )
    scale = 1e-4  # the document's unit
    return {
        "ux_A": u.value(pa, "u_x") / scale,
        "uy_A": u.value(pa, "u_y") / scale,
        "ux_D": u.value(pd, "u_x") / scale,
        "uy_D": u.value(pd, "u_y") / scale,
        "ux_E": u.value(pe, "u_x") / scale,
        "uy_E": u.value(pe, "u_y") / scale,
    }


if __name__ == "__main__":
    print(main("."))
