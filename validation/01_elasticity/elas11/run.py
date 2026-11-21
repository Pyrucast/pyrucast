"""elas11 — thick cylinder-cone-sphere under a linear temperature (NAFEMS LE11).

Problem: an axisymmetric thick solid — a spherical part (radii 1 and 1.4),
a conical part and a cylindrical part — is heated to T = r + z (°C), with
α = 2.3e-4 /°C and the reference temperature 0; u_z = 0 on its two flat
faces. Reference: the axial stress at A (1, 0) is σ_zz = −105 MPa.

The QUA8 mesh has the `.dgibi` file's boundary lines: 2 elements across and 10 along
the meridian (4 on the arcs of 30°, 2 on the arc from 30° to 45°, 4 on the
cone and on the inner straight part, 2 on the outer cylinder), filled by
`transfinite` as the `.dgibi` file's mesher does. `transfinite` takes straight-sided
SEG2 lines and puts the QUA8 mid-nodes on the chords; the `.dgibi` file's are on the
arcs.

In pyrucast's axisymmetric coordinates x is r and y is z: the axial stress
is `sigma_yy`. It is read in the only cell holding A, from its Gauss points.

Running::

    python validation/01_elasticity/elas11/run.py
"""

import math
import os

import pyrucast as pc

E, NU, ALPHA = 210e9, 0.3, 2.3e-4  # Pa, —, /°C
T_REF = 0.0  # °C
COS30, SIN30 = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
COS45 = math.cos(math.radians(45.0))


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords.axisymmetric()
    origin = coords.add_node([0.0, 0.0])
    pa = coords.add_node([1.0, 0.0])
    pb = coords.add_node([1.4, 0.0])
    pc_ = coords.add_node([COS30, SIN30])
    pd = coords.add_node([1.4 * COS30, 0.7])
    pe = coords.add_node([COS45, COS45])
    pg = coords.add_node([1.0, 1.39])
    ph = coords.add_node([0.7071, 1.79])
    pi = coords.add_node([1.0, 1.79])
    # The four sides, counter-clockwise, as in the `.dgibi` file.
    bottom = pc.mesh.line(pa, pb, 2)
    outer = pc.mesh.consolidate(
        pc.mesh.arc(pb, origin, pd, 4)
        | pc.mesh.line(pd, pg, 4)
        | pc.mesh.line(pg, pi, 2)
    )
    top = pc.mesh.line(pi, ph, 2)
    inner = pc.mesh.consolidate(
        pc.mesh.line(ph, pe, 4)
        | pc.mesh.arc(pe, origin, pc_, 2)
        | pc.mesh.arc(pc_, origin, pa, 4)
    )
    solid = pc.mesh.transfinite(bottom, outer, top, inner, "QUA8")
    flat_faces = pc.mesh.consolidate(
        pc.mesh.points_on_line(solid, [1.0, 0.0], [1.4, 0.0])
        | pc.mesh.points_on_line(solid, [0.7071, 1.79], [1.0, 1.79])
    )

    fes = pc.FiniteElementSpace(solid)
    elasticity = pc.model.elasticity(fes, "axisymmetric")
    model = elasticity | pc.model.dirichlet(flat_faces, elasticity, "u_y")
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("alpha", ALPHA)]
    )
    # T = r + z at the nodes: x is r, y is z.
    temperature = pc.node_field.positions(solid, ["X"]).rename_component(
        "X", "T"
    ) + pc.node_field.positions(solid, ["Y"]).rename_component("Y", "T")
    eps_th = pc.element_field.thermal_strain(
        pc.element_field.interp_to_gauss(temperature, fes), materials, fes, T_REF
    )
    sigma_th = pc.element_field.elastic_stress(model, eps_th, materials)
    forces = (
        pc.node_field.divergence(sigma_th, "sigma")
        .rename_component("div_sigma_x", "f_x")
        .rename_component("div_sigma_y", "f_y")
    )
    u = pc.solver.solve(pc.matrix.stiffness(model, materials), forces)

    strain = pc.element_field.strain(model, u, materials)
    stress = pc.element_field.elastic_stress(model, strain, materials) - sigma_th
    at_a = pc.node_field.project(stress, solid, pc.mesh.poi1_from_nodes([pa]))

    if out:
        solid.plot(title="elas11 — QUA8 mesh", save=os.path.join(out, "mesh.svg"))
        solid.plot(
            title="elas11 — axial stress σ_zz (MPa)",
            field=stress * 1e-6,
            component="sigma_yy",
            cmap="viridis",
            save=os.path.join(out, "sigma_zz.svg"),
        )
    return {
        "szz_A": at_a.value(pa, "sigma_yy") / 1e6,
        "ur_B": u.value(pb, "u_x"),
    }


if __name__ == "__main__":
    print(main("."))
