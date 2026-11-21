"""elas11 — thick cylinder-cone-sphere under a linear temperature (NAFEMS LE11).

Problem: an axisymmetric thick solid — a spherical part (radii 1 and 1.4),
a conical part and a cylindrical part — is heated to T = r + z (°C), with
α = 2.3e-4 /°C and the reference temperature 0; u_z = 0 on its two flat
faces. Reference: the axial stress at A (1, 0) is σ_zz = −105 MPa.

The QUA8 mesh is the one of the Cast3M deck, node for node: 2 elements
across and 10 along the meridian (4 on the arcs of 30°, 2 on the arc from
30° to 45°, 4 on the cone and on the inner straight part, 2 on the outer
cylinder). The corner nodes inside are placed by a transfinite (Coons)
interpolation of the boundary nodes, the mid-nodes of the arcs on the arcs,
the other mid-nodes at the middle of their edge.

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
N_ACROSS, N_ALONG = 2, 10  # elements across and along the meridian
PE = (math.cos(math.radians(45.0)), math.sin(math.radians(45.0)))
PH, PI, PG = (0.7071, 1.79), (1.0, 1.79), (1.0, 1.39)
PD = (1.4 * math.cos(math.radians(30.0)), 0.7)


def main(out: str | None = None) -> dict[str, float]:
    # Node grid at half-element steps: i across from the inner side (i = 0)
    # to the outer one (i = NI), j along from z = 0 (j = 0) to z = 1.79.
    ni, nj = 2 * N_ACROSS, 2 * N_ALONG
    bottom, top, inner, outer = [], [], [], []
    for i in range(ni + 1):
        t = i / ni
        bottom.append((1.0 + 0.4 * t, 0.0))
        top.append((PH[0] + t * (PI[0] - PH[0]), 1.79))
    for j in range(nj + 1):
        if j <= 8:  # arc of radius 1 from 0° to 30°, then of radius 1.4
            angle = math.radians(30.0 * j / 8)
            inner.append((math.cos(angle), math.sin(angle)))
            outer.append((1.4 * math.cos(angle), 1.4 * math.sin(angle)))
            continue
        if j <= 12:  # inner arc from 30° to 45°
            angle = math.radians(30.0 + 15.0 * (j - 8) / 4)
            inner.append((math.cos(angle), math.sin(angle)))
        else:  # inner straight part, from 45° up to H
            t = (j - 12) / 8
            inner.append((PE[0] + t * (PH[0] - PE[0]), PE[1] + t * (PH[1] - PE[1])))
        if j <= 16:  # cone, from D to G
            t = (j - 8) / 8
            outer.append((PD[0] + t * (PG[0] - PD[0]), PD[1] + t * (PG[1] - PD[1])))
        else:  # outer cylinder, from G to I
            t = (j - 16) / 4
            outer.append((1.0, PG[1] + t * (PI[1] - PG[1])))

    position = {}
    for i in range(0, ni + 1, 2):
        for j in range(0, nj + 1, 2):
            u, v = i / ni, j / nj
            point = []
            for d in range(2):
                coons = (
                    (1 - v) * bottom[i][d]
                    + v * top[i][d]
                    + (1 - u) * inner[j][d]
                    + u * outer[j][d]
                    - (1 - u) * (1 - v) * bottom[0][d]
                    - u * (1 - v) * bottom[ni][d]
                    - (1 - u) * v * top[0][d]
                    - u * v * top[ni][d]
                )
                point.append(coons)
            position[i, j] = point
    for i in range(ni + 1):
        for j in range(nj + 1):
            if (i % 2, j % 2) == (0, 0) or (i % 2, j % 2) == (1, 1):
                continue
            if j == 0:
                position[i, j] = list(bottom[i])
            elif j == nj:
                position[i, j] = list(top[i])
            elif i == 0:
                position[i, j] = list(inner[j])
            elif i == ni:
                position[i, j] = list(outer[j])
            elif i % 2 == 1:
                a, b = position[i - 1, j], position[i + 1, j]
                position[i, j] = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
            else:
                a, b = position[i, j - 1], position[i, j + 1]
                position[i, j] = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]

    coords = pc.Coords.axisymmetric()
    node = {}
    for key in sorted(position):
        node[key] = coords.add_node(position[key])
    solid = pc.Mesh(coords, "QUA8")
    for i in range(0, ni, 2):
        for j in range(0, nj, 2):
            solid.unit().add_cell(
                [
                    node[i, j],
                    node[i + 2, j],
                    node[i + 2, j + 2],
                    node[i, j + 2],
                    node[i + 1, j],
                    node[i + 2, j + 1],
                    node[i + 1, j + 2],
                    node[i, j + 1],
                ]
            )
    flat_faces = pc.mesh.poi1_from_nodes(
        [node[i, 0] for i in range(ni + 1)] + [node[i, nj] for i in range(ni + 1)]
    )
    point_a, point_b = node[0, 0], node[ni, 0]

    fes = pc.FiniteElementSpace(solid)
    elasticity = pc.model.elasticity(fes, "axisymmetric")
    model = elasticity | pc.model.dirichlet(flat_faces, elasticity, "u_y")
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("alpha", ALPHA)]
    )
    temperature = pc.NodeField(solid, ["T"])
    for (i, j), n in node.items():
        r, z = position[i, j]
        temperature[0].set_value(n, "T", r + z)
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
    at_a = pc.node_field.project(stress, solid, pc.mesh.poi1_from_nodes([point_a]))

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
        "szz_A": at_a.value(point_a, "sigma_yy") / 1e6,
        "ur_B": u.value(point_b, "u_x"),
    }


if __name__ == "__main__":
    print(main("."))
