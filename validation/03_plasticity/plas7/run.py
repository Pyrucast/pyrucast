"""plas7 — pure bending of a rectangle up to full plasticity, plane strain.

Problem: a rectangle 12.5 mm wide (x) and 100 mm high (y, from −50 to
50 mm), of an elastic, perfectly plastic material, in plane strain, is bent
by u_x = 1e-3·t·y/(50 mm) imposed on its side x = 12.5 mm; its side x = 0 is
a symmetry plane (u_x = 0) and the origin is held along y. Reference: the
bending moment of R. Hill's solution at seven load levels, up to the fully
plastic moment, and σ_xx at two nodes of the loaded side at the last one.

Mesh: the `.dgibi` file's regular grid of 32 × 4 QUA4.

The moment is taken from the support forces on the loaded side, Σ y·F_x, in
the `.dgibi` file's unit of 1e4 N·m: exact by equilibrium. The `.dgibi` file integrates y·σ_xx along that side, with
σ_xx extrapolated to the nodes and averaged between cells; the two estimates
differ by the error of that integration. The stresses are read by `project`
in one cell holding the node, from its Gauss points.

Running::

    python validation/03_plasticity/plas6/run.py
"""

import os

import pyrucast as pc

E, NU, SIGMA_Y = 2.1e11, 0.3, 483e6
WIDTH, HALF_HEIGHT = 12.5e-3, 50e-3
N_HEIGHT, N_WIDTH = 32, 4
ROTATION = 1e-3  # u_x at y = 50 mm per unit time
TIMES = [0.0, 0.029435, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
KINEMATICS = "plane_strain"
MOMENT_UNIT = 1e4  # N·m, the `.dgibi` file's


def main(out: str | None = None) -> dict[str, float]:
    coords = pc.Coords(2)
    pa = coords.add_node([WIDTH, HALF_HEIGHT])
    pb = coords.add_node([WIDTH, -HALF_HEIGHT])
    pc_ = coords.add_node([0.0, -HALF_HEIGHT])
    pd = coords.add_node([0.0, HALF_HEIGHT])
    loaded_side = pc.mesh.line(pa, pb, N_HEIGHT)
    symmetry_side = pc.mesh.line(pc_, pd, N_HEIGHT)
    rectangle = pc.mesh.transfinite(
        loaded_side,
        pc.mesh.line(pb, pc_, N_WIDTH),
        symmetry_side,
        pc.mesh.line(pd, pa, N_WIDTH),
    )
    loaded_nodes = pc.mesh.consolidate(loaded_side.to_poi1())
    origin = rectangle.nearest_node([0.0, 0.0])

    fes = pc.FiniteElementSpace(rectangle)
    plasticity = pc.model.plasticity_perfect(fes, KINEMATICS)
    bend = pc.model.dirichlet(loaded_nodes, plasticity, "u_x")
    model = (
        plasticity
        | pc.model.dirichlet(
            pc.mesh.consolidate(symmetry_side.to_poi1()), plasticity, "u_x"
        )
        | pc.model.dirichlet(pc.mesh.poi1_from_nodes([origin]), plasticity, "u_y")
        | bend
    )
    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("sigma_y", SIGMA_Y)]
    )
    # u_x = ROTATION·t·y/HALF_HEIGHT on the loaded side, here for t = 1.
    unit = pc.node_field.positions(bend.multiplier_mesh(), ["Y"]).rename_component(
        "Y", "imposed"
    ) * (ROTATION / HALF_HEIGHT)
    side = [loaded_nodes.node(0, i, 0) for i in range(loaded_nodes.cell_count)]
    data = {
        "model": model,
        "times": TIMES,
        "materials": materials,
        "loads": pc.Evolution([(0.0, unit * 0.0), (1.0, unit)]),
    }
    pc.stepping.step_by_step(data)

    values = {}
    for k, record in enumerate(data["results"][1:], start=1):
        reactions = record["reactions"]
        moment = 0.0
        for node in side:
            moment += node.position()[1] * reactions.value(node, "f_x")
        values[f"M_{k}"] = moment / MOMENT_UNIT

    # σ_xx at A and at the second node of the side, read in one cell each.
    point_a = rectangle.nearest_node([WIDTH, HALF_HEIGHT])
    point_2 = rectangle.nearest_node(
        [WIDTH, HALF_HEIGHT - 2.0 * HALF_HEIGHT / N_HEIGHT]
    )
    stress = pc.node_field.project(
        data["results"][-1]["behaviour"],
        rectangle,
        pc.mesh.poi1_from_nodes([point_a, point_2]),
    )
    values["sxx_A"] = stress.value(point_a, "sigma_xx") / 1e6
    values["sxx_A2"] = stress.value(point_2, "sigma_xx") / 1e6

    if out:
        rectangle.plot(title="plas7 — QUA4 mesh", save=os.path.join(out, "mesh.svg"))
    return values


if __name__ == "__main__":
    print(main("."))
