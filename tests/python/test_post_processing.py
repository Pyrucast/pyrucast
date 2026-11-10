"""Reading a mechanical result: the equivalent stress, a field read along a
line of points, and a disc sector revolved from its axis.

Each check is measured against a value found independently of the operator: a
polynomial the space holds exactly, a uniform tension whose stress is known, a
distance computed by hand.
"""

import math

import pyrucast as pc
import pytest

P = 7.0  # tension of the bar, Pa


def _hex20_bar():
    """A bar 2 × 1 × 1 in two HEX20 (a SEG3 line extruded twice)."""
    coords = pc.Coords(3)
    a = coords.add_node([0.0, 0.0, 0.0])
    b = coords.add_node([2.0, 0.0, 0.0])
    edge = pc.mesh.line(a, b, 2, "SEG3")
    face = pc.mesh.extrude(edge, [0.0, 1.0, 0.0], 1)
    return pc.mesh.extrude(face, [0.0, 0.0, 1.0], 1)


def _points(positions):
    """Points of a fresh `Coords`, as one POI1 line, and their nodes."""
    coords = pc.Coords(3)
    nodes = [coords.add_node(p) for p in positions]
    return pc.mesh.poi1_from_nodes(nodes), nodes


def _quadratic(x, y, z):
    return 1.0 + x - 2.0 * y + 0.5 * z + x * x - x * y + 2.0 * y * z - z * z


def test_a_gauss_field_of_the_space_reads_back_exactly_on_hex20():
    """A quadratic field brought to the 27 Gauss points of each HEX20 and read
    at points inside the cells gives the polynomial back, as the nodal field
    does: each cell's fit holds what its space holds."""
    volume = _hex20_bar()
    nodal = pc.NodeField(volume, ["a"])
    nodes = pc.mesh.consolidate(pc.mesh.to_poi1(volume))
    for i in range(nodes.cell_count):
        node = nodes.node(0, i, 0)
        nodal[0].set_value(node, "a", _quadratic(*node.position()))
    gauss = pc.element_field.interp_to_gauss(nodal, pc.FiniteElementSpace(volume))

    at = [[0.3, 0.2, 0.7], [1.5, 0.5, 0.5], [1.9, 0.9, 0.1]]
    line, points = _points(at)
    from_gauss = pc.node_field.project(gauss, volume, line)
    from_nodes = pc.node_field.project(nodal, volume, line)
    for node, p in zip(points, at):
        assert from_gauss.value(node, "a") == pytest.approx(_quadratic(*p), abs=1e-10)
        assert from_nodes.value(node, "a") == pytest.approx(_quadratic(*p), abs=1e-10)
    # The method forms, on the field and on its zone.
    assert gauss.project(volume, line).value(points[1], "a") == pytest.approx(
        from_gauss.value(points[1], "a"), abs=1e-14
    )
    assert gauss[0].project(volume, line).value(points[1], "a") == pytest.approx(
        from_gauss.value(points[1], "a"), abs=1e-14
    )


def test_a_line_across_two_cells_shows_their_jump():
    """A value per cell, 1 then 5: points on either side of the shared face
    read their own cell's value, nothing averaged — and the curve along the
    line jumps there."""
    volume = _hex20_bar()
    field = pc.ElementField(pc.FiniteElementSpace(volume), ["s"])
    field[0].set_cell_uniform(0, "s", 1.0)
    field[0].set_cell_uniform(1, "s", 5.0)
    xs = [0.25, 0.75, 0.95, 1.05, 1.25, 1.75]
    line, points = _points([[x, 0.5, 0.5] for x in xs])

    read = pc.node_field.project(field, volume, line)
    assert [read.value(n, "s") for n in points] == pytest.approx(
        [1.0, 1.0, 1.0, 5.0, 5.0, 5.0], abs=1e-12
    )
    curve = read.along(line)
    assert len(curve) == 1
    assert curve[0].abscissas() == pytest.approx([x - 0.25 for x in xs], abs=1e-12)


def test_along_gives_one_curve_per_component_against_the_distance():
    """A broken line 3 then 4 long: abscissas 0, 3, 7, one sub-evolution per
    component, in the order asked, named after it."""
    coords = pc.Coords(2)
    nodes = [coords.add_node(p) for p in ([0.0, 0.0], [3.0, 0.0], [3.0, 4.0])]
    line = pc.mesh.poi1_from_nodes(nodes)
    field = pc.NodeField(line, ["T", "q"])
    for node, t in zip(nodes, [10.0, 40.0, 80.0]):
        field[0].set_value(node, "T", t)
        field[0].set_value(node, "q", -t)

    every = pc.evolution.along(field, line)
    assert [sub.ordinate_type() for sub in every] == ["T", "q"]
    asked = field.along(line, ["q"])
    assert len(asked) == 1
    assert asked[0].abscissas() == pytest.approx([0.0, 3.0, 7.0], abs=1e-12)
    assert [asked[0].interpolate(s) for s in (0.0, 3.0, 7.0)] == [-10.0, -40.0, -80.0]
    with pytest.raises(Exception, match="POI1"):
        pc.evolution.along(field, pc.mesh.line(nodes[0], nodes[1], 2))


def _bar_in_tension(kinematics="full_3d"):
    """The HEX20 bar on its three symmetry planes, pulled by P on x = 2:
    a uniform uniaxial stress σxx = P. Returns the model, its solution and the
    material."""
    volume = _hex20_bar()
    fes = pc.FiniteElementSpace(volume)
    model = pc.model.elasticity(fes, kinematics)
    for axis, variable in enumerate(["u_x", "u_y", "u_z"]):
        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0
        plane = pc.mesh.points_on_plane(volume, [0.0, 0.0, 0.0], normal)
        model = model | pc.model.dirichlet(plane, model, variable)
    skin = pc.mesh.consolidate(pc.mesh.skin(volume))
    end = pc.mesh.elements_on(
        skin,
        pc.mesh.points_on_plane(skin, [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        strict=True,
    )
    pull = pc.model.flux(pc.FiniteElementSpace(end), model, "f_x")
    material = pc.element_field.material_field(model, [("E", 200.0), ("nu", 0.3)])
    load = pc.element_field.material_field(pull, [("phi_f_x", P)])
    u = pc.solver.solve(
        pc.matrix.stiffness(model, material), pc.node_field.external_forces(pull, load)
    )
    return model, u, material


def test_von_mises_of_a_uniform_tension_is_the_tension():
    model, u, material = _bar_in_tension()
    stress = pc.element_field.elastic_stress(
        model, pc.element_field.strain(model, u, material), material
    )
    vm = pc.element_field.von_mises(model, stress)
    assert vm.components() == ["von_mises"]
    assert vm.min("von_mises") == pytest.approx(P, rel=1e-9)
    assert vm.max("von_mises") == pytest.approx(P, rel=1e-9)
    # Hill: in uniaxial tension along x, σ_eq² = (G + H)·σxx².
    hill = pc.element_field.von_mises(model, stress, [1.0, 2.0, 3.0, 0.5, 0.5, 0.5])
    assert hill.max("von_mises") == pytest.approx(math.sqrt(5.0) * P, rel=1e-9)
    with pytest.raises(Exception, match="≥ 0"):
        pc.element_field.von_mises(model, stress, [0.5, 0.5, 0.5, -1.0, 1.5, 1.5])


def test_a_line_revolved_from_the_axis_and_extruded_gives_prisms_and_hexahedra():
    """The support of the tutorial's contact step: a radius of 2 SEG2 turned by
    90° in 7 sectors around the y axis, then extruded through a thickness in
    2 layers."""
    coords = pc.Coords(3)
    centre = coords.add_node([0.0, 0.0, 0.0])
    rim = coords.add_node([0.0, 0.0, -1.0])
    radius = pc.mesh.line(centre, rim, 2)
    sector = pc.mesh.revolve(radius, math.pi / 2.0, 7, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert [zone.element_type for zone in sector] == ["QUA4", "TRI3"]
    assert [zone.cell_count for zone in sector] == [7, 7]
    solid = pc.mesh.extrude(sector, [0.0, 0.2, 0.0], 2)
    assert [zone.element_type for zone in solid] == ["HEX8", "PENTA6"]
    assert pc.mesh.consolidate(pc.mesh.to_poi1(solid)).cell_count == 17 * 3
