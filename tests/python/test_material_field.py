"""Python tests for the material-field build operators
(sub_material_field / material_field / material_field_per_sub_model)."""

import pytest

import pyrucast


def _two_zone_model():
    """Build a 3-node 2-SEG2 mesh, FE space and a model with:
      - HC over both zones via Model.heat_conduction(fes), spanning
        subspace 0 (zone A) and subspace 1 (zone B);
      - a Dirichlet constraint on the leftmost node, composed with `|`.
    Sub-model order: [HC_A (model[0]), HC_B (model[1]), Dirichlet (model[2])].
    Returns (coords, [n0, n1, n2], fes, model).
    """
    c = pyrucast.Coords(1)
    nodes = [c.add_node([i * 1.0]) for i in range(3)]
    zone_a = pyrucast.Mesh(c, "SEG2")
    zone_a.unit().add_cell([nodes[0], nodes[1]])
    zone_b = pyrucast.Mesh(c, "SEG2")
    zone_b.unit().add_cell([nodes[1], nodes[2]])
    mesh = zone_a | zone_b
    fes = pyrucast.FiniteElementSpace(mesh)

    imposed = pyrucast.Mesh.poi1_from_nodes([nodes[0]])
    multiplier = pyrucast.mesh.barycenter(imposed)
    model = pyrucast.Model.heat_conduction(fes) | pyrucast.Model.dirichlet(
        "T", "q", imposed, multiplier
    )
    return c, nodes, fes, model


# ─── sub_material_field (one sub-model) ─────────────────────────────────────


def test_sub_model_build_material_field_uniform_value():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([2.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    hc = pyrucast.Model.heat_conduction(fes)[0]

    sub = pyrucast.element_field.sub_material_field(hc, [("k", 2.5)])
    # Pre-filled uniformly at every (cell, gauss).
    for g in range(fes[0].gauss_count()):
        assert sub.value(0, g, "k") == pytest.approx(2.5)


def test_sub_model_build_material_field_errors_on_dirichlet():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    imposed = pyrucast.Mesh.poi1_from_nodes([a])
    multiplier = pyrucast.mesh.barycenter(imposed)
    dir_sub = pyrucast.Model.dirichlet("T", "q", imposed, multiplier)[0]
    with pytest.raises(RuntimeError):
        pyrucast.element_field.sub_material_field(dir_sub, [("k", 1.0)])


def test_sub_model_build_material_field_errors_on_empty_list():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    hc = pyrucast.Model.heat_conduction(fes)[0]
    with pytest.raises(RuntimeError):
        pyrucast.element_field.sub_material_field(hc, [])


# ─── material_field (uniform over the whole model) ──────────────────────────


def test_model_build_material_field_uniform_skips_dirichlet():
    _, nodes, _, model = _two_zone_model()
    materials = pyrucast.element_field.material_field(model, [("k", 1.5)])
    # 2 HC + 1 Dirichlet ⇒ only 2 sub-fields kept.
    assert len(materials) == 2

    # Assemble and verify k/h = 1.5/1 = 1.5 on each side of the shared node.
    K = pyrucast.matrix.stiffness(model, materials)
    tol = 1e-12

    def v(i, j):
        return K.get(i, "q", j, "T")

    n0, n1, n2 = nodes
    assert abs(v(n0, n0) - 1.5) < tol
    # Shared middle node: contributions sum.
    assert abs(v(n1, n1) - 3.0) < tol
    assert abs(v(n2, n2) - 1.5) < tol


# ─── material_field_per_sub_model ───────────────────────────────────────────


def test_model_build_material_field_per_sub_model_different_zones():
    _, nodes, _, model = _two_zone_model()
    materials = pyrucast.element_field.material_field_per_sub_model(
        model,
        [
            [("k", 1.0)],  # zone A (model[0])
            [("k", 4.0)],  # zone B (model[1])
            [],  # Dirichlet (model[2]) — skip
        ],
    )
    assert len(materials) == 2  # only the two HC slots

    K = pyrucast.matrix.stiffness(model, materials)
    n0, n1, n2 = nodes
    tol = 1e-12

    def v(i, j):
        return K.get(i, "q", j, "T")

    assert abs(v(n0, n0) - 1.0) < tol
    # Shared middle node sees both conductivities.
    assert abs(v(n1, n1) - 5.0) < tol
    assert abs(v(n2, n2) - 4.0) < tol


def test_model_build_material_field_per_sub_model_length_mismatch_errors():
    _, _, _, model = _two_zone_model()
    with pytest.raises(RuntimeError):
        pyrucast.element_field.material_field_per_sub_model(model, [[("k", 1.0)]])


def test_sub_model_material_components_lists_required_components():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    hc = pyrucast.Model.heat_conduction(fes)[0]
    assert hc.material_components() == ["k"]

    imposed = pyrucast.Mesh.poi1_from_nodes([a])
    multiplier = pyrucast.mesh.barycenter(imposed)
    dir_sub = pyrucast.Model.dirichlet("T", "q", imposed, multiplier)[0]
    assert dir_sub.material_components() is None


def test_sub_model_build_material_field_filters_extras_and_errors_on_missing():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    hc = pyrucast.Model.heat_conduction(fes)[0]

    # Unknown extras are dropped silently; the required "k" and the optional
    # heat-capacity components ("rho", "cp") survive when supplied.
    mat = pyrucast.element_field.sub_material_field(
        hc, [("k", 2.0), ("rho", 7.0), ("bogus", 1.0)]
    )
    assert mat.value(0, 0, "k") == pytest.approx(2.0)
    assert mat.value(0, 0, "rho") == pytest.approx(7.0)
    with pytest.raises(RuntimeError):
        mat.value(0, 0, "bogus")

    # Missing required component → error.
    with pytest.raises(RuntimeError):
        pyrucast.element_field.sub_material_field(hc, [("rho", 1.0)])


def test_model_indexed_sub_model_builds_its_own_material_field():
    """`sub_material_field(model[i], ...)` builds the SubElementField
    for a single sub-model selected by index."""
    _, _, fes, model = _two_zone_model()
    # model[0] = HC zone A. Building its material there gives a
    # SubElementField on the corresponding FE subspace.
    sub_a_mat = pyrucast.element_field.sub_material_field(model[0], [("k", 7.0)])
    for g in range(fes[0].gauss_count()):
        assert sub_a_mat.value(0, g, "k") == pytest.approx(7.0)

    # model[1] = HC zone B.
    sub_b_mat = pyrucast.element_field.sub_material_field(model[1], [("k", 3.0)])
    for g in range(fes[1].gauss_count()):
        assert sub_b_mat.value(0, g, "k") == pytest.approx(3.0)

    # model[2] is the Dirichlet sub-model — no material to build.
    with pytest.raises(RuntimeError):
        pyrucast.element_field.sub_material_field(model[2], [("k", 1.0)])
