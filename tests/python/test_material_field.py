"""Python tests for the material-field build operators
(sub_material_field / material_field / material_field_per_sub_model)."""

import pytest

import pyrucast


def _two_zone_model():
    """Build a 3-node 2-SEG2 mesh, FE space and a model with:
      - HC on zone A (cells 0..1, on subspace 0)
      - Dirichlet on the leftmost node
      - HC on zone B (cell 1..2, on subspace 1)
    Returns (cfg, [n0, n1, n2], fes, model).
    """
    c = pyrucast.Configuration(1)
    nodes = [c.add_node([i * 1.0]) for i in range(3)]
    mesh = pyrucast.Mesh(c)
    sm_a = pyrucast.SubMesh(c, "SEG2")
    sm_a.add_cell([nodes[0], nodes[1]])
    sm_b = pyrucast.SubMesh(c, "SEG2")
    sm_b.add_cell([nodes[1], nodes[2]])
    mesh.add_sub(sm_a)
    mesh.add_sub(sm_b)
    fes = pyrucast.FiniteElementSpace(mesh)

    model = pyrucast.Model()
    model.add_sub(pyrucast.SubModel.heat_conduction(fes[0]))
    model.add_sub(pyrucast.SubModel.dirichlet("T", "q", [nodes[0]]))
    model.add_sub(pyrucast.SubModel.heat_conduction(fes[1]))
    return c, nodes, fes, model


# ─── sub_material_field (one sub-model) ─────────────────────────────────────


def test_sub_model_build_material_field_uniform_value():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    b = c.add_node([2.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    hc = pyrucast.SubModel.heat_conduction(fes[0])

    sub = pyrucast.sub_material_field(hc, [("k", 2.5)])
    # Pre-filled uniformly at every (cell, gauss).
    for g in range(fes[0].gauss_count()):
        assert sub.value(0, g, "k") == pytest.approx(2.5)


def test_sub_model_build_material_field_errors_on_dirichlet():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    dir_sub = pyrucast.SubModel.dirichlet("T", "q", [a])
    with pytest.raises(RuntimeError):
        pyrucast.sub_material_field(dir_sub, [("k", 1.0)])


def test_sub_model_build_material_field_errors_on_empty_list():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    hc = pyrucast.SubModel.heat_conduction(fes[0])
    with pytest.raises(RuntimeError):
        pyrucast.sub_material_field(hc, [])


# ─── material_field (uniform over the whole model) ──────────────────────────


def test_model_build_material_field_uniform_skips_dirichlet():
    _, nodes, _, model = _two_zone_model()
    materials = pyrucast.material_field(model, [("k", 1.5)])
    # 2 HC + 1 Dirichlet ⇒ only 2 sub-fields kept.
    assert len(materials) == 2

    # Assemble and verify k/h = 1.5/1 = 1.5 on each side of the shared node.
    K = pyrucast.stiffness(model, materials)
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
    materials = pyrucast.material_field_per_sub_model(model, [
        [("k", 1.0)],   # zone A
        [],             # Dirichlet — skip
        [("k", 4.0)],   # zone B
    ])
    assert len(materials) == 2  # only the two HC slots

    K = pyrucast.stiffness(model, materials)
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
        pyrucast.material_field_per_sub_model(model, [[("k", 1.0)]])


def test_sub_model_material_components_lists_required_components():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    hc = pyrucast.SubModel.heat_conduction(fes[0])
    assert hc.material_components() == ["k"]

    dir_sub = pyrucast.SubModel.dirichlet("T", "q", [a])
    assert dir_sub.material_components() is None


def test_sub_model_build_material_field_filters_extras_and_errors_on_missing():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    hc = pyrucast.SubModel.heat_conduction(fes[0])

    # Extras are kept silent — only the declared component ("k") survives.
    mat = pyrucast.sub_material_field(hc, [("k", 2.0), ("rho", 7.0)])
    assert mat.value(0, 0, "k") == pytest.approx(2.0)
    with pytest.raises(RuntimeError):
        mat.value(0, 0, "rho")

    # Missing required component → error.
    with pytest.raises(RuntimeError):
        pyrucast.sub_material_field(hc, [("rho", 1.0)])


def test_model_indexed_sub_model_builds_its_own_material_field():
    """`sub_material_field(model[i], ...)` builds the SubElementField
    for a single sub-model selected by index."""
    _, _, fes, model = _two_zone_model()
    # model[0] = HC zone A. Building its material there gives a
    # SubElementField on the corresponding FE subspace.
    sub_a_mat = pyrucast.sub_material_field(model[0], [("k", 7.0)])
    for g in range(fes[0].gauss_count()):
        assert sub_a_mat.value(0, g, "k") == pytest.approx(7.0)

    # model[1] is the Dirichlet sub-model — no material to build.
    with pytest.raises(RuntimeError):
        pyrucast.sub_material_field(model[1], [("k", 1.0)])

    # model[2] = HC zone B.
    sub_b_mat = pyrucast.sub_material_field(model[2], [("k", 3.0)])
    for g in range(fes[1].gauss_count()):
        assert sub_b_mat.value(0, g, "k") == pytest.approx(3.0)
