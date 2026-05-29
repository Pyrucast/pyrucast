"""Python tests for SubModel/Model.build_material_field."""

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
    sm_a.add_cell([nodes[0].id, nodes[1].id])
    sm_b = pyrucast.SubMesh(c, "SEG2")
    sm_b.add_cell([nodes[1].id, nodes[2].id])
    mesh.add_sub(sm_a)
    mesh.add_sub(sm_b)
    fes = pyrucast.FiniteElementSpace(mesh)

    model = pyrucast.Model()
    model.add_sub(pyrucast.SubModel.heat_conduction(fes[0]))
    model.add_sub(pyrucast.SubModel.dirichlet(c, "T", "q", [nodes[0].id]))
    model.add_sub(pyrucast.SubModel.heat_conduction(fes[1]))
    return c, nodes, fes, model


# ─── SubModel.build_material_field ──────────────────────────────────────────


def test_sub_model_build_material_field_uniform_value():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    b = c.add_node([2.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.add_cell([a.id, b.id])
    fes = pyrucast.FiniteElementSpace(mesh)
    hc = pyrucast.SubModel.heat_conduction(fes[0])

    sub = hc.build_material_field([("k", 2.5)])
    # Pre-filled uniformly at every (cell, gauss).
    for g in range(fes[0].gauss_count()):
        assert sub.value(0, g, "k") == pytest.approx(2.5)


def test_sub_model_build_material_field_errors_on_dirichlet():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    dir_sub = pyrucast.SubModel.dirichlet(c, "T", "q", [a.id])
    with pytest.raises(RuntimeError):
        dir_sub.build_material_field([("k", 1.0)])


def test_sub_model_build_material_field_errors_on_empty_list():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.add_cell([a.id, b.id])
    fes = pyrucast.FiniteElementSpace(mesh)
    hc = pyrucast.SubModel.heat_conduction(fes[0])
    with pytest.raises(RuntimeError):
        hc.build_material_field([])


# ─── Model.build_material_field (uniform mode) ──────────────────────────────


def test_model_build_material_field_uniform_skips_dirichlet():
    _, nodes, _, model = _two_zone_model()
    materials = model.build_material_field([("k", 1.5)])
    # 2 HC + 1 Dirichlet ⇒ only 2 sub-fields kept.
    assert len(materials) == 2

    # Assemble and verify k/h = 1.5/1 = 1.5 on each side of the shared node.
    K = model.stiffness(materials)
    tol = 1e-12

    def v(i, j):
        return K.get(i, "q", j, "T")

    n0, n1, n2 = (n.id for n in nodes)
    assert abs(v(n0, n0) - 1.5) < tol
    # Shared middle node: contributions sum.
    assert abs(v(n1, n1) - 3.0) < tol
    assert abs(v(n2, n2) - 1.5) < tol


# ─── Model.build_material_field_per_sub_model ───────────────────────────────


def test_model_build_material_field_per_sub_model_different_zones():
    _, nodes, _, model = _two_zone_model()
    materials = model.build_material_field_per_sub_model([
        [("k", 1.0)],   # zone A
        [],             # Dirichlet — skip
        [("k", 4.0)],   # zone B
    ])
    assert len(materials) == 2  # only the two HC slots

    K = model.stiffness(materials)
    n0, n1, n2 = (n.id for n in nodes)
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
        model.build_material_field_per_sub_model([[("k", 1.0)]])
