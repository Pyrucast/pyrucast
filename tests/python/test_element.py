"""Python tests for the Element view + FE-space iteration sugar."""

import pyrucast


def _seg2_fes(length=2.0):
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    b = c.add_node([length])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    return c, a, b, fes


def test_fes_element_returns_element_with_expected_geometry():
    _, a, b, fes = _seg2_fes(length=2.0)
    el = fes.element(0, 0)
    assert el.index == 0
    assert el.nodes_per_cell == 2
    assert el.space_dim == 1
    assert el.ref_dim == 1
    # SEG2 of length 2 in 1-D → |J| = 1 at every Gauss point.
    assert el.gauss_count >= 1
    for g in range(el.gauss_count):
        assert abs(el.det_jacobian(g) - 1.0) < 1e-12
    assert [n.id for n in el.nodes()] == [a.id, b.id]


def test_fes_elements_returns_a_list():
    c = pyrucast.Configuration(1)
    nodes = [c.add_node([i * 1.0]) for i in range(3)]
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.add_cell([nodes[0], nodes[1]])
    mesh.add_cell([nodes[1], nodes[2]])
    fes = pyrucast.FiniteElementSpace(mesh)
    elements = fes.elements(0)
    assert len(elements) == 2
    assert [e.index for e in elements] == [0, 1]


def test_subspace_is_iterable_over_elements():
    c = pyrucast.Configuration(1)
    nodes = [c.add_node([i * 1.0]) for i in range(4)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(3):
        mesh.add_cell([nodes[i], nodes[i + 1]])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    # __len__ + __getitem__: sequence iteration works.
    assert len(sub) == 3
    seen_indices = [el.index for el in sub]
    assert seen_indices == [0, 1, 2]
    # Negative index.
    assert sub[-1].index == 2


def test_element_cell_view_matches_underlying_mesh():
    _, a, b, fes = _seg2_fes()
    el = fes.element(0, 0)
    cell = el.cell()
    assert cell.element_type == "SEG2"
    assert [n.id for n in cell.nodes()] == [a.id, b.id]


def test_element_repr_and_str():
    _, _, _, fes = _seg2_fes()
    el = fes.element(0, 0)
    assert "Element" in repr(el)
    assert "SEG2" in str(el)
