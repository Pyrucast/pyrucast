"""A node from another `Coords` is refused wherever a `Node` becomes an id.

An id only names a node inside its own `Coords`; in another one the same
number is some other node. Every API that takes a `Node` and keeps or looks
up its id — a cell, a node union, a field value, a matrix entry, a constraint
right-hand side — must therefore refuse a node from another `Coords` rather
than silently use whichever node carries that number there.
"""

import pyrucast
import pytest


@pytest.fixture
def setup():
    """Two Coords, two nodes each; `a` here and `x` there share id 0."""
    c = pyrucast.Coords(2)
    a, b = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0])
    autre = pyrucast.Coords(2)
    x, y = autre.add_node([5.0, 5.0]), autre.add_node([6.0, 5.0])
    assert x.id == a.id
    return c, a, b, x, y


def _refused(call):
    with pytest.raises(RuntimeError, match="another Coords|mismatched Coords"):
        call()


def test_mesh_refuses_a_foreign_node(setup):
    c, a, b, x, _ = setup
    mesh = pyrucast.Mesh(c, "POI1")
    _refused(lambda: mesh.unit().add_cell([x]))
    assert mesh.cell_count == 0
    assert c.refcount(a.id) == 1  # nothing counted on the homonym
    _refused(lambda: a | x)
    _refused(lambda: (a | b) | x)
    _refused(lambda: pyrucast.mesh.poi1_from_nodes([a, x]))


def test_node_field_refuses_a_foreign_node(setup):
    _, a, b, x, y = setup
    f = pyrucast.NodeField(a | b, ["T"])
    f[0].set_value(a, "T", 7.0)
    zone = f[0]

    def set_item():
        zone[x, "T"] = 1.0

    for call in (
        lambda: f.value(x, "T"),
        lambda: f.values([a, x], "T"),
        lambda: f.values(x | y, "T"),
        lambda: zone.value(x, "T"),
        lambda: zone.set_value(x, "T", 1.0),
        lambda: zone[x, "T"],
        set_item,
        lambda: zone.get_by_node(x, 0),
        lambda: zone.set_by_node(x, 0, 1.0),
    ):
        _refused(call)
    assert f.value(a, "T") == 7.0  # the homonym was never written through
    assert f.values([a, b], "T") == [7.0, 0.0]


def test_matrix_refuses_a_foreign_node(setup):
    _, a, b, x, _ = setup
    support = a | b
    k = pyrucast.Matrix.block(support, support, ["q"], ["T"], symmetry="full")
    _refused(lambda: k[0].add_entry(x, "q", a, "T", 1.0))
    _refused(lambda: k[0].add_entry(a, "q", x, "T", 1.0))
    k[0].add_entry(a, "q", a, "T", 2.0)
    _refused(lambda: k[0].get(a, "q", x, "T"))
    _refused(lambda: k.get(x, "q", a, "T"))
    assert k.get(a, "q", a, "T") == 2.0
    # An empty matrix has no Coords to check against: it still reads zero.
    assert pyrucast.Matrix().get(x, "q", x, "T") == 0.0


def test_constraint_rhs_refuses_a_foreign_node():
    c = pyrucast.Coords(1)
    n0, n1 = c.add_node([0.0]), c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([n0, n1])
    cible = pyrucast.model.heat_conduction(pyrucast.FiniteElementSpace(mesh))
    d = pyrucast.model.dirichlet(pyrucast.mesh.poi1_from_nodes([n0]), cible, "T")
    z = pyrucast.Coords(1).add_node([0.0])
    _refused(lambda: d.constraint_rhs([(z, 1.0)]))
    assert d.constraint_rhs([(n0, 1.0)]).components() == ["imposed"]
