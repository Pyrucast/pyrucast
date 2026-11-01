"""Every aggregate keeps its zones on one `Coords`.

A node id only names a node inside its own `Coords`. An aggregate whose zones
stood on two of them would read two different nodes under one id — a model on
two `Coords` assembled a matrix whose rows mixed both bodies. Every union
therefore refuses a zone from another `Coords`: mesh, FE space, node field,
field at the Gauss points, matrix, model.
"""

import pyrucast
import pytest


def _corps():
    """One triangle on a `Coords` of its own."""
    c = pyrucast.Coords(2)
    a, b, d = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0]), c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([a, b, d])
    return mesh, pyrucast.FiniteElementSpace(mesh), a


@pytest.fixture
def deux():
    return _corps(), _corps()


def _refused(call):
    with pytest.raises(RuntimeError, match="mismatched Coords"):
        call()


def test_mesh_and_fe_space(deux):
    (m1, f1, _), (m2, f2, _) = deux
    _refused(lambda: m1 | m2)
    _refused(lambda: f1 | f2)


def test_fields(deux):
    (m1, f1, _), (m2, f2, _) = deux
    k1, k2 = pyrucast.ElementField(f1, ["k"]), pyrucast.ElementField(f2, ["k"])
    _refused(lambda: k1 | pyrucast.ElementField(f2, ["E"]))
    _refused(lambda: k1 | k2[0])
    # Arithmetic would leave the two supports side by side: it refuses too.
    _refused(lambda: k1 + k2)
    _refused(lambda: pyrucast.NodeField(m1, ["T"]) | pyrucast.NodeField(m2, ["q"]))


def test_matrix_and_model(deux):
    (_, f1, a1), (_, f2, a2) = deux
    h1 = pyrucast.model.heat_conduction(f1)
    h2 = pyrucast.model.heat_conduction(f2)
    _refused(lambda: h1 | h2)
    k1 = pyrucast.matrix.stiffness(
        h1, pyrucast.element_field.material_field(h1, [("k", 1.0)])
    )
    k2 = pyrucast.matrix.stiffness(
        h2, pyrucast.element_field.material_field(h2, [("k", 1.0)])
    )
    _refused(lambda: k1 | k2)
    # A single block whose rows and columns stand on two Coords.
    p1, p2 = pyrucast.mesh.poi1_from_nodes([a1]), pyrucast.mesh.poi1_from_nodes([a2])
    _refused(lambda: pyrucast.Matrix.block(p1, p2, ["q"], ["T"], symmetry="full"))
