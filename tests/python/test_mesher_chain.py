"""`chain`: putting a line's cells back in walking order."""

import pytest

import pyrucast


def _nodes(n):
    """`n` nodes aligned on the x axis, the node of index i at x = i."""
    c = pyrucast.Coords(2)
    return c, [c.add_node([float(i), 0.0]) for i in range(n)]


def _seg2(c, cells):
    mesh = pyrucast.Mesh(c, "SEG2")
    for cell in cells:
        mesh[0].add_cell(cell)
    return mesh


def _conn(mesh):
    """Connectivity of the first submesh, in node numbers."""
    return [[node.id for node in cell] for cell in mesh[0]]


def _ids(cells):
    return [[node.id for node in cell] for cell in cells]


def test_chain_sorts_and_flips_a_shuffled_line():
    c, n = _nodes(5)
    # In no order, and two segments running the wrong way.
    mesh = _seg2(c, [[n[3], n[4]], [n[0], n[1]], [n[3], n[2]], [n[2], n[1]]])

    out = pyrucast.mesh.chain(mesh)
    assert _conn(out) == _ids([[n[0], n[1]], [n[1], n[2]], [n[2], n[3]], [n[3], n[4]]])
    # Idempotent, et disponible aussi en méthode.
    assert _conn(out.chain()) == _conn(out)


def test_chain_walks_a_closed_contour():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    d = c.add_node([1.0, 1.0])
    e = c.add_node([0.0, 1.0])
    carre = pyrucast.Mesh(c, "QUA4")
    carre[0].add_cell([a, b, d, e])

    suite = pyrucast.mesh.chain(pyrucast.mesh.border(carre))
    conn = _conn(suite)
    assert len(conn) == 4
    # Every cell starts where the previous one ended, and the loop closes.
    for prev, cur in zip(conn, conn[1:] + conn[:1]):
        assert prev[1] == cur[0]


def test_chain_rejects_a_branching_or_broken_line():
    c, n = _nodes(4)

    etoile = _seg2(c, [[n[0], n[1]], [n[1], n[2]], [n[1], n[3]]])
    with pytest.raises(Exception, match="3 segments"):
        pyrucast.mesh.chain(etoile)

    disjoint = _seg2(c, [[n[0], n[1]], [n[2], n[3]]])
    with pytest.raises(Exception, match="disjoint"):
        pyrucast.mesh.chain(disjoint)
