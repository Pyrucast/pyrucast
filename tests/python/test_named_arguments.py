"""A name a binding does not understand is a bad argument: a `ValueError`
naming what was refused and what is accepted, whichever binding reads it.

These three bindings used to read their name by hand: the shell formulation
raised a `RuntimeError`, the node order and the matrix ordering wrote messages
of their own, and the ordering did not ignore case. They now take the named
type itself, as every other named argument does."""

import pyrucast
import pytest


def _triangle():
    c = pyrucast.Coords(2)
    n = [c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0])]
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell(n)
    return mesh, n


def test_unknown_shell_formulation_is_a_value_error():
    mesh, n = _triangle()
    fes = pyrucast.FiniteElementSpace(mesh)
    u = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes(n), ["u_x"])
    with pytest.raises(
        ValueError,
        match=r"unknown shell formulation 'foo' \(expected thick\|kirchhoff\)",
    ):
        pyrucast.element_field.shell_deformation(u, fes, "foo")


def test_matrix_ordering_is_read_like_any_name():
    _, n = _triangle()
    support = pyrucast.mesh.poi1_from_nodes(n)
    with pytest.raises(
        ValueError,
        match=r"unknown ordering 'foo' \(expected nodes_then_vars\|vars_then_nodes\)",
    ):
        pyrucast.Matrix.block(support, support, ["q"], ["T"], ordering="foo")
    # Case is ignored, as for every other name.
    pyrucast.Matrix.block(support, support, ["q"], ["T"], ordering="VARS_THEN_NODES")


def test_node_order_is_read_like_any_name():
    mesh, _ = _triangle()
    groups = {"t": mesh}
    with pytest.raises(
        ValueError, match=r"unknown node order 'foo' \(expected pyrucast\|gmsh\|med\)"
    ):
        pyrucast.export.to_arrays(groups, order="foo")
    # "vtk" is pyrucast's own numbering, accepted under that name too.
    own = pyrucast.export.to_arrays(groups)
    vtk = pyrucast.export.to_arrays(groups, order="VTK")
    assert list(vtk["blocks"][0][1]) == list(own["blocks"][0][1])
