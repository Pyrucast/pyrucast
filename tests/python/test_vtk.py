"""Python tests for `export_vtk` — writing a mesh or field to a legacy VTK
file readable by ParaView."""

import pytest

import pyrucast


def _square():
    """Unit square as two TRI3 on a 2-D Coords; returns (coords, mesh, nodes)."""
    c = pyrucast.Coords(2)
    nodes = [c.add_node(p) for p in ([0, 0], [1, 0], [1, 1], [0, 1])]
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([nodes[0], nodes[1], nodes[2]])
    mesh.unit().add_cell([nodes[0], nodes[2], nodes[3]])
    return c, mesh, nodes


def test_export_mesh_only(tmp_path):
    _c, mesh, _n = _square()
    out = tmp_path / "mesh.vtk"
    pyrucast.export.export_vtk(mesh, str(out))
    text = out.read_text()
    assert text.startswith("# vtk DataFile Version 3.0")
    assert "DATASET UNSTRUCTURED_GRID" in text
    assert "POINTS 4 double" in text
    assert "CELLS 2 8" in text
    assert "CELL_TYPES 2" in text
    assert "POINT_DATA" not in text


def test_export_node_field_point_data(tmp_path):
    c, mesh, nodes = _square()
    support = pyrucast.Mesh(c, "POI1")
    for n in nodes:
        support.unit().add_cell([n])
    field = pyrucast.NodeField(support, ["T"])
    for i, n in enumerate(nodes):
        field[0].set_value(n, "T", float(i))

    out = tmp_path / "field.vtk"
    pyrucast.export.export_vtk(mesh, str(out), field=field)
    text = out.read_text()
    assert "POINT_DATA 4" in text
    assert "SCALARS T double 1" in text
    assert "LOOKUP_TABLE default" in text


def test_export_rejects_non_field(tmp_path):
    _c, mesh, _n = _square()
    with pytest.raises(TypeError, match="NodeField or an ElementField"):
        pyrucast.export.export_vtk(mesh, str(tmp_path / "x.vtk"), field=mesh)
