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
    with pytest.raises(TypeError, match="NodeField, an ElementField or an Evolution"):
        pyrucast.export.export_vtk(mesh, str(tmp_path / "x.vtk"), field=mesh)


def test_export_binary(tmp_path):
    _c, mesh, _n = _square()
    out = tmp_path / "mesh.vtk"
    pyrucast.export.export_vtk(mesh, str(out), binary=True)
    data = out.read_bytes()
    assert b"\nBINARY\n" in data
    assert b"POINTS 4 double\n" in data
    # Four 3-D points in big-endian doubles; the third is (1, 1, 0).
    at = data.index(b"POINTS 4 double\n") + len(b"POINTS 4 double\n")
    import struct

    assert struct.unpack(">3d", data[at + 48 : at + 72]) == (1.0, 1.0, 0.0)


def test_export_evolution_as_a_series(tmp_path):
    c, mesh, nodes = _square()
    fes = pyrucast.FiniteElementSpace(mesh)
    cold = pyrucast.ElementField(fes, ["s"])
    hot = pyrucast.ElementField(fes, ["s"])
    hot[0].set_uniform("s", 5.0)
    series = pyrucast.Evolution([(0.0, cold), (10.0, hot)])
    index = tmp_path / "run.vtk.series"
    pyrucast.export.export_vtk(mesh, str(index), field=series)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "run.vtk.series",
        "run_0000.vtk",
        "run_0001.vtk",
    ]
    import json

    files = json.loads(index.read_text())["files"]
    assert files == [
        {"name": "run_0000.vtk", "time": 0},
        {"name": "run_0001.vtk", "time": 10},
    ]
    assert "5\n5\n" in (tmp_path / "run_0001.vtk").read_text()
