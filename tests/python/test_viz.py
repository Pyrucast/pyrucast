"""Python tests for the visualization layer (feature `viz`).

Skipped when pyrucast was built without the `viz` feature (i.e. when
`SubMesh.plot` is not present on the compiled module).
"""

import os
import tempfile

import pytest

import pyrucast


_HAS_VIZ = hasattr(pyrucast.SubMesh(pyrucast.Configuration(2), "TRI3"), "plot")
pytestmark = pytest.mark.skipif(not _HAS_VIZ, reason="pyrucast built without viz feature")


def _make_two_triangles():
    c = pyrucast.Configuration(3)
    a = c.add_node([0.0, 0.0, 0.0])
    b = c.add_node([1.0, 0.0, 0.0])
    cc = c.add_node([1.0, 1.0, 0.0])
    d = c.add_node([0.0, 1.0, 0.5])
    sm = pyrucast.SubMesh(c, "TRI3")
    sm.add_cell([a.id, b.id, cc.id])
    sm.add_cell([a.id, cc.id, d.id])
    return c, sm


def test_face_color_default():
    sm = pyrucast.SubMesh(pyrucast.Configuration(2), "TRI3")
    r, g, b = sm.face_color
    assert (r, g, b) == (180, 200, 230)


def test_face_color_roundtrip():
    sm = pyrucast.SubMesh(pyrucast.Configuration(2), "TRI3")
    sm.face_color = (10, 200, 30)
    assert sm.face_color == (10, 200, 30)


def test_plot_to_png(tmp_path):
    _, sm = _make_two_triangles()
    path = tmp_path / "tri.png"
    sm.plot(save=str(path))
    assert path.stat().st_size > 0


def test_plot_to_svg(tmp_path):
    _, sm = _make_two_triangles()
    path = tmp_path / "tri.svg"
    sm.plot(save=str(path))
    text = path.read_text()
    assert "<svg" in text


def test_plot_with_view(tmp_path):
    _, sm = _make_two_triangles()
    path = tmp_path / "tri-top.png"
    # (yaw, pitch, scale) — top view.
    sm.plot(view=(0.0, 90.0, 1.0), save=str(path))
    assert path.stat().st_size > 0


def test_plot_unsupported_extension(tmp_path):
    _, sm = _make_two_triangles()
    path = tmp_path / "tri.jpg"
    with pytest.raises(RuntimeError):
        sm.plot(save=str(path))


@pytest.mark.parametrize(
    "element_type,coords,connectivity",
    [
        ("POI1", [[0.0, 0.0, 0.0]], [0]),
        ("SEG2", [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0, 1]),
        ("TRI3", [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [0, 1, 2]),
        (
            "QUA4",
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [0, 1, 2, 3],
        ),
        (
            "TET4",
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [0, 1, 2, 3],
        ),
        (
            "HEX8",
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ),
    ],
)
def test_plot_every_element_type(tmp_path, element_type, coords, connectivity):
    c = pyrucast.Configuration(3)
    nodes = [c.add_node(p) for p in coords]
    sm = pyrucast.SubMesh(c, element_type)
    sm.add_cell([nodes[i].id for i in connectivity])
    path = tmp_path / f"{element_type.lower()}.png"
    sm.plot(save=str(path))
    assert path.stat().st_size > 0


def test_mesh_plot_uses_each_submesh_color(tmp_path):
    c = pyrucast.Configuration(3)
    a = c.add_node([0.0, 0.0, 0.0])
    b = c.add_node([1.0, 0.0, 0.0])
    cc = c.add_node([0.0, 1.0, 0.0])
    d = c.add_node([2.0, 0.0, 0.0])
    e = c.add_node([2.0, 1.0, 0.0])

    sm_red = pyrucast.SubMesh(c, "TRI3")
    sm_red.add_cell([a.id, b.id, cc.id])
    sm_red.face_color = (220, 60, 60)

    sm_blue = pyrucast.SubMesh(c, "TRI3")
    sm_blue.add_cell([b.id, d.id, e.id])
    sm_blue.face_color = (60, 60, 220)

    mesh = pyrucast.Mesh(c)
    mesh.add_submesh(sm_red)
    mesh.add_submesh(sm_blue)

    path = tmp_path / "mesh.svg"
    mesh.plot(save=str(path))
    text = path.read_text().lower()
    assert "dc3c3c" in text or "rgb(220,60,60)" in text
    assert "3c3cdc" in text or "rgb(60,60,220)" in text
