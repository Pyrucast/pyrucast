"""Python tests for the visualization layer (feature `viz`).

Skipped when pyrucast was built without the `viz` feature (i.e. when
`Mesh.plot` is not present on the compiled module).
"""

import os
import tempfile

import pytest

import pyrucast


_HAS_VIZ = hasattr(pyrucast.Mesh(pyrucast.Coords(2), "TRI3"), "plot")
pytestmark = pytest.mark.skipif(
    not _HAS_VIZ, reason="pyrucast built without viz feature"
)


def _make_two_triangles():
    c = pyrucast.Coords(3)
    a = c.add_node([0.0, 0.0, 0.0])
    b = c.add_node([1.0, 0.0, 0.0])
    cc = c.add_node([1.0, 1.0, 0.0])
    d = c.add_node([0.0, 1.0, 0.5])
    sm = pyrucast.Mesh(c, "TRI3")[0]
    sm.add_cell([a, b, cc])
    sm.add_cell([a, cc, d])
    return c, sm


def test_face_color_default():
    sm = pyrucast.Mesh(pyrucast.Coords(2), "TRI3")[0]
    r, g, b = sm.face_color
    assert (r, g, b) == (180, 200, 230)


def test_face_color_roundtrip():
    sm = pyrucast.Mesh(pyrucast.Coords(2), "TRI3")[0]
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


def test_plot_title_is_engraved_at_the_bottom(tmp_path):
    _, sm = _make_two_triangles()
    titled = tmp_path / "titled.svg"
    sm.plot(save=str(titled), title="cantilever-beam")
    assert "cantilever-beam" in titled.read_text()

    # No title → the caption is absent.
    plain = tmp_path / "plain.svg"
    sm.plot(save=str(plain))
    assert "cantilever-beam" not in plain.read_text()


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
    c = pyrucast.Coords(3)
    nodes = [c.add_node(p) for p in coords]
    sm = pyrucast.Mesh(c, element_type)[0]
    sm.add_cell([nodes[i] for i in connectivity])
    path = tmp_path / f"{element_type.lower()}.png"
    sm.plot(save=str(path))
    assert path.stat().st_size > 0


def test_mesh_plot_uses_each_submesh_color(tmp_path):
    c = pyrucast.Coords(3)
    a = c.add_node([0.0, 0.0, 0.0])
    b = c.add_node([1.0, 0.0, 0.0])
    cc = c.add_node([0.0, 1.0, 0.0])
    d = c.add_node([2.0, 0.0, 0.0])
    e = c.add_node([2.0, 1.0, 0.0])

    red = pyrucast.Mesh(c, "TRI3")
    red.unit().add_cell([a, b, cc])
    red[0].face_color = (220, 60, 60)

    blue = pyrucast.Mesh(c, "TRI3")
    blue.unit().add_cell([b, d, e])
    blue[0].face_color = (60, 60, 220)

    mesh = red | blue

    path = tmp_path / "mesh.svg"
    mesh.plot(save=str(path))
    text = path.read_text().lower()
    assert "dc3c3c" in text or "rgb(220,60,60)" in text
    assert "3c3cdc" in text or "rgb(60,60,220)" in text


# ─── field-coloured plotting ────────────────────────────────────────────────


def _build_field_on_nodes(c, nodes, components, values_per_component):
    """Helper: build a POI1 NodeField on `nodes` and fill values."""
    poi1 = pyrucast.Mesh(c, "POI1")
    for n in nodes:
        poi1.unit().add_cell([n])
    nf = pyrucast.NodeField(poi1, list(components))
    for ci, comp in enumerate(components):
        for ni, n in enumerate(nodes):
            nf[0].set_value(n, comp, values_per_component[ci][ni])
    return nf


def test_mesh_plot_with_field_writes_overlay_label(tmp_path):
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([a, b, cc])

    nf = _build_field_on_nodes(c, [a, b, cc], ["T"], [[0.0, 1.0, 2.0]])

    path = tmp_path / "field.svg"
    mesh.plot(save=str(path), field=nf)
    text = path.read_text()
    # Overlay label must show the current component and min/max range.
    assert "[T]" in text
    assert "min=" in text and "max=" in text


def test_submesh_plot_with_field_explicit_component(tmp_path):
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.0, 1.0])
    tri = pyrucast.Mesh(c, "TRI3")[0]
    tri.add_cell([a, b, cc])

    nf = _build_field_on_nodes(
        c, [a, b, cc], ["UX", "UY"], [[0.0, 0.0, 0.0], [3.14, 2.71, 1.41]]
    )
    path = tmp_path / "uy.svg"
    # Default would pick "UX" (first component). Ask for "UY" explicitly.
    tri.plot(save=str(path), field=nf, component="UY")
    text = path.read_text()
    assert "[UY]" in text


def test_plot_with_field_unknown_component_errors(tmp_path):
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.0, 1.0])
    tri = pyrucast.Mesh(c, "TRI3")[0]
    tri.add_cell([a, b, cc])
    nf = _build_field_on_nodes(c, [a, b, cc], ["T"], [[0.0, 1.0, 2.0]])
    path = tmp_path / "nope.svg"
    try:
        tri.plot(save=str(path), field=nf, component="DOES_NOT_EXIST")
    except RuntimeError as e:
        assert "DOES_NOT_EXIST" in str(e)
    else:
        raise AssertionError("expected RuntimeError for unknown component")


# ─── wireframe style ────────────────────────────────────────────────────────


def _solid_tetrahedron():
    """A single TET4 cell — a solid volume mesh."""
    c = pyrucast.Coords(3)
    n = [
        c.add_node([0.0, 0.0, 0.0]),
        c.add_node([1.0, 0.0, 0.0]),
        c.add_node([0.0, 1.0, 0.0]),
        c.add_node([0.0, 0.0, 1.0]),
    ]
    mesh = pyrucast.Mesh(c, "TET4")
    mesh[0].add_cell(n)
    return c, mesh


def test_wireframe_has_no_filled_faces(tmp_path):
    """Worked example — a solid tetrahedron drawn two ways.

    The default (``wireframe=False``) fills the opaque outer skin, so the
    SVG carries ``<polygon>`` faces; ``wireframe=True`` draws every edge
    instead, with no fill, so the SVG has no ``<polygon>`` at all. This is
    the Python sibling of the Rust test ``mesh_wireframe_has_no_filled_faces``.
    """
    _, mesh = _solid_tetrahedron()

    surface = tmp_path / "tet_surface.svg"
    wire = tmp_path / "tet_wire.svg"
    mesh.plot(view=(45.0, 35.264, 1.0), save=str(surface))
    mesh.plot(view=(45.0, 35.264, 1.0), save=str(wire), wireframe=True)

    assert "<polygon" in surface.read_text(), "surface fills its faces"
    assert "<polygon" not in wire.read_text(), "wireframe draws edges only"


def test_wireframe_with_field_is_rejected(tmp_path):
    """`wireframe` makes no sense for a field plot — it must raise."""
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.0, 1.0])
    tri = pyrucast.Mesh(c, "TRI3")[0]
    tri.add_cell([a, b, cc])
    nf = _build_field_on_nodes(c, [a, b, cc], ["T"], [[0.0, 1.0, 2.0]])
    with pytest.raises(ValueError):
        tri.plot(save=str(tmp_path / "x.svg"), field=nf, wireframe=True)


# ─── axisymmetric revolution ────────────────────────────────────────────────


def _axisymmetric_section():
    """A 2×2 QUA4 grid in the meridian plane: the section of a hollow tube."""
    c = pyrucast.Coords.axisymmetric()
    n = {}
    for i in range(3):
        for j in range(3):
            n[(i, j)] = c.add_node([1.0 + 0.5 * i, 0.5 * j])
    mesh = pyrucast.Mesh(c, "QUA4")
    for i in range(2):
        for j in range(2):
            mesh[0].add_cell(
                [n[(i, j)], n[(i + 1, j)], n[(i + 1, j + 1)], n[(i, j + 1)]]
            )
    return c, mesh, n


def test_revolve_sweeps_the_section_off_its_plane(tmp_path):
    """Worked example — the same section drawn flat, then revolved.

    Plotted as it stands, an axisymmetric mesh is a flat `(r, z)` section:
    seen from the top (`pitch=90`) it collapses onto a single line. Revolved,
    it becomes the tube it describes and fills the picture from every angle.
    """
    _, mesh, _ = _axisymmetric_section()

    flat = tmp_path / "section.svg"
    body = tmp_path / "tube.svg"
    mesh.plot(view=(0.0, 90.0, 1.0), save=str(flat))
    mesh.plot(view=(0.0, 90.0, 1.0), save=str(body), revolve=True)

    # The swept body carries far more faces than the flat section (4 cells).
    assert flat.read_text().count("<polygon") == 4
    assert body.read_text().count("<polygon") > 50


def test_revolve_angle_cuts_the_body_open(tmp_path):
    """A partial sweep adds the meridian section at both ends of the angle."""
    _, mesh, _ = _axisymmetric_section()
    full = tmp_path / "full.svg"
    part = tmp_path / "part.svg"
    mesh.plot(save=str(full), revolve=True)
    mesh.plot(save=str(part), revolve=True, revolve_angle=90.0)
    # A quarter turn draws far fewer bands, plus the two end sections.
    assert part.read_text().count("<polygon") < full.read_text().count("<polygon")


def test_revolve_carries_the_field_colours(tmp_path):
    """The field keeps colouring the cells the swept surface comes from."""
    c, mesh, n = _axisymmetric_section()
    nodes = list(n.values())
    nf = _build_field_on_nodes(c, nodes, ["T"], [[float(k) for k in range(len(nodes))]])
    path = tmp_path / "field3d.svg"
    mesh.plot(save=str(path), field=nf, revolve=True)
    text = path.read_text()
    assert "[T]" in text
    assert "<polygon" in text


def test_revolve_needs_axisymmetric_coordinates(tmp_path):
    """On a plain Cartesian mesh the abscissa is no radius — it must raise."""
    _, sm = _make_two_triangles()
    with pytest.raises(RuntimeError, match="axisymmetric"):
        sm.plot(save=str(tmp_path / "nope.svg"), revolve=True)


def test_revolve_angle_must_be_a_turn_at_most(tmp_path):
    _, mesh, _ = _axisymmetric_section()
    with pytest.raises(RuntimeError, match="360"):
        mesh.plot(save=str(tmp_path / "nope.svg"), revolve=True, revolve_angle=400.0)
