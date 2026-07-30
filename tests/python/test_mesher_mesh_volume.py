"""Python tests for `pyrucast.mesher.mesh_volume`."""

import pyrucast


def _box(c, lo, hi):
    """The eight corners of a box, ordered 000, 100, 110, 010, 001, 101, 111, 011."""
    return [
        c.add_node(p)
        for p in (
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        )
    ]


# Each square face split along a diagonal of the tetrahedron {0, 2, 5, 7}.
# Which diagonals are chosen matters: most of a box's 64 boundary
# triangulations cannot be filled with tetrahedra on the eight corners alone.
_BOX_FACETS = [
    (0, 3, 2),
    (0, 2, 1),
    (4, 5, 7),
    (5, 6, 7),
    (0, 1, 5),
    (0, 5, 4),
    (1, 2, 5),
    (2, 6, 5),
    (2, 3, 7),
    (2, 7, 6),
    (3, 0, 7),
    (0, 4, 7),
]


def _surface(c, nodes, facets=_BOX_FACETS):
    mesh = pyrucast.Mesh(c, "TRI3")
    for f in facets:
        mesh.unit().add_cell([nodes[f[0]], nodes[f[1]], nodes[f[2]]])
    return mesh


def _tet_count(mesh):
    """How many cells the TET4 submesh holds.

    A result whose envelope had to be cut carries a second submesh of POI1
    naming the nodes that were added, which is not made of cells to measure.
    """
    return mesh.cell_counts()[0]


def _volume(mesh):
    total = 0.0
    for si, n in enumerate(mesh.cell_counts()):
        if mesh.element_types()[si] != "TET4":
            continue
        for ci in range(n):
            p = [mesh.node(si, ci, k).coord() for k in range(4)]
            e = [[p[i][k] - p[0][k] for k in range(3)] for i in (1, 2, 3)]
            total += (
                e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
                - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
                + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])
            ) / 6.0
    return total


def test_mesh_volume_box():
    c = pyrucast.Coords(3)
    mesh = pyrucast.mesher.mesh_volume(_surface(c, _box(c, [0, 0, 0], [1, 1, 1])))
    assert mesh.element_types() == ["TET4"]
    assert abs(_volume(mesh) - 1.0) < 1e-12


def test_mesh_volume_reuses_the_envelope_nodes():
    c = pyrucast.Coords(3)
    nodes = _box(c, [0, 0, 0], [1, 1, 1])
    mesh = pyrucast.mesher.mesh_volume(_surface(c, nodes))
    used = {mesh.node(0, ci, k).id for ci in range(mesh.cell_count()) for k in range(4)}
    assert {n.id for n in nodes} <= used


def test_mesh_volume_leaves_the_surface_alone():
    # Nodes are added inside, never on the skin: peeling the result gives
    # back exactly the twelve facets that went in.
    c = pyrucast.Coords(3)
    envelope = _surface(c, _box(c, [0, 0, 0], [1, 1, 1]))
    mesh = pyrucast.mesher.mesh_volume(envelope)
    assert pyrucast.mesher.skin(mesh).cell_count() == 12


def _extruded_plate(c, per_side, size):
    """A plate meshed, extruded and peeled — the usual way to get an envelope.

    Note the `invert`: `extrude` does not check that its direction is on the
    same side as the source surface's normal, so `skin` of the result comes
    back with its normals pointing into the material.
    """
    pts = []
    for i in range(per_side):
        pts.append([i / per_side, 0.0, 0.0])
    for i in range(per_side):
        pts.append([1.0, 0.0, i / per_side])
    for i in range(per_side):
        pts.append([1.0 - i / per_side, 0.0, 1.0])
    for i in range(per_side):
        pts.append([0.0, 0.0, 1.0 - i / per_side])
    nodes = [c.add_node(p) for p in pts]
    contour = pyrucast.Mesh(c, "SEG2")
    for i in range(len(nodes)):
        contour.unit().add_cell([nodes[i], nodes[(i + 1) % len(nodes)]])
    plate = pyrucast.mesher.triangulate_surface(contour, "TRI3", size=size)
    solid = pyrucast.mesher.extrude(plate, [0.0, 0.4, 0.0], 1)
    skin = pyrucast.mesher.convert(pyrucast.mesher.skin(solid), "TRI3")
    return pyrucast.mesher.invert(skin)


def test_mesh_volume_size_controls_the_fineness():
    # `size` can only be honoured where the envelope is fine enough to allow
    # it: a node may not be placed so near the surface that it would spoil
    # the cells against it, so the surface sets a floor on how fine the
    # inside can get.
    c = pyrucast.Coords(3)
    envelope = _extruded_plate(c, 6, 0.2)
    coarse = pyrucast.mesher.mesh_volume(envelope, size=0.4, allow_surface_nodes=True)
    fine = pyrucast.mesher.mesh_volume(envelope, size=0.1, allow_surface_nodes=True)
    assert _tet_count(fine) > 3 * _tet_count(coarse)
    assert abs(_volume(fine) - 0.4) < 1e-12
    assert abs(_volume(coarse) - 0.4) < 1e-12


def test_mesh_volume_names_the_nodes_it_adds_on_the_envelope():
    # When the envelope has to be cut to fit, the points that did the cutting
    # come back as a second submesh of POI1 — a warning on stderr is not
    # something a script can act on.
    c = pyrucast.Coords(3)
    envelope = _extruded_plate(c, 6, 0.2)
    mesh = pyrucast.mesher.mesh_volume(envelope, allow_surface_nodes=True)
    assert mesh.element_types() == ["TET4", "POI1"]

    added = mesh.cell_counts()[1]
    assert added > 0
    # Each marker is a node the volume actually uses.
    used = {mesh.node(0, ci, k).id for ci in range(_tet_count(mesh)) for k in range(4)}
    for ci in range(added):
        assert mesh.node(1, ci, 0).id in used


def test_mesh_volume_adds_no_submesh_when_it_adds_no_node():
    c = pyrucast.Coords(3)
    envelope = _surface(c, _box(c, [0, 0, 0], [1, 1, 1]))
    mesh = pyrucast.mesher.mesh_volume(envelope, allow_surface_nodes=True)
    assert mesh.element_types() == ["TET4"]


def test_mesh_volume_hollow_box_subtracts_the_cavity():
    # A cavity is declared by its orientation alone: its normals point into
    # the hole, so it takes its volume out of the total.
    c = pyrucast.Coords(3)
    outer = _box(c, [0, 0, 0], [3, 3, 3])
    inner = _box(c, [1, 1, 1], [2, 2, 2])
    mesh = pyrucast.Mesh(c, "TRI3")
    for f in _BOX_FACETS:
        mesh.unit().add_cell([outer[f[0]], outer[f[1]], outer[f[2]]])
    for f in _BOX_FACETS:
        mesh.unit().add_cell([inner[f[0]], inner[f[2]], inner[f[1]]])
    result = pyrucast.mesher.mesh_volume(mesh)
    assert abs(_volume(result) - (27.0 - 1.0)) < 1e-12


def test_mesh_volume_rejects_an_open_surface():
    c = pyrucast.Coords(3)
    nodes = _box(c, [0, 0, 0], [1, 1, 1])
    try:
        pyrucast.mesher.mesh_volume(_surface(c, nodes, _BOX_FACETS[1:]))
    except RuntimeError as e:
        assert "not closed" in str(e)
    else:
        raise AssertionError("expected RuntimeError for an open surface")


def test_mesh_volume_rejects_inward_normals():
    c = pyrucast.Coords(3)
    nodes = _box(c, [0, 0, 0], [1, 1, 1])
    flipped = [(f[0], f[2], f[1]) for f in _BOX_FACETS]
    try:
        pyrucast.mesher.mesh_volume(_surface(c, nodes, flipped))
    except RuntimeError as e:
        assert "invert()" in str(e)
    else:
        raise AssertionError("expected RuntimeError for inward normals")


def test_mesh_volume_rejects_a_quadrangle_envelope():
    c = pyrucast.Coords(3)
    nodes = _box(c, [0, 0, 0], [1, 1, 1])
    quad = pyrucast.Mesh(c, "QUA4")
    quad.unit().add_cell([nodes[0], nodes[1], nodes[2], nodes[3]])
    try:
        pyrucast.mesher.mesh_volume(quad)
    except RuntimeError as e:
        assert "TRI3" in str(e)
    else:
        raise AssertionError("expected RuntimeError for a QUA4 envelope")


def test_mesh_volume_rejects_a_bad_size():
    c = pyrucast.Coords(3)
    envelope = _surface(c, _box(c, [0, 0, 0], [1, 1, 1]))
    try:
        pyrucast.mesher.mesh_volume(envelope, size=0.0)
    except RuntimeError as e:
        assert "size must be > 0" in str(e)
    else:
        raise AssertionError("expected RuntimeError for size=0")


def test_mesh_volume_refuses_a_box_whose_faces_cannot_be_filled():
    # The same corners with every face split the other way: no
    # tetrahedralization exists on those nodes, and saying so is the only
    # right answer.
    c = pyrucast.Coords(3)
    nodes = _box(c, [0, 0, 0], [1, 1, 1])
    unfillable = [
        (0, 3, 2),
        (0, 2, 1),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    try:
        pyrucast.mesher.mesh_volume(_surface(c, nodes, unfillable))
    except RuntimeError as e:
        assert "without adding a node on the surface" in str(e)
    else:
        raise AssertionError("expected RuntimeError for an unfillable box")
