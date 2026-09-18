"""Python tests of node selection by geometric region —
``points_in_*`` / ``points_on_*`` / ``points_below_plane``.

Every operator returns a POI1 mesh modelled on the input: one submesh per
submesh, possibly empty. The "nearest node" query, which can only return one,
is the ``mesh.nearest_node`` method and not an operator of this family.

"""

import math

import pytest

import pyrucast


def _cloud(dim, points):
    """A POI1 mesh of a single submesh, one node per coordinate."""
    c = pyrucast.Coords(dim)
    m = pyrucast.Mesh(c, "POI1")
    ids = [c.add_node(list(p)) for p in points]
    for nid in ids:
        m.unit().add_cell([nid])
    return c, m


def _coords_of(sel, sub=0):
    """Coordinates of the nodes selected in a submesh, in order."""
    return [sel.node(sub, i, 0).position() for i in range(sel.cell_counts()[sub])]


# --- sphères ---------------------------------------------------------------


def test_sphere_in_and_on_2d():
    # A cross of 5 points about the origin, at distance 0 and 1.
    _, m = _cloud(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])

    inside = pyrucast.mesh.points_in_sphere(m, [0.0, 0.0], 1.0)
    assert _coords_of(inside) == [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    # The filled disc loses its centre when only the circle is kept.
    on = pyrucast.mesh.points_on_sphere(m, [0.0, 0.0], 1.0)
    assert _coords_of(on) == [[1.0, 0.0], [0.0, 1.0]]


# --- plans -----------------------------------------------------------------


def test_plane_on_and_below():
    _, m = _cloud(3, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # La face z = 0 : la normale n'a pas besoin d'être unitaire.
    face = pyrucast.mesh.points_on_plane(m, [0.0, 0.0, 0.0], [0.0, 0.0, 3.0])
    assert _coords_of(face) == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

    # Below the z = 1 plane: everyone (the plane is included).
    below = pyrucast.mesh.points_below_plane(m, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    assert len(_coords_of(below)) == 3

    # Normale retournée : l'autre demi-espace, plan compris.
    above = pyrucast.mesh.points_below_plane(m, [0.0, 0.0, 1.0], [0.0, 0.0, -1.0])
    assert _coords_of(above) == [[0.0, 0.0, 1.0]]


# --- droite et cylindre ----------------------------------------------------


def test_line_is_infinite_where_the_cylinder_is_capped():
    _, m = _cloud(2, [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [1.0, 0.0]])

    # The line passes through the diagonal's three points, even beyond b.
    on_line = pyrucast.mesh.points_on_line(m, [0.0, 0.0], [1.0, 1.0])
    assert _coords_of(on_line) == [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]

    # The cylinder, for its part, stops at its end section.
    capped = pyrucast.mesh.points_in_cylinder(m, [0.0, 0.0], [1.0, 1.0], 1e-9)
    assert _coords_of(capped) == [[0.0, 0.0], [1.0, 1.0]]


def test_cylinder_surface_excludes_the_end_discs():
    _, m = _cloud(
        3,
        [
            [1.0, 0.0, 1.0],  # on the tube
            [0.0, 0.0, 1.0],  # on the axis
            [0.5, 0.0, 0.0],  # inside the bottom disc
            [1.0, 0.0, 5.0],  # beyond the top section
        ],
    )
    base, top = [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]

    lateral = pyrucast.mesh.points_on_cylinder(m, base, top, 1.0)
    assert _coords_of(lateral) == [[1.0, 0.0, 1.0]]

    solid = pyrucast.mesh.points_in_cylinder(m, base, top, 1.0)
    assert len(_coords_of(solid)) == 3


# --- cône ------------------------------------------------------------------


def test_cone_defaults_to_an_apex_and_degenerates_to_a_cylinder():
    # Radius 2 at z = 0, apex at z = 2: local radius 1 at mid-height.
    _, m = _cloud(3, [[1.0, 0.0, 1.0], [0.4, 0.0, 1.0], [1.6, 0.0, 1.0]])
    base, top = [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]

    # top_radius is 0 by default: the true cone, `top` being its apex.
    on = pyrucast.mesh.points_on_cone(m, base, top, 2.0)
    assert _coords_of(on) == [[1.0, 0.0, 1.0]]

    inside = pyrucast.mesh.points_in_cone(m, base, top, 2.0)
    assert _coords_of(inside) == [[1.0, 0.0, 1.0], [0.4, 0.0, 1.0]]

    # Equal radii: a cylinder, the three points are inside.
    cyl = pyrucast.mesh.points_in_cone(m, base, top, 2.0, 2.0)
    assert len(_coords_of(cyl)) == 3


# --- tore ------------------------------------------------------------------


def test_torus_tube_around_its_directrix():
    _, m = _cloud(
        3,
        [
            [2.5, 0.0, 0.0],  # on the tube, outer equator
            [2.0, 0.0, 0.0],  # on the directrix, hence inside
            [0.0, 0.0, 0.0],  # centre of the hole, outside
            [0.0, 2.0, 0.5],  # on the tube, a quarter turn further
        ],
    )
    center, axis = [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]

    on = pyrucast.mesh.points_on_torus(m, center, axis, 2.0, 0.5)
    assert _coords_of(on) == [[2.5, 0.0, 0.0], [0.0, 2.0, 0.5]]

    inside = pyrucast.mesh.points_in_torus(m, center, axis, 2.0, 0.5)
    assert len(_coords_of(inside)) == 3


def test_torus_needs_a_3d_mesh():
    _, m = _cloud(2, [[0.0, 0.0]])
    with pytest.raises(RuntimeError):
        pyrucast.mesh.points_in_torus(m, [0.0, 0.0], [0.0, 1.0], 2.0, 0.5)


# --- structure of the result -----------------------------------------------


def test_result_mirrors_the_submeshes_including_empty_zones():
    c = pyrucast.Coords(2)
    a, b = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0])
    far = c.add_node([0.5, 9.0])

    # Zone 0 at y = 0, zone 1 far above.
    m = pyrucast.Mesh(c, "SEG2")
    m.unit().add_cell([a, b])
    high = pyrucast.Mesh(c, "POI1")
    high.unit().add_cell([far])
    m = m | high

    sel = pyrucast.mesh.points_on_plane(m, [0.0, 0.0], [0.0, 1.0])
    assert sel.element_types() == ["POI1", "POI1"]
    # The second zone selects nothing but stays present and empty.
    assert sel.cell_counts() == [2, 0]

    # `consolidate_mesh` is the way back to a single cloud.
    assert pyrucast.mesh.consolidate(sel).cell_counts() == [2]


def test_tolerance_defaults_to_the_model_scale():
    _, m = _cloud(2, [[0.0, 0.0], [1.0, 0.01]])

    # Default precision (1e-6 × diagonal): the second node is out of band.
    assert (
        len(_coords_of(pyrucast.mesh.points_on_plane(m, [0.0, 0.0], [0.0, 1.0]))) == 1
    )

    # An explicit tolerance wider than its offset: it gets in.
    loose = pyrucast.mesh.points_on_plane(m, [0.0, 0.0], [0.0, 1.0], tol=0.02)
    assert len(_coords_of(loose)) == 2


def test_selection_feeds_elements_on():
    """The POI1 output is an ordinary point mesh: it plugs back into
    `elements_on` to get back to the elements the selection carries."""
    c = pyrucast.Coords(2)
    ids = [c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0])]
    m = pyrucast.Mesh(c, "QUA4")
    m.unit().add_cell(ids)

    bottom = pyrucast.mesh.points_on_plane(m, [0.0, 0.0], [0.0, 1.0])
    assert bottom.cell_counts() == [2]
    assert pyrucast.mesh.elements_on(m, bottom, strict=True).cell_counts() == [0]
    assert pyrucast.mesh.elements_on(m, bottom, strict=False).cell_counts() == [1]


def test_nearest_node_is_the_single_node_query():
    """The family's "single node" counterpart: a method of the mesh, which
    returns a `Node` and not a POI1 mesh."""
    _, m = _cloud(2, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    node = m.nearest_node([0.9, 0.9])
    assert isinstance(node, pyrucast.Node)
    assert node.position() == [1.0, 1.0]


def test_invalid_arguments_raise():
    _, m = _cloud(2, [[0.0, 0.0]])

    with pytest.raises(RuntimeError):  # mauvaise dimension
        pyrucast.mesh.points_in_sphere(m, [0.0, 0.0, 0.0], 1.0)
    with pytest.raises(RuntimeError):  # rayon négatif
        pyrucast.mesh.points_in_sphere(m, [0.0, 0.0], -1.0)
    with pytest.raises(RuntimeError):  # tolérance négative
        pyrucast.mesh.points_in_sphere(m, [0.0, 0.0], 1.0, tol=-1e-9)
    with pytest.raises(RuntimeError):  # normale nulle
        pyrucast.mesh.points_on_plane(m, [0.0, 0.0], [0.0, 0.0])
    with pytest.raises(RuntimeError):  # axe de longueur nulle
        pyrucast.mesh.points_on_line(m, [1.0, 1.0], [1.0, 1.0])


def test_axisymmetric_selection_reads_the_meridian_plane():
    """Under axisymmetry the nodes are tested in the (r, z) half-plane where they
    are stored: the "circle" is a circle of the meridian, not a sphere of the
    solide de révolution."""
    c = pyrucast.Coords.axisymmetric()
    m = pyrucast.Mesh(c, "POI1")
    for p in ([1.0, 0.0], [0.0, 1.0], [2.0, 0.0]):
        m.unit().add_cell([c.add_node(list(p))])

    on = pyrucast.mesh.points_on_sphere(m, [0.0, 0.0], 1.0)
    assert _coords_of(on) == [[1.0, 0.0], [0.0, 1.0]]
    assert math.isclose(_coords_of(on)[0][0], 1.0)
