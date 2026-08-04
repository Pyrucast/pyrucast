"""Tests Python de la sélection de nœuds par région géométrique —
``points_in_*`` / ``points_on_*`` / ``points_below_plane``.

Chaque opérateur renvoie un maillage POI1 calqué sur l'entrée : un
sous-maillage par sous-maillage, éventuellement vide. La requête « nœud le
plus proche », qui ne peut en renvoyer qu'un, est la méthode
``mesh.nearest_node`` et non un opérateur de cette famille.
"""

import math

import pytest

import pyrucast


def _cloud(dim, points):
    """Maillage POI1 d'un seul sous-maillage, un nœud par coordonnée."""
    c = pyrucast.Coords(dim)
    m = pyrucast.Mesh(c, "POI1")
    ids = [c.add_node(list(p)) for p in points]
    for nid in ids:
        m.unit().add_cell([nid])
    return c, m


def _coords_of(sel, sub=0):
    """Coordonnées des nœuds sélectionnés dans un sous-maillage, dans l'ordre."""
    return [sel.node(sub, i, 0).position() for i in range(sel.cell_counts()[sub])]


# --- sphères ---------------------------------------------------------------


def test_sphere_in_and_on_2d():
    # Croix de 5 points autour de l'origine, à distance 0 et 1.
    _, m = _cloud(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])

    inside = pyrucast.mesh.points_in_sphere(m, [0.0, 0.0], 1.0)
    assert _coords_of(inside) == [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    # Le disque plein perd son centre quand on ne garde que le cercle.
    on = pyrucast.mesh.points_on_sphere(m, [0.0, 0.0], 1.0)
    assert _coords_of(on) == [[1.0, 0.0], [0.0, 1.0]]


# --- plans -----------------------------------------------------------------


def test_plane_on_and_below():
    _, m = _cloud(3, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # La face z = 0 : la normale n'a pas besoin d'être unitaire.
    face = pyrucast.mesh.points_on_plane(m, [0.0, 0.0, 0.0], [0.0, 0.0, 3.0])
    assert _coords_of(face) == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

    # Sous le plan z = 1 : tout le monde (le plan est inclus).
    below = pyrucast.mesh.points_below_plane(m, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    assert len(_coords_of(below)) == 3

    # Normale retournée : l'autre demi-espace, plan compris.
    above = pyrucast.mesh.points_below_plane(m, [0.0, 0.0, 1.0], [0.0, 0.0, -1.0])
    assert _coords_of(above) == [[0.0, 0.0, 1.0]]


# --- droite et cylindre ----------------------------------------------------


def test_line_is_infinite_where_the_cylinder_is_capped():
    _, m = _cloud(2, [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [1.0, 0.0]])

    # La droite passe par les trois points de la diagonale, même au-delà de b.
    on_line = pyrucast.mesh.points_on_line(m, [0.0, 0.0], [1.0, 1.0])
    assert _coords_of(on_line) == [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]

    # Le cylindre, lui, s'arrête à sa section d'extrémité.
    capped = pyrucast.mesh.points_in_cylinder(m, [0.0, 0.0], [1.0, 1.0], 1e-9)
    assert _coords_of(capped) == [[0.0, 0.0], [1.0, 1.0]]


def test_cylinder_surface_excludes_the_end_discs():
    _, m = _cloud(
        3,
        [
            [1.0, 0.0, 1.0],  # sur le tube
            [0.0, 0.0, 1.0],  # sur l'axe
            [0.5, 0.0, 0.0],  # dans le disque du bas
            [1.0, 0.0, 5.0],  # au-delà de la section haute
        ],
    )
    base, top = [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]

    lateral = pyrucast.mesh.points_on_cylinder(m, base, top, 1.0)
    assert _coords_of(lateral) == [[1.0, 0.0, 1.0]]

    solid = pyrucast.mesh.points_in_cylinder(m, base, top, 1.0)
    assert len(_coords_of(solid)) == 3


# --- cône ------------------------------------------------------------------


def test_cone_defaults_to_an_apex_and_degenerates_to_a_cylinder():
    # Rayon 2 en z = 0, sommet en z = 2 : rayon local 1 à mi-hauteur.
    _, m = _cloud(3, [[1.0, 0.0, 1.0], [0.4, 0.0, 1.0], [1.6, 0.0, 1.0]])
    base, top = [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]

    # top_radius vaut 0 par défaut : le cône vrai, `top` est son sommet.
    on = pyrucast.mesh.points_on_cone(m, base, top, 2.0)
    assert _coords_of(on) == [[1.0, 0.0, 1.0]]

    inside = pyrucast.mesh.points_in_cone(m, base, top, 2.0)
    assert _coords_of(inside) == [[1.0, 0.0, 1.0], [0.4, 0.0, 1.0]]

    # Rayons égaux : un cylindre, les trois points sont dedans.
    cyl = pyrucast.mesh.points_in_cone(m, base, top, 2.0, 2.0)
    assert len(_coords_of(cyl)) == 3


# --- tore ------------------------------------------------------------------


def test_torus_tube_around_its_directrix():
    _, m = _cloud(
        3,
        [
            [2.5, 0.0, 0.0],  # sur le tube, équateur extérieur
            [2.0, 0.0, 0.0],  # sur la directrice, donc dedans
            [0.0, 0.0, 0.0],  # centre du trou, dehors
            [0.0, 2.0, 0.5],  # sur le tube, un quart de tour plus loin
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


# --- structure du résultat -------------------------------------------------


def test_result_mirrors_the_submeshes_including_empty_zones():
    c = pyrucast.Coords(2)
    a, b = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0])
    far = c.add_node([0.5, 9.0])

    # Zone 0 sur y = 0, zone 1 loin au-dessus.
    m = pyrucast.Mesh(c, "SEG2")
    m.unit().add_cell([a, b])
    high = pyrucast.Mesh(c, "POI1")
    high.unit().add_cell([far])
    m = m | high

    sel = pyrucast.mesh.points_on_plane(m, [0.0, 0.0], [0.0, 1.0])
    assert sel.element_types() == ["POI1", "POI1"]
    # La seconde zone ne sélectionne rien mais reste présente et vide.
    assert sel.cell_counts() == [2, 0]

    # `consolidate_mesh` est la voie pour retomber sur un nuage unique.
    assert pyrucast.mesh.consolidate(sel).cell_counts() == [2]


def test_tolerance_defaults_to_the_model_scale():
    _, m = _cloud(2, [[0.0, 0.0], [1.0, 0.01]])

    # Précision par défaut (1e-6 × diagonale) : le second nœud est hors bande.
    assert (
        len(_coords_of(pyrucast.mesh.points_on_plane(m, [0.0, 0.0], [0.0, 1.0]))) == 1
    )

    # Tolérance explicite plus large que son décalage : il rentre.
    loose = pyrucast.mesh.points_on_plane(m, [0.0, 0.0], [0.0, 1.0], tol=0.02)
    assert len(_coords_of(loose)) == 2


def test_selection_feeds_elements_on():
    """La sortie POI1 est un maillage de points ordinaire : elle se rebranche
    sur `elements_on` pour remonter aux éléments portés par la sélection."""
    c = pyrucast.Coords(2)
    ids = [c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0])]
    m = pyrucast.Mesh(c, "QUA4")
    m.unit().add_cell(ids)

    bottom = pyrucast.mesh.points_on_plane(m, [0.0, 0.0], [0.0, 1.0])
    assert bottom.cell_counts() == [2]
    assert pyrucast.mesh.elements_on(m, bottom, strict=True).cell_counts() == [0]
    assert pyrucast.mesh.elements_on(m, bottom, strict=False).cell_counts() == [1]


def test_nearest_node_is_the_single_node_query():
    """Le pendant « un seul nœud » de la famille : une méthode du maillage,
    qui renvoie un `Node` et non un maillage POI1."""
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
    """En axisymétrie les nœuds sont testés dans le demi-plan (r, z) où ils
    sont stockés : le « cercle » est un cercle du méridien, pas une sphère du
    solide de révolution."""
    c = pyrucast.Coords.axisymmetric()
    m = pyrucast.Mesh(c, "POI1")
    for p in ([1.0, 0.0], [0.0, 1.0], [2.0, 0.0]):
        m.unit().add_cell([c.add_node(list(p))])

    on = pyrucast.mesh.points_on_sphere(m, [0.0, 0.0], 1.0)
    assert _coords_of(on) == [[1.0, 0.0], [0.0, 1.0]]
    assert math.isclose(_coords_of(on)[0][0], 1.0)
