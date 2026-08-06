"""Python tests for `pyrucast.mesh.grid_surface`."""

import math

import pytest

import pyrucast as pc


def _on_grid(coords, corners, size):
    """A closed SEG2 loop through `corners`, each side cut into whole cells.

    That last part is what `grid_surface` asks of a contour: a side split into
    a whole number of steps of about `size` puts its nodes on the very lines
    the shape's corners impose on the grid, and the core can then meet the
    boundary instead of stopping short of it.
    """
    nodes = [coords.add_node(list(p)) for p in corners]
    mesh = None
    for i, a in enumerate(corners):
        b = corners[(i + 1) % len(corners)]
        n = max(1, round(math.dist(a, b) / size))
        seg = pc.mesh.line(nodes[i], nodes[(i + 1) % len(corners)], n)
        mesh = seg if mesh is None else mesh | seg
    return pc.mesh.consolidate(mesh)


def _cells(mesh):
    return dict(zip(mesh.element_types(), mesh.cell_counts()))


def test_a_rectangle_comes_out_as_the_plain_grid():
    coords = pc.Coords(2)
    corners = [(0.0, 0.0), (0.6, 0.0), (0.6, 0.3), (0.0, 0.3)]
    mesh = pc.mesh.grid_surface(_on_grid(coords, corners, 0.02), "QUA4", size=0.02)
    # 30 × 15 rectangles and nothing else. The frontal paver cannot do this:
    # its rows meet in the middle and leave four diagonal seams.
    assert _cells(mesh) == {"QUA4": 450}


def test_a_concave_l_is_still_the_plain_grid():
    coords = pc.Coords(2)
    corners = [
        (0.0, 0.0),
        (0.6, 0.0),
        (0.6, 0.2),
        (0.3, 0.2),
        (0.3, 0.4),
        (0.0, 0.4),
    ]
    mesh = pc.mesh.grid_surface(_on_grid(coords, corners, 0.02), "QUA4", size=0.02)
    assert _cells(mesh) == {"QUA4": 450}


def test_a_crenellated_profile_has_no_triangle_and_every_jacobian_one():
    u, v = 0.6 / 9, 0.3 / 40
    levels = [40, 3, 6, 4, 6, 4, 6, 3, 40]
    # The base is cut under each step: whole and straight, its nodes would
    # miss every column the steps impose, by 1.2 %.
    corners = [(i * u, 0.0) for i in range(len(levels) + 1)]
    for i in reversed(range(len(levels))):
        corners.append(((i + 1) * u, levels[i] * v))
        corners.append((i * u, levels[i] * v))

    size = 2 * v / 4
    coords = pc.Coords(2)
    mesh = pc.mesh.grid_surface(_on_grid(coords, corners, size), "QUA4", size=size)
    assert _cells(mesh) == {"QUA4": 18 * 2 * sum(levels)}

    worst = 1.0
    for sub in mesh:
        for cell in sub:
            p = [n.position() for n in cell]
            for i in range(4):
                ux, uy = p[(i + 1) % 4][0] - p[i][0], p[(i + 1) % 4][1] - p[i][1]
                wx, wy = p[i - 1][0] - p[i][0], p[i - 1][1] - p[i][1]
                j = (ux * wy - uy * wx) / (math.hypot(ux, uy) * math.hypot(wx, wy))
                worst = min(worst, abs(j))
    assert worst > 1.0 - 1e-9


def test_a_circle_gets_a_core_and_a_frontal_band():
    coords = pc.Coords(2)
    circle = [
        (math.cos(i / 64 * math.tau), math.sin(i / 64 * math.tau)) for i in range(64)
    ]
    # Nothing axis-aligned to snap to: the grid falls back to the bounding
    # box, the core is the staircase inside the disc, and the front paves the
    # ring left between the two. It must still close on the exact area.
    mesh = pc.mesh.grid_surface(_on_grid(coords, circle, 0.1), "QUA4", size=0.1)
    assert mesh.cell_count() > 100
    assert "QUA4" in _cells(mesh)


def test_coarsening_grades_the_interior():
    """`size` devient la taille AU BORD, l'intérieur grossit.

    Chaque niveau divise à peu près par deux ce qui reste. Les transitions
    coûtent des triangles — un bord à cinq côtés n'admet aucun découpage en
    quadrangles — mais jamais la conformité ni l'aire.
    """
    coords = pc.Coords(2)
    contour = _on_grid(coords, [(0.0, 0.0), (1.6, 0.0), (1.6, 0.8), (0.0, 0.8)], 0.025)

    counts = []
    for c in range(3):
        mesh = pc.mesh.grid_surface(contour, "QUA4", size=0.025, coarsen=c)
        cells = _cells(mesh)
        assert (cells.get("TRI3", 0) == 0) == (c == 0)
        counts.append(mesh.cell_count())
    assert counts[0] == 2048
    assert counts[1] < counts[0] // 2
    assert counts[2] < counts[1]

    # L'autre côté du marché de parité : pas un triangle, presque pas de gain.
    strict = pc.mesh.grid_surface(contour, "QUA4", size=0.025, coarsen=3, all_quad=True)
    assert _cells(strict) == {"QUA4": strict.cell_count()}
    assert strict.cell_count() > counts[0] * 0.9


def test_the_mesh_boundary_is_exactly_the_contour():
    """Le contrat, dans sa formulation la plus forte.

    Un mailleur en grille pose ses propres nœuds : il ne tient ce contrat que
    parce qu'il partage ceux du contour là où ils coïncident. On vérifie donc
    les deux sens — tout segment du contour est une arête de bord du maillage,
    et le maillage n'a pas d'autre arête de bord. Aucun nœud ajouté sur le
    bord, aucun perdu, aucun trou à l'intérieur.
    """
    coords = pc.Coords(2)
    # Un L, que la grille rejoint exactement, et un trou circulaire, qu'elle
    # ne peut pas rejoindre et qui revient au front.
    outer = _on_grid(
        coords,
        [(0.0, 0.0), (3.0, 0.0), (3.0, 0.6), (1.5, 0.6), (1.5, 1.2), (0.0, 1.2)],
        0.1,
    )
    hole = _on_grid(
        coords,
        [
            (
                2.25 + 0.15 * math.cos(-i / 32 * math.tau),
                0.3 + 0.15 * math.sin(-i / 32 * math.tau),
            )
            for i in range(32)
        ],
        0.1,
    )
    mesh = pc.mesh.grid_surface(outer | hole, "QUA4", size=0.1)

    segments = set()
    for loop in (outer, hole):
        for sub in loop:
            for cell in sub:
                a, b = (n.id for n in cell)
                segments.add((min(a, b), max(a, b)))

    seen = {}
    for sub in mesh:
        for cell in sub:
            ids = [n.id for n in cell]
            for i, a in enumerate(ids):
                b = ids[(i + 1) % len(ids)]
                key = (min(a, b), max(a, b))
                seen[key] = seen.get(key, 0) + 1

    boundary = {e for e, count in seen.items() if count == 1}
    assert segments <= boundary, f"{len(segments - boundary)} segments perdus"
    assert boundary <= segments, f"{len(boundary - segments)} arêtes de bord en trop"


def test_the_method_form_matches_the_free_function():
    coords = pc.Coords(2)
    contour = _on_grid(coords, [(0.0, 0.0), (0.4, 0.0), (0.4, 0.2), (0.0, 0.2)], 0.05)
    assert _cells(contour.grid_surface("QUA4", size=0.05)) == _cells(
        pc.mesh.grid_surface(contour, "QUA4", size=0.05)
    )


def test_bad_input_is_rejected():
    coords = pc.Coords(2)
    contour = _on_grid(coords, [(0.0, 0.0), (0.4, 0.0), (0.4, 0.2), (0.0, 0.2)], 0.05)
    with pytest.raises(Exception, match="grid_surface"):
        pc.mesh.grid_surface(contour, "TRI3", size=0.05)
    with pytest.raises(Exception, match="grid_surface"):
        pc.mesh.grid_surface(contour, "QUA4", size=-1.0)
