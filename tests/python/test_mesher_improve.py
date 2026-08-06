"""Python tests for `regularize`, `cleanup` and `merge_triangles`."""

import math

import pytest

import pyrucast as pc


def _circle(n=60, size=0.05):
    """A circle paved by `grid_surface` — the case it finds hardest.

    No axis-aligned edge, so the whole boundary falls to the frontal band, and
    that is where every poor cell and every triangle ends up.
    """
    coords = pc.Coords(2)
    pts = [(math.cos(i / n * math.tau), math.sin(i / n * math.tau)) for i in range(n)]
    nodes = [coords.add_node(list(p)) for p in pts]
    contour = None
    for i, a in enumerate(pts):
        b = pts[(i + 1) % n]
        k = max(1, round(math.dist(a, b) / size))
        seg = pc.mesh.line(nodes[i], nodes[(i + 1) % n], k)
        contour = seg if contour is None else contour | seg
    return pc.mesh.grid_surface(pc.mesh.consolidate(contour), "QUA4", size)


def _look(mesh):
    """Cells, triangles, worst normalised Jacobian, its 1st percentile, boundary."""
    qs, tris, used = [], 0, {}
    for sub in mesh:
        for cell in sub:
            p = [nd.position() for nd in cell]
            ids = [nd.id for nd in cell]
            k = len(p)
            if k == 3:
                tris += 1
            for i in range(k):
                a, b = ids[i], ids[(i + 1) % k]
                key = (min(a, b), max(a, b))
                used[key] = used.get(key, 0) + 1
            s = 0.5 * sum(
                p[i][0] * p[(i + 1) % k][1] - p[(i + 1) % k][0] * p[i][1]
                for i in range(k)
            )
            q = p[::-1] if s < 0 else p
            w = 1.0
            for i in range(k):
                u = (q[(i + 1) % k][0] - q[i][0], q[(i + 1) % k][1] - q[i][1])
                v = (q[i - 1][0] - q[i][0], q[i - 1][1] - q[i][1])
                w = min(
                    w, (u[0] * v[1] - u[1] * v[0]) / (math.hypot(*u) * math.hypot(*v))
                )
            qs.append(w)
    qs.sort()
    boundary = frozenset(e for e, c in used.items() if c == 1)
    return len(qs), tris, qs[0], qs[len(qs) // 100], boundary


def test_regularize_improves_the_mesh_and_keeps_its_boundary():
    mesh = _circle()
    cells, tris, worst, p1, boundary = _look(mesh)
    out = pc.mesh.regularize(mesh, sweeps=40)
    cells2, tris2, worst2, p12, boundary2 = _look(out)

    assert (cells, tris) == (cells2, tris2)
    assert worst2 > worst
    assert p12 > p1
    # Node for node and edge for edge: the boundary is pinned, so it is not
    # merely in the same place, it is the same nodes.
    assert boundary == boundary2


def test_neither_smoothing_rule_ever_inverts_a_cell():
    mesh = _circle()
    _, _, worst, _, _ = _look(mesh)
    for angular in (True, False):
        _, _, w, _, _ = _look(pc.mesh.regularize(mesh, sweeps=40, angular=angular))
        assert w > worst, f"angular={angular}"


def test_merge_triangles_removes_them_two_at_a_time():
    mesh = _circle()
    _, tris, _, _, boundary = _look(mesh)
    _, tris2, worst2, _, boundary2 = _look(pc.mesh.merge_triangles(mesh))

    assert tris2 < tris
    # `4Q + 3T = 2·E_int + E_bord` fixes T's parity to the boundary's, and
    # nothing here may change the boundary.
    assert (tris - tris2) % 2 == 0
    assert worst2 > 0.0
    assert boundary == boundary2


def test_the_three_compose_and_the_composition_converges():
    mesh = _circle()
    _, tris0, worst0, p10, boundary = _look(mesh)
    counts = []
    for _ in range(4):
        mesh = pc.mesh.merge_triangles(mesh)
        mesh = pc.mesh.cleanup(mesh)
        mesh = pc.mesh.regularize(mesh, sweeps=30, angular=True)
        mesh = pc.mesh.regularize(mesh, sweeps=15, angular=False)
        _, t, w, _, b = _look(mesh)
        assert w > 0.0, "a round left an invalid cell"
        assert b == boundary, "a round changed the boundary"
        counts.append(t)
    _, tris1, worst1, p11, _ = _look(mesh)
    assert tris1 < tris0 // 2
    assert worst1 > worst0
    assert p11 > p10
    assert counts[2] == counts[3], f"must settle: {counts}"


def test_the_method_form_matches_the_free_function():
    mesh = _circle()
    assert (
        _look(mesh.regularize(sweeps=10))[2]
        == _look(pc.mesh.regularize(mesh, sweeps=10))[2]
    )


def test_bad_input_is_rejected():
    coords = pc.Coords(2)
    a, b = coords.add_node([0.0, 0.0]), coords.add_node([1.0, 0.0])
    line = pc.mesh.line(a, b, 4)
    with pytest.raises(Exception, match="regularize"):
        pc.mesh.regularize(line)
