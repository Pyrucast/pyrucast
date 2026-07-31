"""Python tests for `pyrucast.mesher.pave_volume`."""

import pytest

import pyrucast as pc


def _box_skin(n=3):
    """The skin of an n³ box of hexahedra: a closed QUA4 shell, normals out."""
    coords = pc.Coords(3)
    a = coords.add_node([0.0, 0.0, 0.0])
    b = coords.add_node([1.0, 0.0, 0.0])
    c = coords.add_node([1.0, 0.0, 1.0])
    d = coords.add_node([0.0, 0.0, 1.0])
    # Wound so the face's normal is +y, the direction it gets extruded in.
    ring = None
    for p, q in ((a, d), (d, c), (c, b), (b, a)):
        seg = pc.mesher.line(p, q, n)
        ring = seg if ring is None else ring | seg
    face = pc.mesher.pave_surface(pc.consolidate(ring), "QUA4", all_quad=True)
    return pc.mesher.skin(pc.mesher.extrude(face, [0.0, 1.0, 0.0], n))


def _cells(mesh):
    return dict(zip(mesh.element_types(), mesh.cell_counts()))


def test_pave_volume_makes_a_hex_layer_over_a_tet_core():
    mesh = pc.mesher.pave_volume(_box_skin(), layers=1, thickness=0.15, size=0.4)
    k = _cells(mesh)
    assert k["HEX8"] == 54, k
    assert k["PYRA5"] == 54, k
    assert k["TET4"] > 0, k


def test_pave_volume_is_conforming():
    """No facet may serve more than two cells, and the boundary is the skin."""
    mesh = pc.mesher.pave_volume(_box_skin(), layers=1, thickness=0.15, size=0.4)
    faces = {
        "HEX8": [
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ],
        "PYRA5": [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
        "TET4": [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)],
        "PENTA6": [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    }
    seen = {}
    for sub in mesh:
        kind = sub.element_type
        for i in range(sub.cell_count()):
            ids = [n.id for n in sub[i].nodes()]
            for f in faces[kind]:
                key = tuple(sorted(ids[j] for j in f))
                seen[key] = seen.get(key, 0) + 1
    assert all(v <= 2 for v in seen.values()), "a facet serves more than two cells"
    assert sum(1 for v in seen.values() if v == 1) == 54, "the boundary is the skin"


def test_pave_volume_stacks_several_layers():
    mesh = pc.mesher.pave_volume(_box_skin(), layers=2, thickness=0.08, size=0.4)
    k = _cells(mesh)
    assert k["HEX8"] == 108, k
    assert k["PYRA5"] == 54, k


def test_pave_volume_refuses_an_inside_out_envelope():
    skin = pc.mesher.invert(_box_skin())
    with pytest.raises(Exception, match="normals point into the material"):
        pc.mesher.pave_volume(skin, layers=1, thickness=0.15, size=0.4)


def test_pave_volume_refuses_zero_layers():
    with pytest.raises(Exception, match="at least 1"):
        pc.mesher.pave_volume(_box_skin(), layers=0)
