"""Python tests for `from_arrays` — the flat-array import every exchange format
goes through (`from_gmsh`, `from_medcoupling`).

They build the arrays by hand and need neither gmsh nor medcoupling, so they
stay in the normal pass.
"""

import pytest

import pyrucast

SQUARE_V2 = """\
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
1 1 "bottom"
2 2 "plate"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
3
1 1 2 1 1 1 2
2 2 2 2 2 1 2 3
3 2 2 2 2 1 3 4
$EndElements
"""


def square_arrays():
    """`SQUARE_V2` as flat arrays: the node table, then one block per
    (element type, groups) with its flattened connectivity."""
    tags = [1, 2, 3, 4]
    coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    blocks = [
        ("SEG2", [1, 2], ["bottom"]),
        ("TRI3", [1, 2, 3, 1, 3, 4], ["plate"]),
    ]
    return tags, coords, blocks


def shape(groups):
    """What an import *is*, comparable from one path to the other."""
    return {k: (m.element_types(), m.cell_counts()) for k, m in groups.items()}


def meshes(*args, **kwargs):
    return pyrucast.mesh.from_arrays(*args, **kwargs)[0]


def test_arrays_agree_with_the_file():
    """The memory path and the file path return the same mesh."""
    tags, coords, blocks = square_arrays()
    memoire = meshes(pyrucast.Coords(dim=2), tags, coords, blocks, order="gmsh")
    fichier = pyrucast.mesh.read_gmsh_str(pyrucast.Coords(dim=2), SQUARE_V2)
    assert shape(memoire) == shape(fichier)


def test_arrays_from_python_lists():
    """A `list` exports no buffer: this is the conversion fallback, and it must
    give exactly the same result as the copy-free path."""
    tags, coords, blocks = square_arrays()
    groups = meshes(pyrucast.Coords(dim=2), tags, coords, blocks)
    assert list(groups) == ["bottom", "plate"]
    assert groups["plate"].cell_counts() == [2]


@pytest.mark.parametrize("dtype", ["uint64", "int64"])
def test_arrays_from_numpy_take_the_buffer_path(dtype):
    """The copy-free path: contiguous numpy arrays, gmsh's `uint64` tags as
    well as medcoupling's `int64` ones."""
    np = pytest.importorskip("numpy")
    tags, coords, blocks = square_arrays()
    groups = meshes(
        pyrucast.Coords(dim=2),
        np.array(tags, dtype=dtype),
        np.array(coords, dtype=np.float64),
        [(t, np.array(conn, dtype=dtype), names) for t, conn, names in blocks],
    )
    assert shape(groups) == shape(
        pyrucast.mesh.read_gmsh_str(pyrucast.Coords(dim=2), SQUARE_V2)
    )


def test_arrays_accept_a_non_contiguous_view():
    """A strided view has no contiguous buffer to lend: the fallback must read it
    all the same, without mistaking the elements."""
    np = pytest.importorskip("numpy")
    tags, coords, _ = square_arrays()
    espace = np.zeros(2 * len(tags), dtype=np.uint64)
    espace[::2] = tags
    groups = meshes(
        pyrucast.Coords(dim=2), espace[::2], coords, [("TRI3", [1, 2, 3], ["plate"])]
    )
    assert groups["plate"].cell_counts() == [1]


def test_arrays_share_one_coords():
    """A node between two groups is the same on both sides."""
    tags, coords, blocks = square_arrays()
    c = pyrucast.Coords(dim=2)
    groups = meshes(c, tags, coords, blocks)
    assert c.node_count() == 4
    assert len(groups) == 2


def test_arrays_ignore_unreferenced_nodes():
    tags, coords, _ = square_arrays()
    tags = tags + [99]
    coords = coords + [7.0, 7.0, 0.0]
    c = pyrucast.Coords(dim=2)
    meshes(c, tags, coords, [("TRI3", [1, 2, 3], ["plate"])])
    assert c.node_count() == 3


def test_arrays_two_coordinates_per_node():
    """medcoupling hands a 2-D mesh with two coordinates per node."""
    c = pyrucast.Coords(dim=2)
    groups = meshes(
        c, [1, 2, 3], [0.0, 0.0, 2.0, 0.0, 0.0, 2.0], [("TRI3", [1, 2, 3], ["t"])]
    )
    assert groups["t"].cell_counts() == [1]
    assert c.node_count() == 3


def test_arrays_med_order():
    """A TET4 in MED order comes out with its base walked back."""
    c = pyrucast.Coords(dim=3)
    xyz = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    # MED lists the base (0, 2, 1): pyrucast's (0, 1, 2).
    groups = meshes(c, [1, 2, 3, 4], xyz, [("TET4", [1, 3, 2, 4], ["t"])], order="med")
    cell = groups["t"].cell(0, 0)
    assert [n.position() for n in cell.nodes()] == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_arrays_fields_come_back_in_order():
    tags, coords, blocks = square_arrays()
    blocks = [
        ("SEG2", [1, 2], [], ["bottom"]),
        ("TRI3", [1, 2, 3, 1, 3, 4], [10, 11], ["plate"]),
    ]
    c = pyrucast.Coords(dim=2)
    groups, nodal, elem = pyrucast.mesh.from_arrays(
        c,
        tags,
        coords,
        blocks,
        node_fields=[(["T"], tags, [1.0, 2.0, 3.0, 4.0])],
        cell_fields=[(["E"], [10, 11], [5.0, 6.0], "cell")],
    )
    assert list(groups) == ["bottom", "plate"]
    node = groups["plate"].node(0, 1, 1)  # second node of the second cell
    assert nodal[0].value(node, "T") == 3.0
    zone = elem[0][0]
    assert zone.gauss_count() == 1
    assert [zone.value(k, 0, "E") for k in range(2)] == [5.0, 6.0]


def test_arrays_unknown_node_raises():
    tags, coords, _ = square_arrays()
    with pytest.raises(RuntimeError, match="unknown node 77"):
        meshes(pyrucast.Coords(dim=2), tags, coords, [("TRI3", [1, 2, 77], ["plate"])])


def test_arrays_ragged_block_raises():
    tags, coords, _ = square_arrays()
    with pytest.raises(RuntimeError, match="whole number of cells"):
        meshes(
            pyrucast.Coords(dim=2), tags, coords, [("TRI3", [1, 2, 3, 4], ["plate"])]
        )


def test_arrays_unknown_type_raises():
    tags, coords, _ = square_arrays()
    with pytest.raises(ValueError, match="TRI10"):
        meshes(pyrucast.Coords(dim=2), tags, coords, [("TRI10", [], ["plate"])])


def test_arrays_malformed_block_raises():
    tags, coords, _ = square_arrays()
    with pytest.raises(ValueError, match="block 0"):
        meshes(pyrucast.Coords(dim=2), tags, coords, [("TRI3", [1, 2, 3])])


def test_gmsh_codes_translate():
    assert pyrucast.mesh.element_type_from_gmsh(4) == "TET4"
    with pytest.raises(RuntimeError, match="unsupported element type 21"):
        pyrucast.mesh.element_type_from_gmsh(21)


# ── to_arrays: the way back ────────────────────────────────────────────────


def test_to_arrays_mirrors_from_arrays():
    """What `to_arrays` lays out, `from_arrays` reads back: same groups, same
    cells, same values."""
    tags, coords, _ = square_arrays()
    blocks = [
        ("SEG2", [1, 2], [], ["bottom"]),
        ("TRI3", [1, 2, 3, 1, 3, 4], [10, 11], ["plate"]),
    ]
    groups, nodal, elem = pyrucast.mesh.from_arrays(
        pyrucast.Coords(dim=2),
        tags,
        coords,
        blocks,
        node_fields=[(["T"], tags, [1.0, 2.0, 3.0, 4.0])],
        cell_fields=[(["E"], [10, 11], [5.0, 6.0], "cell")],
    )
    out = pyrucast.export.to_arrays(groups, node_fields=nodal, element_fields=elem)
    assert out["dim"] == 2
    assert [b[0] for b in out["blocks"]] == ["SEG2", "TRI3"]
    back, nodal2, elem2 = pyrucast.mesh.from_arrays(
        pyrucast.Coords(dim=2),
        out["node_tags"],
        out["node_coords"],
        out["blocks"],
        node_fields=out["node_fields"],
        cell_fields=out["cell_fields"],
    )
    assert shape(back) == shape(groups)
    assert sorted(
        nodal2[0][0].value(n, "T") for n in back["plate"].cell(0, 1).nodes()
    ) == [1.0, 3.0, 4.0]
    assert [elem2[0][0].value(c, 0, "E") for c in range(2)] == [5.0, 6.0]


def test_arrays_out_lend_their_buffer():
    """An `Array` is read without a copy by `memoryview` (and by numpy, when it
    is there): int64 tags, float64 values."""
    tags, coords, blocks = square_arrays()
    groups = meshes(pyrucast.Coords(dim=2), tags, coords, blocks)
    out = pyrucast.export.to_arrays(groups)
    view = memoryview(out["node_tags"])
    assert view.readonly and view.itemsize == 8 and view.ndim == 1
    assert view.tolist() == [1, 2, 3, 4]
    assert memoryview(out["node_coords"]).format == "d"
    assert list(out["node_coords"])[:4] == [0.0, 0.0, 1.0, 0.0]
    assert len(out["blocks"][1][1]) == 6


def test_arrays_out_are_numpy_int64():
    np = pytest.importorskip("numpy")
    tags, coords, blocks = square_arrays()
    out = pyrucast.export.to_arrays(
        meshes(pyrucast.Coords(dim=2), tags, coords, blocks)
    )
    tags_out = np.asarray(out["node_tags"])
    assert tags_out.dtype == np.int64
    assert not tags_out.flags.owndata  # a view on pyrucast's buffer


def test_to_arrays_orders_and_first_tag():
    c = pyrucast.Coords(dim=3)
    xyz = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    groups = meshes(c, [1, 2, 3, 4], xyz, [("TET4", [1, 2, 3, 4], ["t"])])
    med = pyrucast.export.to_arrays(groups, order="med", first_tag=0)
    assert list(med["blocks"][0][1]) == [0, 2, 1, 3]
    assert list(med["node_tags"]) == [0, 1, 2, 3]
