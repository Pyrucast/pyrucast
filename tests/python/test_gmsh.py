"""Python tests for `read_gmsh` / `read_gmsh_str` — importing a gmsh `.msh`
mesh as a `dict[str, Mesh]` keyed by physical group.

The two fixtures describe the **same** unit square (two TRI3) with a named
surface "plate" and a named bottom edge "bottom", once in MSH 2.2 and once
in MSH 4.1.
"""

import textwrap

import pytest

import pyrucast

SQUARE_V2 = textwrap.dedent(
    """\
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
)

SQUARE_V4 = textwrap.dedent(
    """\
    $MeshFormat
    4.1 0 8
    $EndMeshFormat
    $PhysicalNames
    2
    1 1 "bottom"
    2 2 "plate"
    $EndPhysicalNames
    $Entities
    0 1 1 0
    1 0 0 0 1 0 0 1 1 0
    1 0 0 0 1 1 0 1 2 0
    $EndEntities
    $Nodes
    2 4 1 4
    1 1 0 2
    1
    2
    0 0 0
    1 0 0
    2 1 0 2
    3
    4
    1 1 0
    0 1 0
    $EndNodes
    $Elements
    2 3 1 3
    1 1 1 1
    1 1 2
    2 1 2 2
    2 1 2 3
    3 1 3 4
    $EndElements
    """
)


@pytest.mark.parametrize("src", [SQUARE_V2, SQUARE_V4], ids=["v2.2", "v4.1"])
def test_read_returns_dict_by_group(src):
    coords = pyrucast.Coords(dim=2)
    groups = pyrucast.mesh.read_gmsh_str(coords, src)
    assert isinstance(groups, dict)
    assert set(groups) == {"bottom", "plate"}

    plate = groups["plate"]
    assert plate.element_types() == ["TRI3"]
    assert plate.cell_count() == 2

    bottom = groups["bottom"]
    assert bottom.element_types() == ["SEG2"]
    assert bottom.cell_count() == 1


@pytest.mark.parametrize("src", [SQUARE_V2, SQUARE_V4], ids=["v2.2", "v4.1"])
def test_reads_into_the_given_coords(src):
    # The nodes land in the caller's Coords; all groups share it, so a node
    # shared between groups is shared, not duplicated (4 corners total).
    coords = pyrucast.Coords(dim=2)
    pyrucast.mesh.read_gmsh_str(coords, src)
    assert coords.node_count() == 4


@pytest.mark.parametrize("src", [SQUARE_V2, SQUARE_V4], ids=["v2.2", "v4.1"])
def test_mesh_coords_recovers_the_handle(src):
    # Even if the original Coords handle is dropped, mesh.coords() gets it
    # back — and every group reports the very same Coords.
    groups = pyrucast.mesh.read_gmsh_str(pyrucast.Coords(dim=2), src)
    recovered = groups["plate"].coords()
    assert recovered.node_count() == 4
    assert groups["bottom"].coords().node_count() == 4


def test_coords_dimension_decides_kept_coordinates():
    # z = 0 here, so 2-D vs 3-D is observable on the node coordinate length.
    c2 = pyrucast.Coords(dim=2)
    g2 = pyrucast.mesh.read_gmsh_str(c2, SQUARE_V2)
    assert len(g2["plate"].node(0, 0, 0).position()) == 2
    c3 = pyrucast.Coords(dim=3)
    g3 = pyrucast.mesh.read_gmsh_str(c3, SQUARE_V2)
    assert len(g3["plate"].node(0, 0, 0).position()) == 3


def test_read_from_file(tmp_path):
    path = tmp_path / "square.msh"
    path.write_text(SQUARE_V2)
    coords = pyrucast.Coords(dim=2)
    groups = pyrucast.mesh.read_gmsh(coords, str(path))
    assert set(groups) == {"bottom", "plate"}


def test_unsupported_element_type_raises():
    # gmsh type 21 = 10-node third-order triangle, which pyrucast has no
    # element for.
    bad = textwrap.dedent(
        """\
        $MeshFormat
        2.2 0 8
        $EndMeshFormat
        $Nodes
        3
        1 0 0 0
        2 1 0 0
        3 2 0 0
        $EndNodes
        $Elements
        1
        1 21 2 0 1 1 2 3
        $EndElements
        """
    )
    with pytest.raises(Exception, match="unsupported element type"):
        pyrucast.mesh.read_gmsh_str(pyrucast.Coords(dim=2), bad)


def test_reads_a_pyramid():
    """gmsh type 7 is the 5-node pyramid: square base then apex."""
    mesh = textwrap.dedent(
        """\
        $MeshFormat
        2.2 0 8
        $EndMeshFormat
        $Nodes
        5
        1 -1 -1 0
        2 1 -1 0
        3 1 1 0
        4 -1 1 0
        5 0 0 1
        $EndNodes
        $Elements
        1
        1 7 2 0 1 1 2 3 4 5
        $EndElements
        """
    )
    groups = pyrucast.mesh.read_gmsh_str(pyrucast.Coords(dim=3), mesh)
    assert list(groups) == ["<ungrouped>"]
    assert groups["<ungrouped>"].element_types() == ["PYRA5"]


# ── from_gmsh_arrays : le maillage déjà en mémoire ───────────────────────────
# Ces tests n'ont pas besoin de gmsh : ils fabriquent les tableaux à la main,
# dans la forme où gmsh les rend. Ils restent donc dans la passe normale.


def square_arrays():
    """`SQUARE_V2` sous la forme que gmsh tend : table des nœuds, puis un bloc
    par (entité, type d'élément) avec sa connectivité à plat."""
    tags = [1, 2, 3, 4]
    coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    blocks = [
        (1, [1, 2], ["bottom"]),  # SEG2
        (2, [1, 2, 3, 1, 3, 4], ["plate"]),  # TRI3
    ]
    return tags, coords, blocks


def shape(groups):
    """Ce qu'un import *est*, comparable d'une voie à l'autre."""
    return {k: (m.element_types(), m.cell_counts()) for k, m in groups.items()}


def test_arrays_agree_with_the_file():
    """La voie mémoire et la voie fichier rendent le même maillage."""
    tags, coords, blocks = square_arrays()
    memoire = pyrucast.mesh.from_gmsh_arrays(
        pyrucast.Coords(dim=2), tags, coords, blocks
    )
    fichier = pyrucast.mesh.read_gmsh_str(pyrucast.Coords(dim=2), SQUARE_V2)
    assert shape(memoire) == shape(fichier)


def test_arrays_from_python_lists():
    """Une `list` n'exporte aucun tampon : c'est le repli par conversion, et il
    doit donner exactement le même résultat que le chemin sans copie."""
    tags, coords, blocks = square_arrays()
    groups = pyrucast.mesh.from_gmsh_arrays(
        pyrucast.Coords(dim=2), tags, coords, blocks
    )
    assert list(groups) == ["bottom", "plate"]
    assert groups["plate"].cell_counts() == [2]


def test_arrays_from_numpy_take_the_buffer_path():
    """Le chemin sans copie : des tableaux numpy contigus du bon dtype."""
    np = pytest.importorskip("numpy")
    tags, coords, blocks = square_arrays()
    groups = pyrucast.mesh.from_gmsh_arrays(
        pyrucast.Coords(dim=2),
        np.array(tags, dtype=np.uint64),
        np.array(coords, dtype=np.float64),
        [
            (code, np.array(conn, dtype=np.uint64), names)
            for code, conn, names in blocks
        ],
    )
    assert shape(groups) == shape(
        pyrucast.mesh.read_gmsh_str(pyrucast.Coords(dim=2), SQUARE_V2)
    )


def test_arrays_accept_a_non_contiguous_view():
    """Une vue à pas non unitaire n'a pas de tampon contigu à prêter : le repli
    doit la lire quand même, sans se tromper d'éléments."""
    np = pytest.importorskip("numpy")
    tags, coords, _ = square_arrays()
    # Un tableau sur deux : les tags voulus sont aux indices pairs.
    espace = np.zeros(2 * len(tags), dtype=np.uint64)
    espace[::2] = tags
    groups = pyrucast.mesh.from_gmsh_arrays(
        pyrucast.Coords(dim=2), espace[::2], coords, [(2, [1, 2, 3], ["plate"])]
    )
    assert groups["plate"].cell_counts() == [1]


def test_arrays_share_one_coords():
    """Un nœud entre deux groupes est le même des deux côtés."""
    tags, coords, blocks = square_arrays()
    c = pyrucast.Coords(dim=2)
    groups = pyrucast.mesh.from_gmsh_arrays(c, tags, coords, blocks)
    assert c.node_count() == 4
    assert len(groups) == 2


def test_arrays_ignore_unreferenced_nodes():
    tags, coords, _ = square_arrays()
    tags = tags + [99]
    coords = coords + [7.0, 7.0, 0.0]
    c = pyrucast.Coords(dim=2)
    pyrucast.mesh.from_gmsh_arrays(c, tags, coords, [(2, [1, 2, 3], ["plate"])])
    assert c.node_count() == 3


def test_arrays_unknown_node_raises():
    tags, coords, _ = square_arrays()
    with pytest.raises(RuntimeError, match="unknown node 77"):
        pyrucast.mesh.from_gmsh_arrays(
            pyrucast.Coords(dim=2), tags, coords, [(2, [1, 2, 77], ["plate"])]
        )


def test_arrays_ragged_block_raises():
    tags, coords, _ = square_arrays()
    with pytest.raises(RuntimeError, match="whole number of cells"):
        pyrucast.mesh.from_gmsh_arrays(
            pyrucast.Coords(dim=2), tags, coords, [(2, [1, 2, 3, 4], ["plate"])]
        )


def test_arrays_unsupported_type_raises():
    tags, coords, _ = square_arrays()
    with pytest.raises(RuntimeError, match="unsupported element type 21"):
        pyrucast.mesh.from_gmsh_arrays(
            pyrucast.Coords(dim=2), tags, coords, [(21, [], ["plate"])]
        )


def test_arrays_malformed_block_raises():
    tags, coords, _ = square_arrays()
    with pytest.raises(ValueError, match="triple"):
        pyrucast.mesh.from_gmsh_arrays(
            pyrucast.Coords(dim=2), tags, coords, [(2, [1, 2, 3])]
        )
