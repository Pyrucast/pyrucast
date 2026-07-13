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
    groups = pyrucast.mesher.read_gmsh_str(coords, src)
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
    pyrucast.mesher.read_gmsh_str(coords, src)
    assert coords.node_count() == 4


@pytest.mark.parametrize("src", [SQUARE_V2, SQUARE_V4], ids=["v2.2", "v4.1"])
def test_mesh_coords_recovers_the_handle(src):
    # Even if the original Coords handle is dropped, mesh.coords() gets it
    # back — and every group reports the very same Coords.
    groups = pyrucast.mesher.read_gmsh_str(pyrucast.Coords(dim=2), src)
    recovered = groups["plate"].coords()
    assert recovered.node_count() == 4
    assert groups["bottom"].coords().node_count() == 4


def test_coords_dimension_decides_kept_coordinates():
    # z = 0 here, so 2-D vs 3-D is observable on the node coordinate length.
    c2 = pyrucast.Coords(dim=2)
    g2 = pyrucast.mesher.read_gmsh_str(c2, SQUARE_V2)
    assert len(g2["plate"].node(0, 0, 0).coord()) == 2
    c3 = pyrucast.Coords(dim=3)
    g3 = pyrucast.mesher.read_gmsh_str(c3, SQUARE_V2)
    assert len(g3["plate"].node(0, 0, 0).coord()) == 3


def test_read_from_file(tmp_path):
    path = tmp_path / "square.msh"
    path.write_text(SQUARE_V2)
    coords = pyrucast.Coords(dim=2)
    groups = pyrucast.mesher.read_gmsh(coords, str(path))
    assert set(groups) == {"bottom", "plate"}


def test_unsupported_element_type_raises():
    # gmsh type 7 = 5-node pyramid (PYR5), unsupported by pyrucast.
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
        1 7 2 0 1 1 2 3
        $EndElements
        """
    )
    with pytest.raises(Exception, match="unsupported element type"):
        pyrucast.mesher.read_gmsh_str(pyrucast.Coords(dim=2), bad)
