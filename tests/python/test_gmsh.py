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
    groups = pyrucast.read_gmsh_str(src)
    assert isinstance(groups, dict)
    assert set(groups) == {"bottom", "plate"}

    plate = groups["plate"]
    assert plate.element_types() == ["TRI3"]
    assert plate.cell_count() == 2

    bottom = groups["bottom"]
    assert bottom.element_types() == ["SEG2"]
    assert bottom.cell_count() == 1


@pytest.mark.parametrize("src", [SQUARE_V2, SQUARE_V4], ids=["v2.2", "v4.1"])
def test_groups_share_one_coords(src):
    groups = pyrucast.read_gmsh_str(src)
    # The plate's bottom edge nodes are the very same nodes as in "bottom":
    # a node shared between groups is shared, not duplicated.
    plate_coords = groups["plate"].coords()
    bottom_coords = groups["bottom"].coords()
    assert plate_coords.dim == 2  # planar mesh → 2-D Coords
    assert plate_coords.node_count() == bottom_coords.node_count() == 4


def test_planar_is_2d_but_dim_override_keeps_3d():
    g2 = pyrucast.read_gmsh_str(SQUARE_V2)
    assert g2["plate"].coords().dim == 2
    g3 = pyrucast.read_gmsh_str(SQUARE_V2, dim=3)
    assert g3["plate"].coords().dim == 3


def test_read_from_file(tmp_path):
    path = tmp_path / "square.msh"
    path.write_text(SQUARE_V2)
    groups = pyrucast.read_gmsh(str(path))
    assert set(groups) == {"bottom", "plate"}


def test_unsupported_element_type_raises():
    # gmsh type 8 = 3-node second-order line.
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
        1 8 2 0 1 1 2 3
        $EndElements
        """
    )
    with pytest.raises(Exception, match="unsupported element type"):
        pyrucast.read_gmsh_str(bad)
