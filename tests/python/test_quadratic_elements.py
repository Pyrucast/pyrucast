"""Python tests for the quadratic (Lagrange-2) element types:
SEG3, TRI6, QUA8, TET10, PENTA15, HEX20."""

import pytest

import pyrucast


def _tri6(coords):
    """A single TRI6 cell on `coords`: corners then edge midpoints."""
    pts = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
    ]
    ids = [coords.add_node(p) for p in pts]
    m = pyrucast.Mesh(coords, "TRI6")
    m.unit().add_cell(ids)
    return m


def test_mesh_holds_quadratic_cells():
    c = pyrucast.Coords(2)
    m = _tri6(c)
    assert m.element_types() == ["TRI6"]
    assert m.cell_counts() == [1]


def test_lagrange2_fe_space_on_tri6():
    c = pyrucast.Coords(2)
    m = _tri6(c)
    fes = pyrucast.FiniteElementSpace(m, interpolation="LAGRANGE2")
    sub = fes[0]
    assert sub.element_type == "TRI6"
    assert sub.interpolation == "LAGRANGE2"
    assert sub.gauss_count() == 6  # degree-4 triangle rule


def test_degree_must_match_element_type():
    c = pyrucast.Coords(2)
    m = _tri6(c)
    # LAGRANGE1 (linear) on a quadratic mesh is rejected.
    with pytest.raises(Exception):
        pyrucast.FiniteElementSpace(m, interpolation="LAGRANGE1")


def test_export_vtk_quadratic(tmp_path):
    c = pyrucast.Coords(2)
    m = _tri6(c)
    out = tmp_path / "tri6.vtk"
    pyrucast.export_vtk(m, str(out))
    text = out.read_text()
    # VTK_QUADRATIC_TRIANGLE = 22 appears in the CELL_TYPES block.
    assert "\n22\n" in text or text.rstrip().endswith("22")


def test_read_gmsh_quadratic_tet10():
    mesh = """\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
10
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
5 0.5 0 0
6 0.5 0.5 0
7 0 0.5 0
8 0 0 0.5
9 0 0.5 0.5
10 0.5 0 0.5
$EndNodes
$Elements
1
1 11 2 0 1 1 2 3 4 5 6 7 8 9 10
$EndElements
"""
    groups = pyrucast.read_gmsh_str(pyrucast.Coords(3), mesh)
    (_, m) = groups[0] if isinstance(groups, list) else list(groups.items())[0]
    assert m.element_types() == ["TET10"]
    # After the gmsh->pyrucast permutation, local node 8 is the (1,3) midpoint.
    assert m.node(0, 0, 8).coord() == [0.5, 0.0, 0.5]
    assert m.node(0, 0, 9).coord() == [0.0, 0.5, 0.5]
