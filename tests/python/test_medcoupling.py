"""Exchange with **medcoupling** — `mesh.from_medcoupling`,
`export.to_medcoupling`.

These tests require the medcoupling module: they carry the `medcoupling`
marker and are hence out of the normal pass. `script/check_medcoupling.sh` is
what runs them, and it fails outright if medcoupling is missing.

medcoupling is both the tool under test and the reference, so the node
numbering is checked against what medcoupling *computes* — signed measures,
edge and face midpoints — never against a table copied from somewhere.
"""

import pytest

import pyrucast

mc = pytest.importorskip("medcoupling")
np = pytest.importorskip("numpy")

pytestmark = pytest.mark.medcoupling

# pyrucast's reference cell of every type (pyrucast = VTK node order), and
# its measure.
H27 = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    [0, -1, -1], [1, 0, -1], [0, 1, -1], [-1, 0, -1],
    [0, -1, 1], [1, 0, 1], [0, 1, 1], [-1, 0, 1],
    [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
    [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
    [0, 0, 0],
]  # fmt: skip
Q9 = [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]]
P15 = [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1],
    [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0], [0.5, 0, 1], [0.5, 0.5, 1],
    [0, 0.5, 1], [0, 0, 0.5], [1, 0, 0.5], [0, 1, 0.5],
]  # fmt: skip
T10 = [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0, 0], [0.5, 0.5, 0],
    [0, 0.5, 0], [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5],
]  # fmt: skip
TRI6 = [[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]]
PYRA5 = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]]
REFERENCE = {
    "SEG2": ([[-1], [1]], 2.0),
    "SEG3": ([[-1], [1], [0]], 2.0),
    "TRI3": (TRI6[:3], 0.5),
    "TRI6": (TRI6, 0.5),
    "QUA4": (Q9[:4], 4.0),
    "QUA8": (Q9[:8], 4.0),
    "QUA9": (Q9, 4.0),
    "TET4": (T10[:4], 1 / 6),
    "TET10": (T10, 1 / 6),
    "PYRA5": (PYRA5, 4 / 3),
    "PENTA6": (P15[:6], 0.5),
    "PENTA15": (P15, 0.5),
    "HEX8": (H27[:8], 8.0),
    "HEX20": (H27[:20], 8.0),
    "HEX27": (H27, 8.0),
}


def one_cell(element_type):
    """pyrucast's reference cell of `element_type`, in its own Coords."""
    nodes, _ = REFERENCE[element_type]
    dim = len(nodes[0])
    coords = pyrucast.Coords(dim)
    tags = list(range(1, len(nodes) + 1))
    xyz = [float(x) for p in nodes for x in p]
    meshes, _, _ = pyrucast.mesh.from_arrays(
        coords, tags, xyz, [(element_type, tags, ["cell"])]
    )
    return meshes


def midpoint_errors(umesh):
    """The largest distance between a quadratic sub-entity's middle node and
    the middle of its corners, over the edges (and quadrangle faces) medcoupling
    derives from the cell — zero exactly when the middle nodes are where MED's
    numbering says they are."""
    coo = umesh.getCoords().toNumPyArray().reshape(umesh.getNumberOfNodes(), -1)
    worst = 0.0
    level = umesh
    while level.getMeshDimension() > 1:
        level = level.buildDescendingConnectivity()[0]
        for c in range(level.getNumberOfCells()):
            ids = list(level.getNodeIdsOfCell(c))
            if len(ids) == 9:  # QUAD9 face: its centre
                worst = max(worst, np.abs(coo[ids[8]] - coo[ids[:4]].mean(0)).max())
    for c in range(level.getNumberOfCells()):
        ids = list(level.getNodeIdsOfCell(c))
        if len(ids) == 3:  # SEG3 edge: its middle
            worst = max(worst, np.abs(coo[ids[2]] - coo[ids[:2]].mean(0)).max())
    return worst


@pytest.mark.parametrize("element_type", list(REFERENCE))
def test_med_numbering_gives_a_well_formed_cell(element_type):
    """Written in MED order, pyrucast's reference cell has the right measure
    — **positive** for a volume, i.e. oriented as MED wants — and its middle
    nodes sit in the middle of the edges medcoupling derives."""
    meshes = one_cell(element_type)
    data = pyrucast.export.to_medcoupling(meshes)
    umesh = data.getMeshes()[0].getMeshAtLevel(0)
    measure = umesh.getMeasureField(False).getArray().getValues()[0]
    _, expected = REFERENCE[element_type]
    if umesh.getMeshDimension() == 3:
        assert measure == pytest.approx(expected)
    else:
        assert abs(measure) == pytest.approx(expected)
    assert midpoint_errors(umesh) < 1e-12


def test_two_dimensional_cells_turn_like_meds_own():
    """medcoupling signs a 2-D area by its own convention: pyrucast's
    counter-clockwise triangle must carry the same sign as MED's reference
    triangle does."""
    data = pyrucast.export.to_medcoupling(one_cell("TRI3"))
    ours = data.getMeshes()[0].getMeshAtLevel(0).getMeasureField(False)
    ref = mc.MEDCouplingGaussLocalization.GetDefaultReferenceCoordinatesOf(mc.NORM_TRI3)
    m = mc.MEDCouplingUMesh("m", 2)
    m.allocateCells(1)
    m.insertNextCell(mc.NORM_TRI3, [0, 1, 2])
    m.setCoords(ref)
    m.finishInsertingCells()
    theirs = m.getMeasureField(False)
    assert np.sign(ours.getArray().getValues()[0]) == np.sign(
        theirs.getArray().getValues()[0]
    )


# medcoupling's default reference elements for these types do not follow its
# own MED connectivity (TRI6 repeats a corner, PENTA15 and HEXA20 order their
# middle nodes otherwise), and its Gauss-point locator accepts nothing else:
# over an element consistent with the connectivity it raises. Should a later
# medcoupling locate them, this test fails and the set is to be emptied.
MEDCOUPLING_CANNOT_LOCATE = {"TRI6", "PENTA15", "HEX20"}


@pytest.mark.parametrize("element_type", list(REFERENCE))
def test_gauss_rules_are_declared_in_meds_reference_element(element_type, tmp_path):
    """On pyrucast's reference cell, physical and reference coordinates
    coincide: the Gauss points medcoupling **locates** from the rule written
    in the file must be pyrucast's own points — whatever reference element
    the rule is declared in. Then the values come back, point for point."""
    meshes = one_cell(element_type)
    quadratic = element_type in (
        "SEG3",
        "TRI6",
        "QUA8",
        "QUA9",
        "TET10",
        "PENTA15",
        "HEX20",
        "HEX27",
    )
    fes = pyrucast.FiniteElementSpace(
        meshes["cell"], "lagrange2" if quadratic else "lagrange1"
    )
    sub = fes[0]
    field = pyrucast.ElementField(fes, ["s"])
    for g in range(sub.gauss_count()):
        field[0].set_value(0, g, "s", float(g + 1))
    path = tmp_path / "g.med"
    pyrucast.export.to_medcoupling(meshes, {"s": field}).write(str(path), 2)

    data = mc.MEDFileData(str(path))
    mesh = data.getMeshes()[0]
    one = data.getFields()[0][0, -1]
    located = one.getFieldOnMeshAtLevel(mc.ON_GAUSS_PT, 0, mesh)
    ours = np.array([sub.gauss_xi(g) for g in range(sub.gauss_count())])
    if element_type in MEDCOUPLING_CANNOT_LOCATE:
        with pytest.raises(mc.InterpKernelException):
            located.getLocalizationOfDiscr()
    else:
        where = located.getLocalizationOfDiscr().toNumPyArray()
        assert np.abs(where.reshape(ours.shape) - ours).max() < 1e-12

    _, fields = pyrucast.mesh.from_medcoupling(
        pyrucast.Coords(ours.shape[1]), str(path)
    )
    back = fields["s"][0]
    assert [back.value(0, g, "s") for g in range(back.gauss_count())] == [
        float(g + 1) for g in range(sub.gauss_count())
    ]


# ── Round trips ─────────────────────────────────────────────────────────────


def solid():
    """A HEX8 and a TET4 in "solid" (the TET4 also in "right"), a QUA4 face
    in "bottom", two pinned nodes in "pins"."""
    coords = pyrucast.Coords(3)
    xyz = [
        0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
        0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
        2, 0, 0, 2, 1, 0, 2, 0, 1,
    ]  # fmt: skip
    blocks = [
        ("HEX8", [1, 2, 3, 4, 5, 6, 7, 8], [1], ["solid"]),
        ("TET4", [2, 9, 10, 11], [2], ["solid", "right"]),
        ("QUA4", [1, 2, 3, 4], [3], ["bottom"]),
        ("POI1", [1, 5], [], ["pins"]),
    ]
    meshes, _, _ = pyrucast.mesh.from_arrays(
        coords, list(range(1, 12)), [float(x) for x in xyz], blocks
    )
    return meshes


def by_position(mesh, value):
    """`{node position: value(node)}` over the nodes of `mesh`."""
    out = {}
    for s in range(len(mesh)):
        for c in range(mesh[s].cell_count()):
            for n in mesh.cell(s, c).nodes():
                out[tuple(n.position())] = value(n)
    return out


def shape(meshes):
    return {
        k: sorted(zip(m.element_types(), m.cell_counts())) for k, m in meshes.items()
    }


def test_mesh_and_groups_round_trip(tmp_path):
    meshes = solid()
    path = tmp_path / "solid.med"
    pyrucast.export.to_medcoupling(meshes, mesh_name="piece").write(str(path), 2)
    back, fields = pyrucast.mesh.from_medcoupling(pyrucast.Coords(3), str(path))
    assert shape(back) == shape(meshes)
    assert fields == {}
    # A cell of two groups is one MED cell: the level holds two volumes.
    mm = mc.MEDFileData(str(path)).getMeshes()[0]
    assert mm.getName() == "piece"
    assert mm.getMeshAtLevel(0).getNumberOfCells() == 2
    assert sorted(mm.getGroupsOnSpecifiedLev(1)) == ["pins"]


def test_node_fields_and_evolutions_round_trip(tmp_path):
    meshes = solid()
    body = meshes["solid"]
    temperature = pyrucast.NodeField(body, ["T", "p"])
    for z in range(len(temperature)):
        for i in range(temperature[z].node_count()):
            temperature[z].set(i, 0, 1.5 * i)
            temperature[z].set(i, 1, -float(i))
    cold = pyrucast.NodeField(body, ["T", "p"])
    series = pyrucast.Evolution([(0.0, cold), (4.0, temperature)])
    path = tmp_path / "t.med"
    pyrucast.export.to_medcoupling(meshes, {"T": series}).write(str(path), 2)

    back, fields = pyrucast.mesh.from_medcoupling(pyrucast.Coords(3), str(path))
    series_back = fields["T"]
    assert series_back.shared_abscissas() == [0.0, 4.0]
    late = series_back.frames()[1]
    expected = by_position(body, lambda n: temperature.value(n, "T"))
    got = by_position(back["solid"], lambda n: late.value(n, "T"))
    assert got == expected


@pytest.mark.parametrize("gauss", [True, False])
def test_element_fields_round_trip(tmp_path, gauss):
    meshes = solid()
    body = meshes["solid"]
    fes = pyrucast.FiniteElementSpace(body)
    stress = pyrucast.ElementField(fes, ["sxx", "syy"])
    for z in range(len(stress)):
        sub = stress[z]
        for c in range(sub.cell_count()):
            for g in range(sub.gauss_count()):
                sub.set_value(c, g, "sxx", 10.0 * z + g)
                sub.set_value(c, g, "syy", -1.0 * g)
    path = tmp_path / "s.med"
    pyrucast.export.to_medcoupling(meshes, {"S": stress}, gauss=gauss).write(
        str(path), 2
    )
    back, fields = pyrucast.mesh.from_medcoupling(pyrucast.Coords(3), str(path))

    def rows(field):
        """Each zone's values of its single cell, per point — or their mean."""
        out = set()
        for z in range(len(field)):
            zone = field[z]
            values = tuple(zone.value(0, g, "sxx") for g in range(zone.gauss_count()))
            out.add(values if gauss else (round(sum(values) / len(values), 12),))
        return out

    # The TET4 is in "solid" and "right": its values land in both groups.
    assert rows(fields["S"]) == rows(stress)


def test_a_file_written_by_medcoupling(tmp_path):
    """A mesh built with medcoupling's own API: families mixing groups, a node
    group, a cell field defined on part of the cells (a profile)."""
    coo = mc.DataArrayDouble([0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1], 6, 2)
    m2 = mc.MEDCouplingUMesh("m", 2)
    m2.allocateCells()
    m2.insertNextCell(mc.NORM_TRI3, [0, 1, 4])
    m2.insertNextCell(mc.NORM_TRI3, [0, 4, 3])
    m2.insertNextCell(mc.NORM_QUAD4, [1, 2, 5, 4])
    m2.finishInsertingCells()
    m2.setCoords(coo)
    mm = mc.MEDFileUMesh()
    mm.setName("m")
    mm.setMeshAtLevel(0, m2)
    tris = mc.DataArrayInt64([0, 1])
    tris.setName("tris")
    mid = mc.DataArrayInt64([1, 2])
    mid.setName("mid")
    mm.setGroupsAtLevel(0, [tris, mid])
    left = mc.DataArrayInt64([0, 3])
    left.setName("left")
    mm.setGroupsAtLevel(1, [left])
    # A cell field on the triangles only.
    f = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
    f.setName("E")
    part = m2[mc.DataArrayInt64([0, 1])]
    part.setName("m")
    f.setMesh(part)
    f.setArray(mc.DataArrayDouble([5.0, 6.0]))
    f.setTime(0.0, 0, -1)
    one = mc.MEDFileField1TS()
    pfl = mc.DataArrayInt64([0, 1])
    pfl.setName("tri_cells")
    one.setFieldProfile(f, mm, 0, pfl)
    multi = mc.MEDFileFieldMultiTS()
    multi.pushBackTimeStep(one)
    fs = mc.MEDFileFields()
    fs.pushField(multi)
    data = mc.MEDFileData()
    meshes = mc.MEDFileMeshes()
    meshes.pushMesh(mm)
    data.setMeshes(meshes)
    data.setFields(fs)
    path = tmp_path / "direct.med"
    data.write(str(path), 2)

    back, fields = pyrucast.mesh.from_medcoupling(pyrucast.Coords(2), str(path))
    assert shape(back) == {
        "tris": [("TRI3", 2)],
        "mid": [("QUA4", 1), ("TRI3", 1)],
        "left": [("POI1", 2)],
    }
    # A zone per submesh the profile covers entirely: both triangles of
    # "tris", the triangle of "mid" — not the quadrangle of "mid".
    field = fields["E"]
    counts = sorted(field[z].cell_count() for z in range(len(field)))
    assert counts == [1, 2]
    values = sorted(
        field[z].value(c, 0, "E")
        for z in range(len(field))
        for c in range(field[z].cell_count())
    )
    assert values == [5.0, 6.0, 6.0]


def test_import_pyrucast_loads_neither_gmsh_nor_medcoupling():
    import subprocess
    import sys

    code = (
        "import sys, pyrucast; "
        "assert 'medcoupling' not in sys.modules; "
        "assert 'gmsh' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
