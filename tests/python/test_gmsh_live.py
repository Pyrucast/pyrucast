"""Fetching the mesh of a **live** gmsh session — `mesh.from_gmsh`.

These tests require the gmsh module: they carry the `gmsh` marker and are
hence out of the normal pass (`addopts = -m "not gmsh"`). `script/check_gmsh.sh`
is what runs them, and it fails outright if gmsh is missing, rather than
turning green on a volley of *skips*.
"""

import pytest

import pyrucast

try:
    import gmsh
except Exception as e:  # noqa: BLE001
    # Not only `ImportError`: the gmsh wheel imports as a plain Python module,
    # then loads its `libgmsh.so` through `ctypes`. When the system OpenGL
    # libraries are missing, that raises `OSError` — which `pytest.importorskip`
    # does not catch, and collection would break for everyone.

    pytest.skip(f"gmsh unavailable: {e}", allow_module_level=True)

pytestmark = pytest.mark.gmsh


@pytest.fixture
def session():
    """A silent gmsh session, closed whatever happens."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        yield gmsh
    finally:
        gmsh.finalize()


def cube(session, order=1):
    """An OCC cube: one named face, the named volume, one named corner."""
    session.model.occ.addBox(0, 0, 0, 1, 1, 1)
    session.model.occ.synchronize()
    session.model.addPhysicalGroup(2, [1], name="encastrement")
    session.model.addPhysicalGroup(3, [1], name="piece")
    session.model.addPhysicalGroup(0, [1], name="capteur")
    session.model.mesh.generate(3)
    if order > 1:
        session.model.mesh.setOrder(order)


def shape(groups):
    return {k: (m.element_types(), m.cell_counts()) for k, m in groups.items()}


def test_named_groups_become_keys(session):
    """Surfaces, volumes and points named in gmsh are the dict's keys."""
    cube(session)
    regions, _ = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3))

    assert set(regions) == {"capteur", "encastrement", "piece", "<ungrouped>"}
    assert regions["piece"].element_types() == ["TET4"]
    assert regions["encastrement"].element_types() == ["TRI3"]
    # gmsh meshes its point entities: a named point arrives as POI1, ready to
    # carry a boundary condition.
    assert regions["capteur"].element_types() == ["POI1"]
    assert regions["capteur"].cell_counts() == [1]


def test_everything_else_lands_ungrouped(session):
    """The rest of the model is not lost: it falls under `<ungrouped>`, the same
    convention as the file reader."""
    cube(session)
    regions, _ = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3))
    reste = regions["<ungrouped>"]
    # The 7 other corners, the 12 edges, the 5 other faces.
    assert reste.element_types() == ["POI1", "SEG2", "TRI3"]
    assert reste.cell_counts()[0] == 7


def test_a_model_without_physical_groups_yields_everything(session):
    session.model.occ.addRectangle(0, 0, 0, 1, 1)
    session.model.occ.synchronize()
    session.model.mesh.generate(2)
    regions, _ = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=2))
    assert list(regions) == ["<ungrouped>"]
    assert "TRI3" in regions["<ungrouped>"].element_types()


def test_one_coords_shared_by_every_group(session):
    """A node between the clamped face and the volume is *the same* on both
    sides — that is what allows the condition to be set on the named face.

    The proof is in the count: the import only lays gmsh's nodes into the
    ``Coords``, one per tag. If each group carried its own, the total would
    exceed that count, since the volume and its skin share a whole
    face.
    """
    cube(session)
    attendu = len(session.model.mesh.getNodes()[0])

    coords = pyrucast.Coords(dim=3)
    regions, _ = pyrucast.mesh.from_gmsh(coords)

    assert coords.node_count() == attendu
    # The clamped face really is a piece of the part, not a copy.
    assert regions["encastrement"].cell_counts()[0] > 0
    assert regions["piece"].cell_counts()[0] > 0


def test_dim_restricts_the_import(session):
    cube(session)
    surfaces, _ = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3), dim=2)
    assert set(surfaces) == {"encastrement", "<ungrouped>"}
    assert surfaces["encastrement"].element_types() == ["TRI3"]


def test_tag_restricts_to_one_entity(session):
    cube(session)
    une, _ = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3), dim=2, tag=1)
    assert list(une) == ["encastrement"]


def test_tag_without_dim_is_refused(session):
    cube(session)
    with pytest.raises(ValueError, match="specify dim"):
        pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3), tag=1)


def test_matches_the_file_reader(session, tmp_path):
    """The cross-check: writing the `.msh` and reading it back must give, for the
    named groups, exactly the same mesh. At order 2, so that the permutation of
    the quadratic volumes is part of the deal."""
    cube(session, order=2)
    memoire, _ = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3))

    path = tmp_path / "cube.msh"
    session.write(str(path))
    fichier = pyrucast.mesh.read_gmsh(pyrucast.Coords(dim=3), str(path))

    assert memoire["piece"].element_types() == ["TET10"]
    # `gmsh.write` writes only what carries a physical group; the memory path
    # sees the whole model. We compare what they have in common.
    communs = set(memoire) & set(fichier)
    assert communs == {"capteur", "encastrement", "piece"}
    assert {k: shape(memoire)[k] for k in communs} == {
        k: shape(fichier)[k] for k in communs
    }


def test_uninitialized_gmsh_says_so():
    """Without this guard we would return an empty dict without a word: gmsh
    writes its error on the error output, returns empty arrays and does not raise."""
    assert not gmsh.isInitialized()
    with pytest.raises(RuntimeError, match="gmsh is not initialized"):
        pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3))


def test_finalize_does_not_take_the_mesh_with_it(session):
    """gmsh's arrays are views on its memory; pyrucast has made them its own
    data, so `gmsh.finalize()` does not carry them off."""
    cube(session)
    coords = pyrucast.Coords(dim=3)
    regions, _ = pyrucast.mesh.from_gmsh(coords)
    counts = regions["piece"].cell_counts()

    gmsh.finalize()
    gmsh.initialize()  # the fixture will close this one

    assert regions["piece"].cell_counts() == counts
    assert coords.node_count() > 0


# ── Views, and the other direction: to_gmsh ────────────────────────────────


def square(session):
    """A meshed unit square, its surface named "plate"."""
    session.model.occ.addRectangle(0, 0, 0, 1, 1)
    session.model.occ.synchronize()
    session.model.addPhysicalGroup(2, [1], name="plate")
    session.model.mesh.generate(2)


def test_node_views_become_fields_and_steps_an_evolution(session):
    square(session)
    tags, xyz, _ = session.model.mesh.getNodes()
    model = session.model.getCurrent()
    view = session.view.add("T")
    for step, time in enumerate([0.0, 2.0]):
        values = [x + time for x in xyz[0::3]]
        session.view.addHomogeneousModelData(
            view, step, model, "NodeData", tags, values, time, 1
        )
    coords = pyrucast.Coords(dim=2)
    regions, fields = pyrucast.mesh.from_gmsh(coords)
    temperature = fields["T"]
    assert isinstance(temperature, pyrucast.Evolution)
    assert temperature.shared_abscissas() == [0.0, 2.0]
    late = temperature.frames()[1]
    for node in regions["plate"].cell(0, 0).nodes():
        assert late.value(node, "T") == pytest.approx(node.position()[0] + 2.0)


def test_element_views_become_one_point_fields(session):
    square(session)
    types, cells, _ = session.model.mesh.getElements(2)
    model = session.model.getCurrent()
    view = session.view.add("E")
    session.view.addHomogeneousModelData(
        view, 0, model, "ElementData", cells[0], [7.0] * len(cells[0]), 0.0, 1
    )
    regions, fields = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=2))
    zone = fields["E"][0]
    assert zone.gauss_count() == 1
    assert zone.cell_count() == regions["plate"].cell_count()
    assert zone.value(0, 0, "E") == 7.0


def test_views_can_be_left_out(session):
    square(session)
    tags, _, _ = session.model.mesh.getNodes()
    view = session.view.add("T")
    session.view.addHomogeneousModelData(
        view, 0, session.model.getCurrent(), "NodeData", tags, [0.0] * len(tags), 0.0, 1
    )
    _, fields = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=2), views=False)
    assert fields == {}


def test_to_gmsh_then_from_gmsh_gives_the_mesh_and_fields_back(session):
    square(session)
    coords = pyrucast.Coords(dim=2)
    regions, _ = pyrucast.mesh.from_gmsh(coords)
    plate = regions["plate"]
    temperature = pyrucast.NodeField(plate, ["T"])
    for z in range(len(temperature)):
        for i in range(temperature[z].node_count()):
            temperature[z].set(i, 0, float(i))
    fes = pyrucast.FiniteElementSpace(plate)
    stress = pyrucast.ElementField(fes, ["s"])
    stress[0].set_uniform("s", 3.0)
    cold = pyrucast.NodeField(plate, ["T"])
    series = pyrucast.Evolution([(0.0, cold), (1.0, temperature)])

    name = pyrucast.export.to_gmsh(
        {"plate": plate}, {"T": series, "s": stress}, model_name="back"
    )
    assert session.model.getCurrent() == name
    back, fields = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=2))
    assert back["plate"].element_types() == plate.element_types()
    assert back["plate"].cell_counts() == plate.cell_counts()
    assert isinstance(fields["T"], pyrucast.Evolution)
    assert fields["T"].shared_abscissas() == [0.0, 1.0]
    assert fields["s"][0].value(0, 0, "s") == 3.0
    # The values follow the nodes, whatever the numbering: compare by position.
    before = {
        tuple(n.position()): temperature.value(n, "T")
        for c in range(plate.cell_count())
        for n in plate.cell(0, c).nodes()
    }
    after = fields["T"].frames()[1]
    for c in range(back["plate"].cell_count()):
        for n in back["plate"].cell(0, c).nodes():
            assert after.value(n, "T") == before[tuple(n.position())]
