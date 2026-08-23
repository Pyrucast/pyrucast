"""Récupération du maillage d'une session gmsh **vivante** — `mesh.from_gmsh`.

Ces tests exigent le module gmsh : ils portent le marqueur `gmsh` et sortent
donc de la passe normale (`addopts = -m "not gmsh"`). C'est `script/check_gmsh.sh`
qui les lance, et lui échoue franchement si gmsh manque, plutôt que de virer au
vert sur une salve de *skips*.
"""

import pytest

import pyrucast

try:
    import gmsh
except Exception as e:  # noqa: BLE001
    # Pas seulement `ImportError` : la roue gmsh s'importe comme un simple
    # module Python, puis charge son `libgmsh.so` par `ctypes`. Quand les
    # bibliothèques système OpenGL manquent, cela lève `OSError` — que
    # `pytest.importorskip` ne rattrape pas, et la collecte casserait pour
    # tout le monde.
    pytest.skip(f"gmsh indisponible : {e}", allow_module_level=True)

pytestmark = pytest.mark.gmsh


@pytest.fixture
def session():
    """Une session gmsh silencieuse, refermée quoi qu'il arrive."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        yield gmsh
    finally:
        gmsh.finalize()


def cube(session, order=1):
    """Un cube OCC : une face nommée, le volume nommé, un coin nommé."""
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
    """Surfaces, volumes et points nommés dans gmsh sont les clés du dict."""
    cube(session)
    regions = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3))

    assert set(regions) == {"capteur", "encastrement", "piece", "<ungrouped>"}
    assert regions["piece"].element_types() == ["TET4"]
    assert regions["encastrement"].element_types() == ["TRI3"]
    # gmsh maille ses entités ponctuelles : un point nommé arrive en POI1,
    # prêt à porter une condition aux limites.
    assert regions["capteur"].element_types() == ["POI1"]
    assert regions["capteur"].cell_counts() == [1]


def test_everything_else_lands_ungrouped(session):
    """Le reste du modèle n'est pas perdu : il tombe sous `<ungrouped>`, la
    même convention que le lecteur de fichier."""
    cube(session)
    regions = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3))
    reste = regions["<ungrouped>"]
    # Les 7 autres coins, les 12 arêtes, les 5 autres faces.
    assert reste.element_types() == ["POI1", "SEG2", "TRI3"]
    assert reste.cell_counts()[0] == 7


def test_a_model_without_physical_groups_yields_everything(session):
    session.model.occ.addRectangle(0, 0, 0, 1, 1)
    session.model.occ.synchronize()
    session.model.mesh.generate(2)
    regions = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=2))
    assert list(regions) == ["<ungrouped>"]
    assert "TRI3" in regions["<ungrouped>"].element_types()


def test_one_coords_shared_by_every_group(session):
    """Un nœud entre la face encastrée et le volume est *le même* des deux
    côtés — c'est ce qui permet de poser la condition sur la face nommée.

    La preuve tient au compte : l'import ne pose dans le ``Coords`` que les
    nœuds de gmsh, un par tag. Si chaque groupe portait les siens, le total
    dépasserait ce compte, puisque le volume et sa peau partagent une face
    entière.
    """
    cube(session)
    attendu = len(session.model.mesh.getNodes()[0])

    coords = pyrucast.Coords(dim=3)
    regions = pyrucast.mesh.from_gmsh(coords)

    assert coords.node_count() == attendu
    # La face encastrée est bien un morceau de la pièce, pas une copie.
    assert regions["encastrement"].cell_counts()[0] > 0
    assert regions["piece"].cell_counts()[0] > 0


def test_dim_restricts_the_import(session):
    cube(session)
    surfaces = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3), dim=2)
    assert set(surfaces) == {"encastrement", "<ungrouped>"}
    assert surfaces["encastrement"].element_types() == ["TRI3"]


def test_tag_restricts_to_one_entity(session):
    cube(session)
    une = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3), dim=2, tag=1)
    assert list(une) == ["encastrement"]


def test_tag_without_dim_is_refused(session):
    cube(session)
    with pytest.raises(ValueError, match="précisez dim"):
        pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3), tag=1)


def test_matches_the_file_reader(session, tmp_path):
    """Le contrôle croisé : écrire le `.msh` et le relire doit donner, pour
    les groupes nommés, exactement le même maillage. En ordre 2, pour que la
    permutation des volumes quadratiques soit dans le lot."""
    cube(session, order=2)
    memoire = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3))

    path = tmp_path / "cube.msh"
    session.write(str(path))
    fichier = pyrucast.mesh.read_gmsh(pyrucast.Coords(dim=3), str(path))

    assert memoire["piece"].element_types() == ["TET10"]
    # `gmsh.write` n'écrit que ce qui porte un groupe physique ; la voie
    # mémoire, elle, voit tout le modèle. On compare ce qu'ils ont en commun.
    communs = set(memoire) & set(fichier)
    assert communs == {"capteur", "encastrement", "piece"}
    assert {k: shape(memoire)[k] for k in communs} == {
        k: shape(fichier)[k] for k in communs
    }


def test_uninitialized_gmsh_says_so():
    """Sans cette garde on rendrait un dict vide sans rien dire : gmsh écrit
    son erreur sur la sortie d'erreur, rend des tableaux vides et ne lève pas."""
    assert not gmsh.isInitialized()
    with pytest.raises(RuntimeError, match="gmsh n'est pas initialisé"):
        pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3))


def test_finalize_does_not_take_the_mesh_with_it(session):
    """Les tableaux de gmsh sont des vues sur sa mémoire ; pyrucast en a fait
    ses propres données, donc `gmsh.finalize()` ne les emporte pas."""
    cube(session)
    coords = pyrucast.Coords(dim=3)
    regions = pyrucast.mesh.from_gmsh(coords)
    counts = regions["piece"].cell_counts()

    gmsh.finalize()
    gmsh.initialize()  # la fixture refermera celle-ci

    assert regions["piece"].cell_counts() == counts
    assert coords.node_count() > 0
