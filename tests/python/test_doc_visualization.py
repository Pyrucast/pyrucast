"""Source des exemples Python de `book/src/visualization.md`.

Les extraits écrivent des fichiers sous des noms courts (`piece.svg`, …) : les
tests basculent le répertoire courant vers un dossier temporaire, si bien que
l'extrait affiché reste exactement celui qu'un utilisateur écrirait.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import pytest

import pyrucast


@pytest.fixture(autouse=True)
def _dans_un_dossier_temporaire(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _triangle():
    coords = pyrucast.Coords(3)
    n = [
        coords.add_node(p) for p in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    ]
    mesh = pyrucast.Mesh(coords, "TRI3")
    mesh.unit().add_cell(n)
    return coords, mesh, n


def _champ_nodal(noeuds, composantes, valeur=1.0, support=None):
    """Deux appels sans `support` fabriquent deux nuages **distincts** : une
    `Evolution` de champs exige au contraire le même support à chaque pas."""
    support = support or pyrucast.mesh.poi1_from_nodes(noeuds)
    f = pyrucast.NodeField(support, composantes)
    for i, noeud in enumerate(noeuds):
        for composante in composantes:
            f[0].set_value(noeud, composante, valeur * (i + 1))
    return f


# ── Formats de sortie ───────────────────────────────────────────────────────


def test_formats():
    _, mesh, _ = _triangle()
    # ANCHOR: formats
    mesh.plot(save="piece.svg")  # à versionner, à publier
    mesh.plot(save="piece.svgz")  # à empiler par centaines
    # ANCHOR_END: formats


def test_vue_et_export():
    # ANCHOR: vue
    import pyrucast

    coords = pyrucast.Coords(3)
    a = coords.add_node([0.0, 0.0, 0.0])
    b = coords.add_node([1.0, 0.0, 0.0])
    c = coords.add_node([0.0, 1.0, 0.0])

    mesh = pyrucast.Mesh(coords, "TRI3")
    mesh.unit().add_cell([a, b, c])

    # (yaw, pitch, scale) ; save=None ouvre la fenêtre interactive.
    mesh.plot(view=(45.0, 35.264, 1.0), save="triangle.svg")
    # ANCHOR_END: vue


def test_titre():
    _, mesh, n = _triangle()
    t_field = _champ_nodal(n, ["T"], 20.0)
    # ANCHOR: titre
    mesh.plot(
        save="piece.svg", title="poutre encastrée"
    )  # légende centrée en bas du SVG
    mesh.plot(save="t.svg", field=t_field, title="température")  # combinable avec field
    # mesh.plot(title="ma pièce")  # nomme la fenêtre interactive (bloquant)
    # ANCHOR_END: titre


def test_couleur_de_face():
    coords, _, _ = _triangle()
    # ANCHOR: couleur
    sm = pyrucast.Mesh(coords, "TRI3")[0]  # vue du sous-maillage unique
    sm.face_color = (220, 60, 60)
    assert sm.face_color == (220, 60, 60)
    # ANCHOR_END: couleur


# ── Champs et échelles ──────────────────────────────────────────────────────


def test_champ_et_echelle():
    _, mesh, n = _triangle()
    t_field = _champ_nodal(n, ["T"], 20.0)
    u_field = _champ_nodal(n, ["UX", "UY"], 0.5)
    fes = pyrucast.FiniteElementSpace(mesh)
    flux_field = pyrucast.ElementField(fes, ["q"])
    flux_field[0].set_uniform("q", 3.0)
    # ANCHOR: champ
    # Composante par défaut, viridis, échelle auto.
    mesh.plot(save="t.svg", field=t_field)

    # Composante "UY", colormap "coolwarm", bornes fixées.
    mesh.plot(
        save="uy.svg",
        field=u_field,
        component="UY",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )

    # Plafond seul fixé : le plancher suit le minimum des données.
    mesh.plot(save="t.svg", field=t_field, vmax=100.0)

    # Champ aux points de Gauss : strictement le même appel.
    mesh.plot(save="flux.svg", field=flux_field)
    # ANCHOR_END: champ


# ── Style ───────────────────────────────────────────────────────────────────


def test_wireframe():
    _, mesh, n = _triangle()
    t_field = _champ_nodal(n, ["T"], 20.0)
    # ANCHOR: wireframe
    mesh.plot(save="solide.svg")  # peau opaque (défaut)
    mesh.plot(save="fil.svg", wireframe=True)  # fil de fer

    # Sans objet avec un champ : lève ValueError.
    # mesh.plot(save="x.svg", field=t_field, wireframe=True)
    # ANCHOR_END: wireframe


# ── Export VTK ──────────────────────────────────────────────────────────────


def _champ_par_elements(mesh, composante, valeur):
    """Un `ElementField` uniforme, monté sans nommer `pyrucast` chez l'appelant.

    Un `import pyrucast` dans une ancre en fait une variable locale : le module
    devient inutilisable avant l'ancre, dans la même fonction.
    """
    f = pyrucast.ElementField(pyrucast.FiniteElementSpace(mesh), [composante])
    f[0].set_uniform(composante, valeur)
    return f


def test_export_vtk():
    _, mesh, n = _triangle()
    temperature = _champ_nodal(n, ["T"], 20.0)
    stresses = _champ_par_elements(mesh, "sigma_xx", 1.0)
    # ANCHOR: vtk
    import pyrucast

    # Géométrie seule.
    pyrucast.export.export_vtk(mesh, "maillage.vtk")

    # Géométrie + champ aux nœuds (POINT_DATA).
    pyrucast.export.export_vtk(mesh, "solution.vtk", field=temperature)

    # Géométrie + champ aux points de Gauss (CELL_DATA) : une valeur par
    # cellule = moyenne intra-élément des points de Gauss de la cellule.
    pyrucast.export.export_vtk(mesh, "contraintes.vtk", field=stresses)
    # ANCHOR_END: vtk


# ── Évolutions ──────────────────────────────────────────────────────────────


def test_evolution_plot():
    _, mesh, n = _triangle()
    maillage = mesh
    support = pyrucast.mesh.poi1_from_nodes(n)
    champ_t0, champ_t1, champ_t2 = (
        _champ_nodal(n, ["T"], v, support) for v in (10.0, 20.0, 5.0)
    )
    # ANCHOR: evolution
    import pyrucast as pc

    # Courbe scalaire (variable → valeur).
    e = pc.Evolution([(0.0, 10.0), (1.0, 20.0), (2.0, 5.0)])
    e.plot(save="courbe.svg", x_label="temps", y_label="T", title="évolution de T")

    # Évolution d'un champ aux nœuds : un NodeField complet par pas de temps.
    ev = pc.Evolution([(0.0, champ_t0), (1.0, champ_t1), (2.0, champ_t2)])
    ev.plot(save="frame.png", frame=2)  # une valeur tabulée (défaut : la dernière)
    ev.plot(
        save="frame_surf.png", mesh=maillage
    )  # rendu surfacique sur un maillage fourni
    # ANCHOR_END: evolution


# ── Axisymétrie : le corps de révolution ────────────────────────────────────


def test_revolve():
    # ANCHOR: revolve
    import pyrucast

    coords = pyrucast.Coords.axisymmetric()  # (r, z), r ≥ 0
    section = [coords.add_node(p) for p in ([1.0, 0.0], [2.0, 0.0], [1.0, 1.0])]
    mesh = pyrucast.Mesh(coords, "TRI3")
    mesh.unit().add_cell(section)
    # … calcul, puis un champ de température aux nœuds de la section :
    t_field = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes(section), ["T"])

    mesh.plot(save="section.svg")  # la section plane (défaut)
    mesh.plot(save="piece.svg", revolve=True)  # le corps de révolution complet
    mesh.plot(save="coupe.svg", revolve=True, revolve_angle=270.0)  # ouvert à 270°
    mesh.plot(save="t3d.svg", field=t_field, revolve=True)  # champ sur le corps
    # ANCHOR_END: revolve
