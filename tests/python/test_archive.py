"""Sauvegarde et relecture d'un graphe d'objets, côté Python.

L'exigence que ces tests gardent : les objets relus gardent leur cohérence
**sans duplication** — deux champs sur un support restent deux champs sur un
support.
"""

import pytest

import pyrucast


def line(n=3):
    """Une ligne de `n` SEG2, ses coordonnées et son maillage."""
    c = pyrucast.Coords(2)
    nodes = [c.add_node([float(i), 0.0]) for i in range(n + 1)]
    m = pyrucast.Mesh(c, "SEG2")
    for i in range(n):
        m.unit().add_cell([nodes[i], nodes[i + 1]])
    return c, m, nodes


def cloud(c, nodes):
    """Le nuage POI1 de `nodes` — le support sur lequel vit un champ nodal."""
    poi = pyrucast.Mesh(c, "POI1")
    for n in nodes:
        poi.unit().add_cell([n])
    return poi


# ─── L'exigence ─────────────────────────────────────────────────────────────


def test_le_partage_survit(tmp_path):
    """Deux champs sur un support restent deux champs sur UN support.

    L'observable est l'union : elle fusionne les zones qui partagent un
    support, et les laisse côte à côte sinon. Une zone après union ⇒ le
    support est bien un seul objet, pas deux copies aux mêmes nœuds.
    """
    c, m, nodes = line()
    poi = cloud(c, nodes)
    t = pyrucast.NodeField(poi, ["T"])
    f = pyrucast.NodeField(poi, ["f"])
    assert len(t | f) == 1, "le gabarit lui-même doit partager le support"

    chemin = str(tmp_path / "etude.pyr")
    pyrucast.save(chemin, {"temperature": t, "force": f, "maillage": m, "coords": c})

    objets = pyrucast.load(chemin)
    fusion = objets["temperature"] | objets["force"]
    assert len(fusion) == 1, "relus, les deux champs doivent partager UN support"
    assert sorted(fusion[0].components()) == ["T", "f"]

    # Une seule Coords pour tout le fichier : ajouter un nœud par le maillage
    # se voit par la racine `coords`.
    avant = objets["coords"].node_count()
    objets["maillage"].coords().add_node([9.0, 9.0])
    assert objets["coords"].node_count() == avant + 1


def test_les_clefs_sont_libres(tmp_path):
    """Une clef peut porter espace, accent, unité — ce n'est pas un identifiant."""
    _, m, _ = line()
    chemin = str(tmp_path / "clefs.pyr")
    pyrucast.save(chemin, {"maillage très fin": m, "T (°C)": 20.0})

    objets = pyrucast.load(chemin)
    assert set(objets) == {"maillage très fin", "T (°C)"}
    assert objets["T (°C)"] == 20.0


def test_relire_ajoute_a_cote(tmp_path):
    """Relire ne remplace rien : les objets déjà vivants sont intacts."""
    c, m, _ = line()
    chemin = str(tmp_path / "a_cote.pyr")
    pyrucast.save(chemin, {"maillage": m})

    avant = m[0].cell_count()
    relu = pyrucast.load(chemin)["maillage"]
    assert relu[0].cell_count() == avant

    # Les deux Coords sont distinctes : toucher l'une ne touche pas l'autre.
    n_avant = c.node_count()
    relu.coords().add_node([42.0, 42.0])
    assert c.node_count() == n_avant, "la session d'origine n'a pas bougé"


# ─── Les valeurs simples ────────────────────────────────────────────────────


def test_valeurs_simples(tmp_path):
    chemin = str(tmp_path / "valeurs.pyr")
    pyrucast.save(
        chemin,
        {
            "actif": True,
            "pas": 12,
            "dt": 0.05,
            "cas": "charge répartie",
            "instants": [0.0, 0.1, 0.2],
            "noms": ["a", "b"],
            "drapeaux": [True, False],
            "indices": [1, 2, 3],
        },
    )
    o = pyrucast.load(chemin)
    assert o["actif"] is True
    assert o["pas"] == 12
    assert o["dt"] == 0.05
    assert o["cas"] == "charge répartie"
    assert o["instants"] == [0.0, 0.1, 0.2]
    assert o["noms"] == ["a", "b"]
    assert o["drapeaux"] == [True, False]
    assert o["indices"] == [1, 2, 3]


def test_une_liste_imbriquee_est_refusee(tmp_path):
    chemin = str(tmp_path / "imbrique.pyr")
    with pytest.raises(TypeError, match="Nested lists"):
        pyrucast.save(chemin, {"x": [[1.0, 2.0], [3.0]]})


def test_une_liste_heterogene_est_refusee(tmp_path):
    chemin = str(tmp_path / "melange.pyr")
    with pytest.raises(TypeError, match="homogeneous"):
        pyrucast.save(chemin, {"x": [1.0, "deux"]})


def test_un_entier_trop_grand_est_refuse(tmp_path):
    chemin = str(tmp_path / "grand.pyr")
    with pytest.raises(TypeError, match="64 bits"):
        pyrucast.save(chemin, {"n": 2**70})


def test_un_type_non_archivable_le_dit(tmp_path):
    chemin = str(tmp_path / "dict.pyr")
    with pytest.raises(TypeError, match="cannot be archived"):
        pyrucast.save(chemin, {"x": {"imbriqué": 1}})


# ─── Le fichier ─────────────────────────────────────────────────────────────


def test_le_fichier_est_reproductible(tmp_path):
    _, m, _ = line()
    a, b = str(tmp_path / "a.pyr"), str(tmp_path / "b.pyr")
    for chemin in (a, b):
        pyrucast.save(chemin, {"zzz": m, "aaa": 1.0})
    assert open(a, "rb").read() == open(b, "rb").read()


def test_un_fichier_etranger_est_refuse(tmp_path):
    chemin = tmp_path / "faux.pyr"
    chemin.write_bytes(b"ceci n'est pas une archive")
    with pytest.raises(RuntimeError, match="signature"):
        pyrucast.load(str(chemin))


# ─── Une étude complète ─────────────────────────────────────────────────────


def test_une_etude_complete(tmp_path):
    """Maillage, espace, modèle, matériau, chargement : sauvés ensemble, relus,
    réassemblés — la solution doit être celle d'avant."""
    c = pyrucast.Coords(1)
    n = 4
    nodes = [c.add_node([i / n]) for i in range(n + 1)]
    m = pyrucast.Mesh(c, "SEG2")
    for i in range(n):
        m.unit().add_cell([nodes[i], nodes[i + 1]])
    fes = pyrucast.FiniteElementSpace(m, "Lagrange1")

    impose = pyrucast.Mesh(c, "POI1")
    impose.unit().add_cell([nodes[-1]])
    mult = pyrucast.mesh.barycenter(impose)
    cible = pyrucast.model.heat_conduction(fes)

    modele = cible | pyrucast.model.dirichlet(cible, "T", impose, mult)
    materiaux = pyrucast.element_field.material_field(modele, [("k", 1.0)])

    charge = pyrucast.Mesh(c, "POI1")
    charge.unit().add_cell([nodes[0]])
    charge.unit().add_cell([mult[0][0][0]])
    rhs = pyrucast.NodeField(charge, ["imposed_T", "q"])
    rhs[0].set_value(nodes[0], "q", 10.0)
    rhs[0].set_value(mult[0][0][0], "imposed_T", 20.0)

    k = pyrucast.matrix.stiffness(modele, materiaux)
    avant = pyrucast.solver.solve(k, rhs)

    chemin = str(tmp_path / "etude.pyr")
    pyrucast.save(
        chemin,
        {
            "maillage": m,
            "espace": fes,
            "modele": modele,
            "materiaux": materiaux,
            "chargement": rhs,
            "rigidite": k,
            "solution": avant,
            "conductivite": 1.0,
        },
    )

    o = pyrucast.load(chemin)
    assert o["conductivite"] == 1.0

    # Réassembler depuis le modèle relu : la CSR et la factorisation n'étaient
    # pas dans le fichier, elles se rebâtissent.
    k2 = pyrucast.matrix.stiffness(o["modele"], o["materiaux"])
    apres = pyrucast.solver.solve(k2, o["chargement"])

    for node in nodes:
        assert apres.value(node, "T") == avant.value(node, "T")
        assert o["solution"].value(node, "T") == avant.value(node, "T")
