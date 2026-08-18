"""Source des exemples Python de `book/src/sauvegarde.md` et `evolution.md`.

Chaque bloc de ces pages vient d'ici par `{{#include …:ancre}}`.

**Le code vit au niveau module, pas dans des fonctions de test** : mdbook
n'enlève pas l'indentation d'un extrait inclus, si bien qu'un bloc ancré dans
une fonction s'afficherait décalé de quatre espaces. pytest exécute donc ce
fichier à la **collecte** ; un exemple qui casse est une erreur de collecte, au
traceback complet et au code de retour non nul.

Les extraits de sauvegarde écrivent des fichiers sous des noms courts
(`etude.pyr`) : le module bascule une fois pour toutes dans un répertoire
temporaire, ce qui laisse l'extrait affiché tel qu'un utilisateur l'écrirait.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import os
import tempfile

import pyrucast

# Répertoire de travail jetable — les noms de fichiers des extraits restent
# courts. Le module rend le répertoire courant à la fin : au niveau module il
# n'y a pas de fixture, et le laisser déplacé piégerait les autres fichiers.
_TMP = tempfile.TemporaryDirectory()
_CWD = os.getcwd()
os.chdir(_TMP.name)


def _support_et_champs():
    """Un nuage POI1 et deux champs posés dessus, sur le **même** support."""
    c = pyrucast.Coords(2)
    noeuds = [c.add_node([float(i), 0.0]) for i in range(3)]
    support = pyrucast.mesh.poi1_from_nodes(noeuds)
    return c, support, noeuds


def _maillage_et_materiaux():
    c = pyrucast.Coords(2)
    n = [c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0])]
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell(n)
    fes = pyrucast.FiniteElementSpace(mesh)
    mat = pyrucast.ElementField(fes, ["k"])
    mat[0].set_uniform("k", 1.0)
    return c, mesh, fes, mat, n


# ── Sauvegarde : un dictionnaire à l'aller, un dictionnaire au retour ───────

_c, mesh, fes, mat, _n = _maillage_et_materiaux()
temperature = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes(_n), ["T"])

# ANCHOR: save_load
pyrucast.save(
    "etude.pyr",
    {
        "maillage fin": mesh,
        "T (°C)": temperature,
        "materiaux": mat,
        "pas de temps": 0.05,
        "instants": [0.0, 0.1, 0.2],
    },
)

objets = pyrucast.load("etude.pyr")
mesh2 = objets["maillage fin"]
t2 = objets["T (°C)"]
# ANCHOR_END: save_load

assert mesh2.cell_count() == mesh.cell_count()
assert t2.components() == ["T"]


# ── Le partage survit à l'aller-retour ──────────────────────────────────────

_c2, support, _ = _support_et_champs()

# ANCHOR: partage
t = pyrucast.NodeField(support, ["T"])
f = pyrucast.NodeField(support, ["f"])
pyrucast.save("etude.pyr", {"T": t, "f": f})

o = pyrucast.load("etude.pyr")
assert len(o["T"] | o["f"]) == 1  # une seule zone : le support est un seul objet
# ANCHOR_END: partage


# ── Un maillage entraîne ses dépendances ────────────────────────────────────

# ANCHOR: dependances
pyrucast.save(
    "m.pyr", {"maillage": mesh}
)  # écrit aussi la Coords et les sous-maillages
# ANCHOR_END: dependances

assert pyrucast.load("m.pyr")["maillage"].cell_count() == mesh.cell_count()


# ── Ce qui n'est pas sauvé : les états dérivés ──────────────────────────────

_imposed = pyrucast.mesh.poi1_from_nodes([_n[0]])
_mult = pyrucast.mesh.barycenter(_imposed)
modele = pyrucast.Model.heat_conduction(fes) | pyrucast.Model.dirichlet(
    "T", "q", _imposed, _mult
)
materiaux = pyrucast.element_field.material_field(modele, [("k", 1.0)])
chargement = pyrucast.NodeField(_mult, ["imposed_T"])
chargement[0].set_value(_mult.node(0, 0, 0), "imposed_T", 1.0)
pyrucast.save(
    "etude.pyr",
    {"modele": modele, "materiaux": materiaux, "chargement": chargement},
)

# ANCHOR: reassemblage
o = pyrucast.load("etude.pyr")
k = pyrucast.matrix.stiffness(o["modele"], o["materiaux"])  # réassemble
u = pyrucast.solver.solve(k, o["chargement"])  # refactorise
# ANCHOR_END: reassemblage

assert u.node_count() > 0


# ── Les compteurs de nœuds ne traversent pas ────────────────────────────────

_c3 = pyrucast.Coords(2)
_c3.add_node([0.0, 0.0])
pyrucast.save("coords_seules.pyr", {"c": _c3})

# ANCHOR: refcount
c2 = pyrucast.load("coords_seules.pyr")["c"]
c2.gc()  # collecte tout : rien dans le fichier ne retenait ces nœuds
# ANCHOR_END: refcount


# ── Évolution : une courbe, sa zone ─────────────────────────────────────────

pc = pyrucast

# ANCHOR: subevolution
se = pc.SubEvolution(
    [(0.0, 0.0), (100.0, 210e9)], abscissa_type="T", ordinate_type="young"
)
# ANCHOR_END: subevolution

assert se.interpolate(50.0) > 0.0


# ── Loi matériau : un champ en entrée, un champ en sortie ───────────────────

temperature[0].set_value(_n[0], "T", 50.0)
temperature[0].set_value(_n[1], "T", 80.0)
temperature[0].set_value(_n[2], "T", 20.0)

# ANCHOR: loi_materiau
# Loi matériau E(T) : module d'Young fonction de la température.
loi = pc.Evolution(
    [(0.0, 0.0), (100.0, 210e9)], abscissa_type="T", ordinate_type="young"
)
young = loi.interpolate(temperature)  # temperature : NodeField de composante "T"
# young : NodeField de composante "young"
# ANCHOR_END: loi_materiau

assert young.components() == ["young"]


# ── Toutes les formes d'interpolation ───────────────────────────────────────

champ_t0 = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes(_n), ["T"])
champ_t1 = pyrucast.NodeField(champ_t0.support_mesh(), ["T"])

# ANCHOR: interpolate
import pyrucast as pc

# Courbe scalaire (une SubEvolution).
se = pc.SubEvolution([(0.0, 10.0), (1.0, 20.0)])
print(se.interpolate(0.5))  # 15.0
print(se.interpolate(2.0, out_of_range="clamp"))  # 20.0 (sinon : erreur)

# Agrégat scalaire → liste de flottants.
e = pc.Evolution([(0.0, 10.0), (1.0, 20.0)])
print(e.interpolate(0.5))  # [15.0]

# Bas niveau : composition de courbes par zone avec `|`.
agg = pc.SubEvolution([(0.0, 1.0), (1.0, 2.0)]) | pc.SubEvolution(
    [(0.0, 3.0), (1.0, 4.0)]
)
print(agg.interpolate(0.5))  # [1.5, 3.5]

# Haut niveau temps-major : un NodeField complet par pas → NodeField interpolé.
ev = pc.Evolution([(0.0, champ_t0), (2.0, champ_t1)])
champ = ev.interpolate(1.0)  # NodeField à mi-chemin

# Courbe de transfert : passer un champ → champ (loi matériau E(T)).
loi = pc.Evolution(
    [(0.0, 0.0), (100.0, 210e9)], abscissa_type="T", ordinate_type="young"
)
young = loi.interpolate(temperature)  # composante "T" lue → composante "young"
# ANCHOR_END: interpolate

assert se.interpolate(0.5) == 15.0
assert e.interpolate(0.5) == [15.0]


# ── Tracé d'une évolution ───────────────────────────────────────────────────

# ANCHOR: plot
e = pc.Evolution([(0.0, 10.0), (1.0, 20.0), (2.0, 5.0)])
e.plot(save="courbe.svg", x_label="temps", y_label="T")  # courbe scalaire
ev.plot(save="frame.png", frame=1)  # champ tabulé (un pas)
# ANCHOR_END: plot


# Fin des extraits : on rend le répertoire courant.
os.chdir(_CWD)
