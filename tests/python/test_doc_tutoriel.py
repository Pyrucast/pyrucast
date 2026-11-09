"""Source of the Python examples of `book/src/tutoriel/langage-python.md`.

The page teaches the Python a tutorial script needs, one notion at a time,
each followed by the pyrucast shortcut that rides on it, then pyrucast's own
conventions. The excerpts stay as plain as the tutorial scripts: linear code,
no helper function, one notion per anchor. The checks sit after each
`ANCHOR_END`, out of the reader's sight.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show up shifted by four spaces. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero exit code.

Voir `book/src/developper/documentation-et-tests.md`.
"""

# ── Values ──────────────────────────────────────────────────────────────────

# ANCHOR: valeurs
COTE = 0.10  # un nombre décimal s'écrit avec un point, jamais une virgule
SOURCE = 1.0e6  # 1,0 × 10⁶ : l'exposant s'écrit après « e »
FLUX = -40_000.0  # les « _ » ne servent qu'à lire les grands nombres
DEMI_COTE = COTE / 2.0  # un calcul, rangé sous un nouveau nom

element = "QUA4"  # du texte, entre guillemets
# ANCHOR_END: valeurs

assert DEMI_COTE == 0.05
assert SOURCE == 1_000_000.0


# ── Lists, tuples, dictionaries ─────────────────────────────────────────────

# ANCHOR: collections
point = [0.0, 0.10]  # une liste : les coordonnées x, y d'un point
rouge = (255, 0, 0)  # un tuple : une couleur, rouge, vert, bleu
coefficients = [("k", 50.0), ("phi_q", 1.0e6)]  # une liste de paires (nom, valeur)
LONGUEUR, HAUTEUR = 0.30, 0.10  # deux noms d'un coup

print(point[0])  # 0.0 — le premier élément porte le numéro 0
print(coefficients[1])  # ('phi_q', 1000000.0)
# ANCHOR_END: collections

assert coefficients[1] == ("phi_q", 1.0e6)
assert HAUTEUR == 0.10

# ANCHOR: dictionnaire
donnees = {"times": [0.0, 0.5, 1.0]}  # des valeurs rangées par nom
donnees["save_times"] = [1.0]  # une entrée de plus
print(donnees["times"])  # [0.0, 0.5, 1.0]
# ANCHOR_END: dictionnaire

assert list(donnees) == ["times", "save_times"]


# ── Importing, calling a function ───────────────────────────────────────────

# ANCHOR: importer
import os

import pyrucast as pc  # le paquet, sous le nom court « pc »

coords = pc.Coords(2)  # pc.Coords : la classe Coords du paquet
a = coords.add_node([0.0, 0.0])
b = coords.add_node([0.10, 0.0])
ligne = pc.mesh.line(a, b, 20)  # pc.mesh.line : la fonction line du module mesh
# ANCHOR_END: importer

assert ligne.cell_count == 20

# ANCHOR: arguments
plaque = pc.mesh.extrude(ligne, [0.0, 0.10], 20)  # trois arguments, dans l'ordre
cotes = pc.mesh.border(plaque, angle_deg=45)  # un argument donné par son nom
bord = pc.mesh.border(plaque)  # sans lui : sa valeur par défaut
# ANCHOR_END: arguments

assert len(cotes) == 4
assert len(bord) == 1


# ── Objects: methods and properties ─────────────────────────────────────────

# ANCHOR: objets
print(plaque.cell_count)  # 400 — une propriété : sans parenthèses
print(plaque.element_types())  # ['QUA4'] — une méthode : avec parenthèses
print(coords.node_count)  # 441
plaque[0].face_color = (255, 0, 0)  # une propriété qui se modifie
# ANCHOR_END: objets

assert plaque.cell_count == 400
assert coords.node_count == 441

# The first analysis, to have a temperature field at hand.
_fes = pc.FiniteElementSpace(plaque)
_conduction = pc.model.heat_conduction(_fes)
modele = _conduction | pc.model.dirichlet(pc.mesh.to_poi1(bord), _conduction, "T")
modele = modele | pc.model.flux(_fes, modele, "q")
materiaux = pc.element_field.material_field(modele, [("k", 50.0), ("phi_q", 1.0e6)])
temperature = pc.solver.solve(
    pc.matrix.stiffness(modele, materiaux),
    pc.node_field.external_forces(modele, materiaux),
)


# ── Square brackets ─────────────────────────────────────────────────────────

# ANCHOR: crochets
print(len(cotes))  # 4 — les quatre côtés de la plaque, une zone chacun
bas = cotes[0]  # la première zone
dernier = cotes[-1]  # la dernière
sauf_bas = cotes[1:]  # toutes sauf la première : encore un maillage
milieu = cotes[1:-1]  # ni la première ni la dernière

seulement_T = temperature["T"]  # un nom entre crochets : une composante

champ = pc.NodeField(ligne, ["T"])  # un champ nul aux nœuds de la ligne
champ[0][a, "T"] = 20.0  # une valeur : la zone, puis (nœud, composante)
print(champ[0][a, "T"])  # 20.0
# ANCHOR_END: crochets

assert (len(sauf_bas), len(milieu)) == (3, 2)
assert temperature.components() == ["T", "lambda"]
assert seulement_T.components() == ["T"]
assert champ.max("T") == 20.0


# ── Operators ───────────────────────────────────────────────────────────────

# ANCHOR: operateurs
deux_points = a | b  # deux nœuds : un maillage POI1 de deux points
bas_et_droite = cotes[0] | cotes[1]  # deux zones : un maillage à deux zones

kelvin = temperature["T"] + 273.15  # un nombre ajouté à toutes les valeurs
double = 2.0 * temperature["T"]  # toutes les valeurs multipliées
chaud = temperature["T"] >= 10.0  # 1 là où T ≥ 10 °C, 0 ailleurs
# ANCHOR_END: operateurs

assert deux_points.element_types() == ["POI1"] and deux_points.cell_count == 2
assert len(bas_et_droite) == 2
assert abs(kelvin.max("T") - (temperature.max("T") + 273.15)) < 1e-9
assert abs(double.max("T") - 2.0 * temperature.max("T")) < 1e-9
assert chaud.max("T") == 1.0 and chaud.min("T") == 0.0


# ── Loops ───────────────────────────────────────────────────────────────────

# ANCHOR: boucle
for cote in cotes:
    print(cote.element_type, cote.cell_count)  # SEG2 20, quatre fois

PALETTE = [(255, 0, 0), (0, 170, 0), (0, 0, 255)]
for i, cote in enumerate(cotes):
    cote.face_color = PALETTE[i % len(PALETTE)]  # % : le reste de la division
# ANCHOR_END: boucle

assert cotes[3].face_color == PALETTE[0]

# ANCHOR: listes_en_boucle
instants = []  # une liste vide…
for i in range(3):
    instants.append(i * 150.0)  # …qui grandit d'un élément à chaque tour
print(instants)  # [0.0, 150.0, 300.0]

couples = [(0.0, 20.0), (150.0, 35.0)]
for t, valeur in couples:  # chaque couple, défait en deux noms
    print(t, valeur)

for avant, apres in zip([20.0, 35.0], [21.0, 37.0]):  # deux listes à la fois
    print(apres - avant)  # 1.0, puis 2.0

print(max([2, 1, 2]), sum([2, 1, 2]))  # 2 5
# ANCHOR_END: listes_en_boucle

assert instants == [0.0, 150.0, 300.0]
assert [apres - avant for avant, apres in zip([20.0, 35.0], [21.0, 37.0])] == [1.0, 2.0]


# ── Conditions ──────────────────────────────────────────────────────────────

# ANCHOR: condition
OUT = os.environ.get("PYRUCAST_TUTORIEL_IMG_DIR")  # None si la variable n'existe pas
if OUT:
    print("les figures seront écrites dans", OUT)
else:
    print("les figures s'ouvriront dans une fenêtre")
# ANCHOR_END: condition


# ── Listings ────────────────────────────────────────────────────────────────

# ANCHOR: listings
print(ligne)  # Mesh: 1 submesh(es) [#…], 20 cell(s) total — le résumé
print(repr(ligne))  # la structure : chaque zone, son type, son nombre de mailles
ligne.dump()  # tout le contenu, maille par maille
# ANCHOR_END: listings


# ── Reading an error ────────────────────────────────────────────────────────
# The page quotes this message in a `text` block; it is checked here.
try:
    cotes.unit()
except RuntimeError as erreur:
    _message = str(erreur)
assert _message == "expected a unitary Mesh (exactly one submesh(es)), found 4"


# ── Declaring an object ─────────────────────────────────────────────────────

# ANCHOR: declarer
vide = pc.Mesh(coords, "TRI3")  # un maillage de triangles, encore sans maille
fes = pc.FiniteElementSpace(plaque)  # l'espace éléments finis de la plaque
nul = pc.NodeField(ligne, ["T"])  # un champ nul aux nœuds de la ligne
# ANCHOR_END: declarer

assert vide.element_types() == ["TRI3"] and vide.cell_count == 0
assert nul.max("T") == 0.0


# ── The same operation, as a function or as a method ────────────────────────

# ANCHOR: methode
plaque_f = pc.mesh.extrude(ligne, [0.0, 0.10], 20)  # la fonction
plaque_m = ligne.extrude([0.0, 0.10], 20)  # la même, en méthode de `ligne`
# ANCHOR_END: methode

assert plaque_f.cell_count == plaque_m.cell_count == 400

# ANCHOR: maillage_imbrique
plaque = pc.mesh.extrude(pc.mesh.line(a, b, 20), [0.0, 0.10], 20)
bord = pc.mesh.to_poi1(pc.mesh.border(plaque))
# ANCHOR_END: maillage_imbrique

_cells, _nodes = plaque.cell_count, bord.cell_count

# ANCHOR: maillage_enchaine
plaque = pc.mesh.line(a, b, 20).extrude([0.0, 0.10], 20)
bord = plaque.border().to_poi1()
# ANCHOR_END: maillage_enchaine

assert (plaque.cell_count, bord.cell_count) == (_cells, _nodes) == (400, 80)

# ANCHOR: resolution_imbrique
temperature = pc.solver.solve(
    pc.matrix.stiffness(modele, materiaux),
    pc.node_field.external_forces(modele, materiaux),
)
# ANCHOR_END: resolution_imbrique

_t_max = temperature.max("T")

# ANCHOR: resolution_enchaine
second_membre = modele.external_forces(materiaux)
temperature = modele.stiffness_matrix(materiaux).solve(second_membre)
# ANCHOR_END: resolution_enchaine

assert temperature.max("T") == _t_max

# A small 3-D box, for the two chains read in `tutoriel/thermique.py`.
_c3 = pc.Coords(3)
_p, _q = _c3.add_node([0.0, 0.0, 0.0]), _c3.add_node([1.0, 0.0, 0.0])
volume = pc.mesh.line(_p, _q, 2).extrude([0.0, 1.0, 0.0], 2).extrude([0.0, 0.0, 1.0], 2)
TOL = 1e-9

# ANCHOR: chaines_thermique
# thermique.py : pc.mesh.consolidate(pc.mesh.skin(volume))
peau = volume.skin().consolidate()

# thermique.py : pc.mesh.select(pc.node_field.positions(peau, ["Z"]), ge=-TOL, le=TOL)
noeuds_bas = peau.positions(["Z"]).select(ge=-TOL, le=TOL)
# ANCHOR_END: chaines_thermique

assert peau.cell_count == pc.mesh.consolidate(pc.mesh.skin(volume)).cell_count == 24
assert noeuds_bas.cell_count == 9


# ── Aggregates: putting a model together ────────────────────────────────────

# ANCHOR: composer
conduction = pc.model.heat_conduction(fes)
modele = conduction | pc.model.dirichlet(bord, conduction, "T")
modele = modele | pc.model.flux(fes, modele, "q")
print(len(modele))  # 3 — trois termes réunis par |
# ANCHOR_END: composer

assert len(modele) == 3


# ── Component names: primal and dual ────────────────────────────────────────

# ANCHOR: composantes
print(conduction.dual_of("T"))  # q — la duale de la température : le flux
blocage = pc.model.dirichlet(bord, conduction, "T")  # on bloque la primale « T »

mecanique = pc.model.elasticity(fes, "plane_stress")
print(mecanique.dual_of("u_x"))  # f_x — la duale du déplacement : la force
appui = pc.model.dirichlet(bord, mecanique, "u_x")  # on bloque la primale « u_x »
# ANCHOR_END: composantes

assert conduction.dual_of("T") == "q"
assert mecanique.dual_of("u_x") == "f_x"
