"""Source of the Python examples of `book/src/evolution.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show shifted by four spaces in the book. At module level, the
excerpt is exactly what a user would write.

Consequence: pytest runs this file at **collection** time, and an example that
breaks is a collection error, not a failing test.

The plotting excerpt writes files under short names: the module works in a
temporary directory, and gives the current directory back at the end.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import os
import tempfile

_TMP = tempfile.TemporaryDirectory()
_CWD = os.getcwd()
os.chdir(_TMP.name)

# ── Une courbe ──────────────────────────────────────────────────────────────

# ANCHOR: courbe
import pyrucast

# Une température qui monte de 20 à 120 °C en 10 s, puis reste à 120 °C.
montee = pyrucast.SubEvolution([(0.0, 20.0), (10.0, 120.0), (60.0, 120.0)])
print(montee.interpolate(2.5))  # 45.0
print(montee.interpolate(10.0))  # 120.0 : un échantillon, rendu tel quel
print(montee.abscissas())  # [0.0, 10.0, 60.0]
print(len(montee))  # 3
# ANCHOR_END: courbe

assert montee.interpolate(2.5) == 45.0
assert montee.interpolate(10.0) == 120.0
assert montee.abscissas() == [0.0, 10.0, 60.0]
assert len(montee) == 3

# ── Hors plage ──────────────────────────────────────────────────────────────

# ANCHOR: hors_plage
try:
    montee.interpolate(70.0)  # politique "error", par défaut
except RuntimeError as erreur:
    print(erreur)  # evolution: abscissa 70 is outside the tabulated range [0, 60]
print(montee.interpolate(70.0, out_of_range="clamp"))  # 120.0
print(montee.interpolate(-1.0, out_of_range="extrapolate"))  # 10.0

bornee = pyrucast.SubEvolution([(0.0, 20.0), (10.0, 120.0)], out_of_range="clamp")
print(bornee.out_of_range, bornee.interpolate(70.0))  # clamp 120.0
# ANCHOR_END: hors_plage

try:
    montee.interpolate(70.0)
    raise AssertionError("70 is out of range")
except RuntimeError as erreur:
    assert "outside the tabulated range [0, 60]" in str(erreur)
assert montee.interpolate(70.0, out_of_range="clamp") == 120.0
assert montee.interpolate(-1.0, out_of_range="extrapolate") == 10.0
assert (bornee.out_of_range, bornee.interpolate(70.0)) == ("clamp", 120.0)

# ── Des champs comme valeurs ────────────────────────────────────────────────

# ANCHOR: champ
c = pyrucast.Coords(1)
noeuds = [c.add_node([float(i)]) for i in range(3)]
support = pyrucast.mesh.poi1_from_nodes(noeuds)
froid = pyrucast.NodeField(support, ["T"])
froid.unit().set_uniform("T", 20.0)
chaud = pyrucast.NodeField(support, ["T"])
chaud.unit().set_uniform("T", 120.0)

chauffe = pyrucast.Evolution([(0.0, froid), (10.0, chaud)])
t = chauffe.interpolate(2.5)  # un NodeField, sur le même support
print(t.value(noeuds[0], "T"))  # 45.0
print(chauffe.shared_abscissas())  # [0.0, 10.0]
print(len(chauffe.frames()))  # 2 : les champs tabulés, un par abscisse
# ANCHOR_END: champ

assert t.value(noeuds[0], "T") == 45.0
assert chauffe.shared_abscissas() == [0.0, 10.0]
assert len(chauffe.frames()) == 2
# The values were copied at construction: changing a field afterwards leaves
# the evolution as it was.
chaud.unit().set_uniform("T", 500.0)
assert chauffe.interpolate(10.0).value(noeuds[0], "T") == 120.0

# ── Une abscisse à plusieurs variables ──────────────────────────────────────

# ANCHOR: table
# Limite d'élasticité σ_y (MPa) selon la température T et la fraction φ :
# deux valeurs de φ mesurées à 20 °C, trois à 500 °C.
sigma_y = pyrucast.SubEvolution(
    [
        ((20.0, 0.0), 300.0),
        ((20.0, 1.0), 450.0),
        ((500.0, 0.0), 200.0),
        ((500.0, 0.5), 260.0),
        ((500.0, 1.0), 320.0),
    ],
    abscissa_type=("T", "phi"),
    ordinate_type="sigma_y",
)
print(sigma_y.interpolate((300.0, 0.4)))  # 294.666…
# ANCHOR_END: table

# The two steps of the figure, computed by hand.
_a_20 = 300.0 + 0.4 * (450.0 - 300.0)  # 360
_a_500 = 200.0 + (0.4 / 0.5) * (260.0 - 200.0)  # 248
_t = (300.0 - 20.0) / (500.0 - 20.0)
assert abs(sigma_y.interpolate((300.0, 0.4)) - (_a_20 + _t * (_a_500 - _a_20))) < 1e-9

# On a full grid, the reading is the bilinear interpolation.
_grille = pyrucast.SubEvolution(
    [((0.0, 0.0), 1.0), ((0.0, 1.0), 2.0), ((1.0, 0.0), 3.0), ((1.0, 1.0), 5.0)]
)
_x, _y = 0.25, 0.75
_bilineaire = (
    1.0 * (1 - _x) * (1 - _y)
    + 2.0 * (1 - _x) * _y
    + 3.0 * _x * (1 - _y)
    + 5.0 * _x * _y
)
assert abs(_grille.interpolate((_x, _y)) - _bilineaire) < 1e-12


# A function linear in each variable separately is reproduced exactly, even on
# the irregular table.
def _f(T, phi):
    return 1.0 + 2.0 * T + 3.0 * phi + 4.0 * T * phi


_irreguliere = pyrucast.SubEvolution(
    [((T, phi), _f(T, phi)) for T, phi in sigma_y.abscissas()]
)
assert abs(_irreguliere.interpolate((300.0, 0.4)) - _f(300.0, 0.4)) < 1e-9

# A group whose next variable takes a single value is constant along it.
_un_seul = pyrucast.SubEvolution(
    [((0.0, 0.0), 1.0), ((0.0, 1.0), 2.0), ((1.0, 0.5), 10.0)]
)
assert _un_seul.interpolate((1.0, 0.0)) == _un_seul.interpolate((1.0, 1.0)) == 10.0

# On an irregular table, the order of the variables changes the result.
_echangee = pyrucast.SubEvolution(
    [
        ((phi, T), v)
        for (T, phi), v in zip(sigma_y.abscissas(), [300.0, 450.0, 200.0, 260.0, 320.0])
    ]
)
assert (
    abs(_echangee.interpolate((0.25, 300.0)) - sigma_y.interpolate((300.0, 0.25))) > 1.0
)

# At most four variables.
try:
    pyrucast.SubEvolution([((0.0, 0.0, 0.0, 0.0, 0.0), 1.0)])
    raise AssertionError("five variables are refused")
except RuntimeError:
    pass

# ── Types d'abscisse et d'ordonnée ──────────────────────────────────────────

# ANCHOR: types
young = pyrucast.SubEvolution(
    [(20.0, 210e3), (500.0, 170e3)], abscissa_type="T", ordinate_type="E"
)
print(young.abscissa_type(), young.ordinate_type())  # T E
print(sigma_y.abscissa_type())  # ('T', 'phi')
# ANCHOR_END: types

assert (young.abscissa_type(), young.ordinate_type()) == ("T", "E")
assert sigma_y.abscissa_type() == ("T", "phi")
try:
    pyrucast.Evolution([(0.0, froid), (10.0, chaud)], ordinate_type="T")
    raise AssertionError("an evolution of fields has no ordinate type")
except RuntimeError:
    pass

# ── Appliquer une courbe à un champ ─────────────────────────────────────────

# ANCHOR: transfert
etat = pyrucast.NodeField(support, ["T", "phi"])
for noeud, (T, phi) in zip(noeuds, [(20.0, 0.0), (260.0, 0.4), (500.0, 1.0)]):
    etat.unit().set_value(noeud, "T", T)
    etat.unit().set_value(noeud, "phi", phi)

E = young.interpolate(etat[0])  # lit la composante "T", écrit "E"
print(E.components(), E.value(noeuds[1], "E"))  # ['E'] 190000.0
# ANCHOR_END: transfert

assert E.components() == ["E"]
assert E.value(noeuds[1], "E") == 190000.0

# ANCHOR: lois
lois = young | sigma_y  # deux courbes : E(T) et σ_y(T, φ)
materiau = lois.interpolate(etat)  # chaque nœud, en ses propres T et φ
print(materiau.components())  # ['E', 'sigma_y']
print(materiau.value(noeuds[1], "sigma_y"))  # 304.0
# ANCHOR_END: lois

assert materiau.components() == ["E", "sigma_y"]
assert abs(materiau.value(noeuds[1], "sigma_y") - 304.0) < 1e-9

# ── Composer des courbes ────────────────────────────────────────────────────

# ANCHOR: composer
zone_a = pyrucast.SubEvolution([(0.0, 1.0), (1.0, 2.0)], out_of_range="clamp")
zone_b = pyrucast.SubEvolution([(0.0, 3.0), (1.0, 4.0)])
deux = zone_a | zone_b
print(deux.interpolate(0.5))  # [1.5, 3.5] : un flottant par courbe
print(deux.out_of_range)  # error : l'agrégat applique la sienne à ses courbes
print(deux.interpolate(2.0, out_of_range="clamp"))  # [2.0, 4.0]
# ANCHOR_END: composer

assert deux.interpolate(0.5) == [1.5, 3.5]
assert deux.out_of_range == "error"
assert deux[0].out_of_range == "clamp"
assert deux.interpolate(2.0, out_of_range="clamp") == [2.0, 4.0]

# ── Le long d'une ligne ─────────────────────────────────────────────────────

# ANCHOR: le_long
# Une température relevée le long d'un trajet de trois points : la courbe porte
# la valeur en fonction de la distance parcourue — 0, 3, puis 3 + 4 = 7.
plan = pyrucast.Coords(2)
trajet = [plan.add_node(p) for p in ([0.0, 0.0], [3.0, 0.0], [3.0, 4.0])]
ligne = pyrucast.mesh.poi1_from_nodes(trajet)
releve = pyrucast.NodeField(ligne, ["T"])
for noeud, t in zip(trajet, [20.0, 50.0, 90.0]):
    releve[0].set_value(noeud, "T", t)
le_long = pyrucast.Evolution.along(releve, ligne)  # aussi releve.along(ligne)
print(le_long[0].abscissas())  # [0.0, 3.0, 7.0]
# ANCHOR_END: le_long

assert le_long[0].abscissas() == [0.0, 3.0, 7.0]
assert le_long[0].ordinate_type() == "T"
assert releve.along(ligne)[0].interpolate(5.0) == 70.0

# ── Tracer ──────────────────────────────────────────────────────────────────

# ANCHOR: trace
courbe = pyrucast.Evolution(
    [(0.0, 20.0), (10.0, 120.0)], abscissa_type="t", ordinate_type="T"
)
courbe.plot(save="courbe.svg")  # axes étiquetés "t" et "T"
chauffe.plot(save="chauffe.png", frame=1)  # le champ du pas tabulé n° 1
# ANCHOR_END: trace

assert os.path.getsize("courbe.svg") > 0
assert os.path.getsize("chauffe.png") > 0

os.chdir(_CWD)
