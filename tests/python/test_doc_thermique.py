"""Source of the Python examples of `book/src/thermique.md` and
`book/src/rayonnement.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show shifted by four spaces in the book. At module level, the
excerpt is exactly what a user would write.

Consequence: pytest runs this file at **collection** time, and an example that
breaks is a collection error, not a failing test.

Voir `book/src/developper/documentation-et-tests.md`.
"""

# ── Conduction : une barre chauffée ─────────────────────────────────────────

# ANCHOR: conduction
import pyrucast

# Une barre [0, 1] de quatre SEG2 : un flux Q = 10 entre à gauche, T = 20 à droite.
c = pyrucast.Coords(1)
gauche = c.add_node([0.0])
droite = c.add_node([1.0])
barre = pyrucast.mesh.line(gauche, droite, 4)
fes = pyrucast.FiniteElementSpace(barre)

conduction = pyrucast.model.heat_conduction(fes)
appui = pyrucast.model.dirichlet(
    pyrucast.mesh.poi1_from_nodes([droite]), conduction, "T"
)
modele = conduction | appui
materiaux = pyrucast.element_field.material_field(
    modele, [("k", 2.0), ("rho", 1.0), ("cp", 3.0)]
)

source = pyrucast.NodeField(barre, ["q"])  # sur les nœuds de la barre
source.unit().set_value(gauche, "q", 10.0)
second_membre = source | appui.constraint_rhs([(droite, 20.0)])

K = pyrucast.matrix.stiffness(modele, materiaux)
T = K.solve(second_membre)
print(T.value(gauche, "T"))  # 25.0 = 20 + Q·L/k
print(T.value(droite, "T"))  # 20.0
# ANCHOR_END: conduction

assert abs(T.value(gauche, "T") - 25.0) < 1e-12
assert abs(T.value(droite, "T") - 20.0) < 1e-12

# ── Le comportement : gradient, flux, forces internes ───────────────────────

# ANCHOR: comportement
gradient = pyrucast.element_field.strain(modele, T, materiaux)
print(gradient.components())  # ['grad_T_x']
print(gradient[0].value(0, 0, "grad_T_x"))  # -5.0

repos = pyrucast.element_field.initial_state(
    modele, materiaux
)  # l'état de départ de la loi
flux = pyrucast.element_field.integrate_behavior(modele, gradient, materiaux, repos)
print(flux[0].value(0, 0, "flux_x"))  # -10.0 = k·∂T/∂x

f_int = pyrucast.node_field.internal_forces(modele, flux, T, materiaux)
print(f_int.value(gauche, "q"))  # 10.0 : la source que la conduction équilibre
# ANCHOR_END: comportement

assert gradient.components() == ["grad_T_x"]
assert abs(gradient[0].value(0, 0, "grad_T_x") + 5.0) < 1e-12
assert abs(flux[0].value(0, 0, "flux_x") + 10.0) < 1e-12
assert abs(f_int.value(gauche, "q") - 10.0) < 1e-12

# ── Un pas de temps ─────────────────────────────────────────────────────────

# ANCHOR: transitoire
C = pyrucast.matrix.mass(modele, materiaux)  # la capacité ρ·cp ; l'appui n'y entre pas
T0 = pyrucast.NodeField(barre, ["T"])
T0.unit().set_uniform("T", 20.0)

dt = 0.01
A = C / dt + K  # Euler implicite : (C/Δt + K)·T₁ = F + (C/Δt)·T₀
T1 = A.solve(second_membre + (C / dt) * T0)
print(round(T1.value(gauche, "T"), 3))  # 20.306 : la chaleur commence à entrer
# ANCHOR_END: transitoire

assert abs(T1.value(gauche, "T") - 20.30588764486413) < 1e-9

# ── Conduction orthotrope ───────────────────────────────────────────────────

# ANCHOR: orthotrope
c2 = pyrucast.Coords(2)
bas = pyrucast.mesh.line(c2.add_node([0.0, 0.0]), c2.add_node([1.0, 0.0]), 2)
haut = pyrucast.mesh.line(c2.add_node([0.0, 1.0]), c2.add_node([1.0, 1.0]), 2)
carre = pyrucast.mesh.sweep(bas, haut, 2)
plaque = pyrucast.FiniteElementSpace(carre)

orientee = pyrucast.model.heat_conduction(plaque, symmetry="orthotropic")
print(orientee[0].material_components())  # ['k_1', 'k_2', 'k_3', 'V1X', 'V1Y']
materiaux = pyrucast.element_field.material_field(
    orientee,
    [("k_1", 12.0), ("k_2", 3.0), ("k_3", 3.0), ("V1X", 0.6), ("V1Y", 0.8)],
)
K = pyrucast.matrix.stiffness(orientee, materiaux)
# ANCHOR_END: orthotrope

assert orientee[0].material_components() == ["k_1", "k_2", "k_3", "V1X", "V1Y"]
assert K.n_rows == 9

# ── Rayonnement ─────────────────────────────────────────────────────────────

# ANCHOR: rayonnement
conduction = pyrucast.model.heat_conduction(plaque)
rayonnement = pyrucast.model.radiation(pyrucast.FiniteElementSpace(haut), conduction)
print(rayonnement[0].physics())  # ['thermal', 'radiation']
modele = conduction | rayonnement
materiaux = pyrucast.element_field.material_field(
    modele, [("k", 20.0), ("emis", 0.8), ("T_inf", 300.0)]
)

T = pyrucast.NodeField(carre, ["T"])
T.unit().set_uniform("T", 1000.0)  # en kelvins
entree = pyrucast.element_field.strain(modele, T, materiaux)  # ∇T dedans, T au bord
repos = pyrucast.element_field.initial_state(modele, materiaux)
sortie = pyrucast.element_field.integrate_behavior(modele, entree, materiaux, repos)
print(sortie[1].value(0, 0, "flux"))  # 44995.55… = σε(T⁴ − T∞⁴)

K = pyrucast.matrix.stiffness(modele, materiaux)  # film linéarisé en T∞
Kt = pyrucast.matrix.tangent(
    modele, materiaux, entree, repos
)  # film à la température T
coin = haut.node(0, 0, 0)
print(
    Kt.get(coin, "q", coin, "T") - K.get(coin, "q", coin, "T")
)  # 29.425… = 4σε(T³ − T∞³)·L/3
# ANCHOR_END: rayonnement

_s = 5.670374419e-8 * 0.8
assert rayonnement[0].physics() == ["thermal", "radiation"]
assert abs(sortie[1].value(0, 0, "flux") - _s * (1000.0**4 - 300.0**4)) < 1e-6
_ecart = Kt.get(coin, "q", coin, "T") - K.get(coin, "q", coin, "T")
assert abs(_ecart - 4.0 * _s * (1000.0**3 - 300.0**3) * 0.5 / 3.0) < 1e-9
