"""Source of the Python examples of the `book/src/mecanique/*.md` pages.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show shifted by four spaces in the book. At module level, the
excerpt is exactly what a user would write.

Consequence: pytest runs this file at **collection** time, and an example that
breaks is a collection error, not a failing test.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import math

import pyrucast


def _carre(dim_noeud=2):
    """A unit square of 2×2 QUA4, swept between its left and right edges."""
    c = pyrucast.Coords(2)
    gauche = pyrucast.mesh.line(c.add_node([0.0, 0.0]), c.add_node([0.0, 1.0]), 2)
    droite = pyrucast.mesh.line(c.add_node([1.0, 0.0]), c.add_node([1.0, 1.0]), 2)
    return gauche, droite, pyrucast.mesh.sweep(gauche, droite, 2)


def _un_qua4(eps_xx):
    """One unit QUA4 and a displacement giving a uniform strain `eps_xx`."""
    c = pyrucast.Coords(2)
    n = [c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0])]
    maille = pyrucast.Mesh(c, "QUA4")
    maille.unit().add_cell(n)
    u = pyrucast.NodeField(maille, ["u_x", "u_y"])
    for noeud, x in zip(n, [0.0, 1.0, 1.0, 0.0]):
        u.unit().set_value(noeud, "u_x", eps_xx * x)
    return pyrucast.FiniteElementSpace(maille), u


# ── Barre ───────────────────────────────────────────────────────────────────

# ANCHOR: barre
import pyrucast

# Une barre de longueur 2, fixée à gauche, guidée en y à droite, tirée par F = 1000.
c = pyrucast.Coords(2)
a = c.add_node([0.0, 0.0])
b = c.add_node([2.0, 0.0])
ligne = pyrucast.mesh.line(a, b, 1)
barre = pyrucast.model.truss(pyrucast.FiniteElementSpace(ligne))
appuis = (
    pyrucast.model.dirichlet(pyrucast.mesh.poi1_from_nodes([a]), barre, "u_x")
    | pyrucast.model.dirichlet(pyrucast.mesh.poi1_from_nodes([a]), barre, "u_y")
    | pyrucast.model.dirichlet(pyrucast.mesh.poi1_from_nodes([b]), barre, "u_y")
)
modele = barre | appuis
materiaux = pyrucast.element_field.material_field(modele, [("E", 200e3), ("A", 0.01)])

force = pyrucast.NodeField(ligne, ["f_x"])
force.unit().set_value(b, "f_x", 1000.0)
u = pyrucast.matrix.stiffness(modele, materiaux).solve(force)  # valeurs imposées nulles
print(u.value(b, "u_x"))  # 1.0 = F·L/(E·A)

deformation = pyrucast.element_field.strain(modele, u, materiaux)
effort = pyrucast.element_field.integrate_behavior(modele, deformation, materiaux)
print(effort.components(), effort[0].value(0, 0, "n"))  # ['n'] 1000.0
# ANCHOR_END: barre

assert abs(u.value(b, "u_x") - 1.0) < 1e-12
assert effort.components() == ["n"]
assert abs(effort[0].value(0, 0, "n") - 1000.0) < 1e-9

# ── Élasticité ──────────────────────────────────────────────────────────────

# ANCHOR: elasticite
gauche, droite, carre = _carre()  # un carré [0, 1]² de 2×2 QUA4
plaque = pyrucast.model.elasticity(pyrucast.FiniteElementSpace(carre), "plane_stress")
coin = gauche.node(0, 0, 0)
modele = (
    plaque
    | pyrucast.model.dirichlet(gauche.to_poi1(), plaque, "u_x")
    | pyrucast.model.dirichlet(pyrucast.mesh.poi1_from_nodes([coin]), plaque, "u_y")
    | pyrucast.model.flux(pyrucast.FiniteElementSpace(droite), plaque, "f_x")
)
materiaux = pyrucast.element_field.material_field(
    modele,
    [("E", 200e3), ("nu", 0.3), ("phi_f_x", 100.0)],  # traction S = 100 à droite
)
u = pyrucast.matrix.stiffness(modele, materiaux).solve(
    pyrucast.node_field.external_forces(modele, materiaux)
)
haut_droit = droite.node(0, 1, 1)
print(round(u.value(haut_droit, "u_x"), 12))  # 0.0005 = S/E
print(round(u.value(haut_droit, "u_y"), 12))  # -0.00015 = −ν·S/E

eps = pyrucast.element_field.strain(modele, u, materiaux)
sigma = pyrucast.element_field.integrate_behavior(modele, eps, materiaux)
print(sigma.components())  # ['sigma_xx', 'sigma_yy', 'sigma_xy']
print(round(sigma[0].value(0, 0, "sigma_xx"), 9))  # 100.0
# ANCHOR_END: elasticite

assert abs(u.value(haut_droit, "u_x") - 5e-4) < 1e-15
assert abs(u.value(haut_droit, "u_y") + 1.5e-4) < 1e-15
assert abs(sigma[0].value(0, 0, "sigma_xx") - 100.0) < 1e-9

# ── Orthotropie ─────────────────────────────────────────────────────────────


# ANCHOR: orthotrope
gauche, droite, carre = _carre()
plaque = pyrucast.model.elasticity(
    pyrucast.FiniteElementSpace(carre), "plane_stress", symmetry="orthotropic"
)
print(plaque[0].material_components())
# ['E_1', 'E_2', 'E_3', 'nu_12', 'nu_13', 'nu_23', 'G_12', 'G_13', 'G_23', 'V1X', 'V1Y']
angle = 0.0  # l'axe raide, selon x
constantes = [
    ("E_1", 40e3),
    ("E_2", 10e3),
    ("E_3", 10e3),
    ("nu_12", 0.25),
    ("nu_13", 0.25),
    ("nu_23", 0.3),
    ("G_12", 5e3),
    ("G_13", 5e3),
    ("G_23", 4e3),
    ("V1X", math.cos(angle)),
    ("V1Y", math.sin(angle)),
]
# ANCHOR_END: orthotrope


def _traction_orthotrope(angle):
    """The same plate in tension, its stiff axis at `angle` from x."""
    gauche, droite, carre = _carre()
    plaque = pyrucast.model.elasticity(
        pyrucast.FiniteElementSpace(carre), "plane_stress", symmetry="orthotropic"
    )
    coin = gauche.node(0, 0, 0)
    modele = (
        plaque
        | pyrucast.model.dirichlet(gauche.to_poi1(), plaque, "u_x")
        | pyrucast.model.dirichlet(pyrucast.mesh.poi1_from_nodes([coin]), plaque, "u_y")
        | pyrucast.model.flux(pyrucast.FiniteElementSpace(droite), plaque, "f_x")
    )
    axes = [("V1X", math.cos(angle)), ("V1Y", math.sin(angle))]
    materiaux = pyrucast.element_field.material_field(
        modele, constantes[:-2] + axes + [("phi_f_x", 100.0)]
    )
    u = pyrucast.matrix.stiffness(modele, materiaux).solve(
        pyrucast.node_field.external_forces(modele, materiaux)
    )
    return u.value(droite.node(0, 1, 1), "u_x")


# Stiff axis along the traction: S/E₁; across it: S/E₂.
assert abs(_traction_orthotrope(0.0) - 100.0 / 40e3) < 1e-12
assert abs(_traction_orthotrope(math.pi / 2) - 100.0 / 10e3) < 1e-12

# ── Plasticité de von Mises ─────────────────────────────────────────────────

# ANCHOR: plasticite
fes, u = _un_qua4(2e-3)  # une maille, ε_xx = 2·10⁻³ : au-delà de σ_y/E
modele = pyrucast.model.plasticity_perfect(fes, "plane_stress")
materiaux = pyrucast.element_field.material_field(
    modele, [("E", 210e3), ("nu", 0.3), ("sigma_y", 250.0)]
)

eps = pyrucast.element_field.strain(modele, u, materiaux)
etat = pyrucast.element_field.integrate_behavior(
    modele, eps, materiaux
)  # prev=None : premier pas
s = {nom: etat[0].value(0, 0, nom) for nom in ("sigma_xx", "sigma_yy", "sigma_xy")}
q = math.sqrt(
    s["sigma_xx"] ** 2
    + s["sigma_yy"] ** 2
    - s["sigma_xx"] * s["sigma_yy"]
    + 3 * s["sigma_xy"] ** 2
)
print(round(q, 3))  # 250.0 : la contrainte équivalente est plafonnée à σ_y
print(etat[0].value(0, 0, "p") > 0)  # True : de la déformation plastique

# Le pas suivant repart de cet état.
etat = pyrucast.element_field.integrate_behavior(modele, eps, materiaux, prev=etat)
Kt = pyrucast.matrix.tangent(
    modele, materiaux, eps, prev=etat
)  # le module algorithmique
# ANCHOR_END: plasticite

assert abs(q - 250.0) < 1e-3
assert etat[0].value(0, 0, "p") > 0.0
assert Kt.symmetric

# ── Écrouissage isotrope ────────────────────────────────────────────────────

# ANCHOR: ecrouissage
modele = pyrucast.model.plasticity_isotropic(fes, "plane_stress")
materiaux = pyrucast.element_field.material_field(
    modele, [("E", 210e3), ("nu", 0.3), ("sigma_y", 250.0), ("H", 1000.0)]
)
etat = pyrucast.element_field.integrate_behavior(
    modele, pyrucast.element_field.strain(modele, u, materiaux), materiaux
)
s = {nom: etat[0].value(0, 0, nom) for nom in ("sigma_xx", "sigma_yy", "sigma_xy")}
q = math.sqrt(
    s["sigma_xx"] ** 2
    + s["sigma_yy"] ** 2
    - s["sigma_xx"] * s["sigma_yy"]
    + 3 * s["sigma_xy"] ** 2
)
print(round(q - (250.0 + 1000.0 * etat[0].value(0, 0, "p")), 3))  # 0.0 : q = σ_y + H·p
# ANCHOR_END: ecrouissage

assert abs(q - (250.0 + 1000.0 * etat[0].value(0, 0, "p"))) < 1e-3

# ── Drucker-Prager ──────────────────────────────────────────────────────────

# ANCHOR: drucker_prager
modele = pyrucast.model.drucker_prager(fes, "plane_strain")
materiaux = pyrucast.element_field.material_field(
    modele, [("E", 20e3), ("nu", 0.2), ("friction", 0.3), ("k", 10.0), ("psi", 0.1)]
)
eps = pyrucast.element_field.strain(modele, u, materiaux)
etat = pyrucast.element_field.integrate_behavior(modele, eps, materiaux)
Kt = pyrucast.matrix.tangent(modele, materiaux, eps)  # par perturbation, symétrisée
print(Kt.symmetric)  # True
# ANCHOR_END: drucker_prager

assert etat[0].value(0, 0, "p") > 0.0
assert Kt.symmetric

# ── Fluage de Norton ────────────────────────────────────────────────────────

# ANCHOR: fluage
modele = pyrucast.model.creep_norton(fes, "plane_strain")
materiaux = pyrucast.element_field.material_field(
    modele, [("E", 150e3), ("nu", 0.3), ("K", 400.0), ("n", 5.0)]
)
eps = pyrucast.element_field.strain(modele, u, materiaux)  # déformation maintenue
try:
    pyrucast.element_field.integrate_behavior(modele, eps, materiaux)
except RuntimeError as erreur:
    print(erreur)  # Plasticity: this law is rate-dependent and needs a time increment…

etat = pyrucast.element_field.integrate_behavior(modele, eps, materiaux, dt=1.0)
sigma_1 = etat[0].value(0, 0, "sigma_xx")
etat = pyrucast.element_field.integrate_behavior(
    modele, eps, materiaux, prev=etat, dt=1.0
)
print(etat[0].value(0, 0, "sigma_xx") < sigma_1)  # True : la contrainte se relaxe
# ANCHOR_END: fluage

assert etat[0].value(0, 0, "sigma_xx") < sigma_1

# ── Mazars ──────────────────────────────────────────────────────────────────

# ANCHOR: mazars
fes, u = _un_qua4(2e-4)  # ε_xx = 2·10⁻⁴, au-delà du seuil ε_d0 = 10⁻⁴
modele = pyrucast.model.mazars(fes, "plane_stress")
materiaux = pyrucast.element_field.material_field(
    modele,
    [
        ("E", 30e3),
        ("nu", 0.2),
        ("eps_d0", 1e-4),
        ("A_t", 0.8),
        ("B_t", 20e3),
        ("A_c", 1.4),
        ("B_c", 1900.0),
    ],
)
etat = pyrucast.element_field.integrate_behavior(
    modele, pyrucast.element_field.strain(modele, u, materiaux), materiaux
)
print(
    round(etat[0].value(0, 0, "kappa"), 6)
)  # 0.0002 : la plus grande déformation équivalente atteinte
print(0.0 < etat[0].value(0, 0, "damage") < 1.0)  # True
# ANCHOR_END: mazars

assert abs(etat[0].value(0, 0, "kappa") - 2e-4) < 1e-12
assert 0.0 < etat[0].value(0, 0, "damage") < 1.0

# ── Damage TC ───────────────────────────────────────────────────────────────

# ANCHOR: damage_tc
modele = pyrucast.model.damage_tc(fes, "plane_stress")
materiaux = pyrucast.element_field.material_field(
    modele,
    [("E", 30e3), ("nu", 0.2), ("f_t", 3.0), ("f_c", 30.0), ("A_t", 0.9), ("A_c", 0.5)],
)
etat = pyrucast.element_field.integrate_behavior(
    modele, pyrucast.element_field.strain(modele, u, materiaux), materiaux
)
print(
    etat[0].value(0, 0, "d_plus") > 0, etat[0].value(0, 0, "d_minus")
)  # True 0.0 : traction seule
# ANCHOR_END: damage_tc

assert etat[0].value(0, 0, "d_plus") > 0.0
assert etat[0].value(0, 0, "d_minus") == 0.0

# ── Poutres ─────────────────────────────────────────────────────────────────

# ANCHOR: bernoulli
# Une console de longueur 2 dans le plan, encastrée à gauche, F = −10 en y au bout.
c = pyrucast.Coords(2)
a = c.add_node([0.0, 0.0])
b = c.add_node([2.0, 0.0])
ligne = pyrucast.mesh.line(a, b, 2)
poutre = pyrucast.model.bernoulli(
    pyrucast.FiniteElementSpace(ligne, interpolation="HERMITE3")
)
print(poutre[0].primal_vars())  # ['u_x', 'u_y', 'r_z'] : un portique plan
encastrement = pyrucast.mesh.poi1_from_nodes([a])
modele = poutre
for v in ("u_x", "u_y", "r_z"):
    modele = modele | pyrucast.model.dirichlet(encastrement, poutre, v)
materiaux = pyrucast.element_field.material_field(
    modele, [("E", 200e3), ("A", 1e-2), ("I", 1e-4)]
)

force = pyrucast.NodeField(ligne, ["f_y"])
force.unit().set_value(b, "f_y", -10.0)
u = pyrucast.matrix.stiffness(modele, materiaux).solve(force)
print(round(u.value(b, "u_y"), 12))  # -1.333333333333 = F·L³/(3EI), exacte
# ANCHOR_END: bernoulli

assert abs(u.value(b, "u_y") + 10.0 * 8.0 / (3.0 * 200e3 * 1e-4)) < 1e-12

# ANCHOR: timoshenko
poutre = pyrucast.model.timoshenko(
    pyrucast.FiniteElementSpace(ligne, interpolation="MODEL_EMBEDDED")
)
modele = poutre
for v in ("u_x", "u_y", "r_z"):
    modele = modele | pyrucast.model.dirichlet(encastrement, poutre, v)
materiaux = pyrucast.element_field.material_field(
    modele, [("E", 200e3), ("A", 1e-2), ("I", 1e-4), ("G", 80e3), ("A_s", 5e-3)]
)
u = pyrucast.matrix.stiffness(modele, materiaux).solve(force)
print(round(u.value(b, "u_y"), 12))  # -1.383333333333 = F·L³/(3EI) + F·L/(G·A_s)
# ANCHOR_END: timoshenko

_exact = -10.0 * 8.0 / (3.0 * 200e3 * 1e-4) - 10.0 * 2.0 / (80e3 * 5e-3)
assert abs(u.value(b, "u_y") - _exact) < 1e-12

# ── Coques ──────────────────────────────────────────────────────────────────

# ANCHOR: coque
c = pyrucast.Coords(3)
n = [
    c.add_node(p)
    for p in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0])
]
surface = pyrucast.Mesh(c, "QUA4")
surface.unit().add_cell(n)
coque = pyrucast.model.shell(
    pyrucast.FiniteElementSpace(surface), "thick"
)  # ou "kirchhoff"
print(coque[0].primal_vars())  # ['u_x', 'u_y', 'u_z', 'r_x', 'r_y', 'r_z']
print(coque[0].material_components())  # ['E', 'nu', 'h']
materiaux = pyrucast.element_field.material_field(
    coque, [("E", 210e3), ("nu", 0.3), ("h", 0.01)]
)
K = pyrucast.matrix.stiffness(coque, materiaux)
print(K.n_rows)  # 24 : six inconnues par nœud
# ANCHOR_END: coque

assert coque[0].primal_vars() == ["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"]
assert K.n_rows == 24
