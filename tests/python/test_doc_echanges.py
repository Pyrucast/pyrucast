"""Source of the Python examples of `book/src/echanges.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show shifted by four spaces in the book. At module level, the
excerpt is exactly what a user would write.

Consequence: pytest runs this file at **collection** time, and an example that
breaks is a collection error, not a failing test.

Voir `book/src/developper/documentation-et-tests.md`.
"""

# ── Une source à gauche, un film à droite ───────────────────────────────────

# ANCHOR: echange_bord
import pyrucast

# Un carré [0, 1]² : ses bords gauche et droit sont les deux lignes balayées.
c = pyrucast.Coords(2)
gauche = pyrucast.mesh.line(c.add_node([0.0, 0.0]), c.add_node([0.0, 1.0]), 2)
droite = pyrucast.mesh.line(c.add_node([1.0, 0.0]), c.add_node([1.0, 1.0]), 2)
carre = pyrucast.mesh.sweep(gauche, droite, 2)

conduction = pyrucast.model.heat_conduction(pyrucast.FiniteElementSpace(carre))
source = pyrucast.model.flux(pyrucast.FiniteElementSpace(gauche), conduction, "q")
film = pyrucast.model.boundary_transfer(
    pyrucast.FiniteElementSpace(droite), conduction, ["T"]
)
print(source[0].physics(), film[0].physics())  # ['thermal'] ['thermal']
modele = conduction | source | film
materiaux = pyrucast.element_field.material_field(
    modele, [("k", 2.0), ("phi_q", 10.0), ("h_T", 5.0), ("a_ext_T", 20.0)]
)

second_membre = pyrucast.node_field.external_forces(
    modele, materiaux
)  # φ à gauche, h·T_ext à droite
T = pyrucast.matrix.stiffness(modele, materiaux).solve(second_membre)
print(round(T.value(droite.node(0, 0, 0), "T"), 9))  # 22.0 = T_ext + Q/h
print(round(T.value(gauche.node(0, 0, 0), "T"), 9))  # 27.0 = T_ext + Q/h + Q/k
# ANCHOR_END: echange_bord

assert source[0].physics() == film[0].physics() == ["thermal"]
assert abs(T.value(droite.node(0, 0, 0), "T") - 22.0) < 1e-9
assert abs(T.value(gauche.node(0, 0, 0), "T") - 27.0) < 1e-9

# ── Le même échange, sur des déplacements ───────────────────────────────────

# ANCHOR: fondation
plaque = pyrucast.model.elasticity(pyrucast.FiniteElementSpace(carre), "plane_stress")
semelle = pyrucast.model.boundary_transfer(
    pyrucast.FiniteElementSpace(gauche), plaque, ["u_x", "u_y"]
)
print(semelle[0].physics())  # ['mechanical']
print(semelle[0].material_components())  # ['h_u_x', 'h_u_y', 'a_ext_u_x', 'a_ext_u_y']
# ANCHOR_END: fondation

assert semelle[0].physics() == ["mechanical"]
assert semelle[0].material_components() == ["h_u_x", "h_u_y", "a_ext_u_x", "a_ext_u_y"]

# ── Deux carrés et un joint ─────────────────────────────────────────────────

# ANCHOR: interface
# Deux carrés côte à côte ; la ligne x = 1 existe deux fois, avec des nœuds distincts.
c = pyrucast.Coords(2)
x0 = pyrucast.mesh.line(c.add_node([0.0, 0.0]), c.add_node([0.0, 1.0]), 2)
x1_g = pyrucast.mesh.line(c.add_node([1.0, 0.0]), c.add_node([1.0, 1.0]), 2)
x1_d = pyrucast.mesh.line(c.add_node([1.0, 0.0]), c.add_node([1.0, 1.0]), 2)
x2 = pyrucast.mesh.line(c.add_node([2.0, 0.0]), c.add_node([2.0, 1.0]), 2)
corps = pyrucast.model.heat_conduction(
    pyrucast.FiniteElementSpace(pyrucast.mesh.sweep(x0, x1_g, 2))
) | pyrucast.model.heat_conduction(
    pyrucast.FiniteElementSpace(pyrucast.mesh.sweep(x1_d, x2, 2))
)
joint = pyrucast.model.interface_transfer(
    pyrucast.FiniteElementSpace(x1_g), pyrucast.FiniteElementSpace(x1_d), corps, ["T"]
)
source = pyrucast.model.flux(pyrucast.FiniteElementSpace(x0), corps, "q")
fixe = x2.to_poi1()  # les trois nœuds de x = 2
appui = pyrucast.model.dirichlet(fixe, corps, "T")
modele = corps | joint | source | appui
materiaux = pyrucast.element_field.material_field(
    modele, [("k", 1.0), ("h_T", 4.0), ("phi_q", 1.0)]
)

second_membre = pyrucast.node_field.external_forces(
    modele, materiaux
) | appui.constraint_rhs([(fixe.node(0, i, 0), 0.0) for i in range(3)])
T = pyrucast.matrix.stiffness(modele, materiaux).solve(second_membre)
saut = T.value(x1_g.node(0, 0, 0), "T") - T.value(x1_d.node(0, 0, 0), "T")
print(round(saut, 9))  # 0.25 = Q/h : le champ saute à la traversée
# ANCHOR_END: interface

assert abs(saut - 0.25) < 1e-9
assert abs(T.value(x0.node(0, 0, 0), "T") - 2.25) < 1e-9
