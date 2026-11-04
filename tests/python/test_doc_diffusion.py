"""Source of the Python examples of `book/src/diffusion.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show shifted by four spaces in the book. At module level, the
excerpt is exactly what a user would write.

Consequence: pytest runs this file at **collection** time, and an example that
breaks is a collection error, not a failing test.

Voir `book/src/developper/documentation-et-tests.md`.
"""

# ── Une espèce dans une barre ───────────────────────────────────────────────

# ANCHOR: fick
import pyrucast

# Une barre [0, 2] : un flux d'hydrogène de 1 entre à gauche, c = 0 à droite.
c = pyrucast.Coords(1)
gauche = c.add_node([0.0])
droite = c.add_node([2.0])
barre = pyrucast.mesh.line(gauche, droite, 4)
fes = pyrucast.FiniteElementSpace(barre)

fick = pyrucast.model.fick(fes, "H2")
print(fick[0].primal_vars(), fick[0].dual_vars())  # ['c_H2'] ['j_H2']
appui = pyrucast.model.dirichlet(pyrucast.mesh.poi1_from_nodes([droite]), fick, "c_H2")
modele = fick | appui
materiaux = pyrucast.element_field.material_field(modele, [("D_H2", 0.5)])

source = pyrucast.NodeField(barre, ["j_H2"])
source.unit().set_value(gauche, "j_H2", 1.0)
c_h2 = pyrucast.matrix.stiffness(modele, materiaux).solve(
    source | appui.constraint_rhs([(droite, 0.0)])
)
print(round(c_h2.value(gauche, "c_H2"), 9))  # 4.0 = j·L/D
print(
    round(c_h2.value(appui.multiplier_mesh().node(0, 0, 0), "lambda"), 9)
)  # 1.0 : le flux qui sort

gradient = pyrucast.element_field.strain(modele, c_h2, materiaux)
repos = pyrucast.element_field.initial_state(
    modele, materiaux
)  # l'état de départ de la loi
flux = pyrucast.element_field.integrate_behavior(modele, gradient, materiaux, repos)
print(flux.components(), flux[0].value(0, 0, "j_H2_x"))  # ['j_H2_x'] -1.0 = D·∂c/∂x
# ANCHOR_END: fick

assert abs(c_h2.value(gauche, "c_H2") - 4.0) < 1e-12
assert abs(c_h2.value(appui.multiplier_mesh().node(0, 0, 0), "lambda") - 1.0) < 1e-12
assert flux.components() == ["j_H2_x"]
assert abs(flux[0].value(0, 0, "j_H2_x") + 1.0) < 1e-12

# ── Plusieurs espèces, et la conduction, sur un maillage ────────────────────

# ANCHOR: especes
melange = (
    pyrucast.model.fick(fes, "H2")
    | pyrucast.model.fick(fes, "O2")
    | pyrucast.model.heat_conduction(fes)
)
print(melange.primal_vars())  # ['c_H2', 'c_O2', 'T']
print(len(melange.filter("diffusion")), len(melange.filter("thermal")))  # 2 1
materiaux = pyrucast.element_field.material_field(
    melange, [("D_H2", 0.5), ("D_O2", 0.1), ("k", 2.0)]
)
K = pyrucast.matrix.stiffness(melange, materiaux)  # trois blocs, sans couplage
# ANCHOR_END: especes

assert melange.primal_vars() == ["c_H2", "c_O2", "T"]
assert (len(melange.filter("diffusion")), len(melange.filter("thermal"))) == (2, 1)
assert len(K) == 3
