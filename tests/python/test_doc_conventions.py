"""Source of the Python examples of `book/src/conventions.md` and
`thermomecanique-pas-a-pas.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero exit code.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import pyrucast

# ── Montage commun ──────────────────────────────────────────────────────────

_c = pyrucast.Coords(3)
_base = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
_n = [_c.add_node([x, y, z]) for z in (0.0, 1.0) for x, y, _ in _base]
# A HEX8: `skin()` needs a volume to return a skin.
maillage = pyrucast.Mesh(_c, "HEX8")
maillage.unit().add_cell(_n)

_surface = pyrucast.Mesh(_c, "QUA4")
_surface.unit().add_cell(_n[:4])
mesh = _surface
fes = pyrucast.FiniteElementSpace(mesh)
_support = pyrucast.mesh.poi1_from_nodes(_n)
u = pyrucast.NodeField(_support, ["u_x", "u_y", "u_z"])
champ = pyrucast.NodeField(_support, ["v"])
a = pyrucast.NodeField(_support, ["v"])
b = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes(_n[:2]), ["w"])
_imposed = pyrucast.mesh.poi1_from_nodes([_n[0]])
cible = pyrucast.model.heat_conduction(fes)
_appui = pyrucast.model.dirichlet(_imposed, cible, "T")
_mult = _appui.multiplier_mesh()

model = cible | _appui
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])
rhs = pyrucast.NodeField(_mult, ["imposed"])
rhs[0].set_value(_mult.node(0, 0, 0), "imposed", 1.0)


# ── Le verbe exposé aussi en méthode ────────────────────────────────────────

# ANCHOR: chainage
# chaining, when the three conditions hold:
peau = maillage.skin().consolidate()
libre = champ.select(ge=0.0)
eps = u.gradient(fes)

# forme canonique seule, sinon :
eps = pyrucast.element_field.deformation(u, fes)  # exige un déplacement
f = pyrucast.node_field.merge(a, b)  # symétrique : `a | b` suffit
# ANCHOR_END: chainage

assert f.components() == ["v", "w"]


# ── Le miroir Python ────────────────────────────────────────────────────────

# ANCHOR: miroir
import pyrucast

# functions (operators), laid out by produced container — not methods:
poi = pyrucast.mesh.to_poi1(mesh)
coords = pyrucast.node_field.positions(mesh)
eps = pyrucast.element_field.deformation(u, fes)
K = pyrucast.matrix.stiffness(model, materials)
sol = pyrucast.solver.solve(K, rhs)
# ANCHOR_END: miroir

assert sol.node_count() > 0


# ── Erreurs ─────────────────────────────────────────────────────────────────

# ANCHOR: erreurs
import pyrucast

try:
    c = pyrucast.Coords(0)  # dimension nulle
except RuntimeError as e:
    print(f"erreur : {e}")  # erreur : dim must be ≥ 1
# ANCHOR_END: erreurs


# ── Affichage : Debug vs Display ────────────────────────────────────────────

# ANCHOR: affichage
import pyrucast

c = pyrucast.Coords(dim=2)
c.add_node([0.0, 0.0])

print(repr(c))  # vue structurelle — __repr__
print(str(c))  # vue résumée cast3m — __str__
print(c)  # same as str(c)
# ANCHOR_END: affichage


# ── Thermo-mechanics: a result's keys ───────────────────────────────────────
