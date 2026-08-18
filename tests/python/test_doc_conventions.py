"""Source des exemples Python de `book/src/conventions.md` et
`thermomecanique-pas-a-pas.md`.

**Le code vit au niveau module, pas dans des fonctions de test** : mdbook
n'enlève pas l'indentation d'un extrait inclus. pytest exécute donc ce fichier
à la **collecte** ; un exemple qui casse est une erreur de collecte, au
traceback complet et au code de retour non nul.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import pyrucast

# ── Montage commun ──────────────────────────────────────────────────────────

_c = pyrucast.Coords(3)
_base = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
_n = [_c.add_node([x, y, z]) for z in (0.0, 1.0) for x, y, _ in _base]
# Un HEX8 : `skin()` a besoin d'un volume pour rendre une peau.
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
_mult = pyrucast.mesh.barycenter(_imposed)
model = pyrucast.Model.heat_conduction(fes) | pyrucast.Model.dirichlet(
    "T", "q", _imposed, _mult
)
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])
rhs = pyrucast.NodeField(_mult, ["imposed_T"])
rhs[0].set_value(_mult.node(0, 0, 0), "imposed_T", 1.0)


# ── Le verbe exposé aussi en méthode ────────────────────────────────────────

# ANCHOR: chainage
# chaînage, quand les trois conditions tiennent :
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

# fonctions (opérateurs), rangées par conteneur produit — pas des méthodes :
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
print(c)  # même chose que str(c)
# ANCHOR_END: affichage


# ── Thermo-mécanique : les clés d'un résultat ───────────────────────────────

# ANCHOR: cles_resultat
{
    "time",
    "temperature",
    "displacement",
    "state",
    "mech_iters",
    "mech_anderson",
    "converged",
}
# ANCHOR_END: cles_resultat
