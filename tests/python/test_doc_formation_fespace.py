"""Source des exemples Python de `formation/langage-python.md` et `fe-space.md`.

**Le code vit au niveau module, pas dans des fonctions de test** : mdbook
n'enlève pas l'indentation d'un extrait inclus, si bien qu'un bloc ancré dans
une fonction s'afficherait décalé de quatre espaces. pytest exécute donc ce
fichier à la **collecte** ; un exemple qui casse est une erreur de collecte, au
traceback complet et au code de retour non nul.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import pyrucast

# ── Le module et ses sous-modules ───────────────────────────────────────────

# ANCHOR: import
import pyrucast as pc
# ANCHOR_END: import

assert hasattr(pc, "mesh")


# ── Les trois objets de départ ──────────────────────────────────────────────

# ANCHOR: objets
c = pc.Coords(2)  # Cast3M : OPTI 'DIME' 2
n = c.add_node([0.0, 0.0])  # Cast3M : POIN 0. 0. ;
mesh = pc.Mesh(c, "TRI3")  # Cast3M : MAILLAGE (implicite via un opérateur)
# ANCHOR_END: objets

assert mesh.element_types() == ["TRI3"]


# ── Composer un modèle ──────────────────────────────────────────────────────

_c = pyrucast.Coords(2)
_n = [_c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0])]
_volume = pyrucast.Mesh(_c, "TRI3")
_volume.unit().add_cell(_n)
fes = pyrucast.FiniteElementSpace(_volume)
_bord = pyrucast.Mesh(_c, "SEG2")
_bord.unit().add_cell([_n[0], _n[1]])
bord_fes = pyrucast.FiniteElementSpace(_bord)
impose = pyrucast.mesh.poi1_from_nodes([_n[2]])
multiplicateur = pyrucast.mesh.barycenter(impose)

# ANCHOR: composer
modele = pc.Model.heat_conduction(fes) | pc.Model.boundary_transfer(
    bord_fes, [("T", "q")], "thermal"
)
modele = modele | pc.Model.dirichlet("T", "q", impose, multiplicateur)
# ANCHOR_END: composer

assert len(modele) == 3


# ── Les conditions limites, deux physiques ──────────────────────────────────

# ANCHOR: bloquer
pc.Model.dirichlet("T", "q", impose, multiplicateur)  # Cast3M : BLOQ 'T' ...
pc.Model.dirichlet("u_x", "f_x", impose, multiplicateur)  # Cast3M : BLOQ 'UX' ...
# ANCHOR_END: bloquer


# ── Espace éléments finis : le constructeur par défaut ──────────────────────

# ANCHOR: fespace
import pyrucast

c = pyrucast.Coords(dim=2)
n0 = c.add_node([0.0, 0.0])
n1 = c.add_node([2.0, 0.0])
n2 = c.add_node([0.0, 2.0])

mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([n0, n1, n2])

# Constructeur par défaut : Lagrange1 + Gauss partout.
fes = pyrucast.FiniteElementSpace(mesh)
assert len(fes) == 1  # 1 sous-espace = 1 sous-maillage
sub = fes[0]  # vue typée du sous-espace 0
assert sub.element_type == "TRI3"
assert sub.interpolation == "LAGRANGE1"
assert sub.quadrature == "GAUSS"
assert sub.gauss_count() == 3
assert sub.space_dim == 2
assert sub.ref_dim == 2

# Évaluations à un point de Gauss donné.
for g in range(sub.gauss_count()):
    print(sub.gauss_xi(g), sub.gauss_weight(g))
    print(sub.n_at_g(g))  # N_i(ξ_g), flat
    print(sub.dn_at_g(g))  # ∂N_i/∂ξ_j(ξ_g), flat

# Grandeurs physiques (à la volée) sur la cellule 0.
print(sub.jacobian(0, 0))  # J, flat row-major
print(sub.det_jacobian(0, 0))  # |J|, scalaire
print(sub.dn_dx(0, 0))  # ∂N_i/∂x_a, flat row-major
# ANCHOR_END: fespace


# ── Les autres constructeurs ────────────────────────────────────────────────

# ANCHOR: fespace_variantes
# Même Lagrange1 + même Gauss pour tous les sous-maillages, explicite.
fes = pyrucast.FiniteElementSpace(mesh, interpolation="LAGRANGE1", quadrature="GAUSS")

# Forme « class method » équivalente au constructeur par défaut.
fes = pyrucast.FiniteElementSpace.lagrange1(mesh)

# (Interpolation, quadrature) explicites par sous-maillage.
fes = pyrucast.FiniteElementSpace.with_choices(mesh, [("LAGRANGE1", "GAUSS")])
# ANCHOR_END: fespace_variantes

assert len(fes) == 1


# ── Déplacement de maillage : les évaluations suivent ───────────────────────

# ANCHOR: deplacement
print(sub.det_jacobian(0, 0))  # |J| initial

# Déplacement d'un nœud → toutes les évaluations à venir voient les
# nouvelles coordonnées.
n1.set_position([4.0, 0.0])
print(sub.det_jacobian(0, 0))  # |J| recalculé
# ANCHOR_END: deplacement

assert sub.det_jacobian(0, 0) == 8.0
