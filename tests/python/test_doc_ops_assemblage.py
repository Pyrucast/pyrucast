"""Source des exemples de `book/src/operateurs/assemblage.md`.

Chaque section de la page tire son bloc d'ici par `{{#include …:ancre}}` : la page ne peut plus montrer un appel qui a cessé d'exister.
Voir `book/src/developper/documentation-et-tests.md`.

Le montage vit **hors** des ancres — le lecteur du chapitre n'a pas besoin de
revoir la construction d'un maillage à chaque opérateur.

**Le code vit au niveau module, pas dans des fonctions de test** : mdbook
n'enlève pas l'indentation d'un extrait inclus, si bien qu'un bloc ancré dans
une fonction s'afficherait décalé de quatre espaces. pytest exécute donc ce
fichier à la **collecte** ; un exemple qui casse est une erreur de collecte, au
traceback complet et au code de retour non nul.
"""

import pyrucast


def _modele_thermique():
    """Barre 1-D à deux SEG2, conduction, Dirichlet à gauche."""
    c = pyrucast.Coords(1)
    noeuds = [c.add_node([x]) for x in (0.0, 0.5, 1.0)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for a, b in zip(noeuds, noeuds[1:]):
        mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)

    imposed = pyrucast.mesh.poi1_from_nodes([noeuds[0]])
    multiplier = pyrucast.mesh.barycenter(imposed)
    cible = pyrucast.model.heat_conduction(fes)

    model = cible | pyrucast.model.dirichlet(cible, "T", imposed, multiplier)
    return model, fes, multiplier, noeuds


def _modele_mecanique():
    """Plaque QUA4 unique en contraintes planes."""
    c = pyrucast.Coords(2)
    n = [c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0])]
    mesh = pyrucast.Mesh(c, "QUA4")
    mesh.unit().add_cell(n)
    fes = pyrucast.FiniteElementSpace(mesh)
    model = pyrucast.model.elasticity(fes, "plane_stress")
    return model, fes, mesh, n


# ── Rigidité ────────────────────────────────────────────────────────────────


# ── stiffness ──────────────────────────────────────────────

model, _, _, _ = _modele_thermique()
# ANCHOR: stiffness
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])
K = pyrucast.matrix.stiffness(model, materials)
print(K)  # Matrix: n row(s) × n col(s), …
# ANCHOR_END: stiffness
assert K.n_rows() == K.n_cols()
assert K.n_rows() > 0

# ── Masse, et sa version diagonale ──────────────────────────────────────────


# ── mass ───────────────────────────────────────────────────

model, _, _, _ = _modele_mecanique()
# ANCHOR: mass
materials = pyrucast.element_field.material_field(
    model, [("E", 210.0), ("nu", 0.3), ("rho", 7800.0)]
)
M = pyrucast.matrix.mass(model, materials)
# ANCHOR_END: mass
assert M.n_rows() == M.n_cols()

# ── lump ───────────────────────────────────────────────────

model, _, _, _ = _modele_mecanique()
materials = pyrucast.element_field.material_field(
    model, [("E", 210.0), ("nu", 0.3), ("rho", 7800.0)]
)
# ANCHOR: lump
M = pyrucast.matrix.mass(model, materials)
M_lumped = pyrucast.matrix.lump(M)  # diagonale
# ANCHOR_END: lump
assert M_lumped.n_rows() == M.n_rows()

# ── Composition de matrices ─────────────────────────────────────────────────


# ── assemble apres ajout de bloc ───────────────────────────

model, _, _, _ = _modele_thermique()
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])
k_ref = pyrucast.matrix.stiffness(model, materials)
bloc_supplementaire = pyrucast.matrix.stiffness(model, materials)[0]
# ANCHOR: assemble
k = pyrucast.matrix.stiffness(model, materials)
k.add_sub(bloc_supplementaire)
k.assemble()
# ANCHOR_END: assemble
assert k.n_rows() == k_ref.n_rows()

# ── somme de matrices ──────────────────────────────────────

model, _, multiplier, _ = _modele_thermique()
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])
k = pyrucast.matrix.stiffness(model, materials)
m = pyrucast.matrix.stiffness(model, materials)
dt = 0.1
# Chargement : la température imposée, portée par le nœud-multiplicateur.
rhs = pyrucast.NodeField(multiplier, ["imposed_T"])
rhs[0].set_value(multiplier.node(0, 0, 0), "imposed_T", 1.0)
# ANCHOR: somme
sys = (m / dt) | k
sys.assemble()
u = pyrucast.solver.solve(sys, rhs)
# ANCHOR_END: somme
assert u.node_count() > 0

# ── Rigidité géométrique ────────────────────────────────────────────────────


# ── geometric ──────────────────────────────────────────────

model, fes, _, _ = _modele_mecanique()
materials = pyrucast.element_field.material_field(model, [("E", 1.0), ("nu", 0.3)])
stress = pyrucast.ElementField(fes, ["sigma_xx", "sigma_yy", "sigma_xy"])
stress[0].set_uniform("sigma_xx", 3.0)
stress[0].set_uniform("sigma_yy", 0.0)
stress[0].set_uniform("sigma_xy", 0.0)
# ANCHOR: geometric
Kg = pyrucast.matrix.geometric(model, materials, stress)
# ANCHOR_END: geometric
assert Kg.n_rows() == Kg.n_cols()

# ── Tangente cohérente ──────────────────────────────────────────────────────


# ── tangent ────────────────────────────────────────────────

_, fes, _, noeuds = _modele_mecanique()
model = pyrucast.model.plasticity_perfect(fes, "plane_strain")
materials = pyrucast.element_field.material_field(
    model, [("E", 70_000.0), ("nu", 0.3), ("sigma_y", 200.0)]
)
# Déplacement d'essai : un étirement uniforme selon x.
u_mesh = pyrucast.mesh.poi1_from_nodes(noeuds)
u = pyrucast.NodeField(u_mesh, ["u_x", "u_y"])
for noeud, x in zip(noeuds, [0.0, 1.0, 1.0, 0.0]):
    u[0].set_value(noeud, "u_x", 1e-5 * x)
    u[0].set_value(noeud, "u_y", 0.0)
# ANCHOR: tangent
strain = pyrucast.element_field.deformation(u, fes)
Kt = pyrucast.matrix.tangent(model, materials, strain)
# ANCHOR_END: tangent
assert Kt.n_rows() == Kt.n_cols()

# ── Chargement réparti ──────────────────────────────────────────────────────


# ── flux ───────────────────────────────────────────────────

c = pyrucast.Coords(1)
a, b = c.add_node([0.0]), c.add_node([1.0])
edge = pyrucast.Mesh(c, "SEG2")
edge.unit().add_cell([a, b])
edge_fes = pyrucast.FiniteElementSpace(edge)
Q = 2.0
# Un autre chargement, porté par un nœud distinct : l'union les juxtapose.
ailleurs = c.add_node([2.0])
other_loads = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes([ailleurs]), ["q"])
# ANCHOR: flux
# Flux uniforme Q sur le bord gauche (maillage SEG2), versé dans la ligne duale
# « q » du modèle chargé — c'est lui qui possède cette ligne et qui donne sa
# nature à la charge. Celle-ci est un sous-modèle : sa densité vit dans le
# matériau, sous le nom « phi_q », et son terme se demande au modèle.
conduction = pyrucast.model.heat_conduction(edge_fes)
charge = pyrucast.model.flux(edge_fes, conduction, "q")
densite = pyrucast.element_field.material_field(charge, [("phi_q", Q)])
load = pyrucast.node_field.external_forces(charge, densite)
rhs = load | other_loads
# ANCHOR_END: flux
assert rhs.node_count() > 0
