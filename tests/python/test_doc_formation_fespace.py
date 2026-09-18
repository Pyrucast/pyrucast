"""Source of the Python examples of `formation/langage-python.md` and `fe-space.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show up shifted by four spaces. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero exit code.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import pyrucast

# ── The module and its submodules ───────────────────────────────────────────

# ANCHOR: import
import pyrucast as pc
# ANCHOR_END: import

assert hasattr(pc, "mesh")


# ── The three starting objects ──────────────────────────────────────────────

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
conduction = pc.model.heat_conduction(fes)
modele = conduction | pc.model.boundary_transfer(bord_fes, conduction, [("T", "q")])
modele = modele | pc.model.dirichlet(modele, "T", impose, multiplicateur)
# ANCHOR_END: composer

assert len(modele) == 3


# ── The boundary conditions, two physics ────────────────────────────────────

# ANCHOR: bloquer
pc.model.dirichlet(modele, "T", impose, multiplicateur)  # Cast3M : BLOQ 'T' ...
mecanique = pc.model.elasticity(fes, "plane_stress")
pc.model.dirichlet(mecanique, "u_x", impose, multiplicateur)  # Cast3M : BLOQ 'UX' ...
# ANCHOR_END: bloquer


# ── Finite element space: the default constructor ───────────────────────────

# ANCHOR: fespace
import pyrucast

c = pyrucast.Coords(dim=2)
n0 = c.add_node([0.0, 0.0])
n1 = c.add_node([2.0, 0.0])
n2 = c.add_node([0.0, 2.0])

mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([n0, n1, n2])

# Default constructor: Lagrange1 + Gauss everywhere.
fes = pyrucast.FiniteElementSpace(mesh)
assert len(fes) == 1  # 1 sous-espace = 1 sous-maillage
sub = fes[0]  # typed view of subspace 0
assert sub.element_type == "TRI3"
assert sub.interpolation == "LAGRANGE1"
assert sub.quadrature == "GAUSS"
assert sub.gauss_count() == 3
assert sub.space_dim == 2
assert sub.ref_dim == 2

# Evaluations at a given Gauss point.
for g in range(sub.gauss_count()):
    print(sub.gauss_xi(g), sub.gauss_weight(g))
    print(sub.n_at_g(g))  # N_i(ξ_g), flat
    print(sub.dn_at_g(g))  # ∂N_i/∂ξ_j(ξ_g), flat

# Physical quantities (on the fly) on cell 0.
print(sub.jacobian(0, 0))  # J, flat row-major
print(sub.det_jacobian(0, 0))  # |J|, scalaire
print(sub.dn_dx(0, 0))  # ∂N_i/∂x_a, flat row-major
# ANCHOR_END: fespace


# ── The other constructors ──────────────────────────────────────────────────

# ANCHOR: fespace_variantes
# Same Lagrange1 + same Gauss for every submesh, explicitly.
fes = pyrucast.FiniteElementSpace(mesh, interpolation="LAGRANGE1", quadrature="GAUSS")

# "Class method" form equivalent to the default constructor.
fes = pyrucast.FiniteElementSpace.lagrange1(mesh)

# (Interpolation, quadrature) explicit per submesh.
fes = pyrucast.FiniteElementSpace.with_choices(mesh, [("LAGRANGE1", "GAUSS")])
# ANCHOR_END: fespace_variantes

assert len(fes) == 1


# ── Moving the mesh: the evaluations follow ─────────────────────────────────

# ANCHOR: deplacement
print(sub.det_jacobian(0, 0))  # |J| initial

# Moving a node → every evaluation to come sees the
# nouvelles coordonnées.
n1.set_position([4.0, 0.0])
print(sub.det_jacobian(0, 0))  # |J| recalculé
# ANCHOR_END: deplacement

assert sub.det_jacobian(0, 0) == 8.0
