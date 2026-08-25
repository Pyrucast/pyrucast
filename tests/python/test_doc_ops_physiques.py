"""Source des exemples Python des pages d'opérateurs et de physiques.

Couvre `operateurs/{solveur,comportement,construction}.md`, `thermique.md`,
`diffusion.md` et `echanges.md`. Voir
`book/src/developper/documentation-et-tests.md`.

**Le code vit au niveau module, pas dans des fonctions de test** : mdbook
n'enlève pas l'indentation d'un extrait inclus, si bien qu'un bloc ancré dans
une fonction s'afficherait décalé de quatre espaces. pytest exécute donc ce
fichier à la **collecte** ; un exemple qui casse est une erreur de collecte, au
traceback complet et au code de retour non nul.
"""

import pyrucast


def _barre_thermique(n_elements=2, dim=1):
    """Une barre SEG2, son espace EF, et ses nœuds."""
    c = pyrucast.Coords(dim)
    noeuds = []
    for i in range(n_elements + 1):
        p = [0.0] * dim
        p[0] = i / n_elements
        noeuds.append(c.add_node(p))
    mesh = pyrucast.Mesh(c, "SEG2")
    for a, b in zip(noeuds, noeuds[1:]):
        mesh.unit().add_cell([a, b])
    return c, mesh, pyrucast.FiniteElementSpace(mesh), noeuds


def _modele_dirichlet(fes, noeud):
    """Conduction + une température imposée, et le nœud-multiplicateur."""
    imposed = pyrucast.mesh.poi1_from_nodes([noeud])
    mult = pyrucast.mesh.barycenter(imposed)
    model = pyrucast.model.heat_conduction(fes) | pyrucast.model.dirichlet(
        "T", "q", imposed, mult
    )
    return model, mult, mult.node(0, 0, 0)


def _chargement(mult_mesh, mult_node, valeur=1.0):
    rhs = pyrucast.NodeField(mult_mesh, ["imposed_T"])
    rhs[0].set_value(mult_node, "imposed_T", valeur)
    return rhs


def _plaque_2d():
    c = pyrucast.Coords(2)
    n = [c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0])]
    mesh = pyrucast.Mesh(c, "QUA4")
    mesh.unit().add_cell(n)
    return c, mesh, pyrucast.FiniteElementSpace(mesh), n


# ── Solveur ─────────────────────────────────────────────────────────────────


# ── solve et cache ─────────────────────────────────────────

_, _, fes, noeuds = _barre_thermique()
model, mult_mesh, mult_node = _modele_dirichlet(fes, noeuds[0])
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])
rhs = _chargement(mult_mesh, mult_node, 1.0)
autre_rhs = _chargement(mult_mesh, mult_node, 2.0)
some_node = noeuds[0]
# ANCHOR: solve
K = pyrucast.matrix.stiffness(model, materials)
solution = pyrucast.solver.solve(K, rhs)  # factorise puis résout
T = solution.value(some_node, "T")

# Résolutions ultérieures sur la MÊME matrice : la factorisation est réutilisée.
sol2 = pyrucast.solver.solve(K, autre_rhs)  # descente/remontée seulement
sol3 = pyrucast.solver.solve(
    K, autre_rhs, cache=False
)  # refactorise, sans toucher le cache
# ANCHOR_END: solve
assert abs(T - 1.0) < 1e-9
assert abs(sol2.value(some_node, "T") - sol3.value(some_node, "T")) < 1e-12

# ── solve eliminate ────────────────────────────────────────

_, _, fes, noeuds = _barre_thermique()
model, mult_mesh, mult_node = _modele_dirichlet(fes, noeuds[0])
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])
rhs = _chargement(mult_mesh, mult_node, 1.0)
# ANCHOR: eliminate
K = pyrucast.matrix.stiffness(model, materials)
lagrange = pyrucast.solver.solve(K, rhs)  # système augmenté
condense = pyrucast.solver.solve_eliminate(K, model, rhs)  # système réduit — même champ
# ANCHOR_END: eliminate
assert abs(lagrange.value(noeuds[0], "T") - condense.value(noeuds[0], "T")) < 1e-9

# ── solve unilateral ───────────────────────────────────────

_, _, fes, noeuds = _barre_thermique()
imposed = pyrucast.mesh.poi1_from_nodes([noeuds[0]])
mult = pyrucast.mesh.barycenter(imposed)
mult_node = mult.node(0, 0, 0)
model = pyrucast.model.heat_conduction(fes) | pyrucast.model.dirichlet(
    "T", "q", imposed, mult, sense=">="
)
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])
rhs = _chargement(mult, mult_node, 1.0)
# ANCHOR: unilateral
K = pyrucast.matrix.stiffness(model, materials)  # modèle avec sense=">="
solution = pyrucast.solver.solve_unilateral(K, model, rhs)  # "schur" par défaut
reaction = solution.value(mult_node, "lambda_T")  # 0 si la butée est relâchée

# Forcer l'ancienne méthode (refactorisation à chaque pas) :
sol2 = pyrucast.solver.solve_unilateral(K, model, rhs, active_set="refactorize")
# ANCHOR_END: unilateral
assert abs(solution.value(noeuds[0], "T") - sol2.value(noeuds[0], "T")) < 1e-9

# ── Comportement ────────────────────────────────────────────────────────────


# ── boucle pas a pas ───────────────────────────────────────

_, _, fes, n = _plaque_2d()
model = pyrucast.model.plasticity_perfect(fes, "plane_stress")
materials = pyrucast.element_field.material_field(
    model, [("E", 210_000.0), ("nu", 0.3), ("sigma_y", 250.0)]
)
u = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes(n), ["u_x", "u_y"])
nsteps = 2
# ANCHOR: pas_a_pas
state = None  # VAR0 = prev ; None au premier pas
for step in range(1, nsteps + 1):
    ...  # charge du pas → boucle de Newton sur u
    eps = pyrucast.element_field.deformation(u, fes)  # ε(B)
    out = pyrucast.element_field.integrate_behavior(model, eps, materials, prev=state)
    ...  # F_int (BSIG), résidu, correction de u
    state = out  # commit : prev ← VAR1 pour le pas suivant
# ANCHOR_END: pas_a_pas
assert len(state) == 1

# ── beam deformation ───────────────────────────────────────

_, mesh, _, noeuds = _barre_thermique(dim=2)
fes = pyrucast.FiniteElementSpace(mesh, interpolation="MODEL_EMBEDDED")
model = pyrucast.model.timoshenko(fes)
materials = pyrucast.element_field.material_field(
    model,
    [("E", 210_000.0), ("G", 80_000.0), ("A", 1e-2), ("I", 1e-4), ("A_s", 8e-3)],
)
solution = pyrucast.NodeField(
    pyrucast.mesh.poi1_from_nodes(noeuds), ["u_x", "u_y", "r_z"]
)
# ANCHOR: beam_deformation
# Solution (w, theta) déjà obtenue par le solveur.
eps = pyrucast.element_field.beam_deformation(solution, fes, materials)  # (κ, γ)
forces = pyrucast.element_field.integrate_behavior(model, eps, materials)
# forces porte le moment M = E·I·κ et l'effort tranchant V = G·A_s·γ.
# ANCHOR_END: beam_deformation
assert len(forces) == 1

# ── forces internes ────────────────────────────────────────

_, _, fes, n = _plaque_2d()
model = pyrucast.model.elasticity(fes, "plane_stress")
materials = pyrucast.element_field.material_field(model, [("E", 210.0), ("nu", 0.3)])
support = pyrucast.mesh.poi1_from_nodes(n)
solution = pyrucast.NodeField(support, ["u_x", "u_y"])
f_ext = pyrucast.NodeField(support, ["f_x", "f_y"])
# ANCHOR: forces_internes
# Solution déjà obtenue par le solveur.
eps = pyrucast.element_field.deformation(solution, fes)  # ε = B·u
sig = pyrucast.element_field.integrate_behavior(model, eps, materials)  # COMP : σ
f_int = pyrucast.node_field.internal_forces(sig, model)  # BSIG : ∫ Bᵀ σ
residu = f_ext - f_int  # équilibre
# ANCHOR_END: forces_internes
assert residu.node_count() > 0

# ── Construction du champ matériau ──────────────────────────────────────────


def _deux_modeles():
    """Un modèle thermique et un modèle élastique, montés hors de l'ancre."""
    _, _, fes, _ = _plaque_2d()
    return (
        pyrucast.model.heat_conduction(fes),
        pyrucast.model.elasticity(fes, "plane_stress"),
    )


# ── material field ─────────────────────────────────────────

thermique, elastique = _deux_modeles()
# ANCHOR: material_field
import pyrucast

# Thermique : conductivité uniforme.
materials = pyrucast.element_field.material_field(thermique, [("k", 1.0)])

# Élasticité : deux propriétés. Chaque physique déclare les composantes
# qu'elle exige : `material_field` refuse celles qui manquent.
materials = pyrucast.element_field.material_field(
    elastique, [("E", 210e9), ("nu", 0.3)]
)
# ANCHOR_END: material_field
assert len(materials) == 1

# ── Thermique ───────────────────────────────────────────────────────────────


# ── orthotrope ─────────────────────────────────────────────

import math

_, _, fes, _ = _plaque_2d()
cos_a, sin_a = math.cos(0.3), math.sin(0.3)
# ANCHOR: orthotrope
model = pyrucast.model.heat_conduction(fes, symmetry="orthotropic")
materials = pyrucast.element_field.material_field(
    model,
    [("k_1", 12.0), ("k_2", 3.0), ("k_3", 12.0), ("V1X", cos_a), ("V1Y", sin_a)],
)
# ANCHOR_END: orthotrope
assert len(materials) == 1

# ── boundary transfer ──────────────────────────────────────

_, _, bord_fes, _ = _barre_thermique()
# ANCHOR: boundary_transfer
pyrucast.model.boundary_transfer(bord_fes, [("T", "q")], "thermal")
# ANCHOR_END: boundary_transfer

# ── Diffusion ───────────────────────────────────────────────────────────────


# ── fick couple ────────────────────────────────────────────

_, _, fes, _ = _plaque_2d()
# ANCHOR: fick
model = pyrucast.model.fick(fes, "H2") | pyrucast.model.heat_conduction(fes)
materials = pyrucast.element_field.material_field(model, [("D_H2", 2.0), ("k", 5.0)])
k = pyrucast.matrix.stiffness(model, materials)

len(model.filter("diffusion"))  # 1
len(model.filter("thermal"))  # 1
# ANCHOR_END: fick
assert len(model.filter("diffusion")) == 1
assert len(model.filter("thermal")) == 1


def _plaque_et_bord():
    """Une plaque QUA4 et l'espace EF de son bord gauche."""
    c, mesh, fes, n = _plaque_2d()
    bord_mesh = pyrucast.Mesh(c, "SEG2")
    bord_mesh.unit().add_cell([n[0], n[3]])
    return fes, pyrucast.FiniteElementSpace(bord_mesh), c, n


# ── radiation ──────────────────────────────────────────────

volume, bord, _, _ = _plaque_et_bord()
# ANCHOR: radiation
model = pyrucast.model.heat_conduction(volume) | pyrucast.model.radiation(bord)
materials = pyrucast.element_field.material_field(
    model, [("k", 20.0), ("emis", 0.8), ("T_inf", 300.0)]
)
# ANCHOR_END: radiation
assert len(model) == 2


def _deux_corps_en_vis_a_vis():
    """Deux plaques accolées, et l'espace EF de leur face commune de chaque côté."""
    c = pyrucast.Coords(2)
    g = [c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0])]
    d = [c.add_node(p) for p in ([1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0])]
    mg, md = pyrucast.Mesh(c, "QUA4"), pyrucast.Mesh(c, "QUA4")
    mg.unit().add_cell(g)
    md.unit().add_cell(d)
    fg, fd = pyrucast.Mesh(c, "SEG2"), pyrucast.Mesh(c, "SEG2")
    fg.unit().add_cell([g[1], g[2]])
    fd.unit().add_cell([d[0], d[3]])
    return (
        pyrucast.FiniteElementSpace(mg),
        pyrucast.FiniteElementSpace(md),
        pyrucast.FiniteElementSpace(fg),
        pyrucast.FiniteElementSpace(fd),
    )


# ── interface transfer ─────────────────────────────────────

gauche, droite, face_gauche, face_droite = _deux_corps_en_vis_a_vis()
# ANCHOR: interface_transfer
model = (
    pyrucast.model.fick(gauche, "H2")
    | pyrucast.model.fick(droite, "H2")
    | pyrucast.model.interface_transfer(
        face_gauche, face_droite, [("c_H2", "j_H2")], "diffusion"
    )
)
materials = pyrucast.element_field.material_field(
    model, [("D_H2", 2.0), ("h_c_H2", 5.0)]
)
# ANCHOR_END: interface_transfer
assert len(model) == 3

# ── echanges ───────────────────────────────────────────────

fes, peau, _, _ = _plaque_et_bord()
_, _, face_gauche, face_droite = _deux_corps_en_vis_a_vis()
semelle = peau
# ANCHOR: echanges
# Film thermique : entre dans la raideur d'une conduction.
model = pyrucast.model.heat_conduction(fes) | pyrucast.model.boundary_transfer(
    peau, [("T", "q")], "thermal"
)
materials = pyrucast.element_field.material_field(model, [("k", 5.0), ("h_T", 12.0)])

# Résistance de contact entre deux maillages.
joint = pyrucast.model.interface_transfer(
    face_gauche, face_droite, [("T", "q")], "thermal"
)

# Fondation élastique : la même loi, sur des déplacements.
appui = pyrucast.model.boundary_transfer(
    semelle, [("u_x", "f_x"), ("u_y", "f_y")], "mechanical"
)
# ANCHOR_END: echanges
assert len(model) == 2 and len(joint) == 1 and len(appui) == 1
