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
cible = pyrucast.model.heat_conduction(fes)

model = cible | pyrucast.model.dirichlet(cible, "T", _imposed, _mult)
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


# ── Thermo-mécanique pas-à-pas : la mise en donnée complète ─────────────────


def _modele_thermomecanique():
    """Une plaque 2×2 QUA4 : conduction + élasticité + appuis, et son histoire
    de température. Le montage complet est dans `examples/thermomechanique_pas_a_pas.py`."""
    import pyrucast as pc

    nx = ny = 2
    c = pc.Coords(2)
    grid = [c.add_node([i / nx, j / ny]) for j in range(ny + 1) for i in range(nx + 1)]

    def idx(i, j):
        return i + j * (nx + 1)

    mesh = pc.Mesh(c, "QUA4")
    for j in range(ny):
        for i in range(nx):
            mesh.unit().add_cell(
                [
                    grid[idx(i, j)],
                    grid[idx(i + 1, j)],
                    grid[idx(i + 1, j + 1)],
                    grid[idx(i, j + 1)],
                ]
            )
    fes = pc.FiniteElementSpace(mesh)

    gauche = [grid[idx(0, j)] for j in range(ny + 1)]
    droite = [grid[idx(nx, j)] for j in range(ny + 1)]
    bas = [grid[idx(i, 0)] for i in range(nx + 1)]

    def bloquer(target, noeuds, var):
        imposed = pc.mesh.poi1_from_nodes(noeuds)
        return pc.model.dirichlet(target, var, imposed, pc.mesh.barycenter(imposed))

    th_imposed = pc.mesh.poi1_from_nodes(gauche + droite)
    th_mult = pc.mesh.translate(th_imposed, [0.0, 0.0])
    thermal = pc.model.heat_conduction(fes)
    mecanique = pc.model.elasticity(fes, "plane_stress")
    model = (
        thermal
        | mecanique
        | pc.model.dirichlet(thermal, "T", th_imposed, th_mult)
        | bloquer(mecanique, gauche, "u_x")
        | bloquer(mecanique, bas, "u_y")
    )
    materials = pc.element_field.material_field(
        model, [("k", 1.0), ("E", 210_000.0), ("nu", 0.3), ("alpha", 1.2e-5)]
    )
    froid = pc.NodeField(th_mult, ["imposed_T"])
    froid[0].add_to_component("imposed_T", 20.0)
    chaud = pc.NodeField(th_mult, ["imposed_T"])
    chaud[0].add_to_component("imposed_T", 120.0)
    loads = pc.Evolution([(0.0, froid), (1.0, chaud)], out_of_range="clamp")
    return model, materials, loads


model, materials, loads = _modele_thermomecanique()

# ANCHOR: step_by_step
import pyrucast as pc

# … maillage `mesh`, `fes`, modèle thermo-mécanique `model`, `materials`, `loads` …

data = {
    "times": [0.0, 0.25, 0.5, 0.75, 1.0],
    "model": model,  # fespace + maillage déduits du modèle
    "loads": loads,  # NodeField unioné ou Evolution de champ
    "materials": materials,  # ElementField unioné ou Evolution de champ
    "t_ref": 20.0,
}

pc.thermomechanics.step_by_step(data)

for r in data["results"]:
    print(r["time"], r["mech_iters"], r["converged"])
# ANCHOR_END: step_by_step

assert len(data["results"]) == 5
assert all(r["converged"] for r in data["results"])
