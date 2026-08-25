"""Source des exemples de `book/src/operateurs/champs.md`.

Chaque bloc de la page vient d'ici par `{{#include …:ancre}}`. Le montage vit hors des ancres. Voir
`book/src/developper/documentation-et-tests.md`.

**Le code vit au niveau module, pas dans des fonctions de test** : mdbook
n'enlève pas l'indentation d'un extrait inclus, si bien qu'un bloc ancré dans
une fonction s'afficherait décalé de quatre espaces. pytest exécute donc ce
fichier à la **collecte** ; un exemple qui casse est une erreur de collecte, au
traceback complet et au code de retour non nul.
"""

import pyrucast


def _plaque():
    """Une plaque de deux QUA4, son espace EF, et ses nœuds."""
    c = pyrucast.Coords(2)
    n = [
        c.add_node([x, y]) for y in (0.0, 1.0) for x in (0.0, 1.0, 2.0)
    ]  # 0,1,2 en bas ; 3,4,5 en haut
    mesh = pyrucast.Mesh(c, "QUA4")
    mesh.unit().add_cell([n[0], n[1], n[4], n[3]])
    mesh.unit().add_cell([n[1], n[2], n[5], n[4]])
    return c, mesh, pyrucast.FiniteElementSpace(mesh), n


def _champ_nodal(mesh, noeuds, composantes, valeurs, support=None):
    """Un `NodeField` sur le nuage POI1 des nœuds donnés.

    `support` se repasse tel quel d'un champ à l'autre : deux appels à
    `poi1_from_nodes` fabriquent deux supports **distincts**, et les réductions
    qui apparient nœud à nœud (`xty`) rendraient alors zéro.
    """
    support = support or pyrucast.mesh.poi1_from_nodes(noeuds)
    f = pyrucast.NodeField(support, composantes)
    for noeud, ligne in zip(noeuds, valeurs):
        for composante, v in zip(composantes, ligne):
            f[0].set_value(noeud, composante, v)
    return f


# ── Sélection : d'un champ vers un maillage ─────────────────────────────────


# ── select ─────────────────────────────────────────────────

_, mesh, fes, n = _plaque()
temperature = _champ_nodal(
    mesh, n, ["T"], [[t] for t in (10.0, 50.0, 90.0, 30.0, 70.0, 100.0)]
)
sigma = pyrucast.ElementField(fes, ["vm"])
sigma[0].set_uniform("vm", 300e6)
# ANCHOR: select
# Nœuds dont la température est entre 20 et 80 °C (bornes inclusives).
chauds = pyrucast.mesh.select(temperature, ge=20.0, le=80.0)

# Cellules dont la contrainte de von Mises dépasse un seuil (borne basse seule).
critiques = pyrucast.mesh.select(sigma, ge=250e6, components=["vm"])
# ANCHOR_END: select
assert chauds.cell_count() == 3  # 50, 30, 70
assert critiques.cell_count() == 2

# ── Masque : même structure, valeurs réécrites ──────────────────────────────


# ── mask ───────────────────────────────────────────────────

_, mesh, _, n = _plaque()
champ = _champ_nodal(mesh, n, ["v"], [[v] for v in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)])
temperature = _champ_nodal(
    mesh, n, ["T"], [[t] for t in (10.0, 50.0, 90.0, 30.0, 70.0, 100.0)]
)
# ANCHOR: mask
# Remet à zéro les valeurs négatives d'un champ, composante par composante.
positif = champ * champ.mask(ge=0.0)

# Sucre : les comparaisons construisent directement un masque.
positif = champ * (champ >= 0.0)  # même chose
chauds = temperature > 80.0  # NodeField 0/1
# ANCHOR_END: mask
assert positif[0][n[0], "v"] == 0.0
assert positif[0][n[5], "v"] == 3.0
assert chauds[0][n[2], "T"] == 1.0

# ── Composantes : filtrer, renommer ─────────────────────────────────────────


# ── filter et rename ───────────────────────────────────────

_, mesh, fes, n = _plaque()
solution = _champ_nodal(mesh, n, ["u_x", "u_y"], [[0.1, 0.2]] * 6)
model = pyrucast.Model.elasticity(fes, "plane_stress")
# ANCHOR: filter_rename
# Retire les multiplicateurs de Lagrange d'un résultat de solve.
u = solution.filter_components(model.primal_vars())

# Renomme une composante avant export.
export = u.rename_component("u_x", "DX")
# ANCHOR_END: filter_rename
assert u.components() == ["u_x", "u_y"]
assert export.components() == ["DX", "u_y"]

# ── indexation ─────────────────────────────────────────────

_, mesh, _, n = _plaque()
champ = _champ_nodal(mesh, n, ["u_x", "u_y"], [[0.1, 0.2]] * 6)
node = n[0]
# ANCHOR: indexation
ux = champ["u_x"]  # == filter_components(champ, "u_x")
depl = champ[["u_x", "u_y"]]  # == filter_components(champ, ["u_x", "u_y"])
zone = champ[0]  # inchangé : la zone (SubNodeField)
val = champ[0][node, "u_x"]  # inchangé : la valeur au nœud
# ANCHOR_END: indexation
assert ux.components() == ["u_x"]
assert depl.components() == ["u_x", "u_y"]
assert zone.node_count() == 6
assert val == 0.1

# ── alignement de composantes ──────────────────────────────

_, mesh, _, n = _plaque()
u1 = _champ_nodal(mesh, n, ["u_x", "u_y", "u_z"], [[0.1, 0.2, 0.3]] * 6)
u2 = _champ_nodal(mesh, n, ["u_x", "u_y"], [[1.0, 1.0]] * 6)
# ANCHOR: alignement
u = u1[u2.components()]  # u1 réduit au jeu de composantes de u2
# ANCHOR_END: alignement
assert u.components() == ["u_x", "u_y"]

# ── Mathématiques élément par élément ───────────────────────────────────────


# ── maths ──────────────────────────────────────────────────

_, mesh, _, n = _plaque()
temperature = _champ_nodal(
    mesh, n, ["T"], [[t] for t in (10.0, 50.0, 90.0, 30.0, 70.0, 100.0)]
)
signal = _champ_nodal(mesh, n, ["s"], [[-1.0], [2.0], [-3.0], [4.0], [-5.0], [6.0]])
# ANCHOR: maths
# Atténuation exponentielle d'un champ de température.
attenue = pyrucast.field.exp(temperature * -0.1)

# Magnitude d'un champ (combiné à l'arithmétique scalaire de champ).
amplitude = pyrucast.field.abs(signal)
# ANCHOR_END: maths
assert attenue[0][n[0], "T"] > 0.0
assert amplitude[0][n[0], "s"] == 1.0

# ── Réductions à un nombre ──────────────────────────────────────────────────


# ── xty ────────────────────────────────────────────────────

_, mesh, _, n = _plaque()
support = pyrucast.mesh.poi1_from_nodes(n)
forces = _champ_nodal(mesh, n, ["f_x", "f_y"], [[1.0, 2.0]] * 6, support)
deplacements = _champ_nodal(mesh, n, ["f_x", "f_y"], [[0.5, 0.5]] * 6, support)
# ANCHOR: xty
# Énergie de déformation externe : travail des efforts nodaux dans le champ
# de déplacement (mêmes composantes, même maillage).
energie = pyrucast.measure.xty(forces, deplacements)
# ANCHOR_END: xty
assert energie == 6 * (1.0 * 0.5 + 2.0 * 0.5)

# ── psca ───────────────────────────────────────────────────

_, mesh, _, n = _plaque()
vitesse = _champ_nodal(mesh, n, ["v_x", "v_y"], [[3.0, 4.0]] * 6)
# ANCHOR: psca
# Norme au carré d'un champ vectoriel, nœud par nœud.
norme2 = pyrucast.field.psca(vitesse, vitesse)  # champ à une composante "psca"
# ANCHOR_END: psca
assert norme2[0][n[0], "psca"] == 25.0

# ── integral ───────────────────────────────────────────────

_, mesh, fes, n = _plaque()
densite = _champ_nodal(mesh, n, ["f_y"], [[1.0]] * 6)
champ_unite = _champ_nodal(mesh, n, ["u"], [[1.0]] * 6)
# ANCHOR: integral
# Résultante d'une densité de force surfacique f_y sur une plaque (via N_i).
r_y = pyrucast.measure.integral(densite, "f_y", fespace=fes)
# Mesure du domaine : ∫ 1 dΩ.
aire = pyrucast.measure.integral(champ_unite, "u", fespace=fes)
# ANCHOR_END: integral
assert abs(aire - 2.0) < 1e-12
assert abs(r_y - 2.0) < 1e-12

# ── sommes et normes ───────────────────────────────────────

_, mesh, _, n = _plaque()
forces = _champ_nodal(mesh, n, ["f_x", "f_y"], [[1.0, 2.0]] * 6)
residu = _champ_nodal(mesh, n, ["f_x", "f_y"], [[3.0, 4.0]] * 6)
# ANCHOR: sommes
# Résultante d'un champ de forces nodales, composante par composante.
rx = forces.sum("f_x")
ry = forces.sum("f_y")
# Norme du résidu au carré, pour un test de convergence.
r2 = pyrucast.measure.xtx(residu)
# Même norme, restreinte aux seules composantes de translation.
r2_uy = pyrucast.measure.xtx(residu, components=["f_y"])
# Extremums d'une composante nommée…
fy_max = forces.max("f_y")
# …ou, sans argument, de tout le champ, composantes confondues.
partout = forces.min()
# ANCHOR_END: sommes
assert rx == 6.0 and ry == 12.0
assert r2 == 6 * 25.0
assert r2_uy == 6 * 16.0
assert fy_max == 2.0
assert partout == 1.0
