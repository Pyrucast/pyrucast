"""Source des exemples Python de `book/src/installation.md` et
`book/src/formation/maillage.md`.

Ces deux extraits ouvrent une **fenêtre interactive**, ce qui est précisément
leur objet : le premier vérifie que la couche de visualisation est bien
compilée, le second montre la vue à la souris. Ils restent néanmoins
exécutables, parce qu'ils portent la garde qu'un utilisateur écrirait de
toute façon pour un script qui doit tourner aussi en intégration continue —
la même condition que winit teste lui-même (`DISPLAY` ou `WAYLAND_DISPLAY`).

**Le code vit au niveau module, pas dans des fonctions de test** : mdbook
n'enlève pas l'indentation d'un extrait inclus. pytest exécute donc ce fichier
à la **collecte** ; un exemple qui casse est une erreur de collecte, au
traceback complet et au code de retour non nul.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import os
import tempfile

import pyrucast

# Les extraits écrivent des fichiers sous des noms courts ; le module bascule
# dans un dossier jetable et **rend** le répertoire courant à la fin.
_TMP = tempfile.TemporaryDirectory()
_CWD = os.getcwd()
os.chdir(_TMP.name)


# ── Vérifier l'installation ─────────────────────────────────────────────────

# ANCHOR: installation
import os

import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])

mesh = pyrucast.Mesh(c, "SEG2")  # un sous-maillage
mesh.unit().add_cell([a, b])

# Sans écran (intégration continue, session distante), `plot()` échouerait :
# on retombe sur un fichier. C'est la condition que winit teste lui-même.
ecran = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
mesh.plot(save=None if ecran else "apercu.svg")

print(c)
print(mesh)  # Mesh: 1 submesh(es), 1 cell(s) total
mesh.dump()
# ANCHOR_END: installation

assert ecran or os.path.exists("apercu.svg")


# ── Formation : la vue interactive ──────────────────────────────────────────

_c = pyrucast.Coords(2)
_coins = [_c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0])]
plaque = pyrucast.Mesh(_c, "QUA4")
plaque.unit().add_cell(_coins)

# ANCHOR: plot_interactif
plaque.plot(save="plaque.svg")  # export sans fenêtre

# Fenêtre interactive (souris) — seulement s'il y a un écran, sinon `plot`
# lève : ni DISPLAY ni WAYLAND_DISPLAY n'est défini.
if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    plaque.plot(save=None)
# ANCHOR_END: plot_interactif

assert os.path.exists("plaque.svg")


# Fin des extraits : on rend le répertoire courant.
os.chdir(_CWD)
