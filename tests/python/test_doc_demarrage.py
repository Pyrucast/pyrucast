"""Source of the Python examples of `book/src/installation.md` and
`book/src/formation/maillage.md`.

These two excerpts open an **interactive window**, which is precisely
their purpose: the first checks that the visualization layer really is
compiled, the second shows the view under the mouse. They nonetheless stay
exécutables, parce qu'ils portent la garde qu'un utilisateur écrirait de
anyway for a script that must also run in continuous integration —
the same condition winit tests itself (`DISPLAY` or `WAYLAND_DISPLAY`).

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero exit code.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import os
import tempfile

import pyrucast

# The excerpts write files under short names; the module switches
# into a throwaway folder and **gives the current directory back** at the end.
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

# Without a screen (continuous integration, a remote session), `plot()` would fail:
# we fall back to a file. That is the condition winit tests itself.
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
plaque.plot(save="plaque.svg")  # export without a window

# Fenêtre interactive (souris) — seulement s'il y a un écran, sinon `plot`
# raises: neither DISPLAY nor WAYLAND_DISPLAY is set.
if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    plaque.plot(save=None)
# ANCHOR_END: plot_interactif

assert os.path.exists("plaque.svg")


# End of the excerpts: the current directory is given back.
os.chdir(_CWD)
