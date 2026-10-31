"""Source of the Python examples of `book/src/coords.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
a function would show shifted by four spaces in the book. At module level, the
excerpt is exactly what a user would write.

Consequence: pytest runs this file at **collection** time, and an example that
breaks is a collection error, not a failing test. The traceback is
complet et le code de retour non nul ; `--continue-on-collection-errors`
(in `pyproject.toml`) keeps it from interrupting the rest of the suite.

Voir `book/src/developper/documentation-et-tests.md`.
"""

# ── Créer des nœuds ─────────────────────────────────────────────────────────

# ANCHOR: creation
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])  # un pyrucast.Node
b = c.add_node([1.0, 0.0])
print(a.id, b.id)  # 0 1 — identifiants stables
print(c)  # Coords: dim=2, configs=1 (active="default"), nodes=2 …

ligne = pyrucast.mesh.line(a, b, 2)  # pose un nœud au milieu, id 2
print(c.refcount(a.id))  # 2 — la variable a, et la maille qui le contient
print(c.refcount(2))  # 2 — les deux mailles qui le contiennent
milieu = c.acquire(2)  # un Node pour un id existant
print(milieu.position())  # [0.5, 0.0]
# ANCHOR_END: creation

assert (a.id, b.id) == (0, 1)
assert c.refcount(2) == 3  # the two cells, and `milieu`


# ── Repère axisymétrique ────────────────────────────────────────────────────

# ANCHOR: axisymetrique
import pyrucast

c = pyrucast.Coords.axisymmetric()
assert c.dim == 2 and c.is_axisymmetric
c.add_node([1.0, 0.0])  # r = 1, z = 0
try:
    c.add_node([-1.0, 0.0])  # x is a radius: it must be ≥ 0
except RuntimeError as erreur:
    print(erreur)
# ANCHOR_END: axisymetrique

assert c.node_count() == 1


# ── Configurations ──────────────────────────────────────────────────────────

# ANCHOR: configurations
import pyrucast

c = pyrucast.Coords(dim=2)
n = c.add_node([0.0, 0.0])

# Create a second configuration (a clone of the active one).
c2 = c.add_config("deformed")
print(c.names())  # ['default', 'deformed']

# Switch to the deformed configuration and change the coordinates.
c.select(c2)
n.set_position([0.1, 0.05])

# The coordinates read depend on the active configuration.
c.select(0)
print(n.position())  # [0.0, 0.0]  — configuration de référence
c.select(c2)
print(n.position())  # [0.1, 0.05] — configuration déformée
print(c.active)  # 1
# ANCHOR_END: configurations

assert c.names() == ["default", "deformed"]
assert c.active == 1


# ANCHOR: in_config
import pyrucast

c = pyrucast.Coords(dim=2)
n = c.add_node([1.0, 0.0])
deformee = c.add_config("deformed")

# Le bloc voit `deformee` ; la configuration d'avant revient à la sortie,
# même après une exception.
with c.in_config(deformee):
    n.set_position([1.5, 0.0])
    x = n.position()

print(x)  # [1.5, 0.0]
print(n.position())  # [1.0, 0.0] — la référence, de nouveau active
# ANCHOR_END: in_config

assert x == [1.5, 0.0]
assert n.position() == [1.0, 0.0]
assert c.active == 0


# ── Permutation ─────────────────────────────────────────────────────────────

# ANCHOR: permutation
import pyrucast

c = pyrucast.Coords(dim=2)
c.add_node([0.0, 0.0])
c.add_node([1.0, 0.0])
c.add_node([0.5, 1.0])

# One entry per node created, no value twice.
print(c.capacity())  # 3
c.set_permutation([2, 0, 1])
print(c.permutation())  # [2, 0, 1]

# Back to the identity (None = identity).
c.clear_permutation()
print(c.permutation())  # None
# ANCHOR_END: permutation

assert c.permutation() is None
