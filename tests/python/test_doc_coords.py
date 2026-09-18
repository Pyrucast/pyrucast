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


# ── Permutation ─────────────────────────────────────────────────────────────

# ANCHOR: permutation
import pyrucast

c = pyrucast.Coords(dim=2)
c.add_node([0.0, 0.0])
c.add_node([1.0, 0.0])
c.add_node([0.5, 1.0])

# Set a permutation by hand.
c.set_permutation([2, 0, 1])
print(c.permutation())  # [2, 0, 1]

# Back to the identity (None = identity).
c.clear_permutation()
print(c.permutation())  # None
# ANCHOR_END: permutation

assert c.permutation() is None


# ── Cycle de vie d'un nœud ──────────────────────────────────────────────────

# ANCHOR: cycle_de_vie
import pyrucast

c = pyrucast.Coords(dim=2)
n = c.add_node([0.0, 0.0])  # n is a pyrucast.Node; refcount = 1
m = c.add_node([1.0, 0.0])

print(c)  # Coords: dim=2, configs=1 (active="default"), nodes=2 ...
n.set_position([0.5, 0.5])

# GC touches nothing as long as at least one Python Node exists.
assert c.gc() == 0

# del + collect force le Drop côté Rust et libère le refcount.
import gc as pygc

del n
pygc.collect()
assert c.gc() == 1
# ANCHOR_END: cycle_de_vie
