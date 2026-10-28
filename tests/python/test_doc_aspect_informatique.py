"""Source of the Python examples of `book/src/aspect-informatique.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero exit code.

Voir `book/src/developper/documentation-et-tests.md`.
"""

# ── Trois niveaux de listing ────────────────────────────────────────────────

# ANCHOR: listing
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])
ligne = pyrucast.mesh.line(a, b, 10)

print(c)  # résumé : Coords: dim=2, configs=1 (active="default"), nodes=11 …
print(repr(c))  # structure : Coords { dim: 2, frame: Cartesian, … }
c.dump()  # contenu : la table des 11 nœuds, coordonnées et références
# ANCHOR_END: listing
assert ligne.cell_count() == 10
