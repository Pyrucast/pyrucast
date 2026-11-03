"""Source of the Python examples of `book/src/memory-model.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show shifted by four spaces in the book. At module level, the
excerpt is exactly what a user would write.

Consequence: pytest runs this file at **collection** time, and an example that
breaks is a collection error, not a failing test.

Voir `book/src/developper/documentation-et-tests.md`.
"""

# ── Partager plutôt que copier ──────────────────────────────────────────────

# ANCHOR: partage
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])
d = c.add_node([2.0, 0.0])
gauche = pyrucast.mesh.line(a, b, 2)
droite = pyrucast.mesh.line(b, d, 2)
tout = gauche | droite  # les deux zones, sans copie
print(gauche)  # Mesh: 1 submesh(es) [#8baac0], 2 cell(s) total
print(tout)  # Mesh: 2 submesh(es) [#8baac0, #993d70], 4 cell(s) total

T = pyrucast.NodeField(gauche, ["T"])
U = T  # un second nom pour le même champ
U[0].set_uniform("T", 5.0)
print(T.value(a, "T"))  # 5.0 : T voit l'écriture faite par U

S = T + T  # un opérateur rend un objet neuf
S[0].set_uniform("T", 1.0)
print(T.value(a, "T"), S.value(a, "T"))  # 5.0 1.0
# ANCHOR_END: partage

# The zone of `gauche` is the first zone of `tout`: same tag in both prints.
repere = str(gauche).split("[")[1].split("]")[0]
assert str(tout).split("[")[1].startswith(repere)
assert T.value(a, "T") == 5.0
assert S.value(a, "T") == 1.0

# ── Durée de vie ────────────────────────────────────────────────────────────

# ANCHOR: duree_de_vie
print(c.refcount(a.id))  # 3 : la variable a, sa maille dans gauche, le support de T
del gauche, droite, tout  # plus rien ne tient les zones SEG2
print(c.refcount(a.id))  # 2 : la maille est partie avec sa zone
print(T[0].support())  # SubMesh<POI1>: 3 cell(s) — T tient toujours son support
# ANCHOR_END: duree_de_vie

assert c.refcount(a.id) == 2
assert T[0].support().cell_count == 3
