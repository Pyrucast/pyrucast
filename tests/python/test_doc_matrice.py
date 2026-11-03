"""Source of the Python examples of `book/src/matrix.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show shifted by four spaces in the book. At module level, the
excerpt is exactly what a user would write.

Consequence: pytest runs this file at **collection** time, and an example that
breaks is a collection error, not a failing test.

Voir `book/src/developper/documentation-et-tests.md`.
"""

# ── Lire une matrice ────────────────────────────────────────────────────────

# ANCHOR: lire
import pyrucast

# Une barre [0, 1] de deux SEG2 : conduction, T imposée à gauche.
c = pyrucast.Coords(dim=1)
o = c.add_node([0.0])
l = c.add_node([1.0])
barre = pyrucast.mesh.line(o, l, 2)  # crée le nœud du milieu, d'id 2
fes = pyrucast.FiniteElementSpace(barre)
conduction = pyrucast.model.heat_conduction(fes)
appui = pyrucast.model.dirichlet(pyrucast.mesh.poi1_from_nodes([o]), conduction, "T")
modele = conduction | appui
materiaux = pyrucast.element_field.material_field(
    modele, [("k", 1.0), ("rho", 1.0), ("cp", 1.0)]
)

K = pyrucast.matrix.stiffness(modele, materiaux)
print(
    K
)  # Matrix: 3 sub-matrix(es) [#…, #…, #…], 4 row(s) × 4 col(s), symmetric, ~296 B
print(K.row_dofs())  # [(0, 'q'), (2, 'q'), (1, 'q'), (3, 'imposed')]
print(K.col_dofs())  # [(0, 'T'), (2, 'T'), (1, 'T'), (3, 'lambda')]
print(K.get(o, "q", o, "T"))  # 2.0
K.dump()
#               (0,T)   (2,T)   (1,T)  (3,lambda)
#       (0,q)   2.000  -2.000   0.000       1.000
#       (2,q)  -2.000   4.000  -2.000       0.000
#       (1,q)   0.000  -2.000   2.000       0.000
# (3,imposed)   1.000   0.000   0.000       0.000
# ANCHOR_END: lire

assert K.row_dofs() == [(0, "q"), (2, "q"), (1, "q"), (3, "imposed")]
assert K.col_dofs() == [(0, "T"), (2, "T"), (1, "T"), (3, "lambda")]
assert K.get(o, "q", o, "T") == 2.0
assert K.dense() == [2, -2, 0, 1, -2, 4, -2, 0, 0, -2, 2, 0, 1, 0, 0, 0]

# ── Les blocs ───────────────────────────────────────────────────────────────

# ANCHOR: blocs
for bloc in K:
    print(bloc.row_dofs(), bloc.col_dofs(), bloc.physics(), bloc.entry_count)
# [(0, 'q'), (2, 'q'), (1, 'q')] [(0, 'T'), (2, 'T'), (1, 'T')] ['thermal'] 0
# [(3, 'imposed')] [(0, 'T')] ['constraint', 'thermal'] 1
# [(0, 'q')] [(3, 'lambda')] ['constraint', 'thermal'] 1
# ANCHOR_END: blocs

assert [b.entry_count for b in K] == [0, 1, 1]
assert [b.physics() for b in K] == [
    ["thermal"],
    ["constraint", "thermal"],
    ["constraint", "thermal"],
]

# ANCHOR: bloc_ecrit
noeuds = barre.to_poi1()  # les trois nœuds de la barre, chacun une fois
B = pyrucast.Matrix.block(noeuds, noeuds, ["q"], ["T"], symmetry="full")
b = B.unit()
b.add_entry(o, "q", o, "T", 1.0)
b.add_entry(l, "q", l, "T", 1.0)
b.add_entry(o, "q", o, "T", 0.5)  # s'ajoute à la précédente
B.finalize()
print(B.get(o, "q", o, "T"))  # 1.5
print(b.memory_bytes)  # 72 : trois valeurs écrites, 24 octets chacune
# ANCHOR_END: bloc_ecrit

assert B.get(o, "q", o, "T") == 1.5
assert b.memory_bytes == 3 * 24
assert B.memory_bytes > b.memory_bytes

# ── Assemblée ou non ────────────────────────────────────────────────────────

# ANCHOR: assembler
U = K.filter("thermal").exclude("constraint")  # la conduction sans son appui
print(U)  # Matrix: 1 sub-matrix(es) [#…], 3 row(s) × 3 col(s), symmetric, ~0 B
try:
    U.get(o, "q", o, "T")
except RuntimeError as erreur:
    print(
        erreur
    )  # Matrix::get: this matrix carries a computed block that was never assembled…
U.assemble()  # calcule les valeurs, sur la configuration active
print(U.get(o, "q", o, "T"))  # 2.0
# ANCHOR_END: assembler

assert len(U) == 1
assert U.get(o, "q", o, "T") == 2.0

# ── Calculer avec des matrices ──────────────────────────────────────────────

# ANCHOR: calculer
M = pyrucast.matrix.mass(modele, materiaux)  # la capacité : l'appui n'y entre pas
dt = 0.1
A = M / dt + K  # assemblée, prête à résoudre
print(A.row_dofs() == K.row_dofs())  # True

print((K + K).get(o, "q", o, "T"))  # 4.0 : chaque opérande compte
une_fois = K | K  # les mêmes blocs, une seule fois, sans valeurs
une_fois.assemble()
print(une_fois.get(o, "q", o, "T"))  # 2.0

# Le produit par un champ lit T aux colonnes et rend q aux lignes.
x = pyrucast.NodeField(barre, ["T"])
x.unit().set_uniform("T", 1.0)
print((K * x).value(l, "q"))  # 0.0 : une température uniforme ne crée aucun flux
# ANCHOR_END: calculer

assert A.row_dofs() == K.row_dofs()
assert (K + K).get(o, "q", o, "T") == 4.0
assert une_fois.get(o, "q", o, "T") == 2.0
assert (K * x).value(l, "q") == 0.0
assert abs(A.get(o, "q", o, "T") - (M.get(o, "q", o, "T") / dt + 2.0)) < 1e-12

# ── Symétrie ────────────────────────────────────────────────────────────────

# ANCHOR: symetrie
print([bloc.is_symmetric for bloc in K])  # [True, False, False]
print(K.symmetric)  # True : C et Cᵀ sont présents tous les deux
print(K[0:2].symmetric)  # False : Cᵀ a été laissé de côté
# ANCHOR_END: symetrie

assert [bloc.is_symmetric for bloc in K] == [True, False, False]
assert K.symmetric and not K[0:2].symmetric

# ── Résoudre ────────────────────────────────────────────────────────────────

# ANCHOR: resoudre
impose = appui.constraint_rhs([(o, 1.0)])  # T = 1 en x = 0
T = K.solve(impose)  # factorise K, et garde la factorisation dans K
print(T.value(l, "T"))  # 1.0
T = K.solve(2.0 * impose)  # reprend la factorisation : la descente-remontée seule
print(T.value(l, "T"))  # 2.0
# ANCHOR_END: resoudre

assert abs(T.value(l, "T") - 2.0) < 1e-12
