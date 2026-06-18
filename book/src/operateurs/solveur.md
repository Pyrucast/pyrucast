# Opérateurs de solveur

Le module `ops::solver` résout le système linéaire `A · x = b` issu de
l'assemblage. Il expose pour l'instant un **unique back-end LU dense**
(`ops::solver::lu`) ; les solveurs directs creux et itératifs viendront à côté,
sous leurs propres sous-modules, derrière un trait `LinearSolver` enfichable.

## `solve(matrix, rhs)` → `NodeField`

Résout `A x = b` où `A` est la [`Matrice`](../matrix.md) assemblée et `b` le
`NodeField` de chargement. L'opération :

1. lit le `NodeField` de chargement à chacune des **lignes** de la matrice (les
   entrées absentes valent `0.0` par défaut) ;
2. convertit la `Matrix` en `nalgebra::DMatrix<f64>` ;
3. factorise `A = LU` et résout `A x = b` ;
4. emballe la solution dans un `NodeField` indexé sur les **colonnes** de la
   matrice (les primales : déplacements, températures, multiplicateurs…).

```python
K = pyrucast.stiffness(model, materials)
solution = pyrucast.solve(K, rhs)
T = solution.value(some_node, "T")
```

## Statut et feuille de route

`solve` est un **harnais de validation**, pas le solveur final : LU dense via
`nalgebra`, adapté aux petits systèmes de test. La **Phase 3** introduira un
trait `LinearSolver` enfichable (gradient conjugué, direct creux, factorisation
Cholesky pour les cas symétriques détectés via le drapeau `symmetric` de la
[Matrice](../matrix.md)). Les conversions `Matrix::to_csr` / `to_csc` sont déjà
en place pour brancher un back-end creux sans changement d'API.

## Exemples complets

La résolution de bout en bout (assemblage + contraintes + lecture de la
solution **et** des multiplicateurs de réaction) est déroulée sur des cas à
solution analytique :

- [Dirichlet](../contraintes/dirichlet.md) — Poisson 1-D `-u'' = 0`,
  `u(0)=0`, `u(1)=1` ;
- [Conduction thermique](../thermique.md) — ligne chauffée et carré ;
- [Mécanique](../mecanique.md) — treillis, élasticité, poutres.
