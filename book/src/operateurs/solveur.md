# Opérateurs de solveur

Le module `ops::solver` résout le système linéaire `A · x = b` issu de
l'assemblage. Le back-end est un **LU creux parallèle** (faer), confiné à
`ops::solver` : il consomme le CSC assemblé par `nalgebra-sparse`. Chaque crate
garde son rôle — `nalgebra` (primitives), `nalgebra-sparse`
(assemblage/stockage/serde), `faer` (factorise & résout). Voir
[Parallélisme](../developper/parallelisme.md).

## `solve(matrix, rhs)` → `NodeField`

Résout `A x = b` où `A` est la [`Matrice`](../matrix.md) assemblée et `b` le
`NodeField` de chargement. L'opération :

1. obtient la **factorisation** de `A` (cache de la matrice, cf. ci-dessous) ;
2. lit le `NodeField` de chargement à chacune des **lignes** de la matrice (les
   entrées absentes valent `0.0` par défaut) ;
3. effectue la descente/remontée pour ce second membre ;
4. emballe la solution dans un `NodeField` indexé sur les **colonnes** de la
   matrice (les primales : déplacements, températures, multiplicateurs…).

Une matrice singulière (p. ex. conditions aux limites oubliées) produit un pivot
nul ⇒ solution non finie ⇒ erreur explicite.

```python
K = pyrucast.stiffness(model, materials)
solution = pyrucast.solve(K, rhs)                 # factorise puis résout
T = solution.value(some_node, "T")

# Résolutions ultérieures sur la MÊME matrice : la factorisation est réutilisée.
sol2 = pyrucast.solve(K, autre_rhs)               # descente/remontée seulement
sol3 = pyrucast.solve(K, autre_rhs, cache=False)  # refactorise, sans toucher le cache
```

## Factorisation réutilisable (cache transparent)

`solve` **factorise une fois, résout N fois** : la factorisation est mise en
cache **dans la `Matrix`** (état dérivé, non sérialisé, à mutabilité intérieure).
La première résolution factorise et met en cache ; les suivantes sur la même
matrice ne font que la descente/remontée — bien moins cher (cas de charge
multiples, itérations de Newton, transitoire à matrice constante). Le cache est
**invalidé automatiquement** dès que la matrice change (`add_sub`). On ne stocke
**jamais l'inverse explicite** (dense, coûteux, instable) — seulement la
factorisation.

Options de `solve` :

- `method` — méthode directe (`"lu"` par défaut ; l'énum laisse la place à
  d'autres back-ends sans changer les appels) ;
- `cache` — réutiliser/peupler le cache (`True` par défaut ; `False` factorise à
  neuf sans toucher le cache).

## `solve_eliminate(model, matrix, rhs)` → `NodeField`

Voie **alternative** pour un modèle contraint : au lieu de border le système par
des multiplicateurs de Lagrange (ce que fait `solve` sur la matrice augmentée),
on **élimine** les contraintes par condensation maître/esclave. Pour chaque
relation `Σ aₖ·u(nœudₖ, varₖ) = g`, un terme *esclave* `s` est exprimé par les
autres (*maîtres*) : `u_s = (g − Σ_{k≠s} aₖ·u_k)/a_s`. Le système se réduit à
`K̂ û = f̂` avec `K̂ = Tᵀ K T`, résolu par le **même** LU creux (sur une matrice
plus petite et définie, sans degré multiplicateur), puis prolongé `u = T·û + u₀`.

- lit la structure des contraintes via le seam méthode-neutre
  `Constraint::relations()` (partagé avec la voie Lagrange) ; le `K` physique est
  extrait du bloc point-selle assemblé (nœuds multiplicateurs filtrés) ;
- récupère en post-traitement la **réaction** (équivalent du multiplicateur),
  `−(K·u − f)` à la ligne duale de chaque esclave (`= aₛ·λ`) ;
- met en cache la **condensation** (`T`, `K̂` factorisé) sur la matrice, comme la
  factorisation LU ; mêmes options `method` / `cache` ;
- un modèle **sans contrainte** retombe sur un `solve` simple.

**Périmètre v1** : non chaîné, esclaves disjoints — chaque relation élimine un
esclave distinct, jamais réutilisé comme maître ni esclave ailleurs (couvre la
périodicité ; erreur explicite sinon).

```python
K = pyrucast.stiffness(model, materials)
lagrange = pyrucast.solve(K, rhs)                    # système augmenté
condense = pyrucast.solve_eliminate(model, K, rhs)   # système réduit — même champ
```

Voir l'exemple `examples/mpc_condensation.py` et la page
[Contraintes](../contraintes.md).

## Déterminisme

Contrairement au reste des opérateurs (bit-à-bit identiques quel que soit le
nombre de threads), le solveur n'est **pas** bit-à-bit identique à l'ancien LU
dense : pivotage et ordering diffèrent. Les résultats restent dans les
tolérances numériques usuelles.

## Exemples complets

La résolution de bout en bout (assemblage + contraintes + lecture de la
solution **et** des multiplicateurs de réaction) est déroulée sur des cas à
solution analytique :

- [Dirichlet](../contraintes/dirichlet.md) — Poisson 1-D `-u'' = 0`,
  `u(0)=0`, `u(1)=1` ;
- [Conduction thermique](../thermique.md) — ligne chauffée et carré ;
- [Mécanique](../mecanique.md) — treillis, élasticité, poutres.
