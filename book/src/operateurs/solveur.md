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

La solution a **une zone par support colonne des blocs** de la matrice, chaque
zone vivant sur le handle POI1 **du bloc lui-même** (aucun support n'est
reconstruit — ces supports sont créés une fois à la construction des
sous-modèles et réutilisés d'assemblage en assemblage). Elle est donc
`same_support` avec tout champ posé sur ces supports, et deux résolutions
successives partagent les mêmes supports : leur arithmétique (`a - b`, …)
s'aligne zone à zone. Sur un modèle contraint (Lagrange), une zone porte les
multiplicateurs — les réactions s'y lisent directement.

Pour poser un **autre** champ sur ces mêmes supports (p. ex. projeter les
forces externes avant de calculer un résidu `f_ext − K·u`), la matrice expose
ses supports en maillages : `k.row_mesh()` (côté dual, où vivent le second
membre et `mul_field`) et `k.col_mesh()` (côté primal, où vit la solution) —
handles partagés, dédupliqués. `restrict(&f_ext, &k.row_mesh()?)` s'aligne
alors zone à zone avec `K·u` ; pour un résidu **strict** (toute composante
soustraite, absente lue à `0` — et non passée brute par l'union),
`restrict_like(&f_ext, &f_int)` reprojette aussi sur les composantes.

Une matrice singulière (p. ex. conditions aux limites oubliées) produit un pivot
nul ⇒ solution non finie ⇒ erreur explicite.

```python
K = pyrucast.assemble.stiffness(model, materials)
solution = pyrucast.solver.solve(K, rhs)                 # factorise puis résout
T = solution.value(some_node, "T")

# Résolutions ultérieures sur la MÊME matrice : la factorisation est réutilisée.
sol2 = pyrucast.solver.solve(K, autre_rhs)               # descente/remontée seulement
sol3 = pyrucast.solver.solve(K, autre_rhs, cache=False)  # refactorise, sans toucher le cache
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
K = pyrucast.assemble.stiffness(model, materials)
lagrange = pyrucast.solver.solve(K, rhs)                    # système augmenté
condense = pyrucast.solver.solve_eliminate(model, K, rhs)   # système réduit — même champ
```

Voir l'exemple `examples/mpc_condensation.py` et la page
[Contraintes](../contraintes.md).

## `solve_unilateral(model, matrix, rhs)` → `NodeField`

Solveur **actif/inactif** (méthode du statut) pour un modèle portant des
relations **unilatérales** (contraintes construites avec `sense=">="` /
`"<="`) : chaque relation est soit *active* (imposée en égalité, `λ` = la
réaction), soit *inactive* (`λ = 0`, le jeu reste du côté admissible).

### Conditions KKT et boucle de statut

Une relation unilatérale `Cᵣ·u ≥ gᵣ` (ou `≤`) n'obéit pas à une équation mais
aux **conditions de complémentarité** de Karush–Kuhn–Tucker : soit elle est
active (`Cᵣ·u = gᵣ`, le multiplicateur `λᵣ` porte la réaction), soit inactive
(le jeu `Cᵣ·u − gᵣ` est du côté admissible et `λᵣ = 0`). Les deux ne peuvent
être violées à la fois. On ne sait pas *a priori* quelles relations sont
actives ; la méthode du statut itère dessus :

1. partir d'un statut d'essai (toutes actives, ou le statut convergé précédent
   quand le cache est chaud — un *warm start*) ;
2. résoudre le système point-selle avec, pour chaque relation **inactive**, sa
   ligne de contrainte remplacée par `λᵣ = 0` (la matrice garde sa taille) ;
3. vérifier les signes : une relation active dont le `λ` **tire** (signe
   inadmissible pour son sens) est **relâchée** ; une relation inactive dont le
   jeu **pénètre** est **activée** ;
4. aucun changement de statut ⇒ convergence (la boucle finie classique du
   statut) ; sinon on recommence.

La **convention de signe** vient du système point-selle assemblé `K·u + Cᵀ·λ =
f`, `C·u = g` : contre le multiplicateur KKT `μ ≥ 0` d'une contrainte `≥` on a
`λ = −μ`. Donc, dans le champ solution : `≥` active a `λ ≤ 0` (relâchée si `λ >
tol`) ; `≤` active a `λ ≥ 0` (relâchée si `λ < −tol`).

- les relations d'**égalité** du modèle sont imposées inconditionnellement,
  comme par `solve` ; un modèle **sans inégalité** retombe sur un `solve`
  simple ;
- la structure des contraintes est lue via le seam méthode-neutre
  `Constraint::relations()` (partagé avec les voies Lagrange et élimination) ;
- options : `method` / `cache` (comme `solve`), `active_set` (stratégie de
  factorisation, ci-dessous), `max_iter` (borne de la boucle de statut, `100`),
  `tol` (tolérance de signe sur `λ` et sur le jeu, `1e-10`).

### Deux stratégies de factorisation (`active_set`)

Les deux stratégies parcourent **exactement la même** trajectoire de statuts
(mêmes tests KKT, même résultat convergé) — elles ne diffèrent que par la façon
de factoriser le système d'un statut donné.

**`"refactorize"` — refactorisation par itération.** La méthode d'origine :
à chaque changement de statut, on refactorise le point-selle creux complet (une
LU faer par itération). Robuste (aucune hypothèse sur la structure), mais paie
une factorisation creuse à chaque pas.

**`"schur"` (défaut) — complément de Schur / opérateur de Delassus.** On
factorise **une seule fois** le **socle sans inégalités** `A` (physique `K` +
contraintes d'égalité, toutes les relations unilatérales relâchées), on le met
en cache sur la matrice, et on obtient chaque statut par une mise à jour dense.

Le point clé : passer une relation `r` de l'état relâché à l'état actif ne
change **qu'une seule ligne** de `A`. Dans le socle, la ligne relâchée porte
l'identité `λᵣ = 0` (un `1` à la colonne du multiplicateur, notée `λcolᵣ`) ;
l'activer y restaure la vraie ligne de contrainte `Cᵣ`. Restaurer les `k`
relations actives est donc une **mise à jour de rang `k`** :

```text
M = A + Σ_{r actif} e_row(r) · (Cᵣ − e_λcol(r))ᵀ
  = A + U Vᵀ,   U = [e_row(r)],   Vᵣ = Cᵣ − e_λcol(r)
```

La formule de Sherman–Morrison–Woodbury donne alors la solution du statut sans
refactoriser `A` :

```text
x = A⁻¹·b − X · (I + Vᵀ X)⁻¹ · (Vᵀ A⁻¹ b),   X = [A⁻¹·e_row(r)]
```

- les colonnes `xᵣ = A⁻¹·e_row(r)` (une descente/remontée creuse par relation)
  sont **mises en cache paresseusement** : calculées la première fois qu'une
  relation devient active, réutilisées ensuite ;
- le petit système `k × k` `G = I + Vᵀ X` est l'**opérateur de Delassus**
  restreint aux relations actives — dense, factorisé par une LU dense
  (`nalgebra`) à chaque itération (coût `k³/3`, négligeable jusqu'à quelques
  milliers de contacts) ; ses entrées se lisent des colonnes cachées :
  `Gᵢⱼ = δᵢⱼ + Cᵢ·xⱼ − xⱼ[λcolᵢ]` ;
- une itération de statut ne coûte donc plus **aucune factorisation creuse** —
  seulement des descentes/remontées sur `A` (cachée) et une LU dense `k × k`.
  Un re-solve à chargement identique ou proche est quasi gratuit.

**Repli automatique sur socle singulier.** Le socle `A` doit être inversible,
c.-à-d. la structure doit tenir **sans aucun contact** (bloquée par ailleurs).
Un corps simplement *posé* sur un appui n'a pas ce luxe : `A` est singulière
(mode rigide) alors que la méthode du statut converge très bien. Comme la LU
creuse peut factoriser une matrice singulière en valeurs *finies fausses* sans
erreur, la non-singularité du socle est confirmée par un **aller-retour**
`A⁻¹·(A·1) ≈ 1` ; s'il échoue, la voie `"schur"` **retombe automatiquement** sur
`"refactorize"` (marqué une fois pour toutes sur la matrice). Aucune régression
possible : le résultat est le même, seul le coût change.

```python
K = pyrucast.assemble.stiffness(model, materials)              # modèle avec sense=">="
solution = pyrucast.solver.solve_unilateral(model, K, rhs)   # "schur" par défaut
reaction = solution.value(mult_node, "lambda_u_y")    # 0 si la butée est relâchée

# Forcer l'ancienne méthode (refactorisation à chaque pas) :
sol2 = pyrucast.solver.solve_unilateral(model, K, rhs, active_set="refactorize")
```

Voir la section « Relations unilatérales » de la page
[Contraintes](../contraintes.md) pour les conditions de complémentarité et la
convention de signe.

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
