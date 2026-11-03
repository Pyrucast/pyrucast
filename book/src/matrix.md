# Matrice creuse (Matrix/SubMatrix)

Une matrice est ce que rend un assemblage : `pyrucast.matrix.stiffness(modele,
materiaux)` rend **une seule** matrice pour tout le [modèle](model.md),
physiques et contraintes comprises, et `mass`, `geometric` et `tangent` font de
même (voir [Assemblage](operateurs/assemblage.md)). Elle est **creuse** : seuls
ses termes non nuls sont gardés. C'est un [agrégat](aggregate.md) :

- un **bloc** (`SubMatrix`) est la part d'un sous-modèle : la conduction d'une
  zone, ou l'une des deux moitiés d'une contrainte ;
- une **matrice** (`Matrix`) est la liste de ses blocs, plus ses **valeurs
  assemblées** : la matrice que lit le solveur.

## Lire une matrice

Une ligne est un couple (nœud, duale), une colonne un couple (nœud, inconnue) :
la ligne `(0, "q")` est le flux au nœud 0, la colonne `(0, "T")` la
température au même nœud. `row_dofs()` et `col_dofs()` les donnent dans l'ordre
de la matrice, chaque nœud par son identifiant.

```python
{{#include ../../tests/python/test_doc_matrice.py:lire}}
```

Les lignes et les colonnes sont rangées dans l'ordre où les blocs les
présentent, ou selon la [permutation](coords.md) du `Coords` s'il en porte une.
Dans une matrice symétrique, la ligne i et la colonne i se répondent : même
nœud, la duale en ligne et son inconnue en colonne (`q` et `T`, `imposed` et
`lambda`). Une matrice peut aussi être rectangulaire : un bloc de contrainte a
une ligne par relation et une colonne par nœud que les relations relient.

`get` rend 0 pour une case que la matrice ne contient pas. `dense()` rend la
matrice pleine, ligne par ligne, et `dump()` l'affiche en tableau : deux
lectures à réserver aux petites matrices.

## Les blocs

`K[i]` est une vue sur un bloc. Un bloc a deux supports, les nœuds de ses
lignes et ceux de ses colonnes, en zones POI1, et ses noms de lignes et de
colonnes. La matrice ci-dessus a trois blocs : la conduction, puis les deux
moitiés de l'appui, `C` (ligne `imposed`, colonne `T`) et `Cᵀ` (ligne `q`,
colonne `lambda`).

```python
{{#include ../../tests/python/test_doc_matrice.py:blocs}}
```

Un bloc est de l'une de deux sortes :

- **calculé** : il ne garde aucune valeur, seulement de quoi les calculer, son
  sous-modèle, son sous-espace et son matériau. Ses valeurs sont calculées à
  l'assemblage et écrites directement dans la matrice. Les physiques produisent
  des blocs calculés : la conduction n'a aucune valeur écrite (`entry_count`) ;
- **écrit** : il garde ses valeurs, une par une. Ce sont les blocs des
  contraintes, et ceux qu'on écrit soi-même.

Les valeurs se lisent donc sur la matrice, pas sur ses blocs. Sur un bloc
calculé, `get` lève une erreur, `dense` et `mul_dense` lisent des zéros, et
`entries` une liste vide. De même, `entry_count` et `entries()` de la matrice
ne comptent que les valeurs écrites dans ses blocs.

Pour écrire un bloc soi-même, `Matrix.block` rend une matrice d'un seul bloc,
vide, qu'on remplit par `add_entry`. Deux valeurs écrites dans la même case
s'additionnent. `finalize()` assemble ensuite la matrice.

```python
{{#include ../../tests/python/test_doc_matrice.py:bloc_ecrit}}
```

Les nœuds d'un support doivent être **distincts**, comme ceux que rend
`to_poi1` : un nœud répété n'est pas refusé, mais ses valeurs vont à la
mauvaise ligne. Quand un bloc porte plusieurs noms, `ordering` dit comment ses
lignes se suivent : nœud par nœud (`"nodes_then_vars"`, par défaut) ou nom par
nom (`"vars_then_nodes"`). `symmetry` vaut `"full"` ou `"none"` (par défaut),
voir [Symétrie](#symétrie).

## Assemblée ou non

Une matrice a des valeurs une fois **assemblée**. Les assembleurs rendent une
matrice assemblée, sur la configuration active au moment de l'appel. Ce qui
réunit des blocs rend une matrice **sans valeurs** : `|`, `filter`, `exclude`,
une tranche `K[i:j]`. `add_sub` et `add_subs`, qui changent la matrice en place,
effacent ses valeurs.

`assemble()` calcule les valeurs à partir des blocs seuls, sans modèle, sur la
configuration active : un bloc calculé refait son calcul, un bloc écrit apporte
ses valeurs. `finalize()` fait de même pour une matrice de blocs écrits
seulement, et refuse un bloc calculé.

```python
{{#include ../../tests/python/test_doc_matrice.py:assembler}}
```

Sur une matrice sans valeurs, `dense`, `mul_dense`, `K * x` et `solve` lèvent
une erreur, et `get` aussi dès qu'un bloc est calculé.

Une matrice est assemblée **une fois** : ses valeurs ne bougent plus, et un
second `assemble()` ne fait rien. Les valeurs d'une autre configuration
s'obtiennent en refaisant la matrice.

## Calculer avec des matrices

- `2.0 * K`, `K * 2.0`, `K / dt` et `-K` rendent une matrice neuve, assemblée
  si `K` l'était, aux valeurs multipliées. Ses blocs sont des copies qui portent
  le facteur (`factor`) : `K` n'est pas touchée. Diviser par zéro lève
  `ZeroDivisionError`, par un nombre infini ou `nan`, `ValueError`.
- `A + B` et `A - B` additionnent les valeurs **telles qu'elles ont été
  assemblées**, sur la réunion des lignes et des colonnes des deux côtés : le
  résultat est assemblé et se résout tel quel. Rien n'est recalculé, si bien que
  deux matrices assemblées sur deux configurations différentes s'additionnent
  telles quelles. Un opérande dont un bloc calculé n'a jamais été assemblé (une
  sortie de `|` ou de `filter`) lève une erreur ; un opérande fait de blocs
  écrits seulement est assemblé au passage. Chaque côté peut être une matrice
  ou un bloc.
- Les deux côtés n'ont pas besoin des mêmes lignes. Dans `M / dt + K`, l'appui
  n'entre que dans `K`, et `M` ne contribue rien à ses lignes. `A + B` range
  d'abord les lignes de `A`, puis celles que seule `B` apporte.
- `K | K` est `K` : les mêmes blocs une seule fois, sans valeurs. `K + K` vaut
  `2K`. On compose une matrice de morceaux distincts avec `|`, on additionne
  deux matrices avec `+`.
- `K * x`, où `x` est un [champ aux nœuds](node-field.md), est le produit : `x`
  est lu aux colonnes (une valeur absente vaut 0), et le résultat est un champ
  posé sur les lignes.
- `lump()` rend la matrice diagonale dont chaque terme est la somme de sa
  ligne : la masse concentrée.

Deux matrices ne se multiplient pas entre elles.

```python
{{#include ../../tests/python/test_doc_matrice.py:calculer}}
```

## Natures physiques

Chaque bloc garde les natures du sous-modèle qui l'a produit (voir
[Modèle](model.md#nature-physique-et-filtrage)), et `K.physics()` réunit celles
de tous les blocs. `filter(nature)` garde les blocs qui portent cette nature,
`exclude(nature)` ceux qui ne la portent pas ; les blocs sont partagés, et le
résultat est sans valeurs. Un bloc écrit à la main n'a pas de nature :
`filter` ne le garde jamais, `exclude` le garde toujours.

## Symétrie

`K.symmetric` dit si la matrice assemblée est symétrique. Les valeurs ne sont
pas comparées : chaque bloc **déclare** sa part, et la matrice réunit les
déclarations.

- Un bloc de raideur élastique ou de masse est symétrique à lui seul
  (`is_symmetric`).
- Les blocs `C` et `Cᵀ` d'une contrainte ne le sont pas seuls, mais forment
  ensemble une symétrie : la matrice n'est symétrique que s'ils sont présents
  tous les deux.
- Un bloc écrit à la main déclare `symmetry="full"` ou `"none"`.

```python
{{#include ../../tests/python/test_doc_matrice.py:symetrie}}
```

Le solveur lit la matrice colonne par colonne. Une matrice symétrique s'y prête
telle quelle, une autre est d'abord recopiée dans ce sens. Seule une matrice
symétrique accepte `method="cholesky"`. Une déclaration fausse n'est pas
détectée : déclarer `"full"` un bloc qui ne l'est pas fait résoudre le système
de la matrice transposée.

## Résoudre

`K.solve(second_membre)` est `pyrucast.solver.solve(K, second_membre)`. La
matrice porte tout ce qu'il faut : les lignes des relations, leurs coefficients
et leur sens. Les options sont décrites dans [Solveur](operateurs/solveur.md).

```python
{{#include ../../tests/python/test_doc_matrice.py:resoudre}}
```

## Ce qui est mis en cache

Le premier `solve` sur une matrice la **factorise**, c'est-à-dire la décompose
en un produit de matrices triangulaires, l'étape coûteuse, et garde le résultat
dans la matrice. Les `solve` suivants sur la même matrice le reprennent et ne
font que la descente-remontée, bien moins chère : plusieurs chargements, un
transitoire à pas de temps constant, des itérations qui ne changent pas la
matrice. Selon la voie du solveur, ce qui est gardé est la factorisation, la
condensation de l'élimination ou l'état de l'ensemble actif (voir
[Solveur](operateurs/solveur.md#factorisation-réutilisable-cache-transparent)).

Ce qui est gardé est oublié quand la matrice change (`add_sub`, `add_subs`).
Une matrice neuve part sans : `2.0 * K`, `M / dt + K` ou un nouvel assemblage
factorisent de nouveau. Dans une boucle, on construit la matrice une fois et
on la garde dans une variable. `solve(…, cache=False)` ne garde rien.

Une matrice rendue par un assembleur **partage** sa structure, c'est-à-dire
quelles cases sont non nulles et l'ordre des lignes et des colonnes, avec le
motif que le modèle garde pour ce genre de matrice (voir
[Modèle](model.md#ce-qui-est-mis-en-cache)) : deux raideurs du même modèle
n'ont qu'une structure en mémoire, et chacune ne possède que ses valeurs.

## Ce que pèse une matrice

`memory_bytes` estime les octets occupés. Un bloc compte 24 octets par valeur
écrite (deux indices et la valeur), et 0 s'il est calculé. Une matrice compte
ses valeurs assemblées et leur structure (un indice de colonne et une valeur
par terme non nul, un début par ligne, une clé par ligne et par colonne), plus
ce que gardent ses blocs. Une structure partagée entre matrices est comptée
dans chacune. L'estimation apparaît dans l'affichage (`~296 B` plus haut).

La factorisation n'est pas comptée, alors qu'elle est en général bien plus
grosse que la matrice : faer, la bibliothèque qui la calcule, ne donne pas la
taille de ses facteurs. Pour la mémoire d'un `solve`, voir
[Calculs plus gros que la RAM](operateurs/solveur.md#calculs-plus-gros-que-la-ram).

## Sauvegarde

Une matrice se [sauvegarde](sauvegarde.md) avec ses blocs, et avec ce que
désignent ses blocs calculés : sous-modèle, sous-espace, matériau. Ni ses
valeurs assemblées ni sa factorisation ne sont écrites : à la relecture,
`assemble()` recalcule les valeurs depuis les blocs, sur la configuration
active. Une somme de matrices assemblées sur deux configurations est donc
recalculée sur une seule.

## Côté Rust

`Matrix` est l'agrégat de `Handle<SubMatrix>`, avec deux membres de plus :

- `assembled`, les valeurs assemblées et la numérotation des lignes et des
  colonnes, `None` tant que la matrice n'est pas assemblée. Les valeurs sont
  rangées ligne par ligne (forme CSR) : trois tableaux, le début de chaque
  ligne, la colonne de chaque terme et sa valeur. Pour une matrice rendue par
  un assembleur, les deux premiers et la numérotation sont des `Arc` partagés
  avec le motif du modèle (`AssemblyPattern`) ; seules les valeurs lui
  appartiennent.
- `factorization`, ce que garde le solveur, dans un
  `parking_lot::Mutex<Option<Arc<dyn Any + Send + Sync>>>`. `dyn Any` accepte
  n'importe quel type, si bien que `containers` ne dépend pas du solveur :
  `cached_factorization::<T>()` le rend s'il est de type `T`, et
  `store_factorization` le range. Le `Mutex` permet de le remplir depuis
  `solve(&Matrix)`, qui ne reçoit qu'une référence partagée.

`check_push` refuse un bloc posé sur un autre `Coords`, et `post_push` vide
`assembled` et `factorization`.

Une ligne ou une colonne est une `DofKey` : un `u64` qui porte l'identifiant du
nœud dans ses 32 bits hauts, et dans ses 32 bits bas le rang de son nom dans la
table de noms de la matrice (`dof_vars()`). `row_dofs()` rend la forme lisible,
`NamedDof` = `(NodeId, String)`.

`SubMatrix` porte :

- `row_support` et `col_support`, deux `Handle<SubMesh>` POI1 scellés à la
  construction du bloc. Le bloc ne recopie pas leur liste de nœuds : il la lit
  en place et trouve le rang d'un nœud par la table `node_index` du support
  (voir [Maillage](mesh.md#ce-quune-zone-garde-en-cache)). D'où la règle des
  nœuds distincts : cette table donne le rang parmi les nœuds sans répétition ;
- `dual_vars`, `primal_vars` et `ordering` (`DofOrdering`) : la ligne d'un
  couple (nœud, nom) est `rang_du_nœud * n_noms + rang_du_nom` en
  `NodesThenVars`, `rang_du_nom * n_nœuds + rang_du_nœud` en `VarsThenNodes` ;
- `coo`, les valeurs d'un bloc écrit, en liste de triplets (ligne, colonne,
  valeur) (`nalgebra_sparse::CooMatrix`), vide pour un bloc calculé ;
- `recipe`, `Some(ComputedRecipe)` pour un bloc calculé : le sous-modèle, les
  sous-espaces, le matériau, le `MatrixKind` et les `KernelInputs`, ce que le
  calcul lit en plus du matériau (la contrainte pour la raideur géométrique, la
  déformation et l'état précédent pour la tangente) ;
- `symmetry`, `physics`, `factor`, et `row_sense`, le sens (égalité ou
  inégalité) des lignes `imposed`, que lit le solveur.

**Assemblage.** `ops::matrix::stiffness`, comme `mass`, `geometric` et
`tangent`, demande ses blocs à chaque sous-modèle, prend le motif du modèle
(construit au premier appel par `ops::scatter::build_pattern`), calcule les
valeurs en parallèle par coloriage des mailles (voir
[Parallélisme](developper/parallelisme.md)) et les remet à la matrice par
`set_assembled`. `Matrix::finalize` n'assemble que des blocs écrits :
calculer un bloc calculé demande le calcul élémentaire, qui vit dans `models`,
et `containers` n'en dépend pas. `Matrix::assemble`, défini dans `ops::matrix`,
construit le motif depuis les blocs à chaque appel.

**Facteur.** Il s'applique partout où une valeur du bloc est lue ou émise
(`get`, `dense`, les conversions, `mul_dense`, l'assemblage), sauf dans
`local_triplets` et `local_coo_arrays`, les valeurs brutes que chaque lecteur
multiplie lui-même. `&k * s` copie chaque bloc ; `k * s`, qui prend `k`,
multiplie en place les blocs que personne d'autre ne tient
(`Handle::is_sole_owner`). Les valeurs assemblées sont multipliées aussi, la
structure restant partagée : pas de réassemblage. `(Σ v) · s` n'est pas, au bit
près, le `Σ (v · s)` d'un réassemblage, l'addition des nombres à virgule
flottante n'étant pas associative. En Rust, une division par zéro ou par un
nombre non fini arrête le programme (`panic`).

**Somme.** `&a + &b` rend un `Result<Matrix>`. Deux matrices assemblées sur le
même motif additionnent leurs valeurs case à case ; sinon, les lignes et les
colonnes sont renumérotées sur leur réunion.

```rust,ignore
{{#include ../../tests/doc_matrix.rs:somme}}
```

**Symétrie.** `Symmetry` vaut `Full`, `Half(PairId)` ou `None`. Les deux
moitiés d'une paire reçoivent ensemble leur identité par `mint_pair`, calculée
sur leur contenu (nœuds des deux supports, noms, coefficient) : elle ne dépend
pas de l'exécution, et deux sauvegardes des mêmes objets donnent les mêmes
octets. Deux paires distinctes qui tomberaient sur la même identité seraient
rejetées, jamais acceptées à tort. `Matrix::symmetric()` est vrai si chaque bloc
est `Full`, ou `Half` avec ses deux membres présents en nombres égaux : une
contrainte déclarée deux fois reste symétrique.

Dans une matrice symétrique, lignes et colonnes sont numérotées en un seul
parcours, pour que la ligne i et la colonne i se répondent : la ligne k d'un
bloc `Full` fait face à sa colonne k, et une paire `Half` se croise. Deux
incohérences y sont refusées : un même degré de liberté dual associé à deux
primaux différents, et une paire dont les membres n'ont pas la même taille.
Deux blocs rectangulaires qui sont chacun une tranche de lignes d'une matrice
symétrique ne peuvent rien déclarer : la matrice répond `false`, et le solveur
prend le chemin général.

**Solveur.** `csr_arrays()` prête les trois tableaux sans copie. La
factorisation de faer lit la matrice par colonnes (forme CSC) ; or les
tableaux CSR d'une matrice sont exactement les tableaux CSC de sa transposée.
Une matrice symétrique est donc passée telle quelle, une autre est transposée
une fois. `to_csr`, `to_csc`, `to_coo` et `to_dmatrix` fabriquent des objets
nalgebra neufs.

```rust,ignore
{{#include ../../tests/doc_matrix.rs:lecture}}
```

Un bloc écrit se construit par `SubMatrix::new`, puis `add_entry` ;
`SubMatrix::set_physics`, que Python n'expose pas, lui donne des natures, par
exemple `Physics::Other` pour que `filter` puisse le retrouver.

```rust,ignore
{{#include ../../tests/doc_matrix.rs:bloc_ecrit}}
```
