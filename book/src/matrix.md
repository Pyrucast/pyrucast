# Matrice creuse (`Matrix`)

`Matrix` est le **conteneur de sortie** d'un assemblage : c'est ce que produisent les opérateurs `matrix::stiffness(model, materials)` / `matrix::mass(model, materials)` à partir d'un [`Model`](model.md). Elle représente une matrice creuse dont les lignes et les colonnes sont identifiées par des **DOFs nommés**.

## Identification des DOFs : `(NodeId, nom_de_champ)`

Chaque ligne et chaque colonne d'une `Matrix` est identifiée par un couple `(NodeId, ChampId)` :

- **`NodeId`** — l'identifiant stable d'un nœud dans la `Coords`.
- **`ChampId`** — un indice compact dans une petite table de noms portée par la matrice (typiquement 5–10 entrées). Les noms sont des chaînes comme `"T"`, `"q"`, `"ux"`, `"lambda_w"`.

Le type concret est `DofId { node_id, field_idx }`. Cette représentation est compacte (un `u32` par champ partagé sur tous les DOFs qui le portent) et conserve l'information sémantique : à chaque entrée numérique de la matrice est attaché « quel inconnu, à quel nœud ».

Les jeux de DOFs de lignes et de colonnes sont **indépendants** :

- ils peuvent avoir des **tailles différentes** (matrice rectangulaire — par exemple le bloc Lagrange d'une condition de Dirichlet) ;
- ils peuvent porter des **noms de champs différents** (les lignes étiquetées par des duales `q`, les colonnes par des primales `T`).

## Blocs bi-mode : littéral ou calculé

Une `Matrix` est un **agrégat de blocs** `SubMatrix`, et un bloc est de l'un de deux modes :

- **littéral** — il porte ses **valeurs**, stockées en **COO** (coordinate triplet list). Chaque `add_entry(...)` ajoute un triplet `(ligne, colonne, valeur)` ; plusieurs entrées au même couple s'**accumulent** (sommées à l'assemblage), l'ordre d'insertion étant sans effet. C'est le mode historique — celui des contraintes (blocs `C` / `Cᵀ` de Dirichlet) et de tout bloc monté à la main.
- **calculé** — il ne porte **aucune valeur**, seulement une **recette** `{ sous-modèle, sous-espace EF, matériau }`. Ses entrées sont produites **à l'assemblage** par le noyau élémentaire du sous-modèle, dispersées directement dans la matrice globale. C'est le mode des physiques volumiques (raideur), qui évite de matérialiser un COO intermédiaire.

Un bloc calculé garde son lien vers sa physique **via la recette** ; la `Matrix`, elle, reste un simple sac de blocs et **ne référence pas le `Model`**.

### Le bloc ne recopie pas sa liste de nœuds

Un bloc est posé sur deux supports POI1 (lignes et colonnes, souvent le même objet) et **n'en garde aucune copie** : il lit leur connectivité en place à chaque accès, conformément à la règle [Zéro-copie](developper/parallelisme.md). C'est sûr parce que les deux supports sont **scellés** à la construction du bloc — leur connectivité ne peut plus changer, donc la numérotation ne peut pas dériver. Le `NodeId → position` passe par la table que le support porte déjà (`SubMesh::node_index`), partagée avec tous ses autres consommateurs au lieu d'être refaite par bloc.

Corollaire à connaître si l'on monte un bloc à la main : les nœuds d'un support doivent être **distincts**, ce que produit `to_poi1`. Un nœud répété n'est pas rejeté, mais il adresse la mauvaise ligne — la table du support donne un rang dédoublonné, qui s'écarte de la position dès la première répétition.

### Étiquette de nature physique (`physics`)

Chaque bloc porte en plus un **ensemble de natures** `Vec<Physics>` (`Mechanical`,
`Thermal`, `Constraint`, `Other`) — l'assembleur le pose sur **tout** bloc qu'il
émet, sur les deux chemins (calculé **et** littéral), donc le couple `C`/`Cᵀ` d'un
Dirichlet est étiqueté lui aussi. C'est ce qui rend l'étiquette utilisable là où la
recette manque (blocs littéraux). Le tag est un **ensemble** : vide pour un bloc
monté à la main hors assemblage (le cas « rien »), et à plusieurs éléments pour
une physique couplée.

Il alimente `Matrix::filter(Physics)` — le miroir de
[`Model::filter`](model.md#nature-physique-et-filtrage) — qui renvoie une `Matrix`
ne gardant que les blocs dont l'ensemble **contient** la nature donnée (handles
partagés, pas de copie). Le résultat n'est **pas** assemblé : relancer
`Matrix::assemble` avant de résoudre. Un bloc à l'ensemble vide n'est
sélectionné par aucune nature concrète ; l'étiqueter `Physics::Other` le rend
atteignable. `Matrix::physics()` renvoie l'ensemble des natures présentes dans la
matrice (dédupliqué — « plusieurs tags » au niveau de l'agrégat).

```rust,ignore
{{#include ../../tests/doc_matrix.rs:filtrage}}
```

## Assemblage : motif + scatter

Passer d'un agrégat de blocs à une matrice utilisable se fait en deux temps :

1. **Motif creux** (sparsité CSR) — l'**union dédoublée** des DDL des blocs (une liste globale, via table de hachage) et de leurs entrées. Il ne dépend que de la **topologie** (bloc calculé via la connectivité, bloc littéral via sa COO), pas des matériaux ; `stiffness` le **mémoïse donc sur le `Model`** et le réutilise d'un assemblage à l'autre.
2. **Valeurs** — dispersées (scatter) dans le CSR : un bloc calculé lance son noyau élémentaire (en parallèle, par coloration des cellules — voir [Parallélisme](developper/parallelisme.md)) ; un bloc littéral recopie sa COO. Chaque bloc remappe sa numérotation locale `(nœud, variable)` vers l'index global via une **table de traduction** — O(nnz), sans recherche par entrée (`NodeId` est déjà l'index nœud global dense, et `add_entry` retrouve la position d'un nœud en O(1)).

L'ordre des DOFs dans `row_dofs()` / `col_dofs()` est l'**ordre de première rencontre** des blocs — sauf si le `Coords` porte une [`permutation`](coords.md) (ordre solveur), auquel cas la liste globale suit cet ordre (tri stable). Reproductible dans les deux cas.

### `finalize` vs `ops::matrix`

- `Matrix::finalize()` n'assemble que des blocs **littéraux** (somme des COO → CSR). Il **refuse** un bloc calculé : le noyau vit dans `models`, hors de `containers`, et l'y appeler créerait un cycle `matrix ↔ kernel`. Il renvoie alors vers `ops::matrix`.
- `ops::matrix::stiffness(model, materials)` construit les blocs (calculés pour les physiques volumiques, littéraux pour Dirichlet) et assemble, motif mémoïsé sur le `Model`.
- `Matrix::assemble(&mut self)` réassemble une matrice **depuis ses blocs seuls**, sans `Model` : c'est le chemin de **composition** — combiner une sous-matrice neuve (de provenance quelconque) à une matrice existante puis réassembler. La `Matrix` ne dépendant que de ses blocs, cette composabilité de base est ainsi préservée y compris en présence de blocs calculés.

Pour qui veut une matrice creuse d'une autre bibliothèque, des conversions **à la demande** existent — elles fabriquent un objet neuf et ne retiennent rien :

- [`Matrix::to_csr`](#api-rust--accès-en-lecture) → `nalgebra_sparse::CsrMatrix<f64>`
- [`Matrix::to_csc`](#api-rust--accès-en-lecture) → `nalgebra_sparse::CscMatrix<f64>`
- [`Matrix::to_coo`](#api-rust--accès-en-lecture) → `nalgebra_sparse::CooMatrix<f64>`
- [`Matrix::to_dmatrix`](#api-rust--accès-en-lecture) → `nalgebra::DMatrix<f64>`

Mais **le solveur n'en emprunte aucune**. La forme assemblée n'a pas besoin d'être
convertie, seulement d'être regardée sous le bon angle : une CSR est, octet pour
octet, la CSC de la transposée, et une CSC triée sans doublon est exactement ce que
le LU creux de faer demande. `ops::solver::lu` lit donc les tableaux par
[`Matrix::csr_arrays`](#api-rust--accès-en-lecture) — **empruntés**, la sparsité
restant celle du motif — transpose une fois par tri par comptage, et tend à faer
une vue. Aucune copie de la matrice ne coexiste avec la factorisation, qui est
le moment où la mémoire est la plus tendue.

Le produit matrice-vecteur, lui, tire parti de l'orientation **lignes** : lectures
contiguës, un accumulateur par ligne, parallélisable sur les lignes sans atomique.
C'est la raison pour laquelle la forme assemblée reste une CSR.

## Facteur scalaire et somme de matrices

### Le facteur

Chaque `SubMatrix` porte un **facteur** `f64`, `1.0` par défaut, ajusté par
`bloc * s`, `bloc / s` et `-bloc`. Le facteur ne touche **que** ce champ — jamais les
valeurs stockées (`coo`) — ce qui le rend utilisable aussi bien sur un bloc **littéral**
que sur un bloc **calculé** (dont les valeurs n'existent qu'à l'assemblage, produites
par le noyau élémentaire). Il est pris en compte partout où une valeur du bloc est lue
ou émise : les accesseurs directs (`get`, `dense`, `to_dmatrix`, `to_coo`, `to_csr`,
`to_csc`, `mul_dense`) et les deux passes d'assemblage global (`Matrix::finalize` et
`ops::matrix::scatter`, calculé comme littéral). Seules les formes **locales** brutes
(`local_triplets`, `local_coo_arrays`) restent non mises à l'échelle — ce sont des vues
internes destinées au remappage global, chaque consommateur y applique le facteur
lui-même.

`&Matrix * s`, `&Matrix / s` et `-&Matrix` mettent à l'échelle une matrice entière :
chaque bloc est **cloné** dans un nouvel objet avec son facteur ajusté. C'est nécessaire
car `add_sub`/`union`/`filter`/`subset` **partagent** les `Handle<SubMatrix>` (même
objet, compté) plutôt que de les copier ; muter le facteur en place rescalerait
silencieusement toute autre `Matrix` référençant le même bloc. La forme **possédante**
(`matrix * s` sur une valeur, pas une référence) fait l'économie de cette copie pour
tout bloc que personne d'autre ne tient — ce que `Handle::is_sole_owner` établit.

Le scalaire se lit des deux côtés (`2.0 * k` comme `k * 2.0`). Ces opérateurs sont
**infaillibles**, à une exception près : une division refuse un diviseur nul ou non
fini, qui rendrait non finie chaque valeur du résultat. Côté Rust elle interrompt
l'exécution ; côté Python elle lève `ZeroDivisionError` ou `ValueError`.

**La CSR assemblée suit, mise à l'échelle.** Mettre tous les blocs à la même échelle met
chaque entrée à cette échelle et laisse la sparsité intacte : seul le tableau des
valeurs est parcouru, les tableaux d'indices et la table de noms sont des `Arc` partagés.
Une matrice assemblée reste donc assemblée après `* s`, et n'a pas à repasser par un
`assemble()` qui relancerait tous les noyaux élémentaires pour appliquer un scalaire.

> `(Σ v) · s` n'est pas, au bit près, le `Σ (v · s)` que calculerait un réassemblage :
> l'addition flottante n'est pas associative. Les deux valent la même quantité à
> l'arrondi près, et chaque chemin reste reproductible.

```rust,ignore
{{#include ../../tests/doc_matrix.rs:facteur}}
```

### La somme

`a + b` rend une `Matrix` portant les blocs des deux opérandes, **partagés** et
délibérément **non dédoublonnés**. Rien n'est calculé : c'est l'assembleur qui somme ce
qui retombe sur le même `(row, col)` global (`build_global_triplets`,
`scatter_serial`/`scatter_parallel`). Une somme coûte donc quelques incréments de
compteur, ne touche aucune valeur, et laisse un bloc calculé calculé. Comme pour
`filter`, le résultat n'est **pas assemblé** : `assemble()` avant de résoudre.

`a - b` nie les blocs de droite, ce qui les **recopie** (le facteur vit dans le bloc) ;
`a + b` ne copie rien. Les deux opérateurs acceptent indifféremment une `Matrix` ou une
`SubMatrix` de chaque côté, et `-a` nie une matrice entière.

```rust,ignore
{{#include ../../tests/doc_matrix.rs:somme}}
```

Aucun traitement particulier n'est nécessaire quand `K` et `M` n'ont pas le même
ensemble de DOFs (cas courant : un Dirichlet/MPC n'entre que dans la matrice de
raideur, jamais dans la masse) — la somme prend simplement l'union des DOFs des deux
côtés, et les blocs de `M` ne contribuent rien aux DOFs qu'ils ne portent pas.

### `|` compose, `+` additionne

C'est la seule chose qui les sépare, et elle ne se voit que sur des blocs partagés :
l'union **écarte** un bloc dont elle tient déjà l'emplacement, la somme le compte à
chaque fois qu'on le lui donne. Donc `k | k` vaut `k`, tandis que `k + k` vaut `2k`.

Prendre `|` pour **composer un opérateur à partir de morceaux distincts** (une raideur
et son bloc de Dirichlet), `+` pour **additionner deux opérateurs**.

```rust,ignore
{{#include ../../tests/doc_matrix.rs:union_ou_somme}}
```

## Symétrie

Le dernier argument des constructeurs de `SubMatrix` déclare **quelle part de la
symétrie de la matrice ce bloc porte** :

| `Symmetry` | sens |
|---|---|
| `Full` | le bloc est symétrique à lui seul — toute raideur de Galerkine, toute matrice de masse, toute matrice de Gram |
| `Half(id)` | il n'en porte que la **moitié** : sa transposée est l'autre bloc de même identité. Ni l'un ni l'autre n'est symétrique seul |
| `None` | il n'en porte aucune |

La propriété visée est celle du **tableau assemblé** — `A[i][j] == A[j][i]` sur la
CSR — et rien d'autre. Un bloc la **déclare** et on le croit : un modèle sait ce
qu'il écrit, et rien ici ne le vérifie. Déclarer juste est donc tout le travail du
producteur, et l'agrégat additionne les déclarations sans les corriger. Un bloc
**vide**, par exemple, est symétrique : il déclare `Full`, et la règle n'a pas
d'exception à prévoir pour lui.

### La numérotation suit la déclaration

Un tableau n'est symétrique que si le rang `i` désigne des DDL **conjugués** des
deux côtés. Or les deux ordres globaux se construisaient par deux parcours
indépendants des blocs, et rien ne les faisait tomber d'accord : une contrainte
qui introduit deux nœuds neufs d'un coup — `embedded`, dont le nœud immergé
n'appartient à aucune physique — les faisait découvrir en ordre inverse de chaque
côté.

Quand la matrice se déclare symétrique, les deux ordres sont donc construits en
**un seul parcours conjugué**. Ce parcours n'apprend jamais que `q` est le dual
de `T` : il lit seulement quelles positions se correspondent, ce que la
déclaration dit déjà. Un bloc `Full` est carré sur un support unique, donc sa
ligne `k` fait face à sa propre colonne `k` ; une paire `Half` croise, la ligne
de l'un faisant face à la colonne de l'autre.

Deux incohérences y sont **refusées**, jamais rattrapées : un même DDL dual
déclaré conjugué à deux DDL primaux différents, et une paire dont les deux
membres n'ont pas le même nombre de DDL.

Une matrice non symétrique garde les deux parcours indépendants : une matrice
rectangulaire n'a pas de conjugué à apparier.

### Pourquoi une moitié

Une contrainte de Dirichlet introduit deux blocs **rectangulaires**, `C` et `Cᵀ`
(voir [Contraintes](contraintes.md)). Aucun des deux ne peut être symétrique — un
bloc rectangulaire ne l'est jamais — mais ensemble ils le sont. C'est une propriété
du **couple**, que `Half` rend exprimable : le producteur qui écrit le même
coefficient des deux côtés est celui qui les apparie.

L'identité de la paire est une empreinte de son **contenu** : les nœuds des deux
supports, les quatre noms de variables, le coefficient. Déterministe, donc l'archive
reste reproductible ; et si deux paires réellement distinctes venaient à partager
une empreinte, elles seraient **rejetées**, jamais acceptées à tort — on oublie une
symétrie, on n'en invente pas.

### Ce que l'agrégat en conclut

`Matrix::symmetric()` est vrai si chaque bloc est `Full`, ou `Half` avec ses **deux
membres présents en nombres égaux**. Compter les membres plutôt que les blocs permet
à une contrainte déclarée deux fois (quatre blocs, deux de chaque) de tenir, tandis
qu'une paire coupée par un `subset` tombe.

**Le stockage n'est pas dédupliqué** : une matrice symétrique porte quand même ses
deux triangles. Mais la déclaration, elle, est **consultée** : elle décide si la CSR
assemblée peut être tendue telle quelle à la factorisation comme sa propre CSC. Une
matrice symétrique l'est — `CSR(A)` est au bit près `CSC(Aᵀ)`, et `Aᵀ = A` — si bien
que le retournement, et la seconde copie complète de la matrice qui va avec,
disparaissent. C'est elle aussi qui autorise Cholesky (voir
[Résolution](model.md)).

## Cas d'usage typique : matrice de raideur du laplacien

```rust,ignore
{{#include ../../tests/doc_matrix.rs:bloc_carre}}
```

## Matrice rectangulaire : bloc Lagrange

Une contrainte de Dirichlet introduit, par sa nature, un bloc **rectangulaire** : lignes indexées par les nœuds-multiplicateurs (un par contrainte), colonnes par les nœuds primaires contraints.

```rust,ignore
{{#include ../../tests/doc_matrix.rs:bloc_rectangulaire}}
```

## API Rust — accès en lecture

```rust,ignore
{{#include ../../tests/doc_matrix.rs:lecture}}
```

## API Python

```python
{{#include ../../tests/python/test_doc_conteneurs.py:matrix_api}}
```

## Ce que pèse une matrice

`memory_bytes()`, sur un bloc comme sur la matrice, estime les octets de tas
occupés. Pour un bloc, ce sont ses entrées stockées : un indice de ligne, un
indice de colonne et une valeur, soit 24 octets chacune — un bloc *calculé* n'en
stocke aucune et répond `0`. Pour la matrice, c'est la CSR assemblée (un indice
de colonne et une valeur par terme non nul, un décalage par ligne, une clé de
DDL par ligne et par colonne) plus ce que gardent ses blocs.

L'estimation apparaît dans l'affichage : `Matrix: 2 sous-matrice(s), 3 row(s) ×
3 col(s), symmetric, ~1.2 kB`.

**Ce qu'elle ne compte pas, et qui est pourtant le plus gros : la
factorisation.** Ses facteurs pèsent vingt à soixante-cinq fois la matrice,
mais faer garde leur taille privée. Pour la mémoire réellement consommée par un
solve, voir [Calculs plus gros que la RAM](operateurs/solveur.md).

## Sérialisation

`Matrix` implémente `Portable` via `serde` (comme tous les objets pyrucast). Les triplets COO, la table de noms et les DOFs voyagent dans le format binaire portable Linux ↔ Windows. La CSR assemblée et la factorisation, elles, ne sont **pas** écrites : elles se reconstruisent (voir [Sauvegarde et relecture](sauvegarde.md)).

## Limitations actuelles

- **Cache de motif non invalidé par les mutations profondes** : le motif creux mémoïsé sur le `Model` est invalidé à l'ajout d'un sous-modèle (`add_sub`), mais pas si le maillage / l'espace EF sous-jacent change *en place* (remaillage) — reconstruire le modèle dans ce cas. Le chemin de composition `m.assemble()`, lui, reconstruit toujours le motif depuis les blocs.
- **Pas de produit matrice-matrice** : à venir avec les premiers besoins concrets (préconditionneurs, formulations couplées).
- **La somme n'assemble pas de manière opportuniste** : `a + b` rend une matrice non assemblée même quand les deux opérandes le sont. Fusionner leurs CSR — ce qui éviterait de relancer les noyaux élémentaires dans une boucle en temps à pas variable — est possible sans changer la sémantique (l'ordre des DDL d'une concaténation est exactement celui de `a` suivi des DDL que seule `b` apporte), mais demande une addition creuse complète : retable des variables, remappage et retri des colonnes de `b`, fusion ligne à ligne. À faire quand un intégrateur en temps le justifiera.
- La symétrie déclarée n'est pas vérifiée numériquement à l'assemblage, et ne doit pas l'être : c'est une déclaration du modèle, pas une mesure. Des tests unitaires confrontent la déclaration à la CSR réellement assemblée ; le calcul, lui, fait confiance.
- **Une symétrie découpée en tranches de lignes n'est pas exprimable** : deux blocs rectangulaires qui sont chacun une tranche de lignes d'une matrice symétrique ne peuvent rien déclarer — ni `Full`, qui suppose un bloc carré, ni `Half`, les deux n'étant pas transposés l'un de l'autre. Le tableau assemblé est symétrique, le drapeau répond `false`, et le calcul prend le chemin général. C'est le mode de défaillance voulu : on oublie une symétrie, on n'en invente pas.
