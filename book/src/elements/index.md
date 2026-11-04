# Éléments finis

Un type de maille (`"TRI3"`, `"HEX20"`…) est aussi un **élément fini** : un
élément de référence, des fonctions de forme qui interpolent une grandeur à
partir de ses valeurs aux nœuds, et une règle d'intégration. Cette section en
donne une fiche par type. La machinerie commune à tous les éléments (passage de
l'élément de référence à la maille, jacobien, dérivées physiques) est décrite
avec l'[espace éléments finis](../fe-space.md#théorie).

En Python, un élément se lit sur un sous-espace : `gauss_xi`, `gauss_weight`,
`n_at_g` et `dn_at_g` rendent ses tables de référence (voir [Lire un
espace](../fe-space.md#lire-un-espace)). Elles ne dépendent que du type, de
l'interpolation et de la quadrature : un espace posé sur une zone **vide** les
donne déjà, et chaque fiche s'en sert dans son exemple.

## De la fonction de forme à la matrice

Sur chaque maille, une grandeur est interpolée par ses valeurs nodales,
\\( u(\xi) = \sum_i N_i(\xi)\\,u_i \\), et la géométrie de la même façon,
\\( \mathbf{x}(\xi) = \sum_i N_i(\xi)\\,\mathbf{x}_i \\). Toute matrice
élémentaire est une intégrale sur la maille, ramenée à l'élément de référence
par le jacobien \\( J = \partial\mathbf{x}/\partial\xi \\) :

\\[
\int_K \phi(\mathbf{x})\\,d\mathbf{x}
= \int_{\hat K} \phi(\mathbf{x}(\xi))\\,|J(\xi)|\\,d\xi
\approx \sum_{g} w_g\\,\phi(\xi_g)\\,|J(\xi_g)| ,
\\]

et les dérivées physiques viennent de l'inverse du jacobien,
\\( \nabla_x N_i = J^{-\mathsf T}\nabla_\xi N_i \\). Ce qui change d'un élément
à l'autre tient donc en trois choses : les \\( N_i \\), leurs dérivées
\\( \partial N_i/\partial\xi_k \\), et les points et poids
\\( (\xi_g, w_g) \\). C'est le contenu de chaque fiche.

## Catalogue

Deux familles d'interpolation : **Lagrange-1** (`"LAGRANGE1"`, un nœud par
sommet) et **Lagrange-2** (`"LAGRANGE2"`, sommets et milieux d'arête, plus des
nœuds de face et de centre pour les éléments complets). `POI1`, un nœud seul,
n'a pas d'élément de référence : ce n'est pas un élément fini.

### Lagrange-1

| Élément | Nœuds | Dimension | Élément de référence | `"GAUSS"` : points, degré exact |
|---|---:|:---:|---|---|
| [SEG2](seg2.md) | 2 | 1 | \\( \xi\in[-1,1] \\) | Gauss 2, degré 3 |
| [TRI3](tri3.md) | 3 | 2 | triangle \\( \xi+\eta\le 1 \\) | Hammer 3, degré 2 |
| [QUA4](qua4.md) | 4 | 2 | \\( [-1,1]^2 \\) | 2×2, degré 3 par direction |
| [TET4](tet4.md) | 4 | 3 | tétraèdre \\( \xi+\eta+\zeta\le 1 \\) | Hammer 4, degré 2 |
| [PYRA5](pyra5.md) | 5 | 3 | pyramide à base carrée | conique 8, degré 3 |
| [PENTA6](penta6.md) | 6 | 3 | prisme (triangle × \\( \zeta \\)) | 3 × 2 = 6, degré 2 dans le triangle, 3 en \\( \zeta \\) |
| [HEX8](hex8.md) | 8 | 3 | \\( [-1,1]^3 \\) | 2×2×2, degré 3 par direction |

### Lagrange-2

| Élément | Nœuds | Dimension | Parent | Type | `"GAUSS"` : points, degré exact |
|---|---:|:---:|:---:|---|---|
| [SEG3](seg3.md) | 3 | 1 | SEG2 | complet | Gauss 3, degré 5 |
| [TRI6](tri6.md) | 6 | 2 | TRI3 | complet | Dunavant 6, degré 4 |
| [QUA8](qua8.md) | 8 | 2 | QUA4 | sérendipité | 3×3, degré 5 par direction |
| [QUA9](qua9.md) | 9 | 2 | QUA4 | complet | 3×3, degré 5 par direction |
| [TET10](tet10.md) | 10 | 3 | TET4 | complet | Keast 11, degré 4 |
| [PENTA15](penta15.md) | 15 | 3 | PENTA6 | sérendipité | 6 × 3 = 18, degré 4 dans le triangle, 5 en \\( \zeta \\) |
| [HEX20](hex20.md) | 20 | 3 | HEX8 | sérendipité | 3×3×3, degré 5 par direction |
| [HEX27](hex27.md) | 27 | 3 | HEX8 | complet | 3×3×3, degré 5 par direction |

Les éléments **sérendipité** (QUA8, PENTA15, HEX20) n'ont que des nœuds aux
sommets et aux milieux d'arête. Les **complets** ont en plus les nœuds qu'exige
le polynôme de degré 2 entier : un nœud central pour QUA9, des centres de face
et de volume pour HEX27. Le degré exact de chaque règle a été mesuré en
intégrant les monômes \\( \xi^a\eta^b\zeta^c \\) avec les tables que rend
Python.

Tous les types se lisent par `read_gmsh` et `from_arrays`. Chaque fiche dit
quels mailleurs de pyrucast le produisent.

## Les degrés de liberté

Un **degré de liberté** est un couple (nœud, inconnue) : la valeur de `T` au
nœud 12, celle de `u_x` au nœud 3. Une maille de \\( n \\) nœuds, sous une
physique de \\( k \\) inconnues, en porte \\( n \times k \\). Sa matrice
élémentaire les range **nœud par nœud** : toutes les inconnues du nœud 0, puis
celles du nœud 1, et ainsi de suite. Deux mailles voisines partagent les
degrés de liberté de leurs nœuds communs : c'est ce qui rend le champ continu
d'une maille à l'autre.

```python
{{#include ../../../tests/python/test_doc_elements.py:ddl}}
```

Ce que chaque physique pose sur quels éléments :

| physique | éléments | inconnues par nœud |
|---|---|---|
| [conduction](../thermique.md), [diffusion](../diffusion.md) | `SEG2`, `SEG3` ; `TRI3`, `QUA4`, `TRI6`, `QUA8`, `QUA9` ; `TET4`, `PYRA5`, `PENTA6`, `HEX8`, `TET10`, `PENTA15`, `HEX20`, `HEX27` ; en volume ou plongés (un fil, une plaque mince) | `T` ; `c_<espèce>`, une par espèce |
| [échanges](../echanges.md), [rayonnement](../rayonnement.md) | le bord d'une conduction : `SEG2`, `SEG3` en 2-D ; `TRI3`, `QUA4`, `TRI6`, `QUA8`, `QUA9` en 3-D | `T` |
| [élasticité](../mecanique/elasticite.md) et ses lois | 2-D : `TRI3`, `QUA4`, `TRI6`, `QUA8`, `QUA9` ; 3-D : `TET4`, `PYRA5`, `PENTA6`, `HEX8`, `TET10`, `PENTA15`, `HEX20`, `HEX27` | `u_x`, `u_y` ; et `u_z` en 3-D |
| [barre](../mecanique/truss.md) | `SEG2` | un déplacement par dimension |
| [poutres](../mecanique/bernoulli.md) | `SEG2` | `w`, `theta` en 1-D ; `u_x`, `u_y`, `r_z` en plan ; six dans l'espace |
| [coque](../mecanique/coques.md) | `TRI3`, `QUA4`, dans l'espace | `u_x`, `u_y`, `u_z`, `r_x`, `r_y`, `r_z` |

L'élasticité refuse un élément plongé : des segments dans le plan, des
triangles dans l'espace sont un bord, pas un solide. La poutre de Bernoulli
interpole sa flèche avec `"HERMITE3"` : quatre fonctions sur un `SEG2`, une
valeur et une pente par nœud, d'où `w` **et** `theta` (voir [Fonctions de forme
d'Hermite](../fe-space.md#fonctions-de-forme-dhermite-hermite3)).

## Les règles d'intégration

Une intégrale sur l'élément de référence est remplacée par une somme pondérée
de valeurs en quelques points :

\\[
\int_{\hat K} f(\xi)\\,d\xi \approx \sum_{g=1}^{n_g} w_g\\,f(\xi_g) .
\\]

Une règle est **exacte au degré \\( p \\)** quand cette somme donne
l'intégrale exacte de tout polynôme de degré au plus \\( p \\). La somme des
poids vaut toujours la mesure de l'élément de référence : c'est l'intégrale
de la constante 1.

**Sur un segment : Gauss-Legendre.** Avec \\( n \\) points, positions et poids
font \\( 2n \\) inconnues, de quoi rendre la somme exacte pour les \\( 2n \\)
monômes \\( 1, \xi, \dots, \xi^{2n-1} \\). La solution place les points aux
racines du polynôme de Legendre de degré \\( n \\) :

| \\( n \\) | points \\( \xi_g \\) | poids \\( w_g \\) | exacte au degré |
|---:|---|---|---:|
| 2 | \\( \pm 1/\sqrt3 \\) | \\( 1, 1 \\) | 3 |
| 3 | \\( -\sqrt{3/5},\ 0,\ +\sqrt{3/5} \\) | \\( 5/9,\ 8/9,\ 5/9 \\) | 5 |

Sur \\( [0, 1] \\), la hauteur d'un prisme, les mêmes points sont ramenés par
\\( \zeta = (1 + \xi)/2 \\), et les poids divisés par 2.

**Quadrangle, hexaèdre, prisme : des produits.** Le point \\( (a, b) \\) d'une
règle produit est \\( (\xi_a, \eta_b) \\), de poids \\( w_a w_b \\). Elle
intègre exactement \\( \xi^p \eta^q \\) dès que chaque exposant est dans la
portée de la règle à une dimension : l'exactitude se compte **par
direction**. Le prisme croise la règle du triangle et Gauss-Legendre en
\\( \zeta \\).

**Triangle, tétraèdre : des règles symétriques.** Un simplexe n'est pas un
produit. Ses règles sont des tables : des points et des poids calculés une fois
pour rendre la somme exacte jusqu'à un degré donné. Les points y vont par
familles, chaque famille réunissant les images d'un point par permutation des
coordonnées barycentriques, avec un même poids : la règle ne favorise aucun
sommet.

| élément | règle | familles | degré |
|---|---|---|---:|
| `TRI3` | Hammer | 3 milieux de côté | 2 |
| `TET4` | Hammer | 4 points, un vers chaque sommet | 2 |
| `TRI6` | Dunavant | deux familles de 3 | 4 |
| `TET10` | Keast | le centre, une famille de 4, une de 6 | 4 |

Le centre de la règle de Keast a un poids **négatif**, \\( -74/5625 \\) : la
règle reste exacte au degré 4, mais elle n'est pas une moyenne pondérée de
valeurs positives.

**Pyramide : un produit conique.** La section carrée de la pyramide rétrécit
jusqu'au sommet. En écrivant \\( \xi = a(1-\zeta) \\), \\( \eta = b(1-\zeta) \\),
elle devient un produit : Gauss-Legendre 2×2 sur \\( (a, b) \\), et en
\\( \zeta \\) une règle de Gauss-Jacobi qui absorbe le facteur
\\( (1-\zeta)^2 \\) du rétrécissement. Le détail est sur la
[fiche PYRA5](pyra5.md#quadrature).

**Le degré nécessaire.** La masse \\( \int \rho\\,N_i N_j \\) est un produit de
deux fonctions de forme : de degré 2 pour un élément linéaire, 4 pour un
quadratique (par direction pour les produits). La raideur
\\( \int \nabla N_i \cdot D\\,\nabla N_j \\) est de degré moindre. La règle
`"GAUSS"` de chaque type intègre exactement les deux quand la maille est
l'image de l'élément de référence par une transformation affine (côtés droits,
faces en parallélogramme) et que le matériau est constant : \\( J \\) est
alors constant, et l'intégrande reste un polynôme. Sur une maille courbe ou
déformée, \\( J^{-1} \\) n'est plus un polynôme, et aucune règle n'intègre
exactement la raideur.

**La règle `"REDUCED"`** n'a qu'un point, au centre de gravité de l'élément de
référence, de poids égal à sa mesure. Elle intègre exactement les fonctions
affines. Elle porte une valeur par maille (`element_field.cell_integral`), et
les [coques](../mecanique/coques.md) épaisses y intègrent leur cisaillement
transverse.

```python
{{#include ../../../tests/python/test_doc_elements.py:reduite}}
```

## Propriétés communes

Toutes les interpolations de Lagrange vérifient, en tout point de référence :

- **Kronecker** : \\( N_i(\xi_j) = \delta_{ij} \\) aux nœuds, si bien que
  \\( u_i \\) est la valeur du champ au nœud \\( i \\) ;
- **partition de l'unité** : \\( \sum_i N_i = 1 \\), d'où un champ constant
  reproduit exactement ;
- **dérivées de somme nulle** : \\( \sum_i \partial N_i/\partial\xi_k = 0 \\).

Les fonctions des types quadratiques prennent des valeurs **négatives** dans
la maille : celle d'un sommet de `TET10` vaut \\( -1/8 \\) au centre.

## Le plan des fiches

Chaque fiche suit le même plan :

- **En bref** : le nom Python, les nœuds, l'interpolation, la dimension, le
  bord, la quadrature, le type apparenté et les mailleurs qui le produisent ;
- **Élément de référence** : son domaine et la position de chaque nœud ;
- **Fonctions de forme** et **Dérivées de référence** ;
- **Degrés de liberté** : les inconnues qu'il porte sous chaque physique ;
- **Quadrature** : les points et les poids de `"GAUSS"`, et ce qu'elle intègre
  exactement ;
- **Exemple** : ses tables de référence, lues en Python ;
- **Côté Rust**.

## Côté Rust

`ElementType` (`src/atoms/element_type.rs`) énumère les types ;
`ElementType::as_kind` rend l'implémentation de `ElementKind`
(`src/atoms/element_kind/mod.rs`) propre à chacun. Un élément tient dans un
fichier, `src/atoms/element_kind/<nom>.rs` : nœuds de référence, fonctions de
forme et dérivées (`shape_into`, `dshape_into`), règle de Gauss (`gauss`),
faces et arêtes, domaine de référence (`contains_ref`, `clamp_ref`), codes
d'échange VTK et gmsh, type quadratique ou parent linéaire. La règle réduite
n'est écrite nulle part : c'est la méthode par défaut `reduced`, un point en
`ref_centroid` de poids `ref_measure`.

`Interpolation` et `QuadratureRule` (`src/atoms/element_kind/`) sont des
étiquettes : les tables vivent dans chaque élément. Les tests de ce dossier
vérifient, pour tous les types, Kronecker, la partition de l'unité, la somme
des poids, les faces et les arêtes, et pour les types quadratiques et la
pyramide, les dérivées contre des différences finies ; ceux de `quadrature.rs`
intègrent des monômes pour `SEG2`, `TRI3`, `TRI6` et `TET10`.
