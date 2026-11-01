# Espace éléments finis (FiniteElementSpace/SubFiniteElementSpace)

Un [maillage](mesh.md) dit où sont les mailles ; un espace éléments finis dit
comment on calcule dessus. Pour chaque zone du maillage, il fixe :

- une **interpolation** : les fonctions de forme, qui donnent une grandeur en
  tout point d'une maille à partir de ses valeurs aux nœuds ;
- une **quadrature** : les points de Gauss et leurs poids, où l'on évalue les
  intégrales.

C'est un [agrégat](aggregate.md) de **sous-espaces** : un
`SubFiniteElementSpace` par zone de maillage, dans le même ordre. Le maillage
reste purement géométrique, et un même maillage peut porter plusieurs espaces.

Presque tout ce qui calcule s'appuie sur un espace : les
[physiques](model.md) le reçoivent (`model.heat_conduction(fes)`), les
[champs aux points de Gauss](element-field.md) posent leurs zones sur ses
sous-espaces, et les opérateurs qui interpolent ou intègrent
(`element_field.interp_to_gauss`, `element_field.gradient`,
`measure.integral`…) le prennent en argument.

## Construire un espace

`pyrucast.FiniteElementSpace(mesh)` pose sur chaque zone l'interpolation
`"LAGRANGE1"` et la quadrature `"GAUSS"` ; `FiniteElementSpace.lagrange1(mesh)`
en est un autre nom. Les arguments `interpolation=` et `quadrature=` changent
l'un ou l'autre pour toutes les zones à la fois, et
`FiniteElementSpace.with_choices(mesh, [(interpolation, quadrature), …])` en
donne un couple par zone.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:fes_construire}}
```

| interpolation | mailles | ce qu'elle interpole |
|---|---|---|
| `"LAGRANGE1"` (défaut) | linéaires : `SEG2`, `TRI3`, `QUA4`, `TET4`, `PYRA5`, `PENTA6`, `HEX8` | une valeur par nœud |
| `"LAGRANGE2"` | quadratiques : `SEG3`, `TRI6`, `QUA8`, `QUA9`, `TET10`, `PENTA15`, `HEX20`, `HEX27` | une valeur par nœud, sommets et nœuds milieux |
| `"HERMITE3"` | `SEG2` | une valeur **et une pente** par nœud : la poutre d'[Euler-Bernoulli](mecanique/bernoulli.md) |
| `"MODEL_EMBEDDED"` | toutes | rien : la physique porte ses propres fonctions, comme la poutre de [Timoshenko](mecanique/timoshenko.md) |

| quadrature | points par maille |
|---|---|
| `"GAUSS"` (défaut) | la règle du type de maille, voir [Quadrature de Gauss](#quadrature-de-gauss) |
| `"REDUCED"` | un seul, au centre de la maille |

Une interpolation de Lagrange doit avoir le **degré des mailles** : un maillage
qui mêle zones linéaires et quadratiques se pose avec `with_choices`.
`"REDUCED"` sert à porter une valeur par maille, par exemple l'intégrale que
rend `element_field.cell_integral` ; les [coques](mecanique/coques.md)
l'emploient aussi, en interne, pour leur cisaillement transverse.

La construction est refusée pour :

- un maillage sans zone, ou une liste `with_choices` qui ne compte pas un
  couple par zone ;
- une zone `POI1` : un point n'a pas d'élément de référence ;
- une interpolation que le type de maille n'admet pas (`"LAGRANGE2"` sur des
  `TRI3`, `"HERMITE3"` ailleurs que sur des `SEG2`) ;
- des mailles de dimension supérieure à celle des `Coords` (des `TRI3` dans des
  `Coords` 1-D) ;
- un nom d'interpolation ou de quadrature inconnu.

**Construire l'espace scelle les zones de maillage.** Leurs mailles sont
figées : `add_cell` y lève ensuite une erreur. Tout ce qui se pose sur l'espace
— les valeurs d'un champ aux points de Gauss, les lignes d'une matrice, la
[coloration](#ce-qui-est-mis-en-cache) — est rangé par rang de maille, et une
maille ajoutée décalerait tout. Les **positions** des nœuds, elles, restent
libres (voir [Déplacer le maillage](#déplacer-le-maillage)). Le sceau est décrit
avec le [maillage](mesh.md#scellement).

## Lire un espace

`fes[i]`, `fes.unit()` et `for sub in fes` donnent les sous-espaces. Un
sous-espace donne son **élément de référence** (points de Gauss, poids,
fonctions de forme et leurs dérivées en \\( \xi \\)) et, pour chaque maille,
les **grandeurs physiques** tirées de ses nœuds : le jacobien \\( J \\), sa
mesure \\( |J| \\) et les dérivées \\( \partial N_i/\partial x_a \\). Un point
s'y désigne par le rang de la maille dans la zone et le rang du point de Gauss
dans la maille.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:fes_lire}}
```

| `FiniteElementSpace` | |
|---|---|
| `len(fes)`, `fes[i]`, `for sub in fes`, `unit()`, `\|` | l'interface des [agrégats](aggregate.md) |
| `mesh()` | le maillage couvert : ses zones elles-mêmes, une fois chacune |
| `element(i, maille)`, `elements(i)` | la vue sur une maille du sous-espace `i`, ou sur toutes |
| `dump()` | le contenu, points de Gauss compris |

| `SubFiniteElementSpace` (sous-espace) | |
|---|---|
| `element_type`, `interpolation`, `quadrature` | le type de maille et les deux choix de la zone |
| `ref_dim`, `space_dim`, `nodes_per_cell` | la dimension de l'élément de référence, celle des `Coords`, les nœuds par maille |
| `is_axisymmetric` | si les `Coords` décrivent un corps de révolution |
| `cell_count()`, `gauss_count()` | les mailles, les points de Gauss par maille |
| `gauss_xi(g)`, `gauss_weight(g)` | la position du point `g` dans l'élément de référence, son poids |
| `n_at_g(g)`, `dn_at_g(g)` | \\( N_i \\) et \\( \partial N_i/\partial\xi_k \\) au point `g` |
| `jacobian(maille, g)`, `det_jacobian(maille, g)`, `dn_dx(maille, g)` | \\( J \\), \\( \lvert J\rvert \\) et \\( \partial N_i/\partial x_a \\) d'une maille au point `g` |
| `sub[maille]`, `element(maille)`, `elements()`, `len(sub)` | les vues sur les mailles |

Une vue de maille (`Element`) donne les mêmes grandeurs sans le rang de la
maille — `jacobian(g)`, `det_jacobian(g)`, `dn_dx(g)`… —, plus `index`,
`cell()` et `nodes()`. Ses dimensions (`gauss_count`, `nodes_per_cell`,
`ref_dim`, `space_dim`) sont des propriétés, sans parenthèses.

Les tableaux sont rendus **à plat**, ligne par ligne :

| méthode | longueur | case |
|---|---|---|
| `n_at_g(g)` | `nodes_per_cell` | `[i]` = \\( N_i \\) |
| `dn_at_g(g)` | `nodes_per_cell × ref_dim` | `[i * ref_dim + k]` = \\( \partial N_i/\partial\xi_k \\) |
| `jacobian(maille, g)` | `space_dim × ref_dim` | `[a * ref_dim + k]` = \\( \partial x_a/\partial\xi_k \\) |
| `dn_dx(maille, g)` | `nodes_per_cell × space_dim` | `[i * space_dim + a]` = \\( \partial N_i/\partial x_a \\) |

Ces méthodes lisent les fonctions de forme **géométriques**, celles qui placent
les points de la maille. Sous `"LAGRANGE1"` et `"LAGRANGE2"`, ce sont aussi
celles du champ ; sous `"HERMITE3"` et `"MODEL_EMBEDDED"`, elles ne décrivent
que la géométrie (voir [plus bas](#fonctions-de-forme-dhermite-hermite3)).

**Un espace par formulation, construit une fois.** Chaque construction crée
des sous-espaces neufs : deux `FiniteElementSpace(mesh)` sont deux espaces,
même avec les mêmes choix, et les champs posés sur l'un ne se combinent pas
point à point avec ceux de l'autre (voir [Champ aux points de
Gauss](element-field.md#le-support--un-espace-éléments-finis)). On construit
donc l'espace une fois et on le passe à tous les opérateurs. L'union `|`
garde chaque sous-espace une seule fois et refuse deux espaces posés sur des
`Coords` différents.

## Déplacer le maillage

Les grandeurs physiques ne sont pas stockées : chaque appel les recalcule à
partir des positions **actuelles** des nœuds. Déplacer un nœud suffit, sans
rien reconstruire ni signaler :

```python
{{#include ../../tests/python/test_doc_conteneurs.py:fes_deplacer}}
```

L'espace ne contrôle pas la forme des mailles, ni à la construction ni après
un déplacement. `det_jacobian` rend une **valeur absolue** : une maille
retournée (ses nœuds parcourus dans l'autre sens) mesure autant que la même à
l'endroit. Une maille aplatie mesure 0, et `dn_dx` la refuse, faute de pouvoir
inverser son jacobien.

## Ce qui est mis en cache

Deux choses, gardées dans chaque sous-espace pour toute sa durée de vie :

- **Les tables de référence**, calculées à la construction : positions et
  poids des points de Gauss, \\( N_i(\xi_g) \\) et
  \\( \partial N_i/\partial\xi_k(\xi_g) \\) — sous `"HERMITE3"`, en plus, les
  fonctions du champ et leurs dérivées secondes. Elles ne dépendent que du type
  de maille, de l'interpolation et de la quadrature : calculées une fois, elles
  servent ensuite à toutes les mailles et à tous les opérateurs. Leur taille ne
  dépend pas du nombre de mailles, quelques milliers de nombres au plus (un
  `HEX27` à 27 points).
- **La coloration des mailles**, calculée au premier usage : un partage des
  mailles en couleurs, deux mailles de même couleur n'ayant aucun nœud commun.
  Les mailles d'une couleur écrivent alors en parallèle sans se gêner, dans
  l'assemblage des matrices (`matrix.stiffness`, `matrix.mass`,
  `matrix.tangent`…) et dans les opérateurs qui accumulent aux nœuds
  (`node_field.internal_forces`, `node_field.external_forces`,
  `node_field.divergence`, `node_field.sum_to_nodes`). La calculer demande un
  passage sur toute la connectivité ; la garder évite de le refaire à chaque
  appel, alors qu'un calcul non linéaire réassemble à chaque itération. Elle ne
  dépend que de la connectivité, que le sceau fige.

Ce qui n'est **pas** gardé : \\( J \\), \\( |J| \\),
\\( \partial N_i/\partial x_a \\) et la position réelle des points de Gauss.
Les garder coûterait une matrice par point de Gauss — de la mémoire en
proportion du maillage — et les rendrait fausses au premier nœud déplacé. Les
recalculer coûte une somme sur les nœuds de la maille, que chaque assemblage
fait une fois par point de Gauss avant de s'en servir pour tous les termes de
la matrice élémentaire.

## Sauvegarde

Un espace se [sauvegarde](sauvegarde.md) avec ses zones de maillage et ses
`Coords`. À la lecture, le sceau des zones est conservé, et les tables de
référence sont relues telles quelles. La coloration n'est pas écrite : elle se
recalcule au premier usage. Un espace et les champs posés dessus, sauvegardés
dans un même fichier, restent liés à la lecture : les champs relus sont posés
sur l'espace relu.

## Théorie

### Élément de référence

Chaque type de maille a un **élément de référence** \\( \hat K \\), décrit dans
un repère \\( \xi \\), et une numérotation locale de ses nœuds :

| type | repère \\( \xi \\) | ordre des nœuds |
|---|---|---|
| `SEG2` | \\( \xi \in [-1, +1] \\) | nœud 0 en \\( \xi = -1 \\), nœud 1 en \\( \xi = +1 \\) |
| `TRI3` | \\( \xi, \eta \in [0, 1] \\), \\( \xi + \eta \le 1 \\) | \\( (0,0), (1,0), (0,1) \\), sens trigonométrique |
| `QUA4` | \\( \xi, \eta \in [-1, +1] \\) | \\( (-1,-1), (1,-1), (1,1), (-1,1) \\), sens trigonométrique |
| `TET4` | \\( \xi, \eta, \zeta \in [0, 1] \\), \\( \xi + \eta + \zeta \le 1 \\) | \\( (0,0,0), (1,0,0), (0,1,0), (0,0,1) \\) ; face 0-1-2 en sens trigonométrique vue du nœud 3 |
| `PYRA5` | \\( \zeta \in [0, 1] \\), \\( \xi, \eta \in [-(1-\zeta), +(1-\zeta)] \\) | base carrée en sens trigonométrique vue du sommet (nœuds 0 à 3 en \\( \zeta = 0 \\)), puis le sommet : \\( (-1,-1,0), (1,-1,0), (1,1,0), (-1,1,0), (0,0,1) \\) |
| `PENTA6` | \\( \xi, \eta \in [0, 1] \\), \\( \xi + \eta \le 1 \\), \\( \zeta \in [0, 1] \\) | triangle du bas (nœuds 0 à 2 en \\( \zeta = 0 \\)) puis triangle du haut (nœuds 3 à 5 en \\( \zeta = 1 \\)), tous deux en sens trigonométrique |
| `HEX8` | \\( \xi, \eta, \zeta \in [-1, +1] \\) | face du bas (nœuds 0 à 3) puis face du haut (nœuds 4 à 7), en sens trigonométrique |

Les types **quadratiques** reprennent le repère et les sommets de leur parent
linéaire, puis ajoutent les nœuds de **milieu d'arête**, dans l'ordre des
arêtes de la convention VTK :

| type | parent | nœuds ajoutés, arête \\( (a,b) \\) |
|---|---|---|
| `SEG3` | `SEG2` | nœud 2 sur \\( (0,1) \\) (\\( \xi = 0 \\)) |
| `TRI6` | `TRI3` | 3 sur \\( (0,1), (1,2), (2,0) \\) |
| `QUA8` | `QUA4` | 4 sur \\( (0,1), (1,2), (2,3), (3,0) \\) |
| `QUA9` | `QUA4` | les 4 arêtes de `QUA8`, puis un nœud **central** 8 en \\( (0,0) \\) |
| `TET10` | `TET4` | 6 sur \\( (0,1), (1,2), (2,0), (0,3), (1,3), (2,3) \\) |
| `PENTA15` | `PENTA6` | 9 : bas \\( (0,1),(1,2),(2,0) \\), haut \\( (3,4),(4,5),(5,3) \\), verticales \\( (0,3),(1,4),(2,5) \\) |
| `HEX20` | `HEX8` | 12 : bas \\( (0,1),(1,2),(2,3),(3,0) \\), haut \\( (4,5),(5,6),(6,7),(7,4) \\), verticales \\( (0,4),(1,5),(2,6),(3,7) \\) |
| `HEX27` | `HEX8` | les 12 arêtes de `HEX20`, puis 6 **centres de face** (`x∓`, `y∓`, `z∓`, nœuds 20 à 25) et un **centre de volume** 26 |

`QUA8`, `HEX20` et `PENTA15` sont **sérendipité** (nœuds d'arête seulement) ;
`SEG3`, `TRI6`, `TET10`, `QUA9` et `HEX27` sont des Lagrange complets
(`QUA9`/`HEX27` : produits tensoriels \\( Q_2 \\), avec nœuds de face et
central).

Les mailleurs produisent leurs mailles dans ces conventions, l'export VTK les
écrit telles quelles et la lecture gmsh y remet ses nœuds milieux. La fiche de
chaque type, avec ses fonctions de forme et sa quadrature, est dans le
[catalogue des éléments](elements/index.md) ; cette section décrit ce qui leur
est commun.

### Transformation isoparamétrique

Soit une maille à \\( n \\) nœuds, de coordonnées physiques
\\( \mathbf{x}\_1, \dots, \mathbf{x}\_n \in \mathbb{R}^{d_s} \\), avec
\\( d_s \\) la dimension des `Coords`. La **transformation géométrique**
\\( \chi : \hat{K} \to K \\), qui envoie l'élément de référence sur la maille,
est interpolée par les fonctions de forme :

\\[
\mathbf{x}(\xi) = \chi(\xi) = \sum_{i=1}^{n} N_i(\xi)\\, \mathbf{x}\_i
\\]

C'est l'hypothèse **isoparamétrique** : la géométrie s'interpole comme un
champ. En Lagrange-1, \\( \chi \\) est affine sur les simplexes (`SEG2`,
`TRI3`, `TET4`), bilinéaire sur `QUA4` et trilinéaire sur `HEX8`.

### Fonctions de forme de Lagrange

Une fonction de forme de Lagrange vérifie la **propriété de Kronecker** :
\\( N_i(\xi_j) = \delta_{ij} \\) aux nœuds de référence. En Lagrange-1 :

**SEG2** (\\( \xi \in [-1, +1] \\)) :
\\[
N_1(\xi) = \frac{1 - \xi}{2}, \qquad
N_2(\xi) = \frac{1 + \xi}{2}
\\]

**TRI3** (coordonnées barycentriques sur le simplexe \\( \xi + \eta \le 1 \\)) :
\\[
N_1 = 1 - \xi - \eta, \qquad
N_2 = \xi, \qquad
N_3 = \eta
\\]

**QUA4** (bilinéaire sur \\( [-1, +1]^2 \\)) :
\\[
N_1 = \tfrac{1}{4}(1-\xi)(1-\eta), \quad
N_2 = \tfrac{1}{4}(1+\xi)(1-\eta), \quad
N_3 = \tfrac{1}{4}(1+\xi)(1+\eta), \quad
N_4 = \tfrac{1}{4}(1-\xi)(1+\eta)
\\]

**TET4** (coordonnées barycentriques sur le simplexe \\( \xi + \eta + \zeta \le 1 \\)) :
\\[
N_1 = 1 - \xi - \eta - \zeta, \quad
N_2 = \xi, \quad
N_3 = \eta, \quad
N_4 = \zeta
\\]

**HEX8** (trilinéaire sur \\( [-1, +1]^3 \\)). Pour le nœud \\( i \\) de
coordonnées de référence \\( (\xi_i, \eta_i, \zeta_i) \in \\{-1, +1\\}^3 \\),
\\[
N_i(\xi, \eta, \zeta) = \tfrac{1}{8}\\,
(1 + \xi_i\\, \xi)\\,
(1 + \eta_i\\, \eta)\\,
(1 + \zeta_i\\, \zeta)
\\]

**PENTA6** (prisme, produit d'un `TRI3` par un `SEG2` linéaire sur
\\( \zeta \in [0, 1] \\)). Avec les coordonnées barycentriques du triangle
\\( L_1 = 1 - \xi - \eta \\), \\( L_2 = \xi \\), \\( L_3 = \eta \\),
\\[
N_j(\xi, \eta, \zeta) = L_j\\,(1 - \zeta) \quad (j = 1, 2, 3), \qquad
N_{j+3}(\xi, \eta, \zeta) = L_j\\,\zeta \quad (j = 1, 2, 3)
\\]

**PYRA5** a des fonctions **rationnelles**, qui recollent la base carrée au
sommet : voir sa [fiche](elements/pyra5.md).

En Lagrange-2, avec les coordonnées barycentriques \\( L_i \\) pour les
simplexes :

- **SEG3** : \\( N_0 = \tfrac12\xi(\xi-1),\\; N_1 = \tfrac12\xi(\xi+1),\\; N_2 = 1-\xi^2 \\) ;
- **TRI6 / TET10** : sommets \\( L_i(2L_i-1) \\), milieux d'arête \\( 4 L_a L_b \\) ;
- **QUA9** : produit tensoriel \\( N_i(\xi,\eta) = \ell_{a}(\xi)\\,\ell_{b}(\eta) \\) des trois fonctions de Lagrange 1-D \\( \ell_{-}(t)=\tfrac12 t(t-1),\\; \ell_0(t)=1-t^2,\\; \ell_{+}(t)=\tfrac12 t(t+1) \\) ;
- **HEX27** : produit tensoriel 3-D \\( N_i(\xi,\eta,\zeta) = \ell_{a}(\xi)\\,\ell_{b}(\eta)\\,\ell_{c}(\zeta) \\) des mêmes fonctions ;
- **QUA8** (sérendipité) : sommet \\( \tfrac14(1+\xi_i\xi)(1+\eta_i\eta)(\xi_i\xi+\eta_i\eta-1) \\) ; milieu \\( \tfrac12(1-\xi^2)(1+\eta_i\eta) \\) ou \\( \tfrac12(1+\xi_i\xi)(1-\eta^2) \\) selon l'arête ;
- **HEX20** (sérendipité) : sommet \\( \tfrac18(1+\xi_i\xi)(1+\eta_i\eta)(1+\zeta_i\zeta)(\xi_i\xi+\eta_i\eta+\zeta_i\zeta-2) \\) ; milieu \\( \tfrac14(1-\xi^2)(1+\eta_i\eta)(1+\zeta_i\zeta) \\) et ses permutations ;
- **PENTA15** (sérendipité) : produit du `TRI6` par un facteur quadratique en \\( \zeta \\), corrigé aux sommets.

Toutes forment une **partition de l'unité**, \\( \sum_i N_i(\xi) = 1 \\) en
tout \\( \xi \\) : un champ constant est reproduit exactement. Les tests le
vérifient, avec Kronecker, et recoupent les dérivées des types quadratiques par
différences finies.

### Dérivées de référence

Les **dérivées de référence** \\( \partial N_i / \partial \xi_k \\) s'obtiennent
en dérivant les formules. En Lagrange-1, elles sont **constantes** sur
`SEG2`, `TRI3` et `TET4` ; sur les autres types, et sur tous les quadratiques,
elles sont polynomiales en \\( \xi \\). La partition de l'unité dérivée donne
\\( \sum_i \partial N_i / \partial \xi_k = 0 \\) dans chaque direction
\\( k \\), identité elle aussi testée.

### Fonctions de forme d'Hermite (`HERMITE3`)

Les fonctions de Lagrange sont **C⁰** : le champ est continu d'une maille à la
suivante, sa dérivée ne l'est pas. Cela suffit aux équations du second ordre.
Une poutre d'Euler-Bernoulli obéit à \\( (EIw\'\')\'\' = q \\), du
**quatrième** ordre, dont la forme faible exige un champ **C¹** : flèche *et*
pente continues.

C'est ce que fournit `"HERMITE3"`, sur `SEG2` :

\\[
\begin{aligned}
H_1 &= \tfrac14(2 - 3\xi + \xi^3), &\qquad H_2 &= \tfrac14(1 - \xi - \xi^2 + \xi^3),\\\\
H_3 &= \tfrac14(2 + 3\xi - \xi^3), &\qquad H_4 &= \tfrac14(-1 - \xi + \xi^2 + \xi^3).
\end{aligned}
\\]

La propriété de Kronecker y porte sur **deux** grandeurs : à chaque extrémité,
une fonction vaut 1 et a une pente nulle, l'autre vaut 0 et a une pente 1, et
les deux fonctions de l'extrémité opposée s'y annulent avec leur pente. D'où
l'ordre des degrés de liberté, \\( [w_A,\ w\'\_A,\ w_B,\ w\'\_B] \\) : une
valeur et une pente par nœud.

**Quatre fonctions pour deux nœuds.** Le nombre de fonctions de forme n'est
plus le nombre de nœuds : les tables du champ sont dimensionnées sur le nombre
de fonctions, celles de la géométrie sur le nombre de nœuds.

**L'élément est sous-paramétrique.** Un `SEG2` reste un segment droit,
interpolé en Lagrange-1 quel que soit le champ qu'il porte. Le sous-espace
tabule donc **deux** bases : la géométrique, qui donne le jacobien
\\( \partial x/\partial\xi \\) et que lisent `n_at_g`, `dn_at_g` et `dn_dx`, et
celle du champ, qui interpole l'inconnue. Sous Lagrange, les deux coïncident et
seule la géométrique est stockée.

**Dérivées secondes.** `"HERMITE3"` est la seule interpolation à tabuler
\\( \partial^2 N_i/\partial\xi^2 \\) : la courbure est une grandeur
**primaire** d'un élément C¹. Sur `SEG2`, ces dérivées sont linéaires en
\\( \xi \\) : la courbure **varie** dans la maille. Le passage au physique est
une règle de chaîne, sans terme en \\( \partial J/\partial\xi \\) puisqu'un
segment a un jacobien constant :

\\[
\frac{\partial^2 N_i}{\partial x^2} = \frac{1}{J^2}\\,
\frac{\partial^2 N_i}{\partial \xi^2}, \qquad J = \frac{L}{2}.
\\]

**La pente de référence n'est pas la rotation.** \\( H_2 \\) et \\( H_4 \\)
portent une pente \\( \partial w/\partial\xi \\), alors que le degré de liberté
d'une poutre est \\( \theta = \partial w/\partial x \\). Les deux diffèrent du
jacobien, \\( \partial w/\partial\xi = J\\,\partial w/\partial x \\). Ce facteur
n'est pas dans les fonctions de forme : c'est un passage référence → physique,
appliqué là où sont appliqués tous les autres jacobiens. Avec un \\( L \\)
dans la fonction de forme, la base dépendrait de la maille et ne pourrait plus
être tabulée une fois par type.

**Ce qui le vérifie.** La raideur d'Euler-Bernoulli est l'oracle capable de
mettre cette base en défaut :

\\[
K = \int_0^L EI \left(\frac{\partial^2 N}{\partial x^2}\right)^{\\!\top}
\frac{\partial^2 N}{\partial x^2}\\, dx
= \frac{EI}{L^3}
\begin{bmatrix}
 12 & 6L & -12 & 6L \\\\
 6L & 4L^2 & -6L & 2L^2 \\\\
-12 & -6L & 12 & -6L \\\\
 6L & 2L^2 & -6L & 4L^2
\end{bmatrix}.
\\]

`tests/hermite.rs` intègre le membre de gauche depuis la base tabulée et le
compare à la forme fermée, à la précision machine. Une erreur dans les
fonctions, dans leurs dérivées secondes ou dans le facteur jacobien des pentes
apparaît dans cette comparaison. L'intégrande étant quadratique en \\( \xi \\),
les deux points de Gauss du `SEG2` l'intègrent **exactement**.

### Pas de base du tout : `MODEL_EMBEDDED`

Un élément dont la matrice élémentaire est une **forme fermée** n'évalue jamais
de fonction de forme : son intégrale a été faite une fois, à la main, et seul le
résultat figure dans le code. La poutre de Timoshenko exacte en est un : ses
fonctions de forme dépendent du matériau, à travers
\\( \Phi = 12EI/(G A_s L^2) \\), et aucun espace ne peut les tabuler.

Y déclarer une interpolation de Lagrange affirmerait quelque chose que la
physique n'utilise pas, et qui serait **faux** : la forme fermée vient de
fonctions cubiques et quadratiques, pas linéaires. `"MODEL_EMBEDDED"` déclare
qu'il n'y a pas de base du champ :

| interpolation | géométrie | champ |
|---|---|---|
| `"LAGRANGE1"` / `"LAGRANGE2"` | Lagrange | la même |
| `"HERMITE3"` | Lagrange-1 | Hermite, 4 fonctions |
| `"MODEL_EMBEDDED"` | Lagrange | **aucune** |

La géométrie reste entièrement définie — coordonnées, jacobien, mesure — et
tout le reste fonctionne comme sur un autre espace : champ matériau, champs aux
points de Gauss, assemblage, coloration. Seule l'interpolation de **l'inconnue**
n'appartient pas à l'espace, et les opérateurs qui en ont besoin lèvent une
erreur qui le dit, plutôt que de se rabattre sur la base géométrique :

| opérateur | ce qu'il interpole |
|---|---|
| `model.flux` | la fonction test d'une charge répartie |
| `element_field.deformation` | le déplacement, pour la déformation |
| `element_field.shell_deformation` | le déplacement d'une coque |
| `element_field.interp_to_gauss` | un champ aux nœuds, porté aux points de Gauss |
| `measure.integral` | le champ intégré |

Le repli serait plausible, et c'est ce qui le rendrait dangereux : sur une
poutre, la charge répartie cohérente vaut \\( qL/2 \\) **et**
\\( qL^2/12 \\) en moment ; une base linéaire donnerait \\( qL/2 \\) sans
moment.

La **visualisation**, elle, reste sur la base géométrique : ce qu'elle colorie
est ce qu'elle dessine, un segment droit ou une facette plane. C'est une image,
pas une valeur calculée.

La formulation fournit alors elle-même ce que l'espace ne peut plus
interpoler : `element_field.beam_deformation` évalue les déformations de la
poutre depuis les fonctions de forme **de l'élément**, et reçoit pour cela le
champ matériau, dont \\( \Phi \\) dépend.

### Quadrature de Gauss

Pour intégrer une fonction \\( f \\) sur une maille, on se ramène à l'élément
de référence par le changement de variables \\( \mathbf{x} = \chi(\xi) \\) :

\\[
\int_K f(\mathbf{x})\\, d\mathbf{x}
= \int_{\hat{K}} f(\chi(\xi))\\, |J(\xi)|\\, d\xi
\approx \sum_{g=1}^{n_g} w_g\\, f(\chi(\xi_g))\\, |J(\xi_g)|
\\]

avec \\( |J| \\) la mesure du jacobien, définie plus bas. Le couple
\\( (\xi_g, w_g) \\) est la règle de quadrature. La règle `"GAUSS"` de chaque
type intègre **exactement sa matrice de masse** sur une maille à géométrie
droite :

| type | \\( n_g \\) | règle | exactitude polynomiale |
|---|---:|---|---|
| `SEG2` | 2 | Gauss-Legendre sur \\([-1, +1]\\) : \\( \xi_g = \pm 1/\sqrt{3} \\), \\( w_g = 1 \\) | \\( \deg \le 3 \\) |
| `TRI3` | 3 | Hammer, milieux d'arête : \\( (\tfrac{1}{2}, 0), (\tfrac{1}{2}, \tfrac{1}{2}), (0, \tfrac{1}{2}) \\), \\( w_g = 1/6 \\) | \\( \deg \le 2 \\) |
| `QUA4` | 4 | produit 2×2 de Gauss-Legendre : \\( \xi_g = (\pm 1/\sqrt{3}, \pm 1/\sqrt{3}) \\), \\( w_g = 1 \\) | \\( \deg \le 3 \\) par direction |
| `TET4` | 4 | Hammer : \\( \alpha = \tfrac{5 - \sqrt{5}}{20} \\), \\( \beta = \tfrac{5 + 3\sqrt{5}}{20} \\), points permutés, \\( w_g = 1/24 \\) | \\( \deg \le 2 \\) |
| `PYRA5` | 8 | produit **conique** : 2×2 Gauss-Legendre sur la section carrée × Gauss-Jacobi 2 points en \\( \zeta \\) (poids \\( (1-\zeta)^2 \\), nœuds \\( \tfrac13 \mp \tfrac{\sqrt{10}}{15} \\)) | \\( \deg \le 2 \\) |
| `PENTA6` | 6 | produit de la règle `TRI3` (3 points, \\( w = 1/6 \\)) et de Gauss-Legendre 2 points sur \\( \zeta \in [0, 1] \\) (\\( \zeta_g = \tfrac{1}{2} \pm \tfrac{1}{2\sqrt{3}} \\), \\( w = 1/2 \\)) | \\( \deg \le 2 \\) en \\( (\xi, \eta) \\), \\( \le 3 \\) en \\( \zeta \\) |
| `HEX8` | 8 | produit 2×2×2 de Gauss-Legendre : \\( \xi_g = (\pm 1/\sqrt{3})^3 \\), \\( w_g = 1 \\) | \\( \deg \le 3 \\) par direction |
| `SEG3` | 3 | Gauss-Legendre 3 points | \\( \deg \le 5 \\) |
| `TRI6` | 6 | règle symétrique de degré 4 (Dunavant) | \\( \deg \le 4 \\) |
| `QUA8` | 9 | produit 3×3 | \\( \deg \le 5 \\) par direction |
| `QUA9` | 9 | produit 3×3 | \\( \deg \le 5 \\) par direction |
| `TET10` | 11 | règle de Keast de degré 4 (un poids négatif) | \\( \deg \le 4 \\) |
| `PENTA15` | 18 | produit `TRI6` × Gauss 3 points en \\( \zeta \\) | \\( \deg \le 4 \\), \\( \le 5 \\) en \\( \zeta \\) |
| `HEX20` | 27 | produit 3×3×3 | \\( \deg \le 5 \\) par direction |
| `HEX27` | 27 | produit 3×3×3 | \\( \deg \le 5 \\) par direction |

La règle `"REDUCED"` n'a qu'un point, au centre de l'élément de référence, de
poids égal à sa mesure : elle n'intègre exactement que les constantes.

La somme des poids vaut la mesure de l'élément de référence : 2 pour
`SEG2`/`SEG3`, 1/2 pour `TRI3`/`TRI6`, 4 pour `QUA4`/`QUA8`/`QUA9`, 1/6 pour
`TET4`/`TET10`, 4/3 pour `PYRA5`, 1/2 pour `PENTA6`/`PENTA15`, 8 pour
`HEX8`/`HEX20`/`HEX27`. Les tests le vérifient pour tous les types, et
vérifient l'exactitude des règles `TRI6` et `TET10` en intégrant des monômes.

### Jacobien et grandeurs physiques

Le **jacobien** de la transformation \\( \chi \\) est la matrice des dérivées
\\[
J_{a,k}(\xi) = \frac{\partial x_a}{\partial \xi_k}
= \sum_{i=1}^{n} \mathbf{x}\_{i,a}\\, \frac{\partial N_i}{\partial \xi_k}(\xi)
\quad
\text{de taille } d_s \times d_r
\\]
où \\( a \\) parcourt les \\( d_s \\) directions physiques et \\( k \\) les
\\( d_r = \dim \hat K \\) directions de référence.

**Cas plein : \\( d_s = d_r \\).** Quand la maille a la dimension des `Coords`
(un `TRI3` dans des `Coords` 2-D), \\( J \\) est carrée. Son déterminant mesure
la dilatation locale du volume, et c'est sa **valeur absolue** qui entre dans
l'intégration : `det_jacobian` rend \\( |\det J| \\). Les dérivées physiques des
fonctions de forme passent par l'inverse de \\( J \\) :
\\[
\frac{\partial N_i}{\partial x_a}
= \sum_{k=1}^{d_r} (J^{-1})\_{k, a} \\, \frac{\partial N_i}{\partial \xi_k}
\quad \Longleftrightarrow \quad
\nabla_x N_i = J^{-T}\\, \nabla_\xi N_i
\\]

**Cas plongé : \\( d_s > d_r \\).** Une zone peut être **plongée** dans un
espace de dimension supérieure : des `SEG2` dans des `Coords` 2-D ou 3-D (un
contour, une barre), des `TRI3` dans des `Coords` 3-D (une surface, une coque).
\\( J \\) est alors rectangulaire et n'a plus de déterminant, mais la **métrique**
\\[
G(\xi) = J(\xi)^T\\, J(\xi) \quad \text{de taille } d_r \times d_r
\\]
est symétrique définie positive tant que la maille n'est pas dégénérée.
L'élément de mesure devient
\\[
d\mu = \sqrt{\det G}\\, d\xi
\\]
qui joue le rôle de \\( |J| \\) dans la quadrature : `det_jacobian` rend ce
\\( \sqrt{\det G} \\). Le **gradient tangent** d'un champ (la projection du
gradient sur la maille) passe par la pseudo-inverse :
\\[
\nabla_s N_i = J\\, G^{-1}\\, \nabla_\xi N_i
\\]
`dn_dx` rend ces composantes, dans le repère des `Coords`. Pour
\\( d_s = d_r \\), ces formules redonnent celles du cas plein
(\\( J G^{-1} = J^{-T} \\)), et c'est la seule forme calculée.

Dimensions admises :

| type | \\( d_r \\) | \\( d_s \\) admis |
|---|---:|---|
| `SEG2`, `SEG3` | 1 | 1, 2, 3 |
| `TRI3`, `TRI6`, `QUA4`, `QUA8`, `QUA9` | 2 | 2, 3 |
| `TET4`, `TET10`, `PYRA5`, `PENTA6`, `PENTA15`, `HEX8`, `HEX20`, `HEX27` | 3 | 3 |

\\( d_s < d_r \\) n'a pas de sens (pas de maille 2-D dans un espace 1-D) :
la construction le refuse.

**Maille dégénérée.** `dn_dx` doit inverser \\( G \\). Il refuse une maille
dont \\( \det G \\) tombe sous \\( 10^{-12} \\) fois l'échelle de \\( G \\)
(son plus grand terme, à la puissance \\( d_r \\)). Le seuil est **relatif** :
une maille d'un demi-millimètre décrite en mètres a un \\( \det G \\) de
l'ordre de \\( 10^{-21} \\) sans être dégénérée pour autant, et l'unité de
longueur choisie ne change rien au verdict.

**Axisymétrie.** Sur des `Coords` [axisymétriques](coords.md), le plan du
maillage est le plan méridien \\( (r, z) \\), et une maille y représente un
anneau. `det_jacobian` reste le \\( |J| \\) de la section : le facteur
\\( 2\pi r \\) entre dans le poids d'intégration, appliqué par les opérateurs
qui intègrent, de sorte que raideur, masse, flux, volumes et efforts intérieurs
portent sur l'anneau entier.

## Côté Rust

`SubFiniteElementSpace` détient :

- le `Handle<SubMesh>` de sa zone, scellé à la construction
  ([`mesh::seal`](mesh.md#scellement)) ;
- son `Interpolation` et sa `QuadratureRule`, la dimension des `Coords` et leur
  caractère axisymétrique, lus une fois à la construction (un `Coords` n'en
  change pas) ;
- les tables de référence, en vecteurs plats : `gauss_xi`, `gauss_w`, `n_at_g`,
  `dn_at_g` pour la géométrie, et `field_n_at_g`, `field_dn_at_g`,
  `field_d2n_at_g` pour le champ, **vides** quand le champ suit la géométrie
  (toute interpolation de Lagrange) ;
- la coloration, un `OnceLock<Vec<Vec<usize>>>` marqué `#[serde(skip)]`.

`FiniteElementSpace` n'est qu'un `Vec<Handle<SubFiniteElementSpace>>` : il ne
garde pas de `Mesh`. `mesh()` en reconstruit un à partir des zones de ses
sous-espaces, partagées et dédoublonnées par identité, et `check_push` refuse un
sous-espace sur d'autres `Coords` que le premier.

Les constructeurs sont `FiniteElementSpace::lagrange1(&mesh)`,
`FiniteElementSpace::new(&mesh, interpolation)` (Gauss partout) et
`FiniteElementSpace::with(&mesh, &[(interpolation, quadrature), …])`, un couple
par zone ; `SubFiniteElementSpace::new(submesh, interpolation, quadrature)`
construit un sous-espace seul.

```rust,ignore
{{#include ../../tests/doc_fespace.rs:constructeur}}
```

```rust,ignore
{{#include ../../tests/doc_fespace.rs:constructeur_explicite}}
```

Les évaluations sont les mêmes qu'en Python, sur le guard du sous-espace ; les
tables de référence sont prêtées en `&[f64]`, les grandeurs physiques rendues
en `Vec<f64>` :

```rust,ignore
{{#include ../../tests/doc_fespace.rs:evaluations}}
```

Le reste de l'API sert les physiques :

- **La base du champ.** `shape_count()`, `field_n_at_g(g)`, `field_dn_at_g(g)`
  et `field_d2n_at_g(g)` lisent la base qui interpole l'inconnue. Sous Lagrange,
  elles rendent les tables géométriques ; sous `ModelEmbedded`, elles lèvent une
  erreur qui nomme la situation ; `field_d2n_at_g` n'existe que sous `Hermite3`.
- **La coloration.** `coloring(compute)` rend la coloration, en la calculant au
  premier appel par la fermeture fournie : c'est l'assemblage
  (`models::kernel`) qui choisit les clés de conflit, ici les nœuds des
  mailles, et la couche conteneur n'en dépend pas.
- **Les pilotes de `models::kernel`.** Un opérateur ne passe pas par
  `jacobian` ou `dn_dx`, qui allouent leur résultat. Il copie une fois, par zone
  et par appel, les tables de référence du sous-espace, puis parcourt les
  mailles en parallèle. Chaque maille reçoit un `CellGeom` qui lit les
  coordonnées de ses nœuds une seule fois, au premier besoin, et calcule
  \\( \partial N_i/\partial x_a \\) et \\( |J|\\,w_g \\) dans des tampons sur la
  pile (neuf nombres au plus pour \\( J \\), \\( G \\) ou leur inverse).
  `CellGeom::det_j_w` est le seul endroit où se décide la mesure, facteur
  \\( 2\pi r \\) compris, et `CellGeom::field_n_at_g` celui où les pentes
  d'Hermite sont multipliées par \\( J \\).
