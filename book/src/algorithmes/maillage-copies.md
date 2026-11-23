# Copier et transformer un maillage

## Copie sur place : `copy`

{{#pyalgo mesh.copy}}

`copy(mesh, new_nodes=True)` rend une copie du maillage **sans le déplacer** :
mêmes sous-maillages dans le même ordre, mêmes types, mêmes couleurs, même
connectivité. Ce que l'argument tranche, c'est si les deux maillages se tiennent
encore par leur géométrie :

- `new_nodes=True` (par défaut) crée **un nœud neuf par nœud distinct**, à la
  même position, dans la même `Coords` — un nœud partagé entre plusieurs
  cellules de la source reste partagé dans la copie. Les deux maillages sont
  alors indépendants : déplacer un nœud de l'un laisse l'autre en place. C'est
  ce que font les [copies rigides](#copies-rigides--translate-rotate-et-les-symétries),
  moins le déplacement.
- `new_nodes=False` copie **la seule connectivité** : ce sont les mêmes nœuds,
  dont le refcount monte. Déplacer un nœud les déplace tous les deux.

Dans les deux cas la copie est **descellée** ([Maillage](../mesh.md#scellement)),
donc de nouveau modifiable même si l'original a été consommé par un espace
éléments finis, un champ ou une matrice ; c'est la forme fonction libre de la
méthode `duplicate()`, qui est le cas `new_nodes=False`.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-copy}}
```

## Copies rigides : `translate`, `rotate` et les symétries

{{#pyalgo mesh.translate mesh.rotate mesh.symmetry_point mesh.symmetry_line mesh.symmetry_plane}}

`translate(mesh, vector)`, `rotate(mesh, angle, center, axis=None)` et les trois
symétries renvoient une **copie neuve** du maillage — mêmes sous-maillages,
mêmes types, mêmes couleurs, même connectivité — dont **tous les nœuds sont
nouveaux**. Le maillage d'origine (et ses nœuds) reste intact ; un nœud partagé
entre plusieurs cellules de la source reste partagé dans la copie.

- `translate` décale chaque nœud de `vector` (dont la longueur doit valoir la
  dimension du maillage).
- `rotate` tourne de `angle` **radians** autour de `center`. En **2D**, `center`
  est un point et `axis` est ignoré ; en **3D**, la rotation se fait autour de la
  droite passant par `center` dirigée par `axis` (formule de Rodrigues,
  main droite), et `axis` est obligatoire (il n'a pas besoin d'être normé).
- `symmetry_point(mesh, center)` envoie chaque nœud sur `2·center − x` :
  `center` est le milieu de chaque nœud et de son image. C'est le demi-tour
  autour du point en **2D**, l'inversion centrale en **3D**.
- `symmetry_line(mesh, a, b)` réfléchit à travers la droite (infinie) passant
  par `a` et `b` : la composante le long de la droite est gardée, la
  perpendiculaire est retournée. En **2D** c'est l'image miroir dans la droite ;
  en **3D** c'est le **demi-tour** autour d'elle (une rotation de π) — pour
  l'image dans un miroir, c'est `symmetry_plane` qu'il faut.
- `symmetry_plane(mesh, a, b, c)` réfléchit à travers le plan passant par les
  **trois points** `a`, `b` et `c` : `x ↦ x − 2((x − a)·n̂) n̂`, où `n̂` est la
  normale unitaire du plan. Les trois points jouent des rôles symétriques —
  seul compte le plan qu'ils engendrent, pas leur ordre (une permutation
  retourne `n̂`, ce à quoi la formule est insensible). **3D uniquement** : en
  2D, le miroir est `symmetry_line`, qui prend les deux points de la droite.
  Erreur si les trois points sont alignés (ils n'engendrent alors aucun plan).

### Orientation des cellules

Une symétrie peut **retourner l'orientation** : le déterminant de sa partie
linéaire vaut alors `−1`, et la copie brute aurait toutes ses cellules à
l'envers (jacobien négatif, normales rentrantes sur une peau). Ces opérateurs
appliquent donc en plus à chaque cellule la permutation de renversement, comme
[`invert`](maillage.md), de sorte que la copie ait la **même** orientation que
la source et soit directement calculable. Les cas retournés dépendent de la
dimension :

| opérateur | 2D | 3D |
|---|---|---|
| `symmetry_point` | direct (demi-tour) | retourné |
| `symmetry_line` | retourné | direct (demi-tour) |
| `symmetry_plane` | — (3D seulement) | retourné |

Appliquer `invert` au résultat redonne la connectivité miroir brute.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:transformations}}
```

### Projection sur une sphère : `project_on_sphere`

{{#pyalgo mesh.project_on_sphere}}

`project_on_sphere(mesh, center, radius)` renvoie, elle aussi, une **copie
neuve** : chaque nœud — sommets et nœuds milieux — est poussé le long du
rayon issu de `center`, jusqu'à la distance `radius`. La copie garde les zones,
les types, les couleurs et la connectivité de la source ; un nœud partagé le
reste, donc une boucle fermée reste fermée.

Dans un plan qui passe par `center`, la sphère est un **cercle** : un contour
de ce plan s'y pose nœud pour nœud, chacun à son propre angle. C'est la façon
de construire le trou d'une couronne à partir du contour qui l'entoure : un
nœud du trou en face de chaque nœud du contour, ce que [`sweep`](maillage-tisser.md#sweep--une-surface-entre-deux-lignes)
demande.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-project_on_sphere}}
```

`radius` doit être positif et fini, `center` avoir la dimension du maillage.
Un nœud posé sur `center` n'a pas de direction : c'est une erreur, et aucun
nœud n'est créé. Côté Rust, `ops::mesh::project_on_sphere(&mesh, &center,
radius)`.

## Passage à l'ordre quadratique : `to_quadratic`

{{#pyalgo mesh.to_quadratic}}

`to_quadratic(mesh)` construit la **copie quadratique** (Lagrange-2) d'un
maillage **linéaire** : chaque type d'élément est promu vers son homologue
quadratique — `SEG2→SEG3`, `TRI3→TRI6`, `QUA4→QUA8`, `TET4→TET10`,
`PENTA6→PENTA15`, `HEX8→HEX20`.

Les **nœuds sommets sont réutilisés** (refcount incrémenté) ; un **nœud de
milieu d'arête** est créé par arête distincte, au milieu géométrique de l'arête,
et **partagé** entre toutes les cellules (tous sous-maillages confondus) qui
utilisent cette arête — le résultat reste donc conforme. Le maillage d'origine
n'est pas modifié. Un sous-maillage `POI1` ou déjà quadratique lève une erreur.

Les nœuds milieux sont au milieu des **cordes** : sur un bord courbe, la
copie quadratique reste un polygone. Pour suivre une courbe, on maille
directement en quadratique — `arc` ou `circle` en `SEG3`, puis `sweep` ou
`extrude` —, et les nœuds milieux sont alors sur la courbe.

Le maillage obtenu se calcule avec l'interpolation `LAGRANGE2`, que
l'espace éléments finis prend de lui-même (cf.
[Espace éléments finis](../fe-space.md)) :

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-to_quadratic}}
```

## Changement de type d'élément : `convert`

{{#pyalgo mesh.convert}}

`convert(mesh, element_type, alternate=False)` **change le type d'élément**
de chaque sous-maillage vers `element_type`, zone par zone, **en gardant la
géométrie de chaque maille** : aucun nœud n'est déplacé. Les cas couverts :

- **identité** — `element_type` est déjà le type du sous-maillage : celui-ci
  est recopié tel quel ;
- **`QUA4`→`TRI3`** — chaque quadrangle est coupé en deux triangles selon la
  diagonale `(0, 2)` : `(0, 1, 2)` et `(0, 2, 3)` ;
- **`QUA8`→`QUA9`** — chaque maille reçoit son centre, le point du `QUA8`
  lui-même au milieu de la maille, `−¼ Σ coins + ½ Σ milieux` : la moyenne des
  coins quand les bords sont droits, un point de la maille courbe sinon. Le
  `QUA9` garde ainsi exactement la géométrie du `QUA8` ;
- **`QUA9`→`TRI6`** — la découpe de `QUA4`→`TRI3`, le centre devenant le nœud
  milieu de la diagonale ;
- **`QUA8`→`TRI6`** — le centre de `QUA8`→`QUA9`, puis cette découpe ;
- **`HEX8`→`TET4`** — chaque hexaèdre est coupé en six tétraèdres partageant la
  grande diagonale `(0, 6)` (subdivision de Freudenthal/Kuhn), la même dans
  chaque maille : couper un hexaèdre coupe ses faces, et deux voisins coupent
  leur face commune de la même façon, si bien que le résultat reste
  **conforme**.

**`alternate`** concerne les découpes de quadrangles en triangles. Par défaut,
toutes coupent selon `(0, 2)` : un maillage quelconque, comme la peau d'un
solide, n'a pas d'ordre de grille, et le mailleur volumique `triangulate_volume`
retrouve mieux une peau coupée partout pareil. Avec `alternate=True`, la maille
de rang impair dans sa zone est coupée selon `(1, 3)` — `(0, 1, 3)` et
`(1, 2, 3)` — : sur une grille rangée couche par couche, comme celles de
`sweep`, `transfinite` ou `extrude`, la diagonale alterne d'une maille à la
suivante, comme dans les générateurs structurés (voir
[Tisser et révolutionner](maillage-tisser.md#diagonales-alternées)).

Les **sommets et les nœuds milieux sont réutilisés**, seuls les centres d'un
`QUA8` sont créés, et les **couleurs de face sont conservées**. Le maillage
d'origine n'est pas modifié. Tout autre couple `(source, cible)` lève une
erreur : passer à un type **quadratique** (`TRI3`→`TRI6`, …), qui crée des
nœuds de milieu d'arête partagés entre zones, relève de
[`to_quadratic`](#passage-à-lordre-quadratique--to_quadratic).

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-convert}}
```
