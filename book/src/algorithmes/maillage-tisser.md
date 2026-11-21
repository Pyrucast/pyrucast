# Tisser et révolutionner

## `sweep` : une surface entre deux lignes

{{#pyalgo mesh.sweep}}

`sweep(mesh_a, mesh_b, n_layers, element_type=None)` tisse `n_layers` couches
entre deux lignes de même degré et de même nombre d'éléments : le nœud `j` de
la première est relié au nœud `j` de la seconde, et les couches
intermédiaires sont interpolées entre eux.

Entre deux lignes `SEG2`, un maillage `QUA4` est **toujours construit en
premier** (le cœur géométrique du tissage) ; si `element_type` demande autre
chose, il est ensuite **converti** :

- `"TRI3"` — chaque `QUA4` est coupé en deux `TRI3` le long de la diagonale
  `(0, 2)` (pas de nœud créé) ;
- `"QUA8"` — promotion quadratique via `to_quadratic` (nœuds de milieu
  d'arête) ;
- `"QUA9"` — comme `QUA8`, puis un **nœud central** neuf est ajouté par
  cellule (moyenne des 4 coins) — `to_quadratic` ne produit que le `QUA8`
  sérendipité, sans nœud central ;
- `"TRI6"` — coupe en `TRI3` puis promotion quadratique (mêmes deux étapes
  composées).

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-sweep}}
```

Entre deux lignes **`SEG3`**, la surface est construite directement en `QUA8`
(le défaut) ou en `QUA9` : les nœuds milieux des deux lignes sont **repris**,
si bien qu'une couronne entre deux arcs suit exactement leurs courbes. Les
nœuds milieux des arêtes qui vont d'une ligne à l'autre sont interpolés à
mi-couche.

**Deux boucles fermées donnent un anneau fermé.** Quand chaque ligne se termine
sur son propre premier nœud — un cercle, un contour fermé —, la dernière
colonne de chaque couche *est* la première : la couture entre les deux
boucles est partagée, pas doublée. Une boucle fermée balayée vers une ligne
ouverte est une erreur.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:sweep_anneau}}
```

## Surface entre 4 côtés : `transfinite`

{{#pyalgo mesh.transfinite}}

`transfinite(side1, side2, side3, side4, element_type="QUA4")` maille une
surface **structurée** délimitée par quatre lignes `SEG2` — l'équivalent
Cast3M de l'opérateur `DALL(er)`, généralisant `sweep` (2 lignes) à 4 lignes.
`side1`/`side3` et `side2`/`side4` sont les deux paires de côtés
**opposés** ; chaque paire doit avoir le **même nombre d'éléments**. Les
quatre côtés doivent former un **contour fermé, orienté de façon
cohérente** :

```text
side4 dernier nœud == side1 premier nœud
side1 dernier nœud == side2 premier nœud
side2 dernier nœud == side3 premier nœud
side3 dernier nœud == side4 premier nœud
```

Les nœuds des quatre côtés (coins compris) sont **réutilisés** ; seuls les
nœuds intérieurs sont créés, par **interpolation transfinie discrète**
(patch de Coons bilinéaire) : mélange des deux côtés opposés à chaque
direction, corrigé par les quatre coins pour reproduire les côtés **exactement**
sur le bord, quelle que soit leur forme (pas seulement des droites). Comme
`sweep`, un `QUA4` est toujours construit d'abord, puis converti pour
`"TRI3"`/`"QUA8"`/`"QUA9"`/`"TRI6"`.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-transfinite}}
```

> **Différence avec Cast3M.** `DALL` accepte des côtés opposés avec un
> nombre de points **différent** (algorithme de pavage plus général,
> documenté mais non détaillé dans la notice officielle). `transfinite`
> se limite au cas standard de l'interpolation transfinie — côtés opposés
> de **même** nombre d'éléments — largement suffisant en pratique et
> implémentable simplement.

## Tissage d'un solide entre deux surfaces : `sweep_solid`

{{#pyalgo mesh.sweep_solid}}

`sweep_solid(mesh_a, mesh_b, n_layers)` est le **compagnon 3D** de `sweep` :
là où ce dernier relie deux lignes `SEG2` par une bande de `QUA4`, `sweep_solid`
relie deux **surfaces** par un solide. Les faces `TRI3` deviennent des prismes
`PENTA6`, les faces `QUA4` des hexaèdres `HEX8`.

La cellule `i` de `mesh_a` est appariée à la cellule `i` de `mesh_b`, nœud local
par nœud local. Les deux maillages doivent être **mono-sous-maillage**, du
**même** type de surface (`TRI3` ou `QUA4`), avec le même nombre de cellules et
une correspondance de nœuds cohérente, sur le **même** `Coords`. Les `n_layers`
couches de nœuds intermédiaires sont interpolées linéairement ; les nœuds des
deux faces d'extrémité sont réutilisés.

Associé à `translate` / `rotate`, il construit une tranche de solide entre une
surface et sa copie déplacée :

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-sweep_solid}}
```

## Révolution : `revolve`

{{#pyalgo mesh.revolve}}

`revolve(mesh, angle, n_layers, center, axis=None)` est le **compagnon
rotatif** d'`extrude` : là où `extrude` translate le maillage source couche
après couche le long d'un vecteur, `revolve` le fait tourner d'un angle total
`angle` (en **radians**), en `n_layers` couches d'angle égal. Les montées de
type sont les mêmes — `SEG2→QUA4`, `TRI3→PENTA6`, `QUA4→HEX8` — et l'ordre des
nœuds par cellule aussi (couche basse puis couche haute). En quadratique
(`SEG3`, `TRI6`, `QUA8`), le nœud milieu d'une arête dans le sens de la
rotation est pris au **demi-angle** : il est sur l'arc.

- En **2D**, la révolution se fait autour du **point** `center` (sens direct
  pour un `angle` positif) ; `axis` est ignoré. Seul un `SEG2` a du sens : une
  surface engendrerait un solide, que des coordonnées 2D ne peuvent pas
  porter (c'est refusé explicitement).
- En **3D**, elle se fait autour de la **droite** passant par `center` dirigée
  par `axis` (main droite) ; `axis` est alors obligatoire et n'a pas besoin
  d'être normé.

Comme pour `extrude`, la couche 0 **réutilise les nœuds de la source** (les
nœuds partagés entre cellules le restent) et les autres couches sont créées.

**Un tour complet referme l'anneau.** Avec `angle = 2π`, la dernière couche de
nœuds *est* la première : le tore/cylindre engendré n'a ni couture ni nœuds en
double, et rien à souder après coup (pas de `merge_nodes`). Au-delà d'un tour,
la révolution se recouvrirait elle-même : c'est une erreur.

**Une ligne qui touche l'axe s'y referme en triangles.** Un nœud posé sur
l'axe ne bouge pas : il reste le même nœud à chaque couche. Une maille de
ligne qui s'y appuie par un sommet ne balaie donc pas un quadrangle mais un
**triangle** — `SEG2 → TRI3`, `SEG3 → TRI6` —, son sommet répété retiré et
son ordre gardé. Les triangles d'une zone forment un sous-maillage à part,
après ses quadrangles. Un rayon du centre au bord devient ainsi un **secteur
de disque** : un éventail de triangles autour du nœud central, un anneau de
quadrangles autour de l'éventail ; extrudé, il donne des prismes au centre et
des hexaèdres autour. Une maille de surface ne peut pas toucher l'axe — son
prisme s'écraserait —, et une ligne couchée sur l'axe ne balaierait rien :
les deux sont refusées.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:revolve_axe}}
```

**Angle négatif.** Il balaye dans l'autre sens et retourne les cellules :
passez par `invert` sur le résultat, ou révolutionnez d'un angle positif
depuis la source symétrisée. `orient` n'y suffirait pas : les cellules sont
cohérentes entre elles, toutes retournées de la même façon.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-revolve}}
```

`revolve` fait d'un coup ce que `rotate` + `sweep_solid` font tranche par
tranche : une couche de `revolve` équivaut exactement à `sweep_solid(face,
rotate(face, angle, …), 1)`. Le passage par `sweep_solid` reste utile quand
les deux faces ne se déduisent pas l'une de l'autre par une rotation.
