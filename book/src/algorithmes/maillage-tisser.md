# Tisser et révolutionner

## Le type des mailles : `element_type`

Les quatre générateurs structurés de cette page — `sweep`, `transfinite`,
`extrude` et `revolve` — suivent une même règle pour leur option
`element_type` : c'est le type de **toutes** les mailles qu'ils produisent.

- **Absent** (`None`), chaque zone de la source prend son type naturel, celui
  de la colonne « Défaut » ci-dessous. Une source dont les zones sont de types
  différents donne des zones de types différents : `extrude` de lignes `SEG2`
  et de triangles `TRI3` donne des zones `QUA4` et `PENTA6`.
- **Donné**, chaque zone de la source doit pouvoir le produire, et le résultat
  en est fait tout entier, une zone par zone de la source. Les trois axes de la
  [table des types](../elements/index.md#par-forme-et-par-degré) disent ce
  qu'une ligne peut produire :
  - **forme** : une ligne balayée donne des quadrangles, qu'on peut couper en
    triangles ;
  - **degré** : on peut promouvoir au degré 2, jamais rétrograder — une ligne
    `SEG3` ne donne pas de surface linéaire ;
  - **centre** : seuls les quadrangles en ont un (`QUA9`).

  Une surface ne donne que son volume naturel.
- Un type qu'une zone ne peut pas produire est **refusé avant toute création
  de nœud**, par un message qui nomme le type de la zone, le type demandé et
  les types possibles.

| Opérateur | Éléments en entrée | Éléments possibles en sortie | Défaut |
|---|---|---|---|
| `sweep` | deux lignes `SEG2` | `QUA4`, `TRI3`, `QUA8`, `QUA9`, `TRI6` | `QUA4` |
| | deux lignes `SEG3` | `QUA8`, `QUA9`, `TRI6` | `QUA8` |
| | deux surfaces `TRI3`, `QUA4`, `TRI6` ou `QUA8` | `PENTA6`, `HEX8`, `PENTA15`, `HEX20`, une seule sortie par entrée | la même |
| `transfinite` | quatre côtés `SEG2` | `QUA4`, `TRI3`, `QUA8`, `QUA9`, `TRI6` | `QUA4` |
| | quatre côtés `SEG3` | `QUA8`, `QUA9`, `TRI6` | `QUA8` |
| `extrude`, `revolve` | `SEG2` | `QUA4`, `TRI3`, `QUA8`, `QUA9`, `TRI6` | `QUA4` |
| | `SEG3` | `QUA8`, `QUA9`, `TRI6` | `QUA8` |
| | `TRI3`, `QUA4`, `TRI6`, `QUA8` | `PENTA6`, `HEX8`, `PENTA15`, `HEX20`, une seule sortie par entrée | la même |

Chaque générateur construit d'abord une **surface de base**, en `QUA4` à partir
de lignes `SEG2` ou en `QUA8` à partir de lignes `SEG3`, puis la livre dans le
type demandé en trois étapes au plus : couper les quadrangles en triangles,
promouvoir au degré 2, donner aux quadrangles leur centre.

### Où vont les nœuds ajoutés

Le centre d'un `QUA9` et le nœud milieu de la diagonale d'un `TRI6` ne viennent
ni de la source ni des arêtes : il faut les placer. La règle est de les poser
**sur la géométrie déjà construite, sans la modifier** ; l'ordre des étapes en
découle.

- **Mailles à bords droits** (entrée linéaire) : on coupe d'abord, puis on
  promeut. Le milieu d'une diagonale est au milieu de sa corde, et chaque `TRI6`
  garde des bords droits : c'est un triangle affine, à jacobien constant. Le
  centre d'un `QUA9` est la moyenne des quatre coins, le centre de la maille
  bilinéaire.
- **Mailles à bords courbes** (entrée `SEG3`) : on place d'abord le centre, puis
  on coupe. Le centre est le point du `QUA8` lui-même au milieu de la maille,
  `−¼ Σ coins + ½ Σ milieux` : le `QUA9` garde exactement la géométrie du `QUA8`,
  et le `TRI6` prend ce centre comme milieu de sa diagonale. Sur des bords
  droits, cette formule redonne la moyenne des coins.

Pour une maille de 15° d'un arc de rayon 7 extrudé, ce centre est au rayon 7,
sur le cylindre, exactement là où la translation `TRAN` de Cast3M place le nœud
supplémentaire de ses `TRI6` ; la moyenne des coins serait au rayon 6,94011,
0,06 à l'intérieur. Pour une maille de 45° d'une révolution, il est sur l'arc,
au demi-angle.

### Diagonales alternées

Couper un quadrangle en deux triangles demande de choisir une diagonale. Les
générateurs structurés **alternent** : la maille de rang pair dans sa zone est
coupée selon `(0, 2)`, la maille de rang impair selon `(1, 3)`, comme le font
les opérateurs `TRAN`, `ROTA` et `REGL` de Cast3M. Leurs mailles étant rangées
couche par couche, le dessin dépend de la parité du nombre de mailles par
couche (`/` est la diagonale `(0, 2)`, `\` la diagonale `(1, 3)`) :

```text
3 mailles par couche (impair) : damier       2 mailles par couche (pair) : bandes
  couche 1 :   3 \   4 /   5 \                 couche 1 :   2 /   3 \
  couche 0 :   0 /   1 \   2 /                 couche 0 :   0 /   1 \
```

Sur une coque mince comme le cylindre pincé du cas de validation elas2, ce
choix compte : avec la même diagonale partout, la flèche s'écarte de 0,15 % de
la référence ; en alternant, de 0,08 %.

`convert(…, "TRI3")` coupe par défaut tous les quadrangles selon la même
diagonale : un maillage quelconque, comme la peau d'un solide, n'a pas d'ordre
de grille, et le mailleur volumique `triangulate_volume` retrouve mieux une
peau coupée partout pareil. `convert(…, "TRI3", alternate=True)` alterne
comme les générateurs.

## `sweep` : une surface entre deux lignes

{{#pyalgo mesh.sweep}}

`sweep(mesh_a, mesh_b, n_layers, element_type=None)` tisse `n_layers` couches
entre deux lignes de même degré et de même nombre d'éléments : le nœud `j` de
la première est relié au nœud `j` de la seconde, et les couches
intermédiaires sont interpolées entre eux.

Entre deux lignes `SEG2`, un maillage `QUA4` est **toujours construit en
premier** (le cœur géométrique du tissage) ; si `element_type` demande autre
chose, il est ensuite livré selon la règle ci-dessus : `"TRI3"` coupe les
quadrangles en alternant la diagonale, `"QUA8"` les promeut (`to_quadratic`),
`"QUA9"` les promeut puis leur ajoute leur centre, `"TRI6"` les coupe puis
promeut les triangles.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-sweep}}
```

Entre deux lignes **`SEG3`**, la surface de base est un `QUA8` : les nœuds
milieux des deux lignes sont **repris**, si bien qu'une couronne entre deux
arcs suit exactement leurs courbes, et les nœuds milieux des arêtes qui vont
d'une ligne à l'autre sont interpolés à mi-couche. Elle est livrée en `QUA8`
(le défaut), en `QUA9` ou en `TRI6`.

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

`transfinite(side1, side2, side3, side4, element_type=None)` maille une
surface **structurée** délimitée par quatre lignes, quatre `SEG2` ou quatre
`SEG3` — l'équivalent Cast3M de l'opérateur `DALL(er)`, généralisant `sweep`
(2 lignes) à 4 lignes. `side1`/`side3` et `side2`/`side4` sont les deux paires
de côtés **opposés** ; chaque paire doit avoir le **même nombre d'éléments**.
Les quatre côtés doivent former un **contour fermé, orienté de façon
cohérente** :

```text
side4 dernier nœud == side1 premier nœud
side1 dernier nœud == side2 premier nœud
side2 dernier nœud == side3 premier nœud
side3 dernier nœud == side4 premier nœud
```

Les nœuds des quatre côtés (coins compris) sont **réutilisés** ; seuls les
nœuds intérieurs sont créés, par **interpolation transfinie**, dite aussi
patch de Coons (du nom de Steven Coons, qui l'a introduite en conception
assistée par ordinateur dans les années 1960). Un point intérieur, aux
paramètres `(u, v)` compris entre 0 et 1, mélange les deux côtés opposés dans
chaque direction, puis retranche le mélange des quatre coins, sans quoi il les
compterait deux fois :

```text
P(u, v) = (1 − v)·côté1(u) + v·côté3(u) + (1 − u)·côté4(v) + u·côté2(v)
          − [ (1 − u)(1 − v)·P00 + u(1 − v)·P10 + (1 − u)·v·P01 + u·v·P11 ]
```

`côté3` et `côté4` y sont lus à rebours, de façon que `u` et `v` croissent
depuis le coin `P00`. Sur le bord, la formule redonne exactement les côtés,
quelle que soit leur forme (pas seulement des droites).

- Entre côtés **`SEG2`**, la surface de base est un `QUA4`, livrée en `"QUA4"`
  (le défaut), `"TRI3"`, `"QUA8"`, `"QUA9"` ou `"TRI6"`, comme pour `sweep`.
- Entre côtés **`SEG3`**, c'est un `QUA8`, livré en `"QUA8"` (le défaut),
  `"QUA9"` ou `"TRI6"`. Ses coins intérieurs viennent de l'interpolation sur
  les seuls coins des côtés ; les nœuds milieux du bord sont **ceux des
  côtés**, si bien qu'une surface bordée d'arcs les suit ; un nœud milieu
  intérieur est au **milieu de la corde** entre ses deux coins, comme `DALL` le
  place. Les quatre côtés doivent être du même degré.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-transfinite}}
```

> **Différence avec Cast3M.** `DALL` accepte des côtés opposés avec un
> nombre de points **différent** (algorithme de pavage plus général,
> documenté mais non détaillé dans la notice officielle). `transfinite`
> se limite au cas standard de l'interpolation transfinie — côtés opposés
> de **même** nombre d'éléments — largement suffisant en pratique et
> implémentable simplement.

## Un volume entre deux surfaces : `sweep`

Entre deux **surfaces**, `sweep(mesh_a, mesh_b, n_layers)` construit un
volume, comme il construit une surface entre deux lignes. Les faces `TRI3`
deviennent des prismes `PENTA6`, les faces `QUA4` des hexaèdres `HEX8`, et en
quadratique les faces `TRI6` des `PENTA15`, les faces `QUA8` des `HEX20`.

La cellule `i` de `mesh_a` est appariée à la cellule `i` de `mesh_b`, nœud local
par nœud local. Les deux maillages doivent être **mono-sous-maillage**, du
**même** type de surface, avec le même nombre de cellules et une
correspondance de nœuds cohérente, sur le **même** `Coords`. Les `n_layers`
couches de nœuds intermédiaires sont interpolées linéairement ; les nœuds des
deux faces d'extrémité sont réutilisés. Une face quadratique donne aussi les
nœuds milieux des arêtes qui vont d'une face à l'autre, à mi-couche.

Associé à `translate` / `rotate`, il construit une tranche de solide entre une
surface et sa copie déplacée ; entre une face et sa copie translatée, il donne
exactement les mailles qu'`extrude` donne de la face :

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:sweep_volume}}
```

## Révolution : `revolve`

{{#pyalgo mesh.revolve}}

`revolve(mesh, angle, n_layers, center, axis=None, element_type=None)` est le
**compagnon rotatif** d'`extrude` : là où `extrude` translate le maillage
source couche après couche le long d'un vecteur, `revolve` le fait tourner d'un
angle total `angle` (en **radians**), en `n_layers` couches d'angle égal. Les
montées de type sont les mêmes — `SEG2→QUA4`, `TRI3→PENTA6`, `QUA4→HEX8` — et
l'ordre des nœuds par cellule aussi (couche basse puis couche haute). En
quadratique (`SEG3`, `TRI6`, `QUA8`), le nœud milieu d'une arête dans le sens
de la rotation est pris au **demi-angle** : il est sur l'arc. `element_type`
suit la règle du début de page.

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

C'est la seule exception à la règle d'`element_type` : l'éventail reste en
triangles, `TRI3` si le type demandé est linéaire, `TRI6` s'il est
quadratique. Avec `"TRI3"` ou `"TRI6"`, tout le secteur est donc fait de
triangles.

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

`revolve` fait d'un coup ce que `rotate` + `sweep` font tranche par
tranche : une couche de `revolve` équivaut exactement à `sweep(face,
rotate(face, angle, …), 1)`. Le passage par `sweep` reste utile quand
les deux faces ne se déduisent pas l'une de l'autre par une rotation.
