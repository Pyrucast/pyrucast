# Opérateurs de maillage

Les **mailleurs** (`ops::mesh`) construisent et transforment des
[maillages](../mesh.md). Chacun prend ses conteneurs par référence et renvoie
un **nouveau** `Mesh`. Côté Python ils sont exposés à plat
(`pyrucast.mesh.line`, …).

## Inventaire

| Python | Rôle |
|---|---|
| `from_live_nodes(coords)` | un `Mesh` POI1 de **tous** les nœuds vivants d'un `Coords` |
| `poi1_from_nodes(nodes)` | un `Mesh` POI1 sur une liste de nœuds donnée |
| `line(a, b, n_elems, element_type="SEG2")` | une ligne de `n_elems` éléments (`SEG2` ou `SEG3`) entre deux nœuds (nœuds intermédiaires créés) |
| `circle(center, normal, radius, n_elems, element_type="SEG2")` | un cercle fermé (`SEG2` ou `SEG3`, plan défini par `normal`) |
| `arc(a, center, b, n_elems, element_type="SEG2")` | un arc de `a` à `b` sur le cercle de centre `center` passant par les deux (le plus court des deux arcs) |
| `extrude(mesh, direction, n_layers)` | extrude un maillage le long de `direction` (SEG2→QUA4, TRI3→PENTA6, QUA4→HEX8) |
| `revolve(mesh, angle, n_layers, center, axis=None)` | **compagnon rotatif** d'`extrude` : balaye un maillage de `angle` (rad) autour de `center` (axe `axis` en 3D), mêmes montées de type ; un tour complet **referme** l'anneau (voir plus bas) |
| `sweep(mesh_a, mesh_b, n_layers, element_type="QUA4")` | tisse `QUA4`/`TRI3`/`QUA8`/`QUA9`/`TRI6` entre deux lignes `SEG2` (un `QUA4` est toujours construit d'abord, puis converti) |
| `transfinite(side1, side2, side3, side4, element_type="QUA4")` | **généralisation de `sweep` à 4 côtés** (l'équivalent Cast3M de `DALL`) : interpolation transfinie (patch de Coons) entre quatre lignes `SEG2` formant un contour fermé (voir plus bas) |
| `sweep_solid(mesh_a, mesh_b, n_layers)` | **compagnon 3D** de `sweep` : tisse un solide entre deux surfaces (TRI3→PENTA6, QUA4→HEX8) |
| `translate(mesh, vector)` | **copie** du maillage translatée de `vector` (nœuds neufs, original intact) |
| `rotate(mesh, angle, center, axis=None)` | **copie** du maillage tournée de `angle` (rad) autour de `center` (axe `axis` en 3D) |
| `symmetry_point(mesh, center)` | **copie** symétrique par rapport au **point** `center` (Cast3M `SYME`, voir plus bas) |
| `symmetry_line(mesh, a, b)` | **copie** symétrique par rapport à la **droite** passant par `a` et `b` (demi-tour en 3D) |
| `symmetry_plane(mesh, a, b, c)` | **copie** symétrique par rapport au **plan** passant par trois points (3D) |
| `triangulate_surface(contour, type, size=None)` | maille l'intérieur de contours **orientés** (CCW extérieur, CW trous) par **Delaunay contraint + raffinement Ruppert** (voir plus bas) |
| `pave_surface(contour, type, size=None, all_quad=False)` | **pave** l'intérieur des mêmes contours **orientés** en `QUA4`/`QUA8`/`QUA9`, par **front avançant** en rangées parallèles au bord (voir plus bas) |
| `pave_volume(envelope, layers=1, thickness=None, size=None)` | **compagnon 3D** de `pave_surface` : couche limite d'`HEX8`/`PENTA6` poussée vers l'intérieur, raccordée par des `PYRA5` à un cœur `TET4` (voir plus bas) |
| `triangulate_volume(envelope, size=None, allow_surface_nodes=False)` | **compagnon 3D** de `triangulate_surface` : maille l'intérieur d'une **enveloppe TRI3 fermée** en `TET4` — Delaunay exact, récupération du bord, raffinement intérieur et chasse aux slivers (voir plus bas) |
| `border(mesh, angle_deg=None)` | le **bord** d'un maillage de surface (TRI3/QUA4) en boucles `SEG2` (une par sous-maillage) ; avec `angle_deg`, découpé en **arêtes** ouvertes aux coins (voir plus bas) |
| `skin(mesh, angle_deg=None)` | la **peau** d'un maillage volumique (TET4/PENTA6/HEX8) en faces `TRI3`/`QUA4`, **une par face plane** du solide (voir plus bas) |
| `orient(mesh)` | **harmonise** l'orientation des cellules (normales cohérentes), toute dimension (SEG/TRI/QUA/TET/PENTA/HEX), équivalent Cast3M `ORIE` (voir plus bas) |
| `invert(mesh)` | **inverse** l'orientation de toutes les cellules, toute dimension, équivalent Cast3M `INVE` (voir plus bas) |
| `elements_on(mesh, points, strict=True)` | les **éléments** de `mesh` qui s'**appuient** sur les nœuds de `points` (voir plus bas) |
| `points_in_sphere(mesh, center, radius, tol=None)` | les nœuds **dans** la sphère (le disque en 2D) — famille `points_*`, voir plus bas |
| `points_on_sphere(mesh, center, radius, tol=None)` | les nœuds **sur** la sphère (le cercle en 2D) |
| `points_on_plane(mesh, origin, normal, tol=None)` | les nœuds **dans** le plan (la droite en 2D) — la façon usuelle d'attraper une face de bord |
| `points_below_plane(mesh, origin, normal, tol=None)` | les nœuds du **demi-espace** opposé à la normale, plan compris (normale retournée ⇒ l'autre moitié) |
| `points_on_line(mesh, a, b, tol=None)` | les nœuds **sur la droite** (infinie) passant par `a` et `b` |
| `points_in_cylinder(mesh, base, top, radius, tol=None)` | les nœuds **dans** le cylindre fini d'axe `base → top` |
| `points_on_cylinder(mesh, base, top, radius, tol=None)` | les nœuds **sur la surface latérale** du même cylindre (disques d'extrémité **exclus**) |
| `points_in_cone(mesh, base, top, base_radius, top_radius=0.0, tol=None)` | les nœuds **dans** le cône tronqué (`top_radius=0` ⇒ cône vrai de sommet `top`) |
| `points_on_cone(mesh, base, top, base_radius, top_radius=0.0, tol=None)` | les nœuds **sur la surface latérale** du même cône |
| `points_in_torus(mesh, center, axis, major_radius, minor_radius, tol=None)` | les nœuds **dans** le tore à section circulaire (**3D seulement**) |
| `points_on_torus(mesh, center, axis, major_radius, minor_radius, tol=None)` | les nœuds **sur** la surface du même tore |
| `to_poi1(mesh)` | les nœuds **distincts** d'un maillage, en POI1 ; nuage **canonique mis en cache** par sous-maillage (scellé) ⇒ handle reproductible, partagé par `restrict`/blocs de matrice/`divergence`/`flux` (supports appariables) |
| `to_quadratic(mesh)` | la **copie quadratique** (Lagrange-2) d'un maillage linéaire : TRI3→TRI6, HEX8→HEX20, … (voir plus bas) |
| `convert(mesh, element_type)` | **change le type d'élément** sans déplacer ni ajouter de nœud : identité, `QUA4`→`TRI3` (2 triangles), `HEX8`→`TET4` (6 tétraèdres) (voir plus bas) |
| `barycenter(mesh)` | un POI1 au **centre de gravité** de chaque cellule, structure de sous-maillage préservée |
| `mesh.consolidate(mesh)` | fusionne les sous-maillages de même type, en écartant les mailles dupliquées |
| `merge_nodes(mesh, tol, in_place=False)` | **soude** les nœuds distants de moins de `tol` ; remappe la connectivité, abandonne les cellules dégénérées — ou réécrit les sous-maillages **sur place** avec `in_place=True` (voir plus bas) |
| `read_gmsh(coords, path)` | **lit un maillage gmsh** `.msh` (ASCII 2.2 ou 4.1) dans `coords`, renvoie un `dict` `{groupe physique: Mesh}` (voir plus bas) |
| `read_gmsh_str(coords, text)` | comme `read_gmsh` mais depuis le **texte** du fichier déjà en mémoire |

`barycenter` sert notamment à fabriquer les supports de multiplicateurs des
contraintes : POI1 → nœuds **neufs** colocalisés au centre de chaque cellule
(cf. [Dirichlet](../contraintes/dirichlet.md)).

```python
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([4.0, 0.0])

# Ligne de 4 SEG2 entre a et b (3 nœuds intermédiaires créés).
line = pyrucast.mesh.line(a, b, 4)
print(line)  # Mesh: 1 submesh(es), 4 cell(s) total

# Extrusion en QUA4 sur 2 couches selon +y.
surf = pyrucast.mesh.extrude(line, [0.0, 1.0], 2)
print(surf.element_types())  # ['QUA4']

# Ligne quadratique : SEG3 (nœud de milieu d'arête par élément).
line3 = pyrucast.mesh.line(a, b, 4, "SEG3")
print(line3.element_types())  # ['SEG3']
```

## `sweep` : QUA4 par défaut, ou toute variante dérivée

`sweep(mesh_a, mesh_b, n_layers, element_type="QUA4")` tisse `n_layers`
couches entre deux lignes `SEG2`. Un maillage `QUA4` est **toujours construit
en premier** (le cœur géométrique du tissage) ; si `element_type` demande
autre chose, il est ensuite **converti** :

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
tri = pyrucast.mesh.sweep(mesh_a, mesh_b, 2, "TRI3")  # 2× plus de cellules que QUA4
qua8 = pyrucast.mesh.sweep(mesh_a, mesh_b, 2, "QUA8")
qua9 = pyrucast.mesh.sweep(mesh_a, mesh_b, 2, "QUA9")
tri6 = pyrucast.mesh.sweep(mesh_a, mesh_b, 2, "TRI6")
```

## Surface entre 4 côtés : `transfinite`

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
c = pyrucast.Coords(dim=2)
p0 = c.add_node([0.0, 0.0])
p1 = c.add_node([2.0, 0.0])
p2 = c.add_node([2.0, 1.0])
p3 = c.add_node([0.0, 1.0])

side1 = pyrucast.mesh.line(p0, p1, 4)  # bas,   4 éléments
side2 = pyrucast.mesh.line(p1, p2, 2)  # droite, 2 éléments
side3 = pyrucast.mesh.line(p2, p3, 4)  # haut,  4 éléments (= side1)
side4 = pyrucast.mesh.line(p3, p0, 2)  # gauche, 2 éléments (= side2)

surf = pyrucast.mesh.transfinite(side1, side2, side3, side4)
print(surf.element_types(), surf.cell_count())  # ['QUA4'] 8
```

> **Différence avec Cast3M.** `DALL` accepte des côtés opposés avec un
> nombre de points **différent** (algorithme de pavage plus général,
> documenté mais non détaillé dans la notice officielle). `transfinite`
> se limite au cas standard de l'interpolation transfinie — côtés opposés
> de **même** nombre d'éléments — largement suffisant en pratique et
> implémentable simplement.

## Copies rigides : `translate`, `rotate` et les symétries

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
[`invert`](#inventaire), de sorte que la copie ait la **même** orientation que
la source et soit directement calculable. Les cas retournés dépendent de la
dimension :

| opérateur | 2D | 3D |
|---|---|---|
| `symmetry_point` | direct (demi-tour) | retourné |
| `symmetry_line` | retourné | direct (demi-tour) |
| `symmetry_plane` | — (3D seulement) | retourné |

Appliquer `invert` au résultat redonne la connectivité miroir brute.

```python
import math
import pyrucast

# Une face TRI3 (un seul triangle) dans le plan z = 0.
c = pyrucast.Coords(dim=3)
face = pyrucast.Mesh(c, "TRI3")
face.unit().add_cell(
    [
        c.add_node([1.0, 0.0, 0.0]),
        c.add_node([2.0, 0.0, 0.0]),
        c.add_node([1.0, 0.0, 1.0]),
    ]
)

# Copie translatée de 5 selon +z (nœuds neufs ; `face` reste intacte).
haut = pyrucast.mesh.translate(face, [0.0, 0.0, 5.0])

# Copie tournée de 30° autour de l'axe z passant par l'origine.
tournee = pyrucast.mesh.rotate(face, math.pi / 6, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])

# Copie symétrique dans le plan y = 0, donné par trois de ses points : la
# moitié manquante d'une pièce maillée sur son demi-modèle (cellules remises
# à l'endroit).
autre_moitie = pyrucast.mesh.symmetry_plane(
    face, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]
)
```

## Tissage d'un solide entre deux surfaces : `sweep_solid`

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
# `face` et `tournee` : la face TRI3 ci-dessus et sa copie tournée de 30°.
solide = pyrucast.mesh.sweep_solid(face, tournee, 1)
print(solide.element_types())  # ['PENTA6']
```

## Révolution : `revolve`

`revolve(mesh, angle, n_layers, center, axis=None)` est le **compagnon
rotatif** d'`extrude` : là où `extrude` translate le maillage source couche
après couche le long d'un vecteur, `revolve` le fait tourner d'un angle total
`angle` (en **radians**), en `n_layers` couches d'angle égal. Les montées de
type sont les mêmes — `SEG2→QUA4`, `TRI3→PENTA6`, `QUA4→HEX8` — et l'ordre des
nœuds par cellule aussi (couche basse puis couche haute).

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

**Aucun nœud sur l'axe.** Un nœud posé sur l'axe ne bouge pas : toutes les
cellules qui s'y appuient s'écraseraient en éléments dégénérés (jacobien nul).
L'opérateur le refuse plutôt que de produire un maillage incalculable — il
faut décaler la source de l'axe (le trou central d'un disque, l'alésage d'un
tube).

**Angle négatif.** Il balaye dans l'autre sens et retourne les cellules,
exactement comme un `extrude` à contre-normale : passez par `orient` sur le
résultat, ou révolutionnez d'un angle positif depuis la source symétrisée.

```python
import math
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([1.0, 0.0])
b = c.add_node([2.0, 0.0])

# Une couronne complète : le segment radial [1, 2] tourné d'un tour en
# 32 secteurs de QUA4 — refermée, sans couture.
rayon = pyrucast.mesh.line(a, b, 4)
couronne = pyrucast.mesh.revolve(rayon, 2 * math.pi, 32, [0.0, 0.0])
print(couronne.element_types(), couronne.cell_count())  # ['QUA4'] 128

# En 3D : un quart de tube, la section QUA4 balayée autour de l'axe z.
c3 = pyrucast.Coords(dim=3)
section = pyrucast.Mesh(c3, "QUA4")
section.unit().add_cell(
    [
        c3.add_node([1.0, 0.0, 0.0]),
        c3.add_node([2.0, 0.0, 0.0]),
        c3.add_node([2.0, 0.0, 1.0]),
        c3.add_node([1.0, 0.0, 1.0]),
    ]
)
quart = pyrucast.mesh.revolve(section, math.pi / 2, 8, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
print(quart.element_types())  # ['HEX8']
```

`revolve` fait d'un coup ce que `rotate` + `sweep_solid` font tranche par
tranche : une couche de `revolve` équivaut exactement à `sweep_solid(face,
rotate(face, angle, …), 1)`. Le passage par `sweep_solid` reste utile quand
les deux faces ne se déduisent pas l'une de l'autre par une rotation.

## Passage à l'ordre quadratique : `to_quadratic`

`to_quadratic(mesh)` construit la **copie quadratique** (Lagrange-2) d'un
maillage **linéaire** : chaque type d'élément est promu vers son homologue
quadratique — `SEG2→SEG3`, `TRI3→TRI6`, `QUA4→QUA8`, `TET4→TET10`,
`PENTA6→PENTA15`, `HEX8→HEX20`.

Les **nœuds sommets sont réutilisés** (refcount incrémenté) ; un **nœud de
milieu d'arête** est créé par arête distincte, au milieu géométrique de l'arête,
et **partagé** entre toutes les cellules (tous sous-maillages confondus) qui
utilisent cette arête — le résultat reste donc conforme. Le maillage d'origine
n'est pas modifié. Un sous-maillage `POI1` ou déjà quadratique lève une erreur.

Le maillage obtenu se calcule avec l'interpolation `LAGRANGE2` (cf.
[Espace éléments finis](../fe-space.md)) :

```python
lin = pyrucast.mesh.triangulate_surface(contour, "TRI3", 1.0)  # maillage TRI3
quad = pyrucast.mesh.to_quadratic(lin)  # copie TRI6
print(quad.element_types())  # ['TRI6']

fes = pyrucast.FiniteElementSpace(quad, interpolation="LAGRANGE2")
```

## Changement de type d'élément : `convert`

`convert(mesh, element_type)` **change le type d'élément** de chaque
sous-maillage vers `element_type`, en découpant chaque cellule en cellules du
type cible — **sans jamais déplacer ni ajouter de nœud** sur les sommets
existants. Trois cas sont couverts :

- **identité** — `element_type` est déjà le type du sous-maillage : celui-ci
  est recopié tel quel ;
- **`QUA4`→`TRI3`** — chaque quadrangle est coupé en deux triangles selon la
  diagonale `(0, 2)` : `(0, 1, 2)` et `(0, 2, 3)` ;
- **`HEX8`→`TET4`** — chaque hexaèdre est coupé en six tétraèdres partageant la
  grande diagonale `(0, 6)` (subdivision de Freudenthal/Kuhn), un découpage
  qui pave l'espace et reste **conforme** entre hexaèdres voisins (faces
  coupées selon la même diagonale).

Les **nœuds sommets sont réutilisés** (aucune création, aucun déplacement) et
les **couleurs de face sont conservées**. Le maillage d'origine n'est pas
modifié. Tout autre couple `(source, cible)` lève une erreur : passer à un type
**quadratique** (`TRI3`→`TRI6`, …), qui crée des nœuds de milieu d'arête,
relève de [`to_quadratic`](#passage-à-lordre-quadratique--to_quadratic).

```python
faces = pyrucast.mesh.skin(volume)  # peau en QUA4
faces = pyrucast.mesh.convert(faces, "TRI3")  # QUA4 → TRI3
print(faces.element_types())  # ['TRI3']
```

## Maillage d'un contour fermé : `triangulate_surface`

`triangulate_surface(contour, element_type, size=None)` remplit l'intérieur
d'un contour par **triangulation de Delaunay contrainte (CDT)** puis
**raffinement de Ruppert** à taille de maille cible, en créant les nœuds
internes nécessaires. C'est l'équivalent de l'opérateur Cast3M `SURF`. Le
mailleur est rapide (≈ 3·10⁵ mailles/s) et gère nativement les **trous** et
**plusieurs domaines disjoints** en une passe.

Le **contour est figé** : le maillage produit réutilise exactement les nœuds
d'entrée (mêmes identifiants, mêmes positions) et **n'ajoute aucun nœud sur une
arête du contour**. Le raffinement n'insère donc que des nœuds *intérieurs* ;
pour un bord plus fin, discrétisez le contour en amont (`mesh.line(a, b,
15)`, `mesh.arc(...)`, `mesh.circle(...)`).

`contour` est un `Mesh` contenant **une ou plusieurs boucles SEG2** fermées ;
la configuration peut être en dimension **2** (cas direct) ou une boucle
**plane en 3D** (voir *Contrôle de planéité* plus bas). `element_type` vaut
`"TRI3"` ou `"QUA4"` ; `size` fixe la longueur d'arête visée (par défaut :
longueur moyenne des segments de bord de chaque domaine).

Les fondements mathématiques (aire signée, Newell, Delaunay / Bowyer-Watson,
CDT, Ruppert) sont rassemblés dans
[Triangulation : briques mathématiques](../triangulation.md). Cette page-ci
décrit le **comportement** de `triangulate_surface`.

### Convention d'orientation (à la charge de l'appelant)

`triangulate_surface` s'appuie sur l'**orientation** des boucles fournies :

- une boucle **antihoraire** (CCW, aire signée > 0) est la **frontière
  extérieure** d'un domaine ;
- une boucle **horaire** (CW, aire signée < 0) est un **trou**, contenu dans
  une boucle extérieure ;
- plusieurs boucles CCW disjointes maillent **plusieurs domaines** en une fois.

C'est exactement l'orientation produite par [`border`](#bord-dune-surface--border)
(extérieur CCW, trous CW), donc la sortie de `border` réalimente directement
`triangulate_surface`.

### Méthode

1. les points de bord sont insérés un à un dans une triangulation de Delaunay
   par Bowyer-Watson (super-triangle englobant) ;
2. chaque arête de boucle absente est **récupérée** (retrait du corridor +
   ear-clipping des deux polygones adjacents) puis marquée contrainte ;
3. la triangulation est **légalisée** (flips de Delaunay ne traversant aucune
   contrainte) ;
4. **excavation** : un flood-fill depuis un triangle sûrement intérieur, ne
   traversant jamais une arête contrainte, sépare l'intérieur du domaine des
   trous et des poches hors d'un bord concave ;
5. **raffinement de Ruppert** : les triangles trop plats/trop grands sont
   coupés par insertion de leur circoncentre. Le **contour restant figé**, un
   circoncentre qui *empiéterait* une arête de bord (ou tomberait hors du
   domaine) est **abandonné** plutôt que de couper cette arête — on préserve le
   contour au prix d'un triangle un peu moins bon près du bord ;
6. léger lissage laplacien, puis (pour `QUA4`) recombinaison gloutonne des
   paires de triangles.

En `QUA4` le résultat est donc **quad-dominant** : les triangles sont
recombinés par paires en quadrangles, une poignée de triangles de bord pouvant
subsister (sous-maillage `TRI3` annexe). Les éléments sont orientés **CCW**.

### Contrôle de planéité (cas 3D)

Une boucle 3D est ajustée à son plan de meilleure approximation (méthode de
Newell), maillée dans ce repère 2D local, puis relevée dans l'espace 3D. La
déviation maximale d'un nœud du contour à ce plan doit rester inférieure à
`1e-6 × diag` (`diag` = diagonale de la boîte englobante) ; au-delà,
`triangulate_surface` retourne une erreur indiquant la déviation observée et la
tolérance. Ce seuil relatif tolère le bruit numérique tout en refusant les
vrais contours gauches.

### Raffinement — convergence

Le raffinement de Ruppert garantit théoriquement sa convergence pour un angle
minimal `≤ 20.7°` (Shewchuk) ; pyrucast vise 20° et plafonne le nombre
d'insertions pour éviter les divergences (erreur explicite si la limite est
atteinte). Les nouveaux nœuds (« Steiner ») sont **strictement intérieurs** et
créés dans la `Coords` du contour, exactement comme les nœuds utilisateur ;
aucun n'est posé sur le contour, qui reste figé. Cette contrainte peut laisser
subsister, contre un bord grossièrement discrétisé, un triangle plus plat que
l'angle visé : affiner alors le contour d'entrée plutôt que la taille cible.

### Exemple Python

```python
import pyrucast

c = pyrucast.Coords(dim=2)

# Contour extérieur : carré 4×4 (CCW).
outer = pyrucast.Mesh(c, "SEG2")
outer_nodes = [
    c.add_node(list(p)) for p in [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
]
for i in range(4):
    outer.unit().add_cell([outer_nodes[i], outer_nodes[(i + 1) % 4]])

# Trou : carré 2×2 centré, orienté CW.
hole = pyrucast.Mesh(c, "SEG2")
hole_nodes = [
    c.add_node(list(p)) for p in [(1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0)]
]
for i in range(4):
    hole.unit().add_cell([hole_nodes[i], hole_nodes[(i + 1) % 4]])

# Composer les deux contours par l'union | (jamais +).
combined = outer | hole

# Maillage TRI3 de taille ~0.5 (aire = 16 - 4 = 12).
tri = pyrucast.mesh.triangulate_surface(combined, "TRI3", size=0.5)
print(tri.element_types(), tri.cell_count())

# Variante quad-dominante.
quad = pyrucast.mesh.triangulate_surface(combined, "QUA4", size=0.5)
print(quad.element_types())  # ['QUA4', 'TRI3'] en général
```

### Interruption

Un maillage trop long s'**interrompt** par `Ctrl+C` : `triangulate_surface`
sonde les signaux et lève une `KeyboardInterrupt`. Côté Rust,
`triangulate_surface_cancellable(contour, type, size, &cancel)` accepte un
jeton d'interruption (timeout, drapeau partagé…) — voir
[Interrompre une fonction](../developper/interrompre-une-fonction.md).

### Limitations actuelles

- **Taille uniforme** : pas encore de champ de densité variable par nœud
  (un `size` façon `CHPO1`).
- L'orientation est à fournir par l'appelant ; une boucle mal orientée mène à
  une erreur (aucune boucle extérieure) ou à un domaine inattendu.
- Les boucles doivent être deux à deux disjointes (pas de trous emboîtés, pas
  de croisements).

Côté Rust, `ops::mesh::triangulate_surface(&contour, ElementType::TRI3, Some(0.5))`.
Le cœur (CDT + raffinement) opère sur de simples `Vec<Point2>` sans toucher au
store ; lissage et recombinaison QUA4 sont parallélisés (`rayon`). Le module
`pyrucast::ops::mesh::triangulation` regroupe par ailleurs les briques
géométriques réutilisables indépendamment du système `Mesh`
(voir [Triangulation](../triangulation.md)).

## Pavage frontal d'un contour fermé : `pave_surface`

`pave_surface(contour, element_type, size=None, all_quad=False)` remplit le
même contour que `triangulate_surface`, mais en **posant directement des
quadrangles**, par rangées qui avancent depuis le bord vers l'intérieur. C'est
la version quadrangle de l'opérateur Cast3M `SURF`.

### Pourquoi un second mailleur de surface

`triangulate_surface` accepte `"QUA4"`, mais il triangule d'abord et
**recombine** ensuite les triangles deux par deux : ce qu'on obtient dépend de
la chance des appariements, les valences sont désordonnées et rien n'est
aligné sur le bord. `pave_surface` ne recombine pas. Il pose des rangées
**parallèles au contour**, ce qui est précisément la structure recherchée en
éléments finis, où les gradients de contrainte et de flux sont les plus forts
près des frontières.

| | `triangulate_surface` | `pave_surface` |
|---|---|---|
| méthode | Delaunay contraint + Ruppert | front avançant |
| élément naturel | `TRI3` | `QUA4` |
| `QUA4` obtenu par | recombinaison de paires | construction |
| rangées alignées sur le bord | non | oui |
| tout-quadrangle garanti | impossible | `all_quad=True` |

### Convention

Identique à `triangulate_surface`, et volontairement : les deux opérateurs
partagent leur lecture de contour. `contour` est un `Mesh` d'une ou plusieurs
**boucles `SEG2` fermées**, chacune dans **un seul** sous-maillage
(`pyrucast.mesh.consolidate`), orientées par l'appelant — **CCW** pour une frontière
extérieure, **CW** pour un trou. Plusieurs boucles CCW disjointes pavent
plusieurs domaines indépendants en une passe. La configuration peut être en
dimension **2**, ou une boucle **plane en 3D** (ajustée à son plan de meilleur
approximation par la méthode de Newell, pavée dans ce plan, puis relevée).

**Le contour est figé** : les nœuds d'entrée sont réutilisés tels quels (mêmes
identifiants, mêmes positions), ne sont jamais déplacés, et aucun nœud n'est
ajouté sur une arête de bord. Voir *Le contour est intouchable* ci-dessous.

`element_type` vaut `"QUA4"`, `"QUA8"` ou `"QUA9"` (les formes quadratiques
sont dérivées du maillage `QUA4`). `size` fixe la longueur d'arête visée ; par
défaut, la longueur moyenne des segments de bord du domaine.

### Méthode

Le **front** part du bord du domaine et avance vers l'intérieur. Il est
toujours un ensemble de boucles simples et disjointes, matière à gauche ; cet
invariant n'est jamais supposé, il est **maintenu**.

1. **Une rangée par tour.** À chaque nœud du front, le nombre de quadrangles
   voulus est \\( k = \operatorname{round}(\theta / 90°) \\), borné à
   \\( 1..4 \\), où \\( \theta \\) est l'angle intérieur. Ce n'est pas un seuil
   réglé : si \\( k \\) quadrangles entourent le nœud, ses voisins forment un
   chemin de \\( k+1 \\) sommets, dont \\( k-1 \\) sont neufs — vouloir des
   angles droits fixe \\( k \\). Les nouveaux nœuds se placent sur les rayons
   qui découpent le secteur en \\( k \\) parts égales.
2. **Refus et retrait.** Un quadrangle non strictement convexe a un jacobien
   négatif à son coin rentrant : aucun code éléments finis ne peut l'intégrer.
   Une rangée qui en produirait un, ou dont les arêtes croiseraient le front,
   est **refusée** ; le planificateur dit *quels* nœuds sont en cause, et la
   rangée est reprise moins loin à cet endroit seulement.
3. **Couture.** Deux nœuds de front qui se rapprochent à moins d'environ une
   demi-maille sont **identifiés**. La même opération *scinde* une boucle
   quand les deux nœuds lui appartiennent — c'est ainsi qu'une géométrie
   concave se divise — et *joint* deux boucles sinon — c'est ainsi qu'un trou
   est absorbé. Les trous n'ont donc aucun traitement particulier.
4. **Déblocage.** Une boucle qui n'avance plus est **coupée en deux** par une
   corde, et les deux moitiés reprennent le pavage.
5. **Fermeture.** Une boucle réduite à six nœuds ou moins est remplie par
   décomposition, sans jamais découper une arête (ce qui laisserait un nœud
   en T, donc un maillage non conforme).
6. **Nettoyage topologique**, puis **lissage** sous garde de validité qui ne
   déplace jamais un nœud du contour. Dans cet ordre : lisser un nœud qui n'a
   pas le bon nombre de mailles autour de lui ne fait qu'étaler l'erreur sur
   ses voisins.

Le nettoyage corrige ce que le lissage ne peut pas atteindre, parce que c'est
de la **connectivité** et non de la géométrie :

- un **doublet** — un nœud intérieur n'ayant que deux mailles autour de lui,
  qui partagent donc *deux* arêtes. Il reste coincé dans un coin quelles que
  soient les positions ; fusionner les deux mailles supprime le nœud et le coin
  d'un coup ;
- une **valence fautive**. Un nœud intérieur veut quatre mailles : avec trois,
  les angles valent 120° en moyenne, avec cinq, 72°, et aucun lissage n'y peut
  rien puisque les angles autour d'un nœud somment à 2π quelles que soient les
  positions. Deux mailles voisines forment un hexagone, qui se recoupe selon
  l'une de ses trois diagonales ; changer de diagonale déplace une unité de
  valence. C'est le seul geste, il ne change ni le nombre de nœuds ni le bord,
  et il n'est appliqué que s'il fait strictement baisser l'erreur de valence.

Le nombre de mailles voulu à un nœud est le même
\\( \operatorname{round}(\theta / 90°) \\) que dans la classification des
rangées, avec ici \\( \theta \\) la somme des angles incidents : \\( 2\pi \\) à
l'intérieur — d'où le quatre familier — et moins au bord, ce qui donne trois le
long d'une arête droite et deux dans un coin droit. Une seule formule, aucun
cas particulier « nœud de bord ».

Toutes les décisions topologiques — convexité, croisement de segments —
passent par le prédicat **exact** `orient2d` (technique de Shewchuk, partagé
avec le mailleur volumique). Ce ne sont donc pas des estimations.

### Le contour est intouchable

Tous les nœuds du contour reviennent dans le maillage, à leur position, et
**aucun nœud n'est jamais ajouté sur une arête de bord**. La discrétisation du
bord appartient à l'appelant : elle porte en général les conditions aux
limites, et un nœud glissé au milieu d'un segment serait un nœud que personne
n'a demandé. Rien dans le paveur ne découpe une arête de bord, et la couture —
la seule opération qui abandonne un nœud — refuse d'abandonner un nœud de
contour.

Corollaire : un contour avec lequel le paveur ne peut pas travailler est
**signalé**, pas contourné. Deux cas, tous deux renvoyés en erreur nommant le
problème :

- `all_quad` sur une boucle à nombre **impair** de segments. Un polygone à
  nombre impair de côtés n'admet aucun remplissage en quadrangles seuls ; le
  pavage ne peut pas changer cette parité — une rangée la conserve, une couture
  retire deux nœuds — et rééquilibrer le compte reviendrait à ajouter un nœud
  au bord ;
- un contour si grossier, ou si irrégulier pour la taille demandée, que le
  front **se replie sur lui-même** en laissant une région impossible à
  remplir. L'erreur indique l'endroit.

### Le tout-quadrangle

Laissé à lui-même (`all_quad=False`), un contour impair coûte simplement **un**
triangle, rendu dans un sous-maillage `TRI3` séparé — avec les quelques mailles
qu'un polygone résiduel trop déformé n'a pas pu rendre carrées : la fermeture
préfère deux triangles à une maille de jacobien négatif, et la validité n'est
jamais échangée.

Avec `all_quad=True`, la parité devient une exigence sur l'entrée : discrétisez
chaque boucle de bord avec un nombre pair de segments, et le résultat est sans
triangle.

### Exemple Python

```python
import pyrucast as pc

coords = pc.Coords(2)
# … contour extérieur CCW et cercle-trou CW, consolidés en une boucle chacun.
# Chaque boucle du contour a un nombre pair de segments, donc all_quad passe.
plaque = pc.mesh.pave_surface(contour, "QUA4", size=0.002, all_quad=True)
print(plaque.element_types())  # ['QUA4']

# Le solide prismatique vient alors gratuitement, et en hexaèdres purs.
volume = pc.mesh.extrude(plaque, [0, 0.02, 0], 2)
print(volume.element_types())  # ['HEX8']
```

### Interruption

Le pavage interroge les signaux Python entre deux rangées : `Ctrl+C` pendant
un maillage long lève `KeyboardInterrupt`. Côté Rust, la forme
`pave_surface_cancellable(..., cancel)` prend un jeton `Cancel`.

### Qualité

Plaque 30 × 10 cm percée, taille visée 1,6 mm, 15 652 mailles (99,8 % de
quadrangles) :

| indicateur | valeur |
|---|---|
| jacobien normalisé — médiane | **1,000** |
| jacobien normalisé — moyenne / p5 / p1 / min | 0,967 / 0,84 / 0,60 / 0,069 |
| mailles inversées | **0** |
| mailles sous 0,5 | 0,61 % |
| angle médian | **90,0°** |
| angles sous 30° | 0,17 % |
| nœuds intérieurs de valence 4 | **97,4 %** |
| élancement médian | 1,86 |

Autrement dit : le cœur du maillage est fait de rectangles à angle droit et de
valence régulière, ce qui est exactement ce qu'on demande à un maillage
quadrangulaire. Les deux faiblesses résiduelles sont l'**élancement** — les
mailles sont d'équerre mais environ deux fois plus longues que larges, parce
que l'espacement le long du front et la distance d'avance évoluent
indépendamment — et la dispersion de la **taille d'arête**, qui va de 0,4 à
3,7 mm autour des 1,6 mm demandés.

### Coût

Plaque trouée de 30 × 10 cm percée d'un trou de rayon 3,5 cm, taille de maille
0,29 mm, en `--release` :

Plaque 30 × 10 cm percée, contour à 224 segments, `--release` :

| taille visée | mailles | temps | débit | µs/maille |
|---|---|---|---|---|
| 4 mm | 3 171 | 0,033 s | 96 000 /s | 10,4 |
| 1,6 mm | 15 652 | 0,17 s | 91 000 /s | 11,0 |
| 0,4 mm | 181 067 | 1,91 s | 95 000 /s | 10,6 |
| 0,2 mm | 310 446 | 6,14 s | 51 000 /s | 19,8 |

Le coût est **linéaire** — environ 10 µs par maille sur deux ordres de grandeur
— tant que le front avance sans se coincer. Il double quand les blocages se
multiplient, la boucle repassant alors par les cordes de déblocage et les
fermetures.

Le temps se répartit en gros en trois tiers : la pose des rangées, le
nettoyage topologique et le lissage final. À titre de comparaison sur la même
géométrie à 1,6 mm, `triangulate_surface` produit 93 000 mailles/s en `QUA4`
(mais 23 % de triangles) et 178 000 /s en `TRI3`.

1,2 % des mailles ont un jacobien normalisé inférieur à 0,5. Le coût est
essentiellement linéaire : le front croît comme la racine du nombre de mailles,
et l'index spatial est reconstruit à chaque rangée pour ce prix-là.

### Pièges

- **Une boucle par sous-maillage.** Comme pour `triangulate_surface`, une
  boucle fermée doit tenir dans un seul sous-maillage : `pyrucast.mesh.consolidate`
  après avoir uni les côtés.
- **Orientation.** Un trou doit être **CW**. `pyrucast.mesh.invert` retourne
  un cercle construit en CCW.
- **La taille du contour compte.** Le front part de la discrétisation du bord
  et converge vers `size` en quelques rangées. Un contour beaucoup plus
  grossier que `size` donne donc des premières rangées plus grosses que
  demandé.

### Limitations actuelles

- La taille des mailles n'est pas uniforme : l'espacement le long du front et
  la distance d'avance ne sont pas asservis l'un à l'autre, d'où un élancement
  médian proche de 2 et une taille d'arête étalée d'un facteur 10. Les angles,
  eux, restent droits.
- Deux fronts qui se rejoignent de face laissent normalement un éclat de
  recouvrement, dégénéré et sans matière : il est écarté. Au-delà d'une
  demi-maille d'aire, en revanche, c'est une région perdue, et le paveur
  **sort en erreur** en indiquant l'endroit plutôt que de rendre un maillage
  troué.
- **Front convexe sans coin.** Un front ne perd des nœuds que par couture, et
  une couture n'est acceptée que si elle laisse toutes les mailles valides. Un
  contour circulaire n'a aucun coin : son front garde donc son nombre de nœuds
  pendant qu'il se contracte, ses arêtes raccourcissent, et les rangées
  finissent par ne plus tenir. Le paveur s'en sort par des cordes de
  découpage, mais le débit s'effondre — de l'ordre de 10³ mailles/s sur un
  disque finement discrétisé, contre 10⁵ sur la plaque trouée. Une géométrie
  comportant des coins, ou un contour discrétisé près de la taille visée, ne
  rencontre pas ce cas.
- Pas de champ de taille variable : `size` est uniforme par domaine.

## Couche limite hexaédrique : `pave_volume`

`pave_volume(envelope, layers=1, thickness=None, size=None)` remplit la même
enveloppe fermée que `triangulate_volume`, mais met des **hexaèdres là où ils
comptent** — dans la couche contre le bord, où les gradients de contrainte et
de flux sont les plus raides et où la forme d'une maille décide de la
précision — et laisse l'intérieur, où le champ est lisse, aux tétraèdres.

```text
peau (QUA4 / TRI3)  ──►  décalage intérieur  ──►  HEX8 / PENTA6   (couche limite)
                                                        │
                     faces internes carrées ──► PYRA5   │           (le raccord)
                                                        ▼
                                 vide borné par des triangles seulement
                                                        │
                                          triangulate_volume  ──►  TET4
```

### Pourquoi les pyramides ne sont pas facultatives

Les faces internes de la couche sont **carrées**, et un tétraèdre n'en a
aucune. Découper chaque carré en deux triangles ne suffit pas : l'hexaèdre de
l'autre côté continue de voir une face carrée, avec un nœud suspendu en son
milieu. Le maillage n'est plus conforme et aucun solveur ne peut assembler à
travers cette face.

La pyramide est le seul élément qui présente un carré d'un côté et des
triangles de l'autre — c'est exactement ce que réclame le raccord. Ses
fonctions de forme se réduisent à celles d'un `QUA4` sur la base et restent
linéaires le long des arêtes vers le sommet, ce qui assure la continuité des
deux côtés : voir [PYRA5](../elements/pyra5.md).

Une fois chaque face carrée coiffée, ce qui reste du vide n'est plus borné que
par des triangles, et le mailleur tétraédrique existant prend le relais — en
mode **strict**, donc en réutilisant ces triangles tels quels : les deux
parties du maillage se rejoignent nœud à nœud.

### Un front, pas un décalage global

Décaler l'enveloppe entière d'une même distance est la version naïve, et elle
lâche dès que le solide cesse d'être épais partout : un seul étranglement
plafonne l'épaisseur de toutes les couches, sur toute la pièce. Le front fait
trois choses à la place.

**Il avance de ce que la place permet.** Le pas de chaque nœud est borné par la
distance du front à lui-même en ce point — la moitié de la distance à la
facette la plus proche à laquelle il n'appartient pas. Là où le solide est
épais la couche est pleine, là où il se pince elle s'amincit au lieu que tout
le maillage renonce. Demander une couche vingt fois plus épaisse que la pièce
ne produit donc plus une erreur mais une couche adaptée.

**Il place ses éléments localement.** Une facette qui ne peut pas avancer —
parce que la maille sortirait retournée — reste où elle est pendant que ses
voisines poursuivent. La marche ainsi créée est refermée par une **paroi
latérale**, un quadrangle neuf tendu entre l'ancienne arête et la nouvelle. Son
orientation n'est pas choisie mais forcée : si `A` avance et que sa voisine `B`
reste, l'arête `(u,w)` qu'elles partageaient n'est plus empruntée que par `B`,
dans le sens `(w,u)` ; il faut donc que quelque chose l'emprunte en `(u,w)`, et
emprunte la nouvelle arête en `(w',u')`. Le quadrangle `[u, w, w', u']` fait
exactement les deux, et le front reste une surface fermée et orientée.

**Il se coud.** Deux parties du front qui se retrouvent à distance de contact
sont soudées, ce qui referme une région mince au lieu d'y laisser un éclat
qu'aucune maille ne peut remplir. Deux critères, tous deux nécessaires : les
deux nœuds ne doivent pas partager de facette — les souder l'écraserait, ce
n'est pas une couture mais une dégénérescence — et leurs normales doivent
**se faire face**. Sans ce second critère, deux nœuds voisins d'une même nappe
lisse se soudent et replient la surface ; c'est ce qui rendait le cube
impossible à mailler pendant la mise au point.

**Il est lissé.** Les nœuds qu'il a créés sont relâchés sous garde de validité,
si bien qu'un pas raccourci par la place disponible ne reste pas en coude. La
garde est le jacobien normalisé lui-même, pas un indicateur qui lui
ressemblerait, et le balayage est de Gauss–Seidel : chaque déplacement est jugé
sur le maillage tel qu'il est, donc la pire maille ne peut que s'améliorer.

### Décaler n'est pas « déplacer chaque nœud selon sa normale »

Moyenner les normales des facettes incidentes donne une direction, et cette
direction ne suffit pas. Déplacez le coin d'un cube de \\( t \\) le long de la
normale moyenne \\( (1,1,1)/\sqrt3 \\) et chacune des trois faces ne s'écarte
que de \\( t/\sqrt3 \\) : la couche est la plus mince là où la géométrie
tourne, c'est-à-dire là où elle peut le moins se le permettre. Pire, au coin
d'un tétraèdre la normale moyenne est **tangente** à l'une des faces
incidentes, qu'un déplacement le long d'elle ne décale donc pas du tout.
Aucun facteur d'échelle ne rattrape cela : c'est la direction qui est fausse.

Le nœud décalé doit être le point où les facettes incidentes, chacune poussée
de \\( t \\) vers l'intérieur, se rencontrent :

\\[
\mathbf{d} \cdot \mathbf{n}_j = -t \quad \text{pour chaque facette incidente } j,
\\]

soit trois équations à trois inconnues à un coin, davantage sur une surface
lisse, moins sur une arête. Résoudre au sens des **moindres carrés**, par les
équations normales \\( (N^{\!\top}\!N)\,\mathbf{d} = -t\,N^{\!\top}\mathbf{1} \\),
couvre les trois cas d'un coup et rend l'intersection exacte quand elle
existe. Au coin du cube : \\( \mathbf{d} = -t(1,1,1) \\).

### Convention

`envelope` est une surface **fermée** de facettes `QUA4` et/ou `TRI3`, de
normales **sortantes de la matière** — même convention que
`triangulate_volume`, si bien que la peau d'un maillage (`skin`) s'y branche
directement. Ses nœuds sont réutilisés tels quels. Une enveloppe ouverte, ou
dont les facettes se contredisent sur l'orientation, est refusée en nommant
l'arête fautive.

`layers` couches sont poussées vers l'intérieur, chacune de `thickness` de
profondeur ; `thickness=None` prend la longueur d'arête moyenne de
l'enveloppe, ce qui donne des mailles à peu près cubiques. `size` est la
taille visée pour le cœur tétraédrique.

Le résultat porte un sous-maillage `HEX8` (issu des facettes carrées), un
`PENTA6` (des triangulaires), un `PYRA5` (le raccord) et un `TET4` (le cœur),
chacun présent seulement s'il n'est pas vide.

### Exemple Python

```python
import pyrucast as pc

peau = pc.mesh.skin(solide)  # QUA4, normales sortantes
maille = pc.mesh.pave_volume(peau, layers=1, thickness=0.15, size=0.4)
print(dict(zip(maille.element_types(), maille.cell_counts())))
# {'HEX8': 54, 'PYRA5': 54, 'TET4': 408}
```

### Pièges

- **Orientation.** Une enveloppe retournée est refusée en le disant ;
  `pyrucast.mesh.invert` la remet à l'endroit.
- **Épaisseur.** Une couche plus épaisse que le solide ne peut pas rentrer :
  le décalage retourne l'enveloppe et l'opérateur sort en erreur en nommant la
  couche fautive.
- **Le cœur peut refuser.** Le mailleur tétraédrique travaille en mode strict,
  donc sans ajouter de nœud sur la surface intérieure ; s'il n'y arrive pas,
  l'erreur le dit et une couche plus mince, ou une enveloppe plus fine, est la
  réponse habituelle.

### Qualité et coût

Mesuré sur des cubes de 6³ à 16³ mailles de peau, en `--release`
(`cargo test --release -- --ignored volume_report --nocapture`) :

| cas | mailles | temps | débit | jacobien médian | inversées |
|---|---|---|---|---|---|
| cube 6³, 1 couche | 4 054 | 0,22 s | 18 700 /s | 0,374 | 0 |
| cube 8³, 1 couche | 7 422 | 0,38 s | 19 500 /s | 0,345 | 0 |
| cube 12³, 1 couche | 16 454 | 0,80 s | 20 600 /s | 0,418 | 0 |
| cube 16³, 1 couche | 35 015 | 1,72 s | 20 300 /s | 0,431 | 0 |

Le coût est **linéaire**, autour de 20 000 mailles/s et 50 µs par maille. Par
type, sur le cube 16³ :

| type | minimum | médiane |
|---|---|---|
| `HEX8` | 0,577 | **0,987** |
| `PYRA5` | 0,181 | 0,319 |
| `TET4` | 0,040 | 0,431 |

La couche hexaédrique est donc quasi parfaite — c'est le but — et le minimum de
0,577 n'est pas un défaut mais la valeur exacte d'un coin à 60°, celle des
hexaèdres qui suivent une arête du cube. Les pyramides sont les mailles les
plus médiocres du lot, ce qui est attendu d'un élément de raccord aplati.

Une seconde couche coûte cher : le cube 6³ tombe à 2 100 mailles/s, la
tétraédrisation du cœur devenant nettement plus difficile.

### Ce que l'enveloppe devient

Elle est **respectée à la maille près**. Ses nœuds sont réutilisés tels quels
(mêmes identifiants, mêmes positions), ils sont marqués immobiles donc le
lissage ne les touche pas, et chaque facette devient **exactement une face de
maille** — un `QUA4` la face d'un `HEX8`, un `TRI3` celle d'un `PENTA6`. Aucun
nœud n'est ajouté sur le bord. La couture protège explicitement ce contrat :
entre deux nœuds candidats, celui de l'enveloppe est toujours le survivant, et
si les deux en sont, la couture est refusée.

C'est aussi pour cela que le cœur tétraédrique tourne en mode **strict** : en
mode permissif il ajouterait des nœuds sur la surface intérieure et ne
rejoindrait plus les pyramides. Cette exigence est la contrepartie du contrat,
et la principale source d'échec.

### Limitations actuelles

- **Deux cas sur huit échouent**, tous deux au cœur : une plaque mince et un
  barreau, c'est-à-dire les géométries où le front se referme réellement sur
  lui-même. La couture soude bien, mais un repli *partiel* laisse un vide que
  le mailleur tétraédrique, en mode strict, ne sait pas remplir. C'est la
  limite principale aujourd'hui.
- La profondeur des pyramides est fixée au quart de l'arête de leur base. Plus
  profondes, elles seraient mieux formées mais finiraient par se traverser.
- **Les parois latérales ne servent jamais sur les cas mesurés.** Le mécanisme
  est là et testé — retenir une facette lève bien quatre parois et le front
  reste fermé — mais sur les huit géométries du banc, le plafonnement par la
  place suffit toujours à rendre les mailles valides, et aucune facette n'a
  besoin d'être retenue. C'est une capacité en réserve, pas un moteur en
  service.
- Le plastering **complet** — remplir tout le volume d'hexaèdres par front
  avançant, sans cœur tétraédrique — reste un problème ouvert. Sandia, qui l'a
  inventé, l'a abandonné : la fermeture du vide central bute sur des
  obstructions topologiques qu'une méthode locale ne voit pas. Le cœur
  tétraédrique n'est donc pas un raccourci mais l'état de l'art.

## Mailleur volumique : `triangulate_volume`

```python
solide = pyrucast.mesh.triangulate_volume(
    enveloppe, size=None, allow_surface_nodes=False
)
```

Le **compagnon 3D** de `triangulate_surface` : il remplit l'intérieur d'une
**enveloppe fermée en TRI3** avec des `TET4`. Les
normales de l'enveloppe doivent pointer **vers l'extérieur de la matière** ;
une forme concave est admise, et une cavité interne n'est qu'une autre
surface fermée dont les normales pointent vers le trou — elle se soustrait
d'elle-même, sans argument dédié.

```python
peau = pyrucast.mesh.convert(pyrucast.mesh.skin(solide_penta6), "TRI3")
peau = pyrucast.mesh.invert(peau)  # voir « pièges », plus bas
volume = pyrucast.mesh.triangulate_volume(peau, size=0.01)
```

L'enveloppe est **respectée exactement** : ses nœuds sont réutilisés tels
quels (mêmes `NodeId`, mêmes positions), et aucun nœud n'est posé sur la
surface — les nœuds ajoutés le sont strictement à l'intérieur. Ce n'est pas
un effort au mieux : avant d'écrire quoi que ce soit, l'opérateur vérifie que
le bord du maillage produit est *exactement* l'ensemble des facettes reçues.

### Ce qui se passe, et pourquoi

Le mailleur enchaîne cinq étapes. Chacune répond à une difficulté précise, et
il vaut la peine de savoir laquelle, parce que les messages d'erreur les
nomment.

#### 1. Prédicats exacts

Tout repose sur deux questions posées des millions de fois : *de quel côté du
plan `(a, b, c)` se trouve `d` ?* et *le point `e` est-il dans la sphère
passant par `a, b, c, d` ?* Ce sont les signes de deux déterminants :

\\[
\mathrm{orient3d}(a,b,c,d)=\begin{vmatrix}
b_x-a_x & b_y-a_y & b_z-a_z\\\\
c_x-a_x & c_y-a_y & c_z-a_z\\\\
d_x-a_x & d_y-a_y & d_z-a_z
\end{vmatrix}
\qquad
\mathrm{insphere}(a,b,c,d,e)=-\begin{vmatrix}
a_x-e_x & a_y-e_y & a_z-e_z & \lVert a-e\rVert^2\\\\
b_x-e_x & b_y-e_y & b_z-e_z & \lVert b-e\rVert^2\\\\
c_x-e_x & c_y-e_y & c_z-e_z & \lVert c-e\rVert^2\\\\
d_x-e_x & d_y-e_y & d_z-e_z & \lVert d-e\rVert^2
\end{vmatrix}
\\]

`orient3d` vaut six fois le volume signé, et il est **positif** exactement
quand le tétraèdre `(a,b,c,d)` est bien orienté au sens de `TET4` — face
`0-1-2` vue en sens direct depuis le nœud 3.

Ce qui compte n'est pas la précision de ces valeurs mais leur **cohérence** :
si `orient3d(a,b,c,d)` répond « au-dessus », alors `orient3d(b,a,c,d)` doit
répondre « en dessous », et un point ne peut pas être à la fois dans un
tétraèdre et hors de ses quatre faces. En `f64` nu, près d'une
dégénérescence, cette cohérence tombe — et une seule réponse contradictoire
corrompt le graphe d'adjacence. C'est de cette façon qu'un mailleur
incrémental tourne en boucle ou produit des cellules qui se recouvrent.

Ni une tolérance ni un jitter n'y remédient : ils rendent le prédicat
*généralement* juste, pas cohérent avec lui-même. `triangulate_volume` calcule donc
le **signe exact**, par la technique des expansions flottantes de Shewchuk :
une estimation en `f64` comparée à une borne d'erreur rigoureuse, puis, dans
le seul cas où le signe reste indécidable, une réévaluation en arithmétique
exacte. Un cube, une grille régulière, des coins cosphériques sont ainsi
**décidés** et non devinés.

#### 2. Triangulation de Delaunay des nœuds de l'enveloppe

Construite point par point selon **Bowyer-Watson** : on supprime tous les
tétraèdres dont la sphère circonscrite contient le nouveau point `p`, puis on
rebouche la cavité en joignant `p` à chaque face de son bord. L'appartenance
à la cavité est `insphere > 0` et rien d'autre, ce qui garantit que la cavité
reste **étoilée** — visible en entier depuis `p` — et donc que le
rebouchage produit des cellules bien formées.

Les points cosphériques ne sont pas perturbés : `insphere == 0` signifie
simplement « hors cavité ». On obtient l'une des triangulations de Delaunay
valides de la configuration dégénérée, choisie de façon cohérente.

#### 3. Récupération du bord

La triangulation précédente pave l'**enveloppe convexe** des nœuds et ne doit
rien à la surface d'où ils viennent : une arête de l'enveloppe peut être
traversée par un tétraèdre, une facette percée par une arête. Avant de
pouvoir distinguer l'intérieur de l'extérieur, chacune doit *apparaître*
dans la triangulation.

C'est la partie difficile, et pour une raison de fond : **certains polyèdres
n'admettent aucune tétraédrisation sur leurs propres sommets**. Le prisme
tordu de Schönhardt est l'exemple d'école. Plus près de vous : sur les 64
façons de trianguler le bord d'un cube, la plupart ne se remplissent pas. Ce
n'est donc pas un algorithme perfectible, c'est une question dont la réponse
est parfois « il n'y en a pas ».

La récupération procède obstacle par obstacle — bascules locales, puis
reconstruction de la poche qui bloque — et, quand elle n'y arrive pas, elle
nomme l'arête ou la facette en cause plutôt que de rendre un maillage qui ne
correspond pas à la surface reçue.

**Reconstruire une poche : deux remplisseurs.** Le premier *cherche*, en
faisant croître un pavage cellule par cellule depuis la surface de la poche.
Il est complet — lui seul peut prouver qu'*aucun* remplissage n'existe — mais
il est exponentiel et s'arrête à quelques cellules. Le second ne cherche pas :
il **calcule** l'unique candidat canonique, la triangulation de Delaunay des
sommets de la poche, et lui pose une seule question — contient-elle chaque
face de la surface de la poche ? Si oui, les cellules qui tombent à
l'intérieur pavent exactement la poche, pour le prix d'une triangulation quel
que soit son volume ; si non, il **nomme la face** sur laquelle il a buté.

Ce nom est une instruction, pas un diagnostic : en absorbant la cellule située
de l'autre côté de cette face, celle-ci cesse d'être sur la surface de la
poche, et l'obstacle ne peut pas se représenter. On repose alors une question
strictement plus grande, sans arbre de recherche ni retour arrière. C'est ce
qui rend abordable la récupération d'une facette prise dans un plan interne —
la diagonale d'un quadrilatère de paroi d'extrusion, par exemple — là où la
recherche exhaustive renonçait. La facette à récupérer est passée comme un
**mur à deux faces** : la poche est coupée en deux le long d'elle et chaque
moitié est remplie depuis son propre côté, de sorte qu'une facette contenue
dans la triangulation est une facette récupérée.

> **Ce que cela ne fait pas.** Une triangulation de Delaunay est ce qu'elle
> est : on ne peut pas lui *demander* une arête. Or une arête d'enveloppe
> manquante est manquante précisément parce qu'elle n'est pas de Delaunay.
> La retriangulation de cavité traite donc les **facettes**, pas les arêtes ;
> une arête bloquée reste l'affaire des bascules, et si elles n'y suffisent
> pas, du mode `allow_surface_nodes`.

#### 4. Séparation matière / vide

Les facettes de l'enveloppe deviennent des **murs**. Deux tétraèdres séparés
par un mur sont de part et d'autre de la surface ; toute autre paire de
voisins est du même côté. L'intérieur s'obtient donc par inondation depuis
des cellules connues intérieures — et « connue intérieure » découle de
l'orientation que vous avez fournie, la matière étant du côté opposé à la
normale.

L'inondation est menée **des deux côtés**, et les deux résultats doivent
partitionner le maillage : chaque cellule intérieure ou extérieure, aucune
les deux, aucune ni l'une ni l'autre. Une cellule qui échappe à cela est une
fuite, signalée et non maillée en silence.

#### 5. Qualité : raffinement puis chasse aux slivers

Un maillage *valide* n'est pas un maillage *utilisable*. Une tétraédrisation
des seuls nœuds d'une surface contient toujours des cellules dont les quatre
coins sont presque coplanaires, et une poignée suffit à rendre une matrice
élémentaire singulière.

**Raffinement de Delaunay.** On prend la cellule la plus mal formée et on
pose un nœud au centre de sa sphère circonscrite. Deux critères décident :

- le **rapport rayon-arête** \\( \rho = R/\ell \\), rayon circonscrit sur
  arête la plus courte. Un tétraèdre régulier vaut \\( \rho = \sqrt6/4
  \approx 0{,}61 \\) ; une aiguille ou un coin en donne un grand. Découper
  au-dessus d'un seuil \\( B \\) **termine pour tout \\( B > 2 \\)** — d'où
  le seuil juste au-dessus de 2 ;
- la **taille**, \\( R \\) comparé à `size`. Pour un tétraèdre régulier
  d'arête \\( a \\), \\( R = a\sqrt6/4 \\), ce qui convertit la longueur
  d'arête demandée en rayon visé.

Une règle est indispensable à la terminaison : **l'empiètement**. Un nœud
posé dans la sphère d'une facette du bord dégrade les cellules contre cette
facette au lieu de les améliorer, et le raffinement le redemande sans fin. La
réponse classique est de découper la facette ; l'enveloppe ne nous
appartenant pas, la cellule est simplement laissée telle quelle.

**C'est aussi ce qui borne `size` par le bas** : on ne peut pas mailler plus
fin que la surface ne le permet. Sur un cube à huit coins, tout centre
circonscrit empiète, et `size` n'a aucun effet ; il faut une enveloppe déjà
discrétisée à la finesse voulue.

**Chasse aux slivers.** Le raffinement ne peut rien contre le **sliver** :
quatre coins proches d'un même plan, répartis régulièrement sur un cercle.
Son arête la plus courte est honorable et sa sphère circonscrite petite —
\\( \rho \\) ne voit rien d'anormal — alors que son volume est presque nul.
C'est un théorème, pas une lacune d'implémentation : aucun nœud inséré ne le
casse. Le maillage est donc *amélioré* plutôt que subdivisé, par les deux
seuls mouvements qui ne changent pas ce qu'il remplit :

- **reconnexion** : les mêmes nœuds, joints autrement ;
- **retrait d'arête** : les arêtes du sliver ôtées, une à une ;
- **relaxation** : les mêmes liens, un nœud déplacé — les nœuds **intérieurs**
  seulement, jamais les vôtres.

Les trois sont jugés sur le plus petit **angle dièdre** des cellules touchées
et appliqués seulement s'ils l'améliorent, ce qui rend la passe monotone.

Le deuxième est celui qui tue réellement les slivers, et la raison est
géométrique. Un sliver est plat **en travers** d'une arête : une bascule 2-3
sur une de ses faces n'a donc souvent nulle part où aller — la paire de
cellules n'y est pas convexe — tandis que vider l'anneau de cellules autour de
l'arête fautive et le remplir autrement a toujours la place de le faire. Le
retrait d'arête généralise la bascule 3-2 et tout ce qui vient après.

Mesuré sur la plaque percée de `formation/maillage_test.py`, en ajoutant cette
passe (l'angle dièdre médian reste à 47° dans tous les cas) :

| mailles | part sous 10° | part sous 1° | cellule la plus plate / moyenne |
|---|---|---|---|
| 28 000 | 0,59 % → **0,00 %** | 0 → 0 | \\( 5{,}5\cdot10^{-2} \\) → \\( 1{,}5\cdot10^{-1} \\) |
| 117 000 | 0,81 % → **0,02 %** | 0,009 % → **0,000 %** | \\( 2{,}9\cdot10^{-2} \\) → \\( 7{,}8\cdot10^{-2} \\) |
| 402 000 | 0,94 % → **0,11 %** | 0,022 % → **0,000 %** | \\( 7{,}3\cdot10^{-5} \\) → \\( 3{,}6\cdot10^{-3} \\) |
| 977 000 | 0,91 % → **0,18 %** | 0,028 % → **0,001 %** | \\( 4{,}0\cdot10^{-3} \\) → \\( 7{,}1\cdot10^{-4} \\) |

Le second chiffre est l'enjeu : une cellule sous 1° d'angle dièdre donne une
matrice élémentaire quasi singulière, et il n'en faut pas beaucoup pour
couler un calcul. La passe coûte environ **1,35×** le temps de maillage, à
toutes les tailles.

> **Ce que cela a demandé.** Une passe qui améliore un maillage doit être
> moins chère que sa construction, ce qui interdit deux réflexes. On ne
> **teste pas en faisant puis défaisant** : copier le maillage à chaque
> candidat coûte \\( O(n) \\) pour un échange qui coûte \\( O(1) \\), donc une
> passe qui fait \\( O(n) \\) échanges devient quadratique — mesuré, cela
> portait le maillage d'un million de mailles à 325 s au lieu de 77 s. Le
> remplacement de région décide donc **avant de muter** si les nouvelles
> cellules pavent bien la région, à partir de leurs seules faces. Et on ne
> paie pas la **recherche exhaustive** de remplissage pour une arête qu'on
> peut aussi bien laisser en place : la récupération du bord a besoin de
> complétude, une passe de qualité n'a besoin que d'une réponse.

### Quand l'enveloppe ne convient pas

Deux refus vous sont rendus, tous deux avec un lieu et une action.

**« cannot fit the envelope's edge/facet … »** — la surface ne peut pas être
retrouvée dans la triangulation. Soit elle n'admet réellement aucun maillage
sur ses nœuds, soit la récupération n'y arrive pas.

**« the mesh has N flat cell(s) that cannot be improved … »** — un sliver
dont les quatre coins sont des nœuds de l'enveloppe. Rien à insérer pour le
casser, rien à bouger puisque ses coins sont les vôtres, et le retrait
d'arête n'y arrive pas non plus. C'est devenu rare : aucune des enveloppes de
la suite de tests ne l'obtient plus. La mesure est
\\( \eta = 12\,(3V)^{2/3} / \sum \ell^2 \\), qui vaut 1 pour un tétraèdre
régulier et 0 pour un plat ; le seuil de \\( 10^{-4} \\) est calibré et non
choisi — les cellules saines restent au-dessus de \\( 4\cdot10^{-2} \\), une
cellule réellement plate tombe sous \\( 10^{-7} \\).

Dans les deux cas, la réponse est de **rediscrétiser la surface** à cet
endroit — ou de laisser le mailleur le faire.

### `allow_surface_nodes` : ce qu'on échange

```python
solide = pyrucast.mesh.triangulate_volume(peau, allow_surface_nodes=True)
```

Autorise le mailleur à **couper l'enveloppe plus fin** là où il ne peut ni la
retrouver ni la rendre utilisable.

Ce qui est conservé : la **forme**. Chaque nœud ajouté est posé sur l'arête
ou la facette qu'il divise, donc la surface reste la même surface, seule sa
triangulation s'affine. Ce qui est perdu : la **discrétisation** — la peau du
résultat ne coïncide plus avec le maillage surfacique fourni. Cela compte si
deux solides doivent partager une interface conforme ; cela ne compte pas si
l'enveloppe ne servait qu'à décrire une forme. Un avertissement sur `stderr`
indique combien de nœuds ont été ajoutés — et prévient que le maillage rendu
porte de ce fait un sous-maillage de plus.

**Et surtout, le résultat vous dit lesquels.** Un message sur `stderr` n'est
pas quelque chose sur quoi un script peut agir ; quand des nœuds ont été
posés sur l'enveloppe, le maillage rendu porte un **second sous-maillage de
`POI1`** qui les nomme. Il n'apparaît que dans ce cas — un maillage obtenu
sans rien ajouter n'a qu'un sous-maillage `TET4` :

```python
solide = pyrucast.mesh.triangulate_volume(peau, allow_surface_nodes=True)
if solide.element_types() == ["TET4", "POI1"]:
    ajoutes = solide.cell_counts()[1]
    print(f"{ajoutes} nœud(s) posé(s) sur la peau")
```

Ce sous-maillage se visualise, se soustrait, sert de support de champ comme
n'importe quel autre. Attention en revanche si vous enchaînez : les opérateurs
qui attendent un maillage volumique pur veulent le sous-maillage `TET4` seul.

C'est aussi ce qui débloque. Une arête qu'on n'arrive pas à faire rentrer n'a
autrement aucune issue ; la couper en deux scinde le problème en deux plus
faciles, et une arête suffisamment entourée de ses propres subdivisions est
récupérée par le Delaunay tout seul. Ce qui a été ajouté est ensuite **rendu**
partout où le maillage veut bien s'en séparer : sur la plaque de la formation,
15 nœuds posés, 12 repris, 3 restants.

### Pièges

**Orientation après `extrude`.** `extrude` ne vérifie pas que sa direction
est du côté de la normale de la surface source. Un `skin` de son résultat
revient donc souvent avec les normales **rentrantes**, et `triangulate_volume` le
refuse en vous renvoyant vers `invert`. C'est le cas du pipeline
`triangulate_surface` → `extrude` → `skin` ci-dessus.

**Uniquement du TRI3.** Un quadrangle n'a pas de plan unique à respecter ; il
est refusé plutôt que découpé en silence. Passez par `convert(peau, "TRI3")`.

**Nœuds confondus.** Deux nœuds distincts au même endroit déchirent la
surface ; utilisez `merge_nodes` au préalable. Le mailleur le signale
explicitement.

### Coût

Sur la plaque percée de la formation, extrudée puis pelée :

| `size` | facettes | tétraèdres | temps |
| --- | --- | --- | --- |
| 0,01 | 3 308 | 28 081 | 1,9 s |
| 0,006 | 8 032 | 117 245 | 7,0 s |
| 0,004 | 17 144 | 405 400 | 24 s |
| 0,003 | 29 604 | 984 415 | 57 s |

Le coût est essentiellement **linéaire** en nombre de mailles produites.
L'opération est interruptible : `Ctrl+C` pendant un long maillage lève
`KeyboardInterrupt` sans rien laisser derrière.

Côté Rust, `ops::mesh::triangulate_volume(envelope, size, allow_surface_nodes)`,
et `triangulate_volume_cancellable(…, cancel)` pour la forme interruptible.

## Bord d'une surface : `border`

`border(mesh, angle_deg=None)` est l'**inverse** de `triangulate_surface` : il
prend un maillage de surface (cellules `TRI3` / `QUA4`) et renvoie son **bord**
sous forme de boucles `SEG2` fermées.

Une arête de cellule utilisée par **exactement une** cellule est une arête de
bord ; les arêtes intérieures sont partagées par deux cellules (orientations
opposées) et s'annulent. Les arêtes de bord de **tous** les sous-maillages de
surface sont regroupées — la sortie `QUA4` + `TRI3` de `triangulate_surface` donne donc un
bord commun unique — puis chaînées en boucles fermées.

Le résultat est un `Mesh` avec **un sous-maillage SEG2 par boucle** : une seule
boucle pour un domaine simplement connexe, plusieurs quand le domaine a des
trous ou des morceaux disjoints. Chaque boucle garde
l'orientation CCW du bord (boucle extérieure CCW, trous CW) : le résultat peut
donc **réalimenter directement** `triangulate_surface`. Les nœuds d'origine
sont réutilisés (et re-référencés).

```python
import pyrucast

c = pyrucast.Coords(dim=2)
center = c.add_node([0.0, 0.0])
disc = pyrucast.mesh.triangulate_surface(
    pyrucast.mesh.circle(center, [0.0, 0.0, 1.0], 2.0, 16), "TRI3"
)

bord = pyrucast.mesh.border(disc)
print(len(bord))  # 1  (domaine simplement connexe)
print(bord.element_types())  # ['SEG2']
print(bord.cell_counts())  # [16]
```

### Découpe par angle (`angle_deg`)

Avec un `angle_deg`, chaque boucle est en plus **découpée en arêtes ouvertes**
à ses **coins** — le pendant 1D du découpage en faces planes de
[`skin`](#peau-dun-volume--skin). Un nœud est un coin quand le bord y **tourne**
de plus de `angle_deg` degrés (l'angle entre les directions des arêtes entrante
et sortante). Chaque arête — une suite maximale de segments quasi alignés entre
deux coins — devient son propre sous-maillage `SEG2` (un côté droit d'un carré,
même subdivisé, reste **une** arête). Une boucle sans aucun coin (bord courbé
dont tous les virages restent sous le seuil) est conservée comme une boucle
fermée. `angle_deg=None` (défaut) garde chaque bord en une boucle fermée.

```python
carre = pyrucast.mesh.triangulate_surface(contour_carre, "TRI3", 0.5)
aretes = pyrucast.mesh.border(carre, angle_deg=45.0)
print(len(aretes))  # 4  (les quatre côtés, arêtes ouvertes)
```

Les sous-maillages POI1 (un point n'a pas d'arête) sont ignorés. La fonction
lève une erreur si le maillage n'a aucune cellule de surface, s'il porte des
cellules autres que POI1/TRI3/QUA4 (les bords 1D et 3D ne sont pas gérés ici —
voir [`skin`](#peau-dun-volume--skin) pour le bord d'un volume), ou si le bord
n'est pas un ensemble propre de boucles fermées (arête ouverte ou non-manifold).

Côté Rust, `ops::mesh::border(&mesh, angle_deg)`.

## Peau d'un volume : `skin`

`skin(mesh, angle_deg=None)` est le pendant **3D** de `border` : il prend un
maillage volumique (cellules `TET4` / `PENTA6` / `HEX8`) et renvoie sa **peau**
— la surface extérieure — découpée en **faces planes**, un sous-maillage par
face.

Une facette d'élément volumique (une face de `TET4`, de `HEX8`, …) utilisée par
**exactement une** cellule est une facette de bord ; les facettes intérieures
sont partagées par deux cellules et s'annulent. Les facettes de bord de **tous**
les sous-maillages volumiques sont regroupées, puis réparties en **faces
planes** : deux facettes adjacentes (partageant une arête) appartiennent à la
même face tant qu'elles restent **quasi coplanaires** — l'angle entre leurs
normales sortantes est inférieur ou égal à `angle_deg` degrés (défaut **1°**).

Chaque groupe devient un sous-maillage `TRI3` et/ou `QUA4` (une face mêlant
triangles et quadrangles, p. ex. à une interface `TET4`/`HEX8`, en produit un de
chaque). Un cube donne ainsi six sous-maillages, un prisme cinq (deux chapeaux
triangulaires, trois flancs quadrangulaires). Les facettes conservent leur
orientation sortante ; les nœuds d'origine sont réutilisés (et re-référencés).

```python
import pyrucast

# Un pavé PENTA6 : carré triangulé, extrudé selon +z.
c = pyrucast.Coords(dim=3)
coins = [c.add_node(p) for p in [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]]
contour = pyrucast.Mesh(c, "SEG2")
for i in range(4):
    contour[0].add_cell([coins[i], coins[(i + 1) % 4]])
surf = pyrucast.mesh.triangulate_surface(contour, "TRI3", 0.34)
solide = pyrucast.mesh.extrude(surf, [0.0, 0.0, 1.0], 3)  # TRI3 -> PENTA6

peau = pyrucast.mesh.skin(solide)
print(len(peau))  # 6  (deux chapeaux + quatre flancs)
print(peau.element_types())  # ['TRI3', 'TRI3', 'QUA4', 'QUA4', 'QUA4', 'QUA4']
```

Un `angle_deg` plus grand regroupe des faces plus courbées (un cylindre facetté
devient une seule paroi) ; un `angle_deg` proche de 0 isole chaque facette. Les
sous-maillages POI1 sont ignorés. La fonction lève une erreur si le maillage n'a
aucune cellule volumique, s'il porte des cellules autres que
POI1/TET4/PENTA6/HEX8, ou si l'espace n'est pas 3D.

Côté Rust, `ops::mesh::skin(&mesh, angle_deg)`.

## Orientation des cellules : `orient` et `invert`

`orient(mesh)` (Cast3M `ORIE`) **harmonise** l'orientation des cellules d'un
maillage, et `invert(mesh)` (Cast3M `INVE`) l'**inverse**. Les deux travaillent
en **toute dimension** — segments `SEG*` (1D), faces `TRI*`/`QUA*` (2D),
volumes `TET*`/`PENTA*`/`HEX*` (3D), variantes linéaires **et quadratiques** — et
renvoient un **maillage neuf** qui reflète l'entrée sous-maillage par
sous-maillage (mêmes types, mêmes couleurs, mêmes nœuds partagés) ; l'entrée est
laissée intacte.

### Cadre unifié : facettes orientées

Le bord orienté d'une cellule est une somme signée de ses **facettes de
codimension 1** :

- `SEG*` (d = 1) : les deux nœuds extrémité — queue (`−1`) et tête (`+1`) ;
- `TRI*` / `QUA*` (d = 2) : les arêtes orientées ;
- `TET*` / `PENTA*` / `HEX*` (d = 3) : les faces orientées sortantes.

Chaque occurrence se réduit à un couple `(clé, signe)` où `clé` est la liste
**triée** des nœuds (coins) de la facette et `signe ∈ {−1, +1}` encode son
orientation par rapport à une orientation canonique de la clé. **Deux cellules
partageant une facette sont cohérentes ssi elles lui donnent des signes
opposés.** Les clés de dimensions différentes (1 / 2 / ≥ 3 nœuds) ne se
confondent jamais : un maillage mixte se sépare en composantes par dimension.

### `orient` — cohérence, pas de sens absolu

`orient` propage une orientation cohérente à travers les facettes partagées par
un parcours en largeur du graphe dual. Chaque **composante connexe** est amorcée
par sa cellule d'indice le plus bas, qui **garde** son orientation (choix
déterministe, reproductible bit à bit) ; les autres sont retournées au besoin.

`orient` **ne choisit pas** de sens absolu « sortant » : pour une surface fermée
il laisse le tout entièrement sortant *ou* entièrement rentrant selon la graine.
Pour choisir le sens (p. ex. définir l'intérieur d'un trou), composer avec
`invert`. Les facettes **non-manifold** (partagées par plus de deux cellules)
n'imposent aucune contrainte et sont ignorées.

### `invert` — retournement inconditionnel

`invert` applique à **chaque** cellule sa permutation d'inversion
(`ElementType::reversal_permutation`, la réflexion échangeant les deux premiers
axes de référence — pour `SEG*` la négation de l'axe unique). Les `POI1` (sans
orientation) sont inchangés. Appliqué deux fois, `invert` redonne le maillage de
départ.

```python
import pyrucast

# Une plaque trouée : contour extérieur + bord du trou, orientations quelconques.
surf = pyrucast.mesh.triangulate_surface(contour, "TRI3")

propre = pyrucast.mesh.orient(surf)  # toutes les mailles cohérentes
trou_dedans = pyrucast.mesh.invert(propre)  # sens inversé (intérieur/extérieur)
```

Côté Rust, `ops::mesh::orient(&mesh)` et `ops::mesh::invert(&mesh)`.

## Éléments s'appuyant sur des nœuds : `elements_on`

`elements_on(mesh, points, strict=True)` renvoie le **sous-ensemble** des
éléments de `mesh` qui **s'appuient** sur les nœuds de `points` — l'opérateur
historique `ELEM … APPUYE`. Seul l'**ensemble des nœuds référencés** par
`points` compte (typiquement un maillage de points POI1) ; ni le type ni la
connectivité de `points` n'importent.

Le critère dépend de `strict` :

- `strict=True` — on garde une cellule lorsque **tous** ses nœuds sont dans
  l'ensemble (`APPUYE STRICTEMENT`) ;
- `strict=False` — on garde une cellule dès qu'**au moins un** de ses nœuds y
  est (`APPUYE`).

Le résultat **épouse la structure** de `mesh` sous-maillage par sous-maillage
(même ordre, mêmes types d'éléments, mêmes couleurs) : chaque sous-maillage de
sortie porte les cellules retenues du sous-maillage d'entrée correspondant,
**éventuellement vide**. Les zones restent séparées (jamais de fusion). Au
besoin, `mesh.consolidate(mesh)` élimine ou fond ensuite les zones vides ou
redondantes. Les cellules retenues réutilisent les nœuds d'origine (refcount
incrémenté) ; `mesh` est laissé intact.

Les deux maillages doivent vivre sur la **même `Coords`** (un identifiant de
nœud n'a de sens qu'au sein d'une `Coords`), sinon une erreur est levée. Un
`points` vide ne retient rien.

```python
import pyrucast

c = pyrucast.Coords(dim=2)
nodes = [c.add_node(p) for p in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 0.0)]]

mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([nodes[0], nodes[1], nodes[2]])  # cellule 0
mesh.unit().add_cell([nodes[1], nodes[3], nodes[2]])  # cellule 1

# Points = {0, 1, 2} : seule la cellule 0 a tous ses nœuds dedans.
pts = pyrucast.Mesh.poi1_from_nodes([nodes[0], nodes[1], nodes[2]])

strict = pyrucast.mesh.elements_on(mesh, pts, strict=True)
print(strict.cell_count())  # 1  (cellule 0)

loose = pyrucast.mesh.elements_on(mesh, pts, strict=False)
print(loose.cell_count())  # 2  (les deux touchent un nœud de pts)
```

Côté Rust, `ops::mesh::elements_on(&mesh, &points, strict)`.

## Sélection de nœuds par région géométrique : la famille `points_*`

Pour poser une condition aux limites il faut d'abord **désigner des nœuds**.
La famille `points_*` répond à cette question par une **région géométrique** —
l'équivalent de l'opérateur historique `POIN … PLAN / DROIT / CYLI / SPHE` :

```text
points_<in|on|below>_<forme>(mesh, …géométrie…, tol=None) -> Mesh POI1
```

Toutes ces fonctions renvoient un **maillage POI1 calqué sur l'entrée** : un
sous-maillage par sous-maillage de `mesh`, dans le même ordre, **éventuellement
vide**. La sélection conserve donc le zonage de sa source — on sait de *quelle*
zone vient chaque nœud, et on peut travailler zone par zone en indexant le
résultat. `mesh.consolidate(sel)` retombe sur un nuage unique quand ce découpage ne
sert pas.

Les nœuds sont **dédoublonnés dans l'ordre de première apparition** dans la
connectivité, exactement comme [`to_poi1`](#inventaire) — une sélection totale
reproduit `to_poi1` nœud pour nœud.

### `in` et `on`

Deux familles, deux lectures :

- `points_in_*` — **dans** la région fermée, élargie de `tol` ;
- `points_on_*` — à moins de `tol` de la **surface** de la région, des deux
  côtés.

Deux formes sont **fermées par des faces planes**, et la distinction compte :
`points_on_cylinder` et `points_on_cone` ne retiennent que la **surface
latérale**, pas les disques d'extrémité — ceux-ci sont plats, c'est
`points_on_plane` qui les coupe. Le tore, lui, est une surface fermée : la
question ne se pose pas.

Le plan n'a pas de « dedans » : il a deux côtés, d'où `points_below_plane`, le
demi-espace **opposé** à la normale (plan compris). Il n'existe pas de
`points_above_plane` : retourner la normale donne l'autre moitié.

Enfin, la droite est **infinie** là où le cylindre est **borné** : pour une
sélection le long d'un axe mais limitée au segment, c'est `points_in_cylinder`
avec un petit rayon.

### La tolérance `tol`

`tol` est la **précision géométrique** du test, mesurée comme une **distance à
la surface** de la région. `tol=None` demande la valeur par défaut :
`1e-6 ×` la diagonale de la boîte englobante du maillage. C'est ce qui rend les
opérateurs **sans échelle** — le même appel marche sur une équerre en
millimètres et sur un barrage en kilomètres — et ce qui rend `points_on_plane`
utilisable sur des nœuds sortis d'un mailleur plutôt que d'une arithmétique
exacte.

Pour le cône, la distance est prise **perpendiculairement** à la surface
inclinée, et non radialement : la bande reste large de `tol` quelle que soit la
pente.

### Le cas « un seul nœud »

La requête du **nœud le plus proche** d'un point ne peut en renvoyer qu'un ;
elle n'est donc pas dans cette famille et ne renvoie pas de POI1, mais un
`Node` : c'est la méthode `mesh.nearest_node([x, y])`, des deux côtés — voir
[Opérateurs géométriques](geometrie.md).

### Repère de travail

Les nœuds sont testés dans les coordonnées où ils sont **stockés**. En
[axisymétrie](../coords.md), c'est le demi-plan méridien `(r, z)` et non le
solide de révolution : une « sphère » y est un cercle du méridien. Le tore,
qui a besoin d'un axe hors du plan pour être un tore, est **3D seulement**.

```python
import pyrucast

# Une plaque carrée maillée en TRI3.
plaque = pyrucast.mesh.triangulate_surface(contour, "TRI3", size=0.1)

# Le bord gauche (x = 0) : le plan de normale +x passant par l'origine.
gauche = pyrucast.mesh.points_on_plane(plaque, [0.0, 0.0], [1.0, 0.0])

# Les nœuds du congé : dans le disque de rayon 0.2 autour du coin rentrant.
conge = pyrucast.mesh.points_in_sphere(plaque, [1.0, 1.0], 0.2)

# La sélection sert directement de support imposé à un Dirichlet — le nuage
# POI1 est ce que `Model.dirichlet` attend (cf. Contraintes / Dirichlet).
blocage = pyrucast.Model.dirichlet("UX", "RX", gauche, pyrucast.mesh.barycenter(gauche))

# La sortie POI1 est un maillage ordinaire : elle se rebranche sur les autres
# opérateurs, ici pour remonter aux éléments portés par la sélection.
bande = pyrucast.mesh.elements_on(plaque, conge, strict=True)
```

En 3D, les formes de révolution sélectionnent alésages, arbres et gorges :

```python
# L'alésage d'un tube : la surface latérale du cylindre de rayon intérieur.
alesage = pyrucast.mesh.points_on_cylinder(tube, [0.0, 0.0, 0.0], [0.0, 0.0, 10.0], 5.0)

# Un chanfrein conique (rayon 8 en z = 0, sommet fictif en z = 8).
chanfrein = pyrucast.mesh.points_on_cone(piece, [0.0, 0.0, 0.0], [0.0, 0.0, 8.0], 8.0)

# La matière autour d'une gorge torique de rayon 1 sur un cercle de rayon 5.
gorge = pyrucast.mesh.points_in_torus(piece, [0.0, 0.0, 3.0], [0.0, 0.0, 1.0], 5.0, 1.0)
```

Côté Rust, `ops::mesh::points_on_plane(&mesh, &origin, &normal, tol)` et
consorts, avec `tol: Option<f64>`.

## Soudure des nœuds proches : `merge_nodes`

`merge_nodes(mesh, tol)` **soude** entre eux les nœuds distants de moins de
`tol` (distance euclidienne). C'est l'opération de « recollage » classique :
quand deux morceaux maillés séparément se rejoignent le long d'une interface,
leurs nœuds y sont colocalisés mais **distincts** ; `merge_nodes` les fond en
un seul, rendant le maillage topologiquement connexe.

Chaque cluster de nœuds proches est représenté par **un seul** nœud — celui de
plus petit identifiant —, et ce représentant **garde ses propres coordonnées**
(aucune moyenne : on ne déplace jamais la géométrie en douce). La connectivité
de chaque sous-maillage est réécrite pour pointer vers les représentants ; la
structure de sous-maillages (types, ordre, couleurs) est préservée.

Une cellule qui **s'effondre** — c'est-à-dire qui référence deux fois le même
représentant après soudure (un `SEG2` dont les deux bouts fusionnent, un `TRI3`
à deux coins confondus, …) — est **abandonnée** : elle est dégénérée. Les
cellules `POI1` (un seul nœud) ne s'effondrent jamais et sont toujours
conservées ; dédupliquer des points colocalisés reste le rôle de
`mesh.consolidate`, pas celui-ci.

`tol` doit être ≥ 0 ; `tol = 0` ne soude que les nœuds **exactement**
colocalisés. Seuls les nœuds **référencés** par le maillage sont concernés. Le
maillage d'entrée est laissé intact ; les nœuds soudés disparaissent de la
connectivité du résultat et deviennent récupérables par le GC de la `Coords`
une fois plus rien ne les référence.

```python
import pyrucast

# Un maillage dont l'interface porte des nœuds colocalisés mais distincts
# (deux SEG2 qui se touchent par un bout dupliqué).
c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])
b2 = c.add_node([1.0, 0.0])  # superposé à b, mais nœud distinct
d = c.add_node([2.0, 0.0])

mesh = pyrucast.Mesh(c, "SEG2")
mesh.unit().add_cell([a, b])
mesh.unit().add_cell([b2, d])

joined = pyrucast.mesh.merge_nodes(mesh, 1e-6)  # b2 est soudé sur b
```

**Bilan à l'écran.** Chaque appel imprime une ligne sur la sortie standard —
nœuds soudés, mailles supprimées, tolérance employée :

```text
merge_nodes: 12 node(s) welded, 3 cell(s) dropped, tol = 0.000001
merge_nodes (in place): 12 node(s) welded, cells untouched, tol = 0.000001
```

C'est une étape qu'on veut voir passer dans un journal de construction :
`tol` est un pari sur la géométrie, et cette ligne est ce qui dit s'il était
bon. En place, la ligne ne parle pas de mailles supprimées : il ne peut pas y
en avoir, l'opérateur refuse plutôt (voir plus bas).

> `merge_nodes` opère **au sein d'une même `Coords`** (l'invariant du `Mesh`
> impose déjà une `Coords` commune à tous les sous-maillages). Deux pièces
> maillées dans des `Coords` séparées ne se soudent donc pas : il faut d'abord
> les amener dans la même `Coords`.

### Souder sur place : `in_place=True`

Par défaut `merge_nodes` **copie** : il rend un maillage neuf et laisse ses
entrées intactes. Les maillages d'origine, eux, gardent donc leurs nœuds
dupliqués — ce qui oblige à ne plus manipuler qu'un troisième maillage, et à
recâbler tout ce qui pointait vers les deux premiers.

`merge_nodes(mesh, tol, in_place=True)` réécrit à la place la connectivité des
sous-maillages **existants** — effet de bord assumé et voulu — et renvoie le
maillage lui-même. Comme l'union `mesh_a | mesh_b` **partage** les
sous-maillages (elle ne les copie pas), souder l'union soude du même coup
`mesh_a` et `mesh_b` :

```python
gauche = pyrucast.mesh.line(a, b, 4)
droite = pyrucast.mesh.line(b2, d, 4)  # b2 colocalisé avec b, mais distinct

pyrucast.mesh.merge_nodes(gauche | droite, 1e-6, in_place=True)

# Les deux morceaux partagent maintenant réellement le nœud d'interface.
assert droite.node(0, 0, 0).id == b.id
```

Ce que la mutation ne touche pas : **la structure du maillage**. Mêmes
sous-maillages, mêmes types, même nombre de cellules dans le même ordre — seul
*quel nœud* une cellule référence change. C'est ce qui rend l'effet de bord
tenable : tout indice déjà détenu sur ces sous-maillages (numéros de cellules,
et donc les champs par élément qui s'appuient dessus) reste valide. Les
coordonnées des nœuds ne bougent pas non plus.

D'où deux refus, vérifiés **sur tout le maillage avant la moindre écriture**
(un appel rejeté ne modifie donc rien) :

- une cellule qui **s'effondrerait** est une **erreur**, là où la variante
  copiante l'abandonne : l'abandonner changerait le nombre de mailles, c'est-à-
  dire précisément l'invariant sur lequel repose l'appel sur place. Baissez
  `tol`, ou passez par la variante copiante ;
- un sous-maillage **scellé** est une erreur : un espace d'éléments finis, un
  champ ou une matrice l'a capturé et lit sa numérotation de nœuds. Soudez
  avant de les construire (ou repartez d'un `duplicate()`).

Les caches dérivés de la connectivité (index des nœuds, compagnon POI1 de
`to_poi1`) sont invalidés par la réécriture, et les refcounts suivent : chaque
emplacement réécrit incrémente son nouveau nœud et décrémente l'ancien.

Le retour est **exactement le maillage passé** — les mêmes sous-maillages,
dont l'intérieur a changé —, pas une copie : en Python, `out is mesh`. On peut
donc l'ignorer, ou chaîner dessus, au choix.

Côté Rust, c'est le **même** opérateur avec le même drapeau :
`ops::mesh::merge_nodes(&mesh, tol, in_place)` — miroir strict de la forme
Python, qui n'ajoute que la valeur par défaut. La brique de conteneur
sous-jacente est `SubMesh::remap_nodes(&map)`, un **renommage** de nœuds à
structure constante.

## Lecture d'un maillage gmsh : `read_gmsh`

`read_gmsh` importe un maillage produit par **gmsh** (fichier `.msh`,
versions **MSH 2.2** et **MSH 4.1**, en **ASCII comme en binaire** —
l'endianness est lue dans le fichier). L'appelant fournit la `Coords` dans
laquelle lire (il garde ainsi la main sur les nœuds) ; le résultat est un
`dict` Python qui associe à **chaque groupe physique** son `Mesh` :

```python
import pyrucast

coords = pyrucast.Coords(dim=2)
regions = pyrucast.mesh.read_gmsh(coords, "piece.msh")
# {'plate': Mesh<…>, 'bottom': Mesh<…>, …}  — ordre du fichier préservé

plate = regions["plate"]
print(plate.element_types())  # p.ex. ['TRI3']
print(plate.cell_count())
```

### Groupes, types et `Coords` partagée

- **Un `Mesh` par groupe physique**, et **un sous-maillage par type
  d'élément** à l'intérieur de chaque groupe. Les éléments sans groupe
  physique sont rangés sous la clé `"<ungrouped>"`.
- **Tous les `Mesh` partagent la `Coords` fournie** : un nœud à la frontière
  de deux groupes (p. ex. un nœud du bord `"bottom"` qui appartient aussi à
  la surface `"plate"`) est **le même nœud** des deux côtés — pas un
  doublon. Comme c'est *votre* `Coords`, vous gardez le handle pour poser
  des conditions aux limites sur une région nommée lue dans le fichier :

  ```python
  coords = pyrucast.Coords(dim=2)
  regions = pyrucast.mesh.read_gmsh(coords, "piece.msh")
  plate = regions["plate"]
  bottom = regions["bottom"]  # même Coords que plate

  fes = pyrucast.FiniteElementSpace(plate)
  # ... assemblage sur 'plate', blocage des nœuds de 'bottom', etc.
  ```

  La `Coords` peut déjà contenir de la géométrie : l'import s'y **ajoute**.
  Si vous égarez le handle côté Python, `mesh.coords()` le récupère depuis
  n'importe quel `Mesh` du `dict`.

### Types d'éléments reconnus

Les codes gmsh sont traduits vers les types pyrucast ; l'ordre local des
nœuds coïncide déjà avec le repère de référence, la connectivité est donc
copiée telle quelle.

| Code gmsh | Type pyrucast |
|---|---|
| `1`  | `SEG2` |
| `2`  | `TRI3` |
| `3`  | `QUA4` |
| `4`  | `TET4` |
| `5`  | `HEX8` |
| `6`  | `PENTA6` |
| `7`  | `PYRA5` |
| `15` | `POI1` |
| `8`  | `SEG3` |
| `9`  | `TRI6` |
| `16` | `QUA8` |
| `10` | `QUA9` |
| `11` | `TET10` |
| `17` | `HEX20` |
| `18` | `PENTA15` |
| `12` | `HEX27` |

Pour les types quadratiques volumiques (`TET10`, `HEX20`, `PENTA15`, `HEX27`),
gmsh numérote les nœuds de milieu d'arête (et de face pour `HEX27`) dans un
ordre différent de la convention pyrucast (VTK) : la connectivité est
**réalignée** à la lecture (même permutation que meshio). Tout autre type gmsh
(pyramide, ordre 3+…) lève une erreur explicite.

### Dimension

gmsh stocke toujours trois coordonnées par nœud ; c'est la **dimension de la
`Coords` fournie** qui décide combien sont conservées. On lit donc dans une
`Coords(dim=2)` pour aplatir un maillage planaire sur `xy`, ou dans une
`Coords(dim=3)` pour garder le relief.

> Seuls les nœuds **référencés** par un élément sont matérialisés dans la
> `Coords` ; les nœuds isolés listés mais utilisés par aucun élément sont
> ignorés.

`read_gmsh_str(coords, text)` fait la même chose à partir du **texte** du
fichier déjà chargé en mémoire (utile pour les tests ou un `.msh` reçu sur
le réseau). Côté Rust, `ops::mesh::read_gmsh(coords, path)` et
`ops::mesh::read_gmsh_str(coords, text)` renvoient un
`Vec<(String, Mesh)>` ordonné.
