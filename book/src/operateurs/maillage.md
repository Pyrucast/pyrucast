# Opérateurs de maillage

Les **mesher** (`ops::mesher`) construisent et transforment des
[maillages](../mesh.md). Chacun prend ses conteneurs par référence et renvoie
un **nouveau** `Mesh`. Côté Python ils sont exposés à plat
(`pyrucast.mesher.line`, …).

## Inventaire

| Python | Rôle |
|---|---|
| `from_live_nodes(coords)` | un `Mesh` POI1 de **tous** les nœuds vivants d'un `Coords` |
| `poi1_from_nodes(nodes)` | un `Mesh` POI1 sur une liste de nœuds donnée |
| `line(a, b, n_elems, element_type="SEG2")` | une ligne de `n_elems` éléments (`SEG2` ou `SEG3`) entre deux nœuds (nœuds intermédiaires créés) |
| `circle(center, normal, radius, n_elems, element_type="SEG2")` | un cercle fermé (`SEG2` ou `SEG3`, plan défini par `normal`) |
| `arc(a, center, b, n_elems, element_type="SEG2")` | un arc de `a` à `b` sur le cercle de centre `center` passant par les deux (le plus court des deux arcs) |
| `extrude(mesh, direction, n_layers)` | extrude un maillage le long de `direction` (SEG2→QUA4, TRI3→PENTA6, QUA4→HEX8) |
| `sweep(mesh_a, mesh_b, n_layers, element_type="QUA4")` | tisse `QUA4`/`TRI3`/`QUA8`/`QUA9`/`TRI6` entre deux lignes `SEG2` (un `QUA4` est toujours construit d'abord, puis converti) |
| `transfinite(side1, side2, side3, side4, element_type="QUA4")` | **généralisation de `sweep` à 4 côtés** (l'équivalent Cast3M de `DALL`) : interpolation transfinie (patch de Coons) entre quatre lignes `SEG2` formant un contour fermé (voir plus bas) |
| `sweep_solid(mesh_a, mesh_b, n_layers)` | **compagnon 3D** de `sweep` : tisse un solide entre deux surfaces (TRI3→PENTA6, QUA4→HEX8) |
| `translate(mesh, vector)` | **copie** du maillage translatée de `vector` (nœuds neufs, original intact) |
| `rotate(mesh, angle, center, axis=None)` | **copie** du maillage tournée de `angle` (rad) autour de `center` (axe `axis` en 3D) |
| `triangulate_surface(contour, type, size=None)` | maille l'intérieur de contours **orientés** (CCW extérieur, CW trous) par **Delaunay contraint + raffinement Ruppert** (voir plus bas) |
| `pave_surface(contour, type, size=None, all_quad=False)` | **pave** l'intérieur des mêmes contours **orientés** en `QUA4`/`QUA8`/`QUA9`, par **front avançant** en rangées parallèles au bord (voir plus bas) |
| `triangulate_volume(envelope, size=None, allow_surface_nodes=False)` | **compagnon 3D** de `triangulate_surface` : maille l'intérieur d'une **enveloppe TRI3 fermée** en `TET4` — Delaunay exact, récupération du bord, raffinement intérieur et chasse aux slivers (voir plus bas) |
| `border(mesh, angle_deg=None)` | le **bord** d'un maillage de surface (TRI3/QUA4) en boucles `SEG2` (une par sous-maillage) ; avec `angle_deg`, découpé en **arêtes** ouvertes aux coins (voir plus bas) |
| `skin(mesh, angle_deg=None)` | la **peau** d'un maillage volumique (TET4/PENTA6/HEX8) en faces `TRI3`/`QUA4`, **une par face plane** du solide (voir plus bas) |
| `orient(mesh)` | **harmonise** l'orientation des cellules (normales cohérentes), toute dimension (SEG/TRI/QUA/TET/PENTA/HEX), équivalent Cast3M `ORIE` (voir plus bas) |
| `invert(mesh)` | **inverse** l'orientation de toutes les cellules, toute dimension, équivalent Cast3M `INVE` (voir plus bas) |
| `elements_on(mesh, points, strict=True)` | les **éléments** de `mesh` qui s'**appuient** sur les nœuds de `points` (voir plus bas) |
| `to_poi1(mesh)` | les nœuds **distincts** d'un maillage, en POI1 ; nuage **canonique mis en cache** par sous-maillage (scellé) ⇒ handle reproductible, partagé par `restrict`/blocs de matrice/`divergence`/`flux` (supports appariables) |
| `to_quadratic(mesh)` | la **copie quadratique** (Lagrange-2) d'un maillage linéaire : TRI3→TRI6, HEX8→HEX20, … (voir plus bas) |
| `convert(mesh, element_type)` | **change le type d'élément** sans déplacer ni ajouter de nœud : identité, `QUA4`→`TRI3` (2 triangles), `HEX8`→`TET4` (6 tétraèdres) (voir plus bas) |
| `barycenter(mesh)` | un POI1 au **centre de gravité** de chaque cellule, structure de sous-maillage préservée |
| `consolidate(mesh)` | fusionne les sous-maillages de même type (dispatch partagé avec `NodeField`) |
| `merge_nodes(mesh, tol)` | **soude** les nœuds distants de moins de `tol` ; remappe la connectivité, abandonne les cellules dégénérées (voir plus bas) |
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
line = pyrucast.mesher.line(a, b, 4)
print(line)  # Mesh: 1 submesh(es), 4 cell(s) total

# Extrusion en QUA4 sur 2 couches selon +y.
surf = pyrucast.mesher.extrude(line, [0.0, 1.0], 2)
print(surf.element_types())  # ['QUA4']

# Ligne quadratique : SEG3 (nœud de milieu d'arête par élément).
line3 = pyrucast.mesher.line(a, b, 4, "SEG3")
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
tri = pyrucast.mesher.sweep(mesh_a, mesh_b, 2, "TRI3")  # 2× plus de cellules que QUA4
qua8 = pyrucast.mesher.sweep(mesh_a, mesh_b, 2, "QUA8")
qua9 = pyrucast.mesher.sweep(mesh_a, mesh_b, 2, "QUA9")
tri6 = pyrucast.mesher.sweep(mesh_a, mesh_b, 2, "TRI6")
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

side1 = pyrucast.mesher.line(p0, p1, 4)  # bas,   4 éléments
side2 = pyrucast.mesher.line(p1, p2, 2)  # droite, 2 éléments
side3 = pyrucast.mesher.line(p2, p3, 4)  # haut,  4 éléments (= side1)
side4 = pyrucast.mesher.line(p3, p0, 2)  # gauche, 2 éléments (= side2)

surf = pyrucast.mesher.transfinite(side1, side2, side3, side4)
print(surf.element_types(), surf.cell_count())  # ['QUA4'] 8
```

> **Différence avec Cast3M.** `DALL` accepte des côtés opposés avec un
> nombre de points **différent** (algorithme de pavage plus général,
> documenté mais non détaillé dans la notice officielle). `transfinite`
> se limite au cas standard de l'interpolation transfinie — côtés opposés
> de **même** nombre d'éléments — largement suffisant en pratique et
> implémentable simplement.

## Copies rigides : `translate` et `rotate`

`translate(mesh, vector)` et `rotate(mesh, angle, center, axis=None)` renvoient
une **copie neuve** du maillage — mêmes sous-maillages, mêmes types, mêmes
couleurs, même connectivité — dont **tous les nœuds sont nouveaux**. Le maillage
d'origine (et ses nœuds) reste intact ; un nœud partagé entre plusieurs cellules
de la source reste partagé dans la copie.

- `translate` décale chaque nœud de `vector` (dont la longueur doit valoir la
  dimension du maillage).
- `rotate` tourne de `angle` **radians** autour de `center`. En **2D**, `center`
  est un point et `axis` est ignoré ; en **3D**, la rotation se fait autour de la
  droite passant par `center` dirigée par `axis` (formule de Rodrigues,
  main droite), et `axis` est obligatoire (il n'a pas besoin d'être normé).

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
haut = pyrucast.mesher.translate(face, [0.0, 0.0, 5.0])

# Copie tournée de 30° autour de l'axe z passant par l'origine.
tournee = pyrucast.mesher.rotate(face, math.pi / 6, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
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
solide = pyrucast.mesher.sweep_solid(face, tournee, 1)
print(solide.element_types())  # ['PENTA6']
```

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
lin = pyrucast.mesher.triangulate_surface(contour, "TRI3", 1.0)  # maillage TRI3
quad = pyrucast.mesher.to_quadratic(lin)  # copie TRI6
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
faces = pyrucast.mesher.skin(volume)  # peau en QUA4
faces = pyrucast.mesher.convert(faces, "TRI3")  # QUA4 → TRI3
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
pour un bord plus fin, discrétisez le contour en amont (`mesher.line(a, b,
15)`, `mesher.arc(...)`, `mesher.circle(...)`).

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
tri = pyrucast.mesher.triangulate_surface(combined, "TRI3", size=0.5)
print(tri.element_types(), tri.cell_count())

# Variante quad-dominante.
quad = pyrucast.mesher.triangulate_surface(combined, "QUA4", size=0.5)
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

Côté Rust, `ops::mesher::triangulate_surface(&contour, ElementType::TRI3, Some(0.5))`.
Le cœur (CDT + raffinement) opère sur de simples `Vec<Point2>` sans toucher au
store ; lissage et recombinaison QUA4 sont parallélisés (`rayon`). Le module
`pyrucast::ops::mesher::triangulation` regroupe par ailleurs les briques
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
(`pyrucast.consolidate`), orientées par l'appelant — **CCW** pour une frontière
extérieure, **CW** pour un trou. Plusieurs boucles CCW disjointes pavent
plusieurs domaines indépendants en une passe. La configuration peut être en
dimension **2**, ou une boucle **plane en 3D** (ajustée à son plan de meilleur
approximation par la méthode de Newell, pavée dans ce plan, puis relevée).

**Le contour est figé** : les nœuds d'entrée sont réutilisés tels quels (mêmes
identifiants, mêmes positions) et ne sont jamais déplacés — la seule exception
est décrite sous *Le tout-quadrangle* ci-dessous.

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
6. **Lissage** sous garde de validité, qui ne déplace jamais un nœud du
   contour.

Toutes les décisions topologiques — convexité, croisement de segments —
passent par le prédicat **exact** `orient2d` (technique de Shewchuk, partagé
avec le mailleur volumique). Ce ne sont donc pas des estimations.

### Le tout-quadrangle

Un polygone à nombre **pair** de côtés se remplit toujours de quadrangles
seuls ; un polygone impair laisse toujours **exactement un** triangle. Or le
pavage ne peut pas changer cette parité — une rangée la conserve, une couture
retire deux nœuds. **Elle est donc décidée par le contour, avant que le
maillage ne commence.**

D'où le paramètre :

- `all_quad=False` (défaut) — les quelques triangles résiduels reviennent dans
  un sous-maillage `TRI3` séparé ;
- `all_quad=True` — toute boucle de bord à nombre impair de segments reçoit
  **un** nœud supplémentaire, au milieu de son plus long segment. C'est le prix
  minimal, et il n'y a pas d'alternative : découper une arête plus tard
  laisserait un nœud en T.

### Exemple Python

```python
import pyrucast as pc

coords = pc.Coords(2)
# … contour extérieur CCW et cercle-trou CW, consolidés en une boucle chacun.
plaque = pc.mesher.pave_surface(contour, "QUA4", size=0.002, all_quad=True)
print(plaque.element_types())  # ['QUA4'] — aucun triangle

# Le solide prismatique vient alors gratuitement, et en hexaèdres purs.
volume = pc.mesher.extrude(plaque, [0, 0.02, 0], 2)
print(volume.element_types())  # ['HEX8']
```

### Interruption

Le pavage interroge les signaux Python entre deux rangées : `Ctrl+C` pendant
un maillage long lève `KeyboardInterrupt`. Côté Rust, la forme
`pave_surface_cancellable(..., cancel)` prend un jeton `Cancel`.

### Coût

Plaque trouée de 30 × 10 cm percée d'un trou de rayon 3,5 cm, taille de maille
0,29 mm, en `--release` :

| mailles | temps | débit | quadrangles | mailles inversées |
|---|---|---|---|---|
| 209 167 | 1,39 s | 150 000 /s | 100,0 % | 0 |

1,2 % des mailles ont un jacobien normalisé inférieur à 0,5. Le coût est
essentiellement linéaire : le front croît comme la racine du nombre de mailles,
et l'index spatial est reconstruit à chaque rangée pour ce prix-là.

### Pièges

- **Une boucle par sous-maillage.** Comme pour `triangulate_surface`, une
  boucle fermée doit tenir dans un seul sous-maillage : `pyrucast.consolidate`
  après avoir uni les côtés.
- **Orientation.** Un trou doit être **CW**. `pyrucast.mesher.invert` retourne
  un cercle construit en CCW.
- **La taille du contour compte.** Le front part de la discrétisation du bord
  et converge vers `size` en quelques rangées. Un contour beaucoup plus
  grossier que `size` donne donc des premières rangées plus grosses que
  demandé.

### Limitations actuelles

- Pas encore de nettoyage topologique (résorption des doublets, valences
  ramenées vers 4) : la qualité du pire élément reste inférieure à ce qu'un
  paveur mûr obtient, même si aucune maille n'est inversée.
- Sur un contour très grossier, deux parties du front peuvent finir par
  s'effleurer et laisser une boucle d'aire négative. Elle est **abandonnée**
  plutôt que remplie de mailles inintégrables, ce qui coûte un éclat d'aire.
  Le paveur peut donc dégrader ; il ne peut pas rendre un maillage invalide.
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

## Mailleur volumique : `triangulate_volume`

```python
solide = pyrucast.mesher.triangulate_volume(
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
peau = pyrucast.mesher.convert(pyrucast.mesher.skin(solide_penta6), "TRI3")
peau = pyrucast.mesher.invert(peau)  # voir « pièges », plus bas
volume = pyrucast.mesher.triangulate_volume(peau, size=0.01)
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
solide = pyrucast.mesher.triangulate_volume(peau, allow_surface_nodes=True)
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
solide = pyrucast.mesher.triangulate_volume(peau, allow_surface_nodes=True)
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

Côté Rust, `ops::mesher::triangulate_volume(envelope, size, allow_surface_nodes)`,
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
disc = pyrucast.mesher.triangulate_surface(
    pyrucast.mesher.circle(center, [0.0, 0.0, 1.0], 2.0, 16), "TRI3"
)

bord = pyrucast.mesher.border(disc)
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
carre = pyrucast.mesher.triangulate_surface(contour_carre, "TRI3", 0.5)
aretes = pyrucast.mesher.border(carre, angle_deg=45.0)
print(len(aretes))  # 4  (les quatre côtés, arêtes ouvertes)
```

Les sous-maillages POI1 (un point n'a pas d'arête) sont ignorés. La fonction
lève une erreur si le maillage n'a aucune cellule de surface, s'il porte des
cellules autres que POI1/TRI3/QUA4 (les bords 1D et 3D ne sont pas gérés ici —
voir [`skin`](#peau-dun-volume--skin) pour le bord d'un volume), ou si le bord
n'est pas un ensemble propre de boucles fermées (arête ouverte ou non-manifold).

Côté Rust, `ops::mesher::border(&mesh, angle_deg)`.

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
surf = pyrucast.mesher.triangulate_surface(contour, "TRI3", 0.34)
solide = pyrucast.mesher.extrude(surf, [0.0, 0.0, 1.0], 3)  # TRI3 -> PENTA6

peau = pyrucast.mesher.skin(solide)
print(len(peau))  # 6  (deux chapeaux + quatre flancs)
print(peau.element_types())  # ['TRI3', 'TRI3', 'QUA4', 'QUA4', 'QUA4', 'QUA4']
```

Un `angle_deg` plus grand regroupe des faces plus courbées (un cylindre facetté
devient une seule paroi) ; un `angle_deg` proche de 0 isole chaque facette. Les
sous-maillages POI1 sont ignorés. La fonction lève une erreur si le maillage n'a
aucune cellule volumique, s'il porte des cellules autres que
POI1/TET4/PENTA6/HEX8, ou si l'espace n'est pas 3D.

Côté Rust, `ops::mesher::skin(&mesh, angle_deg)`.

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
surf = pyrucast.mesher.triangulate_surface(contour, "TRI3")

propre = pyrucast.mesher.orient(surf)  # toutes les mailles cohérentes
trou_dedans = pyrucast.mesher.invert(propre)  # sens inversé (intérieur/extérieur)
```

Côté Rust, `ops::mesher::orient(&mesh)` et `ops::mesher::invert(&mesh)`.

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
besoin, `consolidate(mesh)` élimine ou fond ensuite les zones vides ou
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
pts = pyrucast.mesher.poi1_from_nodes([nodes[0], nodes[1], nodes[2]])

strict = pyrucast.mesher.elements_on(mesh, pts, strict=True)
print(strict.cell_count())  # 1  (cellule 0)

loose = pyrucast.mesher.elements_on(mesh, pts, strict=False)
print(loose.cell_count())  # 2  (les deux touchent un nœud de pts)
```

Côté Rust, `ops::mesher::elements_on(&mesh, &points, strict)`.

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
`consolidate`, pas celui-ci.

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

joined = pyrucast.mesher.merge_nodes(mesh, 1e-6)  # b2 est soudé sur b
```

> `merge_nodes` opère **au sein d'une même `Coords`** (l'invariant du `Mesh`
> impose déjà une `Coords` commune à tous les sous-maillages). Deux pièces
> maillées dans des `Coords` séparées ne se soudent donc pas : il faut d'abord
> les amener dans la même `Coords`.

Côté Rust, `ops::mesher::merge_nodes(&mesh, tol)`.

## Lecture d'un maillage gmsh : `read_gmsh`

`read_gmsh` importe un maillage produit par **gmsh** (fichier `.msh`,
versions **MSH 2.2** et **MSH 4.1**, en **ASCII comme en binaire** —
l'endianness est lue dans le fichier). L'appelant fournit la `Coords` dans
laquelle lire (il garde ainsi la main sur les nœuds) ; le résultat est un
`dict` Python qui associe à **chaque groupe physique** son `Mesh` :

```python
import pyrucast

coords = pyrucast.Coords(dim=2)
regions = pyrucast.mesher.read_gmsh(coords, "piece.msh")
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
  regions = pyrucast.mesher.read_gmsh(coords, "piece.msh")
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
le réseau). Côté Rust, `ops::mesher::read_gmsh(coords, path)` et
`ops::mesher::read_gmsh_str(coords, text)` renvoient un
`Vec<(String, Mesh)>` ordonné.
