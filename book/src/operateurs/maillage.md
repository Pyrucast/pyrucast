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
| `volume(envelope, size=None)` | maille l'intérieur d'une **enveloppe TRI3 fermée** en `TET4` par **Delaunay** (voir plus bas) |
| `border(mesh)` | le **bord** d'un maillage de surface (TRI3/QUA4) en boucles `SEG2`, une par sous-maillage (voir plus bas) |
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

## Mailleur volumique : `volume`

`volume(envelope, size=None)` est le **compagnon 3D** de `triangulate_surface` : il remplit
l'intérieur d'une **enveloppe surfacique fermée** avec des tétraèdres `TET4`, à
taille de maille imposée, en créant les nœuds internes nécessaires. Le
remplissage d'un intérieur vide est précisément ce que fait un opérateur de
volume ; il s'appuie ici sur un cœur **Delaunay** robuste.

L'`envelope` doit être une **surface `TRI3` fermée et orientée de façon
cohérente** (un ou plusieurs sous-maillages, tous `TRI3`) sur un `Coords` **3D**.
`size` fixe la longueur d'arête cible ; `None` prend la longueur moyenne des
arêtes de l'enveloppe. Le résultat est un `Mesh` à un sous-maillage `TET4`
(orientation à volume signé positif — convention `TET4`) ; les nœuds de bord
sont **réutilisés**, les nœuds internes créés dans le même `Coords`.

### Méthode

Le remplissage est le pendant 3D du pipeline 2D de `triangulate_surface` :
Delaunay du bord, puis **récupération contrainte de la peau** et **excavation
par flood-fill** (au lieu d'une découpe par centroïde, qui laissait des
tétraèdres traverser les concavités et les trous).

1. **Delaunay du bord.** Les seuls nœuds de bord sont tétraédrisés par
   l'algorithme incrémental de **Bowyer–Watson**. Un *jitter* déterministe
   minuscule est appliqué au calcul de **connectivité** pour lever sans
   ambiguïté les dégénérescences cosphériques (les huit coins d'un cube, par
   exemple) ; la sortie conserve les coordonnées exactes.
2. **Récupération de la peau.** Chaque face de peau absente du Delaunay (une
   diagonale de quad prise « à l'envers », une arête de concavité) est
   récupérée en re-tétraédrisant le petit corridor qu'elle traverse, puis
   **marquée contrainte**. Toute face de peau qui resterait irrécupérable →
   **erreur claire** (polyèdre de type Schönhardt), jamais un maillage faux.
3. **Points internes.** Une grille de nœuds candidats à la taille cible est
   insérée *dans le maillage déjà contraint* (aucun nœud si la taille dépasse
   la géométrie), la cavité de Bowyer–Watson étant **coupée aux faces
   contraintes** : un nœud interne ne peut jamais raboter une face de peau.
4. **Excavation.** Un *flood-fill* depuis un tétraèdre intérieur garde
   exactement ce que la peau enferme, sans jamais traverser la surface : les
   concavités, les trous et l'autre côté d'une pièce mince sont creusés
   exactement.

> **Portée.** Les enveloppes convexes ou peu concaves sont maillées
> directement. Les surfaces fortement non convexes riches en **arêtes réflexes**
> (le pourtour d'un trou facetté, par exemple) peuvent encore déclencher
> l'erreur de récupération : une récupération de bord 3D complète (prédicats
> exacts) reste à faire.

### Exemple Python

```python
import pyrucast

# Enveloppe : la surface d'un cube, en TRI3 fermés et orientés.
# (par ex. obtenue en assemblant des faces TRI3, ou via les mailleurs de
#  surface puis une mise en volume.)
env = construire_enveloppe_cube()  # Mesh TRI3 fermé sur un Coords 3D

# Remplissage en tétraèdres de taille ~0,5.
tet = pyrucast.mesher.volume(env, 0.5)
print(tet.element_types())  # ['TET4']
```

### Interruption

Comme `triangulate_surface`, un maillage trop long s'**interrompt** par `Ctrl+C` : `volume`
sonde les signaux pendant la génération des points et l'insertion Delaunay, et
lève une `KeyboardInterrupt`. Côté Rust,
`volume_cancellable(envelope, size, &cancel)` accepte un jeton
d'interruption (timeout, drapeau partagé…) — voir
[Interrompre une fonction](../developper/interrompre-une-fonction.md).

### Limitations actuelles

- **Entrée `TRI3` seulement** : les enveloppes `QUA4` (découpe en triangles) ne
  sont pas encore acceptées.
- **Convexe ou peu concave** : la découpe se fait au centroïde, sans
  *recouvrement de frontière* (boundary recovery). Pour une enveloppe convexe le
  pavage est exact ; sur une **forte concavité**, le bord Delaunay s'écarte de
  la surface d'entrée (le creux peut être comblé) — les cavités internes et les
  surfaces de genre > 0 ne sont pas non plus garanties.
- **Taille uniforme** : pas encore de champ de densité variable.
- **Sortie `TET4`** : le remplissage hexaédrique (`HEX8`) n'est pas couvert.
- Le maillage des nœuds **de bord** suit la triangulation de Delaunay : les
  faces externes des `TET4` peuvent diviser les facettes d'entrée selon d'autres
  diagonales (même surface, autre découpe).

Côté Rust, `ops::mesher::volume(&envelope, Some(0.5))`. Le cœur géométrique
(`pave_volume`) opère sur de simples `Vec<Point3>` sans toucher au store —
frontière nette pour un futur parallélisme intra-opérateur.

## Bord d'une surface : `border`

`border(mesh)` est l'**inverse** de `triangulate_surface` : il prend un
maillage de surface (cellules `TRI3` / `QUA4`) et renvoie son **bord** sous
forme de boucles `SEG2` fermées.

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

Les sous-maillages POI1 (un point n'a pas d'arête) sont ignorés. La fonction
lève une erreur si le maillage n'a aucune cellule de surface, s'il porte des
cellules autres que POI1/TRI3/QUA4 (les bords 1D et 3D ne sont pas gérés ici —
voir [`skin`](#peau-dun-volume--skin) pour le bord d'un volume), ou si le bord
n'est pas un ensemble propre de boucles fermées (arête ouverte ou non-manifold).

Côté Rust, `ops::mesher::border(&mesh)`.

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
