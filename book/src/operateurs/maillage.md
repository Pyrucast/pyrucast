# Opérateurs de maillage

Les **mesher** (`ops::mesher`) construisent et transforment des
[maillages](../mesh.md). Chacun prend ses conteneurs par référence et renvoie
un **nouveau** `Mesh`. Côté Python ils sont exposés à plat
(`pyrucast.line_seg2`, …).

## Inventaire

| Python | Rôle |
|---|---|
| `from_live_nodes(coords)` | un `Mesh` POI1 de **tous** les nœuds vivants d'un `Coords` |
| `poi1_from_nodes(nodes)` | un `Mesh` POI1 sur une liste de nœuds donnée |
| `line_seg2(a, b, n_elems)` | une ligne de `n_elems` `SEG2` entre deux nœuds (nœuds intermédiaires créés) |
| `circle_seg2(center, normal, radius, n_elems)` | un cercle de `SEG2` (plan défini par `normal`) |
| `extrude(mesh, direction, n_layers)` | extrude un maillage le long de `direction` (SEG2→QUA4, QUA4→HEX8, …) |
| `sweep_qua4(mesh_a, mesh_b, n_layers)` | tisse des `QUA4` entre deux lignes `SEG2` |
| `fill_surface(contour, type, …)` | triangule l'intérieur d'un contour fermé (voir plus bas) |
| `surface(contour, type, size=None)` | maille l'intérieur d'un contour par **front avançant** avec création de nœuds internes (voir plus bas) |
| `to_poi1(mesh)` | les nœuds **distincts** d'un maillage, en POI1 |
| `barycenter(mesh)` | un POI1 au **centre de gravité** de chaque cellule, structure de sous-maillage préservée |
| `consolidate(mesh)` | fusionne les sous-maillages de même type (dispatch partagé avec `NodeField`) |

`barycenter` sert notamment à fabriquer les supports de multiplicateurs des
contraintes : POI1 → nœuds **neufs** colocalisés au centre de chaque cellule
(cf. [Dirichlet](../contraintes/dirichlet.md)).

```python
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([4.0, 0.0])

# Ligne de 4 SEG2 entre a et b (3 nœuds intermédiaires créés).
line = pyrucast.line_seg2(a, b, 4)
print(line)               # Mesh: 1 submesh(es), 4 cell(s) total

# Extrusion en QUA4 sur 2 couches selon +y.
surf = pyrucast.extrude(line, [0.0, 1.0], 2)
print(surf.element_types())   # ['QUA4']
```

## Triangulation d'un contour fermé : `fill_surface`

`fill_surface(contour, element_type, max_edge_length=None, min_angle_deg=None)`
prend un `Mesh` contenant **un ou plusieurs sous-maillages SEG2** (chacun une
boucle fermée) et remplit la surface ainsi définie avec des éléments 2D. La
configuration peut être en dimension **2** (cas direct) ou **3** (boucles
quasi co-planaires — voir plus bas).

Pour l'instant un seul type cible est supporté : **`TRI3`**.

Les fondements mathématiques (aire signée, ear clipping, Newell, Delaunay /
Bowyer-Watson, CDT, Ruppert) sont rassemblés dans
[Triangulation : briques mathématiques](../triangulation.md). Cette page-ci
décrit le **comportement** de `fill_surface`.

### Cas d'un seul contour (sans trous)

Avec un unique sous-maillage SEG2, la fonction prend un chemin rapide via *ear
clipping* :

1. les segments sont chaînés dans l'ordre (un seul cycle, sinon erreur) ;
2. en 3D, la normale du plan moyen est calculée par la **méthode de Newell** ;
   les points sont projetés sur ce plan via une base orthonormée locale
   `(u, v)` ;
3. l'orientation du polygone est détectée (aire signée) ;
4. à chaque itération on retire une **oreille** (sommet convexe dont le
   triangle prev-curr-next ne contient aucun autre sommet) ;
5. on recommence jusqu'à ne plus avoir que 3 sommets.

Le résultat contient exactement `n − 2` triangles pour `n` nœuds de contour.
**Aucun nœud interne n'est créé** (pas de raffinement). Les nœuds du contour
sont réutilisés (refcount incrémenté). En 3D, les triangles vivent dans
l'espace 3D global — seule la triangulation est faite dans le plan moyen. Les
triangles produits sont orientés **CCW** dans le plan de projection, quel que
soit le sens du contour d'entrée.

### Cas avec trous (plusieurs contours)

Quand `contour` contient deux sous-maillages SEG2 ou plus, `fill_surface`
bascule sur une **triangulation de Delaunay contrainte (CDT)** maison :

1. chaque sous-maillage est traité comme une boucle fermée indépendante ;
2. la boucle d'aire absolue la plus grande est automatiquement désignée
   **contour extérieur** ; les autres deviennent des **trous** ;
3. les points sont insérés un à un par Bowyer-Watson (avec un super-triangle
   englobant) ;
4. chaque arête de boucle est ensuite **forcée** dans la triangulation ;
5. un *flood-fill* par parité depuis le super-triangle retire l'extérieur et
   l'intérieur des trous.

L'orientation des boucles d'entrée n'a pas d'importance (détection par aire
absolue). Aucun nœud interne n'est créé sans raffinement. L'algorithme
géométrique est exposé via `triangulate_polygon_with_holes(outer, holes)` pour
les besoins indépendants du système `Mesh`.

### Contrôle de planéité (cas 3D)

En 3D, la déviation maximale d'un nœud du contour au plan moyen doit rester
inférieure à `1e-6 × diag` (`diag` = diagonale de la boîte englobante). Au-delà,
`fill_surface` retourne une erreur claire indiquant la déviation observée et la
tolérance. Ce seuil relatif tolère le bruit numérique tout en refusant les
vrais contours gauches.

### Raffinement — points Steiner

Quand `max_edge_length` ou `min_angle_deg` est renseigné, l'algorithme bascule
sur la CDT avec **raffinement de Ruppert** :

- `max_edge_length` — longueur d'arête maximale tolérée ;
- `min_angle_deg` — angle minimum garanti (en degrés).

La convergence n'est théoriquement garantie que pour `min_angle_deg ≤ 20.7°`
(Shewchuk). pyrucast plafonne le nombre d'insertions à `50 · n_contour + 1000`
pour éviter les divergences ; l'erreur est explicite si la limite est atteinte.
Les nouveaux nœuds (« Steiner ») sont créés dans la `Coords` du contour,
exactement comme les nœuds utilisateur.

### Limitations actuelles

- seul `TRI3` est supporté en sortie ;
- en 3D, l'algorithme refuse les contours franchement non plans ;
- les boucles doivent être deux à deux disjointes (pas de trous emboîtés, pas
  de croisements) ;
- la fonction de taille reste **globale** (un seul `max_edge_length`) — une
  carte de taille variable `h(x, y)` viendra plus tard.

### Exemple Python (2D)

```python
import pyrucast

c = pyrucast.Coords(dim=2)
nodes = [c.add_node(p) for p in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]

contour = pyrucast.Mesh(c, "SEG2")
for i in range(4):
    contour.unit().add_cell([nodes[i], nodes[(i + 1) % 4]])

surface = pyrucast.fill_surface(contour, "TRI3")
print(surface)        # Mesh: 1 submesh(es), 2 cell(s) total
```

### Exemple Python (avec trou et raffinement)

```python
import pyrucast

c = pyrucast.Coords(dim=2)

# Contour extérieur : carré 4×4.
outer = pyrucast.Mesh(c, "SEG2")
outer_nodes = [c.add_node(list(p)) for p in [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]]
for i in range(4):
    outer.unit().add_cell([outer_nodes[i], outer_nodes[(i + 1) % 4]])

# Trou : carré 2×2 centré.
hole = pyrucast.Mesh(c, "SEG2")
hole_nodes = [c.add_node(list(p)) for p in [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]]
for i in range(4):
    hole.unit().add_cell([hole_nodes[i], hole_nodes[(i + 1) % 4]])

# Composer les deux contours par l'union | (jamais +).
combined = outer | hole

# Sans raffinement : 6 triangles « bruts ».
brut = pyrucast.fill_surface(combined, "TRI3")

# Avec raffinement : arête max 1.0 + angle min 20° → maillage fin et de qualité.
fin = pyrucast.fill_surface(combined, "TRI3", max_edge_length=1.0, min_angle_deg=20.0)
# Aire triangulée = 16 - 4 = 12, mais bien plus de cellules.
```

Côté Rust, `ops::mesher::fill_surface(&combined, ElementType::TRI3, Some(opts))`
prend un `Option<RefinementOptions>` (`max_edge_length`, `min_angle_deg`). Le module
`pyrucast::ops::mesher::triangulation` regroupe les briques géométriques
(`signed_area`, `ear_clip_2d`, `newell_normal`, `in_plane_basis`,
`delaunay_2d`, `constrained_delaunay_2d`, `triangulate_polygon_with_holes`) —
toutes opèrent sur des tableaux bruts, réutilisables indépendamment du système
`Mesh` (voir [Triangulation](../triangulation.md)).

## Mailleur frontal : `surface`

`surface(contour, element_type, size=None)` remplit l'intérieur d'un contour
fermé par un **front avançant** qui **crée des nœuds internes** pour respecter
une taille de maille cible — là où `fill_surface` ne triangule que les nœuds
du bord. C'est l'analogue de l'opérateur `SURF` historique.

`element_type` vaut `"TRI3"` ou `"QUA4"` ; `size` fixe la longueur d'arête
visée (par défaut : longueur moyenne des segments du contour). Le contour peut
être en 2D, ou une boucle **quasi planaire** en 3D (projetée sur son plan de
meilleure approximation, maillée, puis relevée — même contrôle de planéité que
`fill_surface`).

### Méthode

1. **Épluchage des coins.** On retire itérativement le coin convexe le plus
   aigu du front sous forme de triangle (« oreille ») ; une arête de fermeture
   plus longue que `1.5 × size` est **bissectée** par un nœud neuf (deux
   triangles) pour ne pas créer d'élément trop grand.
2. **Couche frontale.** Quand aucun coin n'est assez aigu, tout le front est
   **décalé vers l'intérieur** d'environ une taille de maille (bissectrice
   intérieure), et une bande d'éléments est pavée entre le front et son
   décalé ; le décalé devient le nouveau front et le procédé récurse. Une
   cellule de bande est un quadrangle en mode QUA4, deux triangles en TRI3.
3. **Fermeture par éventail.** Lorsque le front convexe s'est réduit à un
   point, on le ferme par un éventail autour de son centroïde : triangles en
   TRI3, éventail de quadrangles (avec au plus un triangle résiduel si le
   nombre de nœuds est impair) en QUA4.

Comme l'opérateur d'origine, un maillage **QUA4 est *quad-dominant*** : le
résultat peut porter à la fois un sous-maillage `QUA4` et un sous-maillage
`TRI3` (coins aigus, repli concave, éventail de fermeture). Les éléments sont
orientés **CCW**.

> **`surface` vs `fill_surface`.** `fill_surface` triangule les nœuds donnés
> (ear clipping / Delaunay contraint, raffinement optionnel) et gère
> nativement les **trous**. `surface` crée des nœuds internes pour une taille
> contrôlée façon front avançant. Choisir `surface` pour un maillage de taille
> homogène imposée ; `fill_surface` pour trianguler un contour (avec trous) ou
> raffiner par critère d'arête/angle.

### Exemple Python

```python
import pyrucast, math

c = pyrucast.Coords(dim=2)
# Contour : cercle discrétisé (rayon 5, 40 segments).
nodes = [
    c.add_node([5.0 * math.cos(2*math.pi*i/40), 5.0 * math.sin(2*math.pi*i/40)])
    for i in range(40)
]
contour = pyrucast.Mesh(c, "SEG2")
sm = contour[0]
for i in range(40):
    sm.add_cell([nodes[i], nodes[(i + 1) % 40]])

# Maillage frontal en triangles de taille ~1.0 (nœuds internes créés).
tri = pyrucast.surface(contour, "TRI3", 1.0)
print(tri.element_types(), tri.cell_count())

# Variante quad-dominante.
quad = pyrucast.surface(contour, "QUA4", 1.0)
print(quad.element_types())   # ['QUA4', 'TRI3'] en général
```

### Limitations actuelles

- **Un seul contour** (pas de trous) : pour mailler un domaine troué, utiliser
  `fill_surface` (CDT) en attendant la séparation/fusion de contours.
- **Taille uniforme** : le champ de densité par nœud (un `size` variable, façon
  `CHPO1`) n'est pas encore exposé.
- Le pavage en couches utilise un décalage 1:1 sans rééchantillonnage
  circonférentiel : la qualité près du centre est correcte mais perfectible.

Côté Rust, `ops::mesher::surface(&contour, ElementType::TRI3, Some(1.0))`. Le
cœur géométrique (`pave_single`) opère sur de simples `Vec<Point2>` sans
toucher au store — frontière nette pour un futur parallélisme intra-opérateur.
