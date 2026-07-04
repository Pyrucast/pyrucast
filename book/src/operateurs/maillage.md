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
| `extrude(mesh, direction, n_layers)` | extrude un maillage le long de `direction` (SEG2→QUA4, TRI3→PENTA6, QUA4→HEX8) |
| `sweep_qua4(mesh_a, mesh_b, n_layers)` | tisse des `QUA4` entre deux lignes `SEG2` |
| `sweep_solid(mesh_a, mesh_b, n_layers)` | **compagnon 3D** de `sweep_qua4` : tisse un solide entre deux surfaces (TRI3→PENTA6, QUA4→HEX8) |
| `translate(mesh, vector)` | **copie** du maillage translatée de `vector` (nœuds neufs, original intact) |
| `rotate(mesh, angle, center, axis=None)` | **copie** du maillage tournée de `angle` (rad) autour de `center` (axe `axis` en 3D) |
| `fill_surface(contour, type, …)` | triangule l'intérieur d'un contour fermé (voir plus bas) |
| `surface(contour, type, size=None)` | maille l'intérieur d'un contour par **front avançant** avec création de nœuds internes (voir plus bas) |
| `volume(envelope, size=None)` | maille l'intérieur d'une **enveloppe TRI3 fermée** en `TET4` par **Delaunay** (voir plus bas) |
| `contour(mesh)` | le **bord** d'un maillage de surface (TRI3/QUA4) en boucles `SEG2`, une par sous-maillage (voir plus bas) |
| `elements_on(mesh, points, strict=True)` | les **éléments** de `mesh` qui s'**appuient** sur les nœuds de `points` (voir plus bas) |
| `to_poi1(mesh)` | les nœuds **distincts** d'un maillage, en POI1 |
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
line = pyrucast.line_seg2(a, b, 4)
print(line)  # Mesh: 1 submesh(es), 4 cell(s) total

# Extrusion en QUA4 sur 2 couches selon +y.
surf = pyrucast.extrude(line, [0.0, 1.0], 2)
print(surf.element_types())  # ['QUA4']
```

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
face.unit().add_cell([
    c.add_node([1.0, 0.0, 0.0]),
    c.add_node([2.0, 0.0, 0.0]),
    c.add_node([1.0, 0.0, 1.0]),
])

# Copie translatée de 5 selon +z (nœuds neufs ; `face` reste intacte).
haut = pyrucast.translate(face, [0.0, 0.0, 5.0])

# Copie tournée de 30° autour de l'axe z passant par l'origine.
tournee = pyrucast.rotate(face, math.pi / 6, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
```

## Tissage d'un solide entre deux surfaces : `sweep_solid`

`sweep_solid(mesh_a, mesh_b, n_layers)` est le **compagnon 3D** de `sweep_qua4` :
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
solide = pyrucast.sweep_solid(face, tournee, 1)
print(solide.element_types())  # ['PENTA6']
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
print(surface)  # Mesh: 1 submesh(es), 2 cell(s) total
```

### Exemple Python (avec trou et raffinement)

```python
import pyrucast

c = pyrucast.Coords(dim=2)

# Contour extérieur : carré 4×4.
outer = pyrucast.Mesh(c, "SEG2")
outer_nodes = [
    c.add_node(list(p)) for p in [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
]
for i in range(4):
    outer.unit().add_cell([outer_nodes[i], outer_nodes[(i + 1) % 4]])

# Trou : carré 2×2 centré.
hole = pyrucast.Mesh(c, "SEG2")
hole_nodes = [
    c.add_node(list(p)) for p in [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]
]
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
    c.add_node(
        [5.0 * math.cos(2 * math.pi * i / 40), 5.0 * math.sin(2 * math.pi * i / 40)]
    )
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
print(quad.element_types())  # ['QUA4', 'TRI3'] en général
```

### Interruption

Un maillage trop long s'**interrompt** par `Ctrl+C` : `surface` sonde les
signaux à chaque couche/oreille et lève une `KeyboardInterrupt`. Côté Rust,
`surface_cancellable(contour, type, size, &cancel)` accepte un jeton
d'interruption (timeout, drapeau partagé…) — voir
[Interrompre une fonction](../developper/interrompre-une-fonction.md).

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

## Mailleur frontal volumique : `volume`

`volume(envelope, size=None)` est le **compagnon 3D** de `surface` : il remplit
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

1. **Points internes.** Une grille de nœuds candidats à la taille cible est
   générée à l'intérieur de l'enveloppe (aucun nœud si la taille dépasse la
   géométrie : le remplissage se fait alors avec les seuls nœuds de bord).
2. **Tétraédrisation de Delaunay.** Les nœuds de bord et les nœuds internes
   sont tétraédrisés par l'algorithme incrémental de **Bowyer–Watson**. Un
   *jitter* déterministe minuscule est appliqué au calcul de **connectivité**
   pour lever sans ambiguïté les dégénérescences cosphériques (les huit coins
   d'un cube, par exemple) ; la sortie conserve les coordonnées exactes.
3. **Découpe.** Les tétraèdres dont le centroïde tombe **hors** de l'enveloppe
   d'origine sont écartés (test d'angle solide / *winding number*). Pour une
   enveloppe convexe, tous les tétraèdres sont conservés et le remplissage pave
   exactement le domaine ; pour une enveloppe peu concave, la découpe retire le
   débord.

### Exemple Python

```python
import pyrucast

# Enveloppe : la surface d'un cube, en TRI3 fermés et orientés.
# (par ex. obtenue en assemblant des faces TRI3, ou via les mailleurs de
#  surface puis une mise en volume.)
env = construire_enveloppe_cube()  # Mesh TRI3 fermé sur un Coords 3D

# Remplissage en tétraèdres de taille ~0,5.
tet = pyrucast.volume(env, 0.5)
print(tet.element_types())  # ['TET4']
```

### Interruption

Comme `surface`, un maillage trop long s'**interrompt** par `Ctrl+C` : `volume`
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

## Bord d'une surface : `contour`

`contour(mesh)` est l'**inverse** de `fill_surface` / `surface` : il prend un
maillage de surface (cellules `TRI3` / `QUA4`) et renvoie son **bord** sous
forme de boucles `SEG2` fermées.

Une arête de cellule utilisée par **exactement une** cellule est une arête de
bord ; les arêtes intérieures sont partagées par deux cellules (orientations
opposées) et s'annulent. Les arêtes de bord de **tous** les sous-maillages de
surface sont regroupées — la sortie `QUA4` + `TRI3` de `surface` donne donc un
bord commun unique — puis chaînées en boucles fermées.

Le résultat est un `Mesh` avec **un sous-maillage SEG2 par boucle** : une seule
boucle pour un domaine simplement connexe, plusieurs quand le domaine a des
trous ou des morceaux disjoints — d'où les « n contours ». Chaque boucle garde
l'orientation CCW du bord (boucle extérieure CCW, trous CW) : le résultat peut
donc **réalimenter directement** `surface` / `fill_surface`. Les nœuds d'origine
sont réutilisés (et re-référencés).

```python
import pyrucast

c = pyrucast.Coords(dim=2)
center = c.add_node([0.0, 0.0])
disc = pyrucast.fill_surface(
    pyrucast.circle_seg2(center, [0.0, 0.0, 1.0], 2.0, 16), "TRI3"
)

bord = pyrucast.contour(disc)
print(len(bord))  # 1  (domaine simplement connexe)
print(bord.element_types())  # ['SEG2']
print(bord.cell_counts())  # [16]
```

Les sous-maillages POI1 (un point n'a pas d'arête) sont ignorés. La fonction
lève une erreur si le maillage n'a aucune cellule de surface, s'il porte des
cellules autres que POI1/TRI3/QUA4 (les bords 1D et 3D ne sont pas encore
gérés), ou si le bord n'est pas un ensemble propre de boucles fermées (arête
ouverte ou non-manifold).

Côté Rust, `ops::mesher::contour(&mesh)`.

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
pts = pyrucast.poi1_from_nodes([nodes[0], nodes[1], nodes[2]])

strict = pyrucast.elements_on(mesh, pts, strict=True)
print(strict.cell_count())  # 1  (cellule 0)

loose = pyrucast.elements_on(mesh, pts, strict=False)
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

joined = pyrucast.merge_nodes(mesh, 1e-6)  # b2 est soudé sur b
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
regions = pyrucast.read_gmsh(coords, "piece.msh")
# {'plate': Mesh<…>, 'bottom': Mesh<…>, …}  — ordre du fichier préservé

plate = regions["plate"]
print(plate.element_types())   # p.ex. ['TRI3']
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
  regions = pyrucast.read_gmsh(coords, "piece.msh")
  plate = regions["plate"]
  bottom = regions["bottom"]            # même Coords que plate

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
| `11` | `TET10` |
| `17` | `HEX20` |
| `18` | `PENTA15` |

Pour les types quadratiques volumiques (`TET10`, `HEX20`, `PENTA15`), gmsh
numérote les nœuds de milieu d'arête dans un ordre d'arêtes différent de la
convention pyrucast (VTK) : la connectivité est **réalignée** à la lecture (même
permutation que meshio). Tout autre type gmsh (Lagrange complet `QUA9`/`HEX27`,
pyramide, ordre 3+…) lève une erreur explicite.

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
