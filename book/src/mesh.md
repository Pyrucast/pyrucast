# Maillage

Le maillage de pyrucast se compose à deux niveaux :

- **`SubMesh`** : regroupe toutes les cellules d'un même `ElementType`. Stocke la connectivité à plat (un `Vec<NodeId>` de longueur `cell_count × nodes_per_cell`).
- **`Mesh`** : agrège plusieurs `SubMesh` liés à la même `Configuration`.

Le cas POI1 est volontairement dégénéré : un sous-maillage POI1 est exactement une liste de nœuds, ce qui sert de support naturel aux `NodeField` (à venir).

## Types d'éléments

L'enum `ElementType` liste les types supportés ; chaque variante porte sa propre méta-information (nombre de nœuds, dimension topologique, nom court cast3m).

| Variante | Nœuds | Dim. topo. | Cas usuel |
|---|---:|---:|---|
| `POI1` | 1 | 0 | liste de nœuds |
| `SEG2` | 2 | 1 | segment linéaire |
| `TRI3` | 3 | 2 | triangle linéaire |
| `QUA4` | 4 | 2 | quadrangle linéaire |
| `TET4` | 4 | 3 | tétraèdre linéaire |
| `HEX8` | 8 | 3 | hexaèdre linéaire |

Ajouter un nouveau type d'élément est purement additif : nouvelle variante + métadonnées dans `src/element_type.rs`.

## Refcount sur les nœuds — interaction avec le GC

Chaque appel à `SubMesh::add_cell` **incrémente** le refcount interne de chaque nœud dans la `Configuration` (cf. [Configuration](configuration.md)). Le `Drop` du `SubMesh` les **décrémente**. Tant qu'un `SubMesh` référence un nœud, le ramasse-miettes le protège — même si tous les `Node` utilisateurs ont disparu.

```text
   Configuration             ◀── refcount par NodeId
        │                          (le nœud est-il vivant ?)
        ├── Node(s) utilisateur(s) ── chacun +1
        └── SubMesh(s)             ── chacun +1 par cellule incidente
```

En cas d'échec partiel d'`add_cell` (par exemple un nœud déjà ramassé), les incréments déjà effectués pour la cellule courante sont **annulés** (rollback transactionnel à l'échelle d'une cellule).

## API Rust

```rust,ignore
use pyrucast::mesh::configuration::Configuration;
use pyrucast::mesh::element_type::ElementType;
use pyrucast::mesh::{Mesh, SubMesh};
use pyrucast::mesh::node::Node;
use pyrucast::store::{insert, with, with_mut};

let cfg = insert(Configuration::new(2).unwrap());
let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

let sm_handle = insert(sm);
let mut mesh = Mesh::new(cfg.clone());
mesh.add_submesh(sm_handle).unwrap();
assert_eq!(mesh.cell_count().unwrap(), 1);
```

## API Python

```python
import pyrucast

c = pyrucast.Configuration(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])
n3 = c.add_node([0.5, 1.0])

mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([a, b, n3])
print(mesh)               # Mesh: 1 submesh(es), 1 cell(s) total
```

## Triangulation d'un contour fermé : `fill_surface`

`Mesh::fill_surface(contour, element_type)` prend un maillage `Mesh` contenant **un ou plusieurs sous-maillages SEG2** (chacun représentant une boucle fermée) et remplit la surface ainsi définie avec des éléments 2D. La configuration peut être en dimension **2** (cas le plus direct) ou **3** (les boucles doivent alors être quasi co-planaires — voir plus bas).

Pour l'instant un seul type cible est supporté : **`TRI3`**.

### Cas d'un seul contour (sans trous)

Avec un unique sous-maillage SEG2, la fonction prend un chemin rapide via *ear clipping* :

1. les segments sont chaînés dans l'ordre (un seul cycle, sinon erreur),
2. en 3D, la normale du plan moyen est calculée par la **méthode de Newell** ; les points sont projetés sur ce plan via une base orthonormée locale `(u, v)` orthogonale à la normale,
3. l'orientation du polygone est détectée (aire signée),
4. à chaque itération on retire un sommet *convexe* dont le triangle prev-curr-next ne contient aucun autre sommet du polygone (une « oreille »),
5. le triangle est ajouté au maillage, le sommet retiré, on recommence — jusqu'à ne plus avoir que 3 sommets.

Le résultat contient exactement `n − 2` triangles pour `n` nœuds de contour. **Aucun nœud interne n'est créé** (pas de raffinement). Les nœuds du contour sont réutilisés (leur compteur de références est incrémenté). En 3D, les triangles vivent dans l'espace 3D global — seule la triangulation est faite dans le plan moyen. Les triangles produits sont orientés **CCW** dans le plan de projection, quel que soit le sens du contour d'entrée.

### Cas avec trous (plusieurs contours)

Quand `contour` contient deux sous-maillages SEG2 ou plus, `fill_surface` bascule sur une **triangulation de Delaunay contrainte (CDT)** maison :

1. chaque sous-maillage est traité comme une boucle fermée indépendante (chaînage + validation),
2. la boucle dont l'aire absolue (dans le plan de projection) est la plus grande est automatiquement désignée comme **contour extérieur** ; les autres deviennent des **trous** ;
3. les points sont insérés un à un dans une triangulation de Bowyer-Watson (avec un super-triangle englobant),
4. chaque arête de boucle est ensuite **forcée** dans la triangulation (retrait des triangles qui la croisent + re-triangulation des deux polygones formés par *ear clipping*),
5. un *flood-fill* par parité depuis le super-triangle marque ce qui est extérieur au domaine (chaque traversée d'arête contrainte inverse le statut « extérieur / intérieur »), puis les triangles extérieurs et les triangles à l'intérieur des trous sont retirés.

L'orientation des boucles d'entrée n'a pas d'importance : la détection extérieur/trous se base uniquement sur l'aire absolue. Les triangles produits sont CCW dans le plan de projection. Aucun nœud interne n'est créé pour le moment ; ce sera le rôle de l'étape de raffinement à venir.

L'algorithme géométrique vit dans `pyrucast::ops::mesher::triangulation::cdt`, exposé via `triangulate_polygon_with_holes(outer, holes)` pour les besoins indépendants du système `Mesh`.

### Contrôle de planéité (cas 3D)

Lorsque la configuration est en 3D, la déviation maximale d'un nœud du contour au plan moyen doit rester inférieure à `1e-6 × diag`, où `diag` est la diagonale de la boîte englobante du contour. Si ce seuil est dépassé, `fill_surface` retourne une erreur claire indiquant la déviation observée et la tolérance. Ce seuil relatif tolère le bruit numérique habituel tout en refusant les vrais contours gauches.

### Raffinement (étape 4) — points Steiner

`Mesh::fill_surface` accepte un troisième argument optionnel `refinement: Option<RefinementOptions>`. La struct expose deux critères indépendants :

- `max_edge_length: Option<f64>` — longueur d'arête maximale tolérée ;
- `min_angle_deg: Option<f64>` — angle minimum garanti (en degrés).

Quand au moins un est renseigné, l'algorithme bascule sur la CDT (Bowyer-Watson contraint + raffinement de **Ruppert**) :

1. tant qu'une arête contrainte est *encroachée* (un sommet de la triangulation tombe dans son disque diamétral), on la coupe en son milieu — un nouveau nœud est inséré et l'ancienne contrainte est remplacée par les deux moitiés ;
2. sinon on cherche un triangle « mauvais » (arête trop longue OU rapport circumrayon/arête plus courte dépassant `1/(2 sin α)`) ;
3. on calcule son centre du cercle circonscrit ; s'il *encroache* une contrainte, on la coupe (retour à 1) ; sinon on l'insère par Bowyer-Watson contraint ;
4. on itère jusqu'à ce qu'aucun triangle ne viole les critères.

La convergence n'est théoriquement garantie que pour `min_angle_deg ≤ 20.7°` (Shewchuk). pyrucast plafonne le nombre total d'insertions à `50 · n_contour + 1000` pour éviter les divergences sur entrées pathologiques ; l'erreur est explicite si la limite est atteinte.

Les nouveaux nœuds (« Steiner ») sont créés dans la `Configuration` du contour, exactement comme les nœuds utilisateur. En 3D, leurs coordonnées sont calculées dans le plan local puis ré-injectées dans l'espace 3D via la base `(u, v, n)`.

### Limitations actuelles

- seul `TRI3` est supporté en sortie ;
- en 3D, l'algorithme refuse les contours franchement non plans (par construction) ;
- les boucles doivent être deux à deux disjointes (pas de trous emboîtés, pas de croisements) ;
- la fonction de taille reste **globale** (un seul `max_edge_length`) — une carte de taille variable `h(x, y)` viendra plus tard.

### Exemple Python (2D)

```python
import pyrucast

c = pyrucast.Configuration(dim=2)
nodes = [c.add_node(p) for p in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]

contour = pyrucast.Mesh(c, "SEG2")
for i in range(4):
    contour.unit().add_cell([nodes[i], nodes[(i + 1) % 4]])

surface = pyrucast.Mesh.fill_surface(contour, "TRI3")
print(surface)        # Mesh: 1 submesh(es), 2 cell(s) total
```

### Exemple Python (3D plan)

```python
import math
import pyrucast

c = pyrucast.Configuration(dim=3)
s = 1.0 / math.sqrt(2.0)
# Carré unitaire incliné à 45° autour de l'axe x.
pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, s, s), (0.0, s, s)]
nodes = [c.add_node(list(p)) for p in pts]

contour = pyrucast.Mesh(c, "SEG2")
for i in range(4):
    contour.unit().add_cell([nodes[i], nodes[(i + 1) % 4]])

surface = pyrucast.Mesh.fill_surface(contour, "TRI3")
# 2 triangles dont les sommets vivent dans le repère 3D global.
```

### Exemple Python (avec trou et raffinement)

```python
import pyrucast

c = pyrucast.Configuration(dim=2)

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

# Sans raffinement : 6 triangles « bruts ».
brut = pyrucast.Mesh.fill_surface(outer + hole, "TRI3")

# Avec raffinement : arête max 1.0 + angle min 20° → maillage fin et de bonne qualité.
fin = pyrucast.Mesh.fill_surface(
    outer + hole, "TRI3",
    max_edge_length=1.0,
    min_angle_deg=20.0,
)
# Aire triangulée = 16 - 4 = 12, mais bien plus de cellules.
```

Côté Rust :

```rust,ignore
use pyrucast::ops::mesher::triangulation::RefinementOptions;

let opts = RefinementOptions {
    max_edge_length: Some(1.0),
    min_angle_deg: Some(20.0),
};
let fin = Mesh::fill_surface(&combined, ElementType::TRI3, Some(opts))?;
```

Le module `pyrucast::ops::mesher::triangulation` regroupe les briques géométriques (`signed_area`, `ear_clip_2d`, `newell_normal`, `in_plane_basis`, `delaunay_2d`, `constrained_delaunay_2d`, `triangulate_polygon_with_holes`) — toutes opèrent sur des tableaux bruts et restent réutilisables indépendamment du système `Mesh`.

## Sûreté du swap

Les `SubMesh` et `Mesh` portent des effets de bord dans leur `Drop` (décrément du refcount des nœuds). Le store traite leur swap correctement :

- `swap_out` n'exécute **pas** le `Drop` de la valeur évincée (`std::mem::forget` interne) — l'objet est logiquement vivant, juste relocalisé.
- Le `Drop` final s'exécute après rechargement depuis le disque si nécessaire, garantissant que les refcounts sont décrémentés exactement une fois sur la durée de vie de l'objet.

Le détail est documenté dans le chapitre [Modèle mémoire](memory-model.md).
