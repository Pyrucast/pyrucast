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
use pyrucast::configuration::Configuration;
use pyrucast::element_type::ElementType;
use pyrucast::mesh::{Mesh, SubMesh};
use pyrucast::node::Node;
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
assert_eq!(with(&insert(mesh), |m| m.cell_count()).unwrap().unwrap(), 1);
```

## API Python

```python
import pyrucast

c = pyrucast.Configuration(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])
n3 = c.add_node([0.5, 1.0])

sm_tri = pyrucast.SubMesh(c, "TRI3")
sm_tri.add_cell([a.id, b.id, n3.id])

mesh = pyrucast.Mesh(c)
mesh.add_submesh(sm_tri)
print(mesh)               # Mesh: 1 submesh(es), 1 cell(s) total
```

## Triangulation d'un contour fermé : `fill_2d`

`Mesh::fill_2d(contour, element_type)` prend un maillage **SEG2** dont les segments forment une **boucle fermée simple** et le remplit avec des éléments 2D. La configuration doit être en dimension 2.

Pour l'instant un seul type cible est supporté : **`TRI3`**. La fonction utilise un *ear clipping* élémentaire :

1. les segments sont chaînés dans l'ordre (un seul cycle, sinon erreur),
2. l'orientation du polygone est détectée (l'aire signée est calculée),
3. à chaque itération on retire un sommet *convexe* dont le triangle prev-curr-next ne contient aucun autre sommet du polygone (une « oreille »),
4. le triangle est ajouté au maillage, le sommet retiré, on recommence — jusqu'à ne plus avoir que 3 sommets.

Le résultat contient exactement `n − 2` triangles pour `n` nœuds de contour ; **aucun nœud interne n'est créé** dans cette première itération (pas de raffinement). Les nœuds du contour sont réutilisés (leur compteur de références est incrémenté). Les triangles produits sont orientés **CCW** dans le plan, quel que soit le sens du contour d'entrée.

### Limitations actuelles

- un seul contour simple, pas de trous ;
- dimension 2 uniquement (pas de projection 3D plane) ;
- pas de point Steiner — la qualité géométrique dépend entièrement du contour ;
- seul `TRI3` est supporté en sortie.

Ces limitations seront levées par itérations : projection 3D plane → trous → raffinement (taille cible + critère de qualité).

### Exemple Python

```python
import pyrucast

c = pyrucast.Configuration(dim=2)
nodes = [c.add_node(p) for p in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]

contour = pyrucast.Mesh(c, "SEG2")
for i in range(4):
    contour.add_cell([nodes[i].id, nodes[(i + 1) % 4].id])

surface = pyrucast.Mesh.fill_2d(contour, "TRI3")
print(surface)        # Mesh: 1 submesh(es), 2 cell(s) total
```

L'algorithme géométrique est isolé dans le module `pyrucast::triangulation` (`signed_area`, `ear_clip_2d`) — il opère sur des tableaux 2D bruts et reste réutilisable indépendamment du système `Mesh`.

## Sûreté du swap

Les `SubMesh` et `Mesh` portent des effets de bord dans leur `Drop` (décrément du refcount des nœuds). Le store traite leur swap correctement :

- `swap_out` n'exécute **pas** le `Drop` de la valeur évincée (`std::mem::forget` interne) — l'objet est logiquement vivant, juste relocalisé.
- Le `Drop` final s'exécute après rechargement depuis le disque si nécessaire, garantissant que les refcounts sont décrémentés exactement une fois sur la durée de vie de l'objet.

Le détail est documenté dans le chapitre [Modèle mémoire](memory-model.md).
