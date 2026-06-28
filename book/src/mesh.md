# Maillage (Mesh / SubMesh)

Le maillage de pyrucast se compose à deux niveaux, selon le motif
[agrégat / zone](aggregate.md) :

- **`SubMesh`** : regroupe toutes les cellules d'un même `ElementType`. Stocke
  la connectivité à plat (un `Vec<NodeId>` de longueur `cell_count ×
  nodes_per_cell`).
- **`Mesh`** : agrège plusieurs `SubMesh` liés à la même `Coords`.

Le cas POI1 est volontairement dégénéré : un sous-maillage POI1 est exactement
une **liste de nœuds** (un nœud par cellule), ce qui sert de support naturel
aux [`NodeField`](node-field.md).

> Ce chapitre décrit l'**objet** maillage (structure, types d'éléments,
> refcount). La **construction** de maillages par des générateurs
> (`line_seg2`, `fill_surface`, `extrude`…) relève des
> [opérateurs de maillage](operateurs/maillage.md).

## Types d'éléments

L'enum `ElementType` liste les types supportés ; chaque variante porte sa
propre méta-information (nombre de nœuds, dimension topologique, nom court
cast3m).

| Variante | Nœuds | Dim. topo. | Cas usuel |
|---|---:|---:|---|
| `POI1` | 1 | 0 | liste de nœuds |
| `SEG2` | 2 | 1 | segment linéaire |
| `TRI3` | 3 | 2 | triangle linéaire |
| `QUA4` | 4 | 2 | quadrangle linéaire |
| `TET4` | 4 | 3 | tétraèdre linéaire |
| `HEX8` | 8 | 3 | hexaèdre linéaire |

Ajouter un nouveau type d'élément est purement additif (nouvelle variante +
métadonnées dans `src/containers/mesh/element_type.rs`) — voir
[Ajouter un élément fini](developper/ajouter-un-element-fini.md).

## Cellule (`Cell`)

`mesh.cell(submesh_idx, cell_idx)` (ou `mesh[submesh_idx][cell_idx]`) renvoie
une **vue** `Cell` sur une cellule : `len(cell)` donne son nombre de nœuds et
`cell[k]` le `k`-ième `Node`. C'est l'accès lecture à la connectivité ;
l'ajout passe par `submesh.add_cell([...])`.

## Refcount sur les nœuds — interaction avec le GC

Chaque appel à `SubMesh::add_cell` **incrémente** le refcount interne de chaque
nœud dans la `Coords` (cf. [Coordonnées](coords.md)). Le `Drop` du `SubMesh`
les **décrémente**. Tant qu'un `SubMesh` référence un nœud, le ramasse-miettes
le protège — même si tous les [`Node`](node.md) utilisateurs ont disparu.

```text
   Coords             ◀── refcount par NodeId
        │                          (le nœud est-il vivant ?)
        ├── Node(s) utilisateur(s) ── chacun +1
        └── SubMesh(s)             ── chacun +1 par cellule incidente
```

En cas d'échec partiel d'`add_cell` (par exemple un nœud déjà ramassé), les
incréments déjà effectués pour la cellule courante sont **annulés** (rollback
transactionnel à l'échelle d'une cellule).

## API Rust

```rust,ignore
use pyrucast::containers::mesh::Coords;
use pyrucast::mesh::element_type::ElementType;
use pyrucast::mesh::{Mesh, SubMesh};
use pyrucast::mesh::node::Node;
use pyrucast::store::insert;

let coords = insert(Coords::new(2).unwrap());
let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

let sm_handle = insert(sm);
let mut mesh = Mesh::new(coords.clone());
mesh.add_sub(sm_handle).unwrap();
assert_eq!(mesh.cell_count().unwrap(), 1);
```

## API Python

```python
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])
n3 = c.add_node([0.5, 1.0])

# Mesh(coords, type) crée un maillage à un seul sous-maillage ; unit() en
# donne la vue, add_cell ajoute une cellule.
mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([a, b, n3])
print(mesh)  # Mesh: 1 submesh(es), 1 cell(s) total
print(mesh.element_types())  # ['TRI3']
print(mesh.cell_counts())  # [1]

# Composer plusieurs zones : l'union | (jamais +).
quad = pyrucast.Mesh(c, "QUA4")
# … add_cell … ;  combined = mesh | quad
```

## Sûreté du swap

Les `SubMesh` et `Mesh` portent des effets de bord dans leur `Drop`
(décrément du refcount des nœuds). Le store traite leur swap correctement :

- `swap_out` n'exécute **pas** le `Drop` de la valeur évincée
  (`std::mem::forget` interne) — l'objet est logiquement vivant, juste
  relocalisé ;
- le `Drop` final s'exécute après rechargement depuis le disque si nécessaire,
  garantissant que les refcounts sont décrémentés **exactement une fois** sur
  la durée de vie de l'objet.

Le détail est dans le chapitre [Modèle mémoire](memory-model.md).

## Visualisation

`mesh.plot(...)` trace le maillage (chaque sous-maillage avec sa propre couleur,
ou coloré par un champ). Voir [Visualisation](visualization.md).
