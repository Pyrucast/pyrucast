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
> (`line`, `triangulate_surface`, `extrude`…) relève des
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
| `PYRA5` | 5 | 3 | pyramide linéaire (raccord hexaèdre ↔ tétraèdre) |
| `PENTA6` | 6 | 3 | prisme linéaire (extrusion d'un TRI3) |
| `HEX8` | 8 | 3 | hexaèdre linéaire |
| `SEG3` | 3 | 1 | segment quadratique |
| `TRI6` | 6 | 2 | triangle quadratique |
| `QUA8` | 8 | 2 | quadrangle quadratique (sérendipité) |
| `QUA9` | 9 | 2 | quadrangle biquadratique (Lagrange complet, nœud central) |
| `TET10` | 10 | 3 | tétraèdre quadratique |
| `PENTA15` | 15 | 3 | prisme quadratique (sérendipité) |
| `HEX20` | 20 | 3 | hexaèdre quadratique (sérendipité) |
| `HEX27` | 27 | 3 | hexaèdre tri-quadratique (Lagrange complet, centres de face + nœud central) |

Les huit derniers types sont **quadratiques** (Lagrange-2) : ils reprennent la
numérotation des sommets de leur parent linéaire puis ajoutent les nœuds de
**milieu d'arête**, dans l'ordre d'arêtes de la convention VTK (voir le rustdoc
d'`ElementType`). `QUA8`, `HEX20` et `PENTA15` sont **sérendipité** (nœuds
d'arête seulement) ; `SEG3`, `TRI6`, `TET10`, `QUA9` et `HEX27` sont des
Lagrange complets (`QUA9`/`HEX27` = quadrangle/hexaèdre bi-/tri-quadratiques,
avec nœuds de face et central). Ils se posent avec l'interpolation `LAGRANGE2`
(cf. [Espace éléments finis](fe-space.md)).

Ajouter un nouveau type d'élément est purement additif : un fichier
`src/atoms/element_kind/<nom>.rs` et une variante — voir
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

## Scellement (connectivité figée après consommation)

Par convention, un maillage n'est plus modifié une fois construit : un
[espace éléments finis](fe-space.md), un [champ](node-field.md) ou une
[matrice](matrix.md) indexent ses cellules, et lui ajouter une maille par la
suite les laisserait dans un état incohérent.

Cette convention est désormais **imposée**. Dès qu'un objet autre qu'un
maillage capture un `SubMesh` (construction d'un `SubFiniteElementSpace`, d'un
`SubNodeField`, d'un support de `SubMatrix`…), ce sous-maillage est **scellé** :
sa connectivité est figée pour toujours. `add_cell` / `add_cell_taking`
renvoient alors l'erreur `MeshSealed`. Un `Mesh` qui se contente de contenir le
sous-maillage ne le scelle **pas** — il peut continuer à grossir tant qu'aucun
consommateur ne s'y attache. On teste l'état avec `is_sealed`.

Pour repartir d'un maillage scellé et le modifier à nouveau, on en prend une
**copie profonde** avec `duplicate()` : un `SubMesh` (ou `Mesh`) neuf, **non
scellé**, avec la même connectivité (les nœuds sont partagés — même `Coords` —,
seuls leurs refcounts augmentent).

```python
mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([a, b, n3])

pyrucast.FiniteElementSpace(mesh)  # scelle mesh[0]
assert mesh[0].is_sealed
# mesh[0].add_cell([...])           # → RuntimeError (MeshSealed)

copie = mesh.duplicate()  # neuf, modifiable
copie.unit().add_cell([b, n3, n4])  # OK
```

## API Rust

```rust,ignore
use pyrucast::coords::Coords;
use pyrucast::atoms::ElementType;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::atoms::Node;
use pyrucast::store::Handle;

let coords = Handle::new(Coords::new(2).unwrap());
let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

let sm_handle = Handle::new(sm);
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

## Durée de vie et refcount

Les `SubMesh` et `Mesh` portent un effet de bord dans leur `Drop` : ils
décrémentent le refcount des nœuds qu'ils référencent dans la `Coords`. Cet
effet a lieu **exactement une fois**, quand le dernier `Handle` sur le maillage
disparaît. Le détail est dans le chapitre
[Modèle mémoire](memory-model.md).

## Visualisation

`mesh.plot(...)` trace le maillage (chaque sous-maillage avec sa propre couleur,
ou coloré par un champ). Voir [Visualisation](visualization.md).
