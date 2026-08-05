# Introduction

**pyrucast** est une librairie d'éléments finis dont le cœur est écrit en Rust et qui expose une API Python. Elle s'inspire des principes de cast3m : un noyau d'objets typés, accompagné de fonctions opérant sur ces objets.

> 📖 **Référence API Rust (rustdoc).** Cette documentation couvre les principes,
> l'architecture et les exemples ; la référence par item (signatures, types,
> modules) est générée par rustdoc et publiée à côté de ce book :
> <https://pyrucast.github.io/pyrucast/rust/>.

## Philosophie

- **Simplicité avant tout.** Le code doit rester maintenable et éditable par un humain non expert ; on évite la sophistication gratuite.
- **Dépendances minimales.** Tout ajout de dépendance externe (Rust ou Python) requiert un accord explicite.
- **Vérification continue.** Chaque objet est livré avec des tests unitaires Rust, des doctests, des tests Python et un chapitre de cette documentation.

## Modèle d'objets (arbre de dépendances)

Chaque structure ne dépend que des structures qui la précèdent. La plupart
viennent par **paire** zone / agrégat (`SubMesh`/`Mesh`…) — c'est le motif
[Agrégat](aggregate.md), avec sa composition par union `|`.

```text
Coords ── Node                          (NodeId u32 stable ; Node = accesseur RAII)
   │
   ├── NodeField  (agrège des SubNodeField)        valeurs par nœud × composante
   │
   └── Mesh  (agrège des SubMesh, + ElementType)   géométrie
          └── FiniteElementSpace  (agrège des SubFiniteElementSpace)
                 │                  (+ Interpolation, QuadratureRule)
                 ├── ElementField  (agrège des SubElementField)   valeurs aux Gauss
                 └── Model  (agrège des SubModel : physiques + contraintes)
                        └── ops::matrix ──► Matrix   (matrice creuse, DOFs nommés)

Model + ElementField (matériau) ──► stiffness ──► Matrix
Matrix + NodeField (second membre) ──► solve ──► NodeField (solution)
```

Deux **traits transverses** factorisent le comportement commun :
[`Aggregate`](aggregate.md) (accès `len`/`[i]`/union `|`) et
[`Field`/`SubField`](field.md) (composantes nommées, `min`/`max`, arithmétique),
partagés entre `NodeField` et `ElementField`.

Résumé des rôles :

| Structure | Rôle |
|---|---|
| [`Coords`](coords.md) | Référentiel de coordonnées de nœuds (plusieurs configurations) |
| [`Node`](node.md) | Accesseur utilisateur d'un nœud, avec protection GC automatique |
| [`SubMesh`](mesh.md) / `Mesh` | Cellules d'un même `ElementType` / union de sous-maillages |
| [`SubNodeField`](node-field.md) / `NodeField` | Valeurs par nœud × composante (zone / agrégat) |
| [`SubFiniteElementSpace`](fe-space.md) / `FiniteElementSpace` | Formulation EF (interpolation + quadrature) / union |
| [`SubElementField`](element-field.md) / `ElementField` | Valeurs par cellule × point de Gauss × composante |
| [`SubModel`](model.md) / `Model` | Physique ou contrainte locale / problème complet |
| [`SubMatrix`](matrix.md) / `Matrix` | Matrice creuse dont les lignes/colonnes sont des DOFs `(NodeId, champ)` |
| [`SubEvolution`](evolution.md) / `Evolution` | Valeur (scalaire ou champ) tabulée vs une variable, interpolée linéairement (zone / agrégat) |

## Premiers pas

Exemple minimal en Rust :

```rust,ignore
use pyrucast::coords::Coords;
use pyrucast::atoms::ElementType;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::atoms::Node;
use pyrucast::store::insert;

let coords = insert(Coords::new(2).unwrap());
let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();

let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
sm.add_cell(&[a.id(), b.id()]).unwrap();

let sm_h = insert(sm);
let mut mesh = Mesh::new(coords);
mesh.add_sub(sm_h).unwrap();
println!("{}", mesh); // Mesh: 1 submesh(es), 1 cell(s) total
```

Exemple équivalent en Python :

```python
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])

mesh = pyrucast.Mesh(c, "SEG2")
mesh.unit().add_cell([a, b])
print(mesh)  # Mesh: 1 submesh(es), 1 cell(s) total
```
