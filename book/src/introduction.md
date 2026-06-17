# Introduction

**pyrucast** est une librairie d'éléments finis dont le cœur est écrit en Rust et qui expose une API Python. Elle s'inspire des principes de cast3m : un noyau d'objets typés, accompagné de fonctions opérant sur ces objets.

## Philosophie

- **Simplicité avant tout.** Le code doit rester maintenable et éditable par un humain non expert ; on évite la sophistication gratuite.
- **Dépendances minimales.** Tout ajout de dépendance externe (Rust ou Python) requiert un accord explicite.
- **Vérification continue.** Chaque objet est livré avec des tests unitaires Rust, des doctests, des tests Python et un chapitre de cette documentation.

## Modèle d'objets (arbre de dépendances)

Chaque structure ne dépend que des structures qui la précèdent dans le graphe.
Les noms entre parenthèses sont des types auxiliaires (enum / primitif) sans état partagé.

```text
Coords
├── NodeId            (u32 opaque)
├── Node              (accesseur RAII — maintient le refcount GC)
├── NodeField         (valeurs par nœud × composante)
└── SubMesh           (+ ElementType)
      └── Mesh        (agrège N × SubMesh)
            └── SubFiniteElementSpace  (+ Interpolation, QuadratureRule)
                  ├── FiniteElementSpace  (agrège N × SubFiniteElementSpace)
                  └── ElementField        (valeurs par cellule × point de Gauss)

SubFiniteElementSpace + ElementField ──► SubModel::HeatConduction ──┐
Coords              ──► SubModel::Dirichlet       ├──► Model
                                                         ┘
NodeId ──► DofId ──► Matrix  (matrice creuse, DOFs nommés par (NodeId, champ))

Model + NodeField (second membre) + Matrix ──► solve() ──► NodeField (solution)
```

Résumé des rôles :

| Structure              | Rôle                                                              |
|------------------------|-------------------------------------------------------------------|
| `Coords`        | Référentiel de coordonnées de nœuds (plusieurs jeux possibles)    |
| `NodeId`               | Identifiant opaque u32 d'un nœud                                  |
| `Node`                 | Accesseur utilisateur avec protection GC automatique              |
| `SubMesh`              | Cellules d'un même `ElementType` (SEG2, TRI3, …)                  |
| `Mesh`                 | Union de sous-maillages                                           |
| `NodeField`            | Valeurs scalaires ou vectorielles par nœud × composante           |
| `SubFiniteElementSpace`           | Formulation EF sur un sous-maillage (interpolation + quadrature)  |
| `FiniteElementSpace`   | Union de sous-espaces EF                                          |
| `ElementField`         | Valeurs par cellule × point de Gauss × composante                 |
| `SubModel`             | Physique locale : `HeatConduction` ou `Dirichlet`                 |
| `Model`                | Problème physique complet (agrège N × SubModel)                   |
| `DofId`                | Degré de liberté : `(NodeId, index de champ)`                     |
| `Matrix`               | Matrice creuse dont les lignes/colonnes sont des `DofId`          |

## Premiers pas

Exemple minimal en Rust :

```rust,ignore
use pyrucast::containers::mesh::Coords;
use pyrucast::mesh::element_type::ElementType;
use pyrucast::mesh::{Mesh, SubMesh};
use pyrucast::mesh::node::Node;
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

## Feuille de route

Le déroulé des phases (0 à 6) est décrit dans `ROADMAP.md` à la racine du dépôt.
