# Introduction

**pyrucast** est une librairie d'éléments finis dont le cœur est écrit en Rust et qui expose une API Python. Elle s'inspire des principes de cast3m : un noyau d'objets typés, accompagné de fonctions opérant sur ces objets.

## Philosophie

- **Simplicité avant tout.** Le code doit rester maintenable et éditable par un humain non expert ; on évite la sophistication gratuite.
- **Dépendances minimales.** Tout ajout de dépendance externe (Rust ou Python) requiert un accord explicite.
- **Vérification continue.** Chaque objet est livré avec des tests unitaires Rust, des doctests, des tests Python et un chapitre de cette documentation.

## Modèle d'objets (ordre de dépendance)

1. **Configuration** — jeux de coordonnées de nœuds (plusieurs jeux possibles).
2. **Node** — accesseur utilisateur d'un nœud et de ses coordonnées.
3. **Mesh / SubMesh** — un sous-maillage groupe les cellules d'un même type d'élément. POI1 = élément à 1 nœud ; un sous-maillage POI1 est une liste de nœuds.
4. **NodeField** — valeurs par nœud (sur un maillage POI1), multi-composantes.
5. **FiniteElementSpace** — maillage + formulation EF.
6. **ElementField** — valeurs par point de Gauss × composante.
7. **Model** — modèle physique (élasticité, plasticité, thermique…) sur un FE space.
8. **Matrix** — matrice creuse, hand-built ou assemblée depuis un Model.

## Premiers pas

Exemple minimal en Rust :

```rust,ignore
use pyrucast::configuration::Configuration;
use pyrucast::element_type::ElementType;
use pyrucast::mesh::{Mesh, SubMesh};
use pyrucast::node::Node;
use pyrucast::store::insert;

let cfg = insert(Configuration::new(2).unwrap());
let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();

let mut sm = SubMesh::new(cfg.clone(), ElementType::SEG2);
sm.add_cell(&[a.id(), b.id()]).unwrap();

let sm_h = insert(sm);
let mut mesh = Mesh::new(cfg);
mesh.add_submesh(sm_h).unwrap();
println!("{}", mesh); // Mesh: 1 submesh(es), 1 cell(s) total
```

Exemple équivalent en Python :

```python
import pyrucast

c = pyrucast.Configuration(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])

sm = pyrucast.SubMesh(c, "SEG2")
sm.add_cell([a.id, b.id])

mesh = pyrucast.Mesh(c)
mesh.add_submesh(sm)
print(mesh)  # Mesh: 1 submesh(es), 1 cell(s) total
```

## Feuille de route

Le déroulé des phases (0 à 6) est décrit dans `ROADMAP.md` à la racine du dépôt.
