# Conventions & philosophie

## Trois natures de types

L'arborescence sépare trois choses que le mot « objet » confond, et c'est
elle qui rend la convention lisible sans qu'on ait à la relire :

- **conteneurs** (`containers/`) — ce dont **une partie est encore de la
  même nature** : la moitié d'un maillage est un maillage, la moitié d'un
  champ est un champ. Les sept agrégats et leurs vues `Sub*`. Seul un
  conteneur peut être le **sujet** d'un opérateur ;
- **atomes** (`atoms/`) — les insécables : `Node`, `Cell`, `Element`
  (désignateurs), `ElementType`, `Point3`, `RgbColor` (valeurs). Un atome
  compose (`node | node` → `Mesh`) mais ne se décompose pas ; il est donc
  toujours argument, jamais `self`. Piège à connaître : `Cell` s'indexe
  (`cell[k]` → `Node`) sans être un conteneur, car ses parties sont d'une
  *autre* nature ;
- **magasin** (`coords.rs`) — `Coords`, ni conteneur ni atome : il contient
  des nœuds, d'une autre nature que lui, et n'offre ni `[]`, ni `|`, ni vue
  `Sub*`.

## Méthodes vs fonctions libres

La règle qui décide si une opération est une **méthode** d'un conteneur ou
une **fonction libre** d'un module `ops/` est figée dans le fichier
`CONVENTIONS.md` à la racine du dépôt. En résumé :

- **méthode** — accesseur, mutation préservant l'invariant, ou vue dérivée
  d'**un seul** conteneur (`mesh.cell_count()`, `field.set(...)`,
  `field.components()`) ;
- **fonction libre `ops::<module>`** — tout opérateur qui croise des
  conteneurs ou appartient à une famille d'opérateurs
  (`ops::node_field::restrict(field, mesh)`, `ops::matrix::stiffness(model,
  mat)`, `ops::mesh::consolidate(mesh)`).

## Où vit une fonction libre : le conteneur produit

**Un module d'`ops/` rassemble les opérateurs qui produisent le même
conteneur, et porte son nom.** On range par la **sortie**, jamais par
l'entrée : `gradient(field, fespace)` produit un `ElementField`, donc il vit
dans `element_field`, à côté de `deformation` — pas dans `node_field`.

| module | produit |
|---|---|
| `mesh` | un `Mesh` (mailleurs, transformations, `select`) |
| `node_field` | un `NodeField` (dérivations, assemblage nodal) |
| `element_field` | un `ElementField` (cinématique, matériaux, comportement) |
| `matrix` | une `Matrix` (les assembleurs) |
| `coords` | écrit dans le magasin (`set`, `displace`) |

Ceux qui ne produisent **aucun** conteneur échappent à la règle par
construction et se rangent par activité : `measure` (réductions à un
nombre), `geom` (requêtes géométriques), `export` (effets de bord).

**Une seule exception, nommée et assumée** : `solver` produit un `NodeField`
et devrait rejoindre `node_field`. Il garde son nom parce que plusieurs
familles distinctes produisent un champ nodal — dérivation, assemblage,
résolution — et que seule la résolution se cherche par son propre nom.

Corollaire pratique : un module ne contient jamais deux opérateurs qui ne
diffèrent que par leur conteneur. Le qualificatif appartient au nom du
module, pas à celui de la fonction — d'où trois `consolidate` homonymes et
sans ambiguïté, au lieu de trois fonctions suffixées au même endroit.

## Le miroir Python

Le binding Python est un **miroir 1:1** : une fonction Rust devient une
fonction Python, une méthode reste une méthode. On vise le style
numpy/scipy (et l'héritage cast3m) — des opérateurs **nommés** plutôt que
des chaînes de méthodes. Le module de production est **reflété par un
sous-module** : une fonction libre `ops::<module>::f` est exposée comme
`pyrucast.<module>.f`. Les conteneurs et les atomes restent des classes au
top-level (`pyrucast.Coords`, `pyrucast.Mesh`, `pyrucast.Node`, …) :

```python
import pyrucast

# fonctions (opérateurs), rangées par conteneur produit — pas des méthodes :
poi = pyrucast.mesh.to_poi1(mesh)
coords = pyrucast.node_field.coordinates(mesh)
eps = pyrucast.element_field.deformation(u, fes)
K = pyrucast.matrix.stiffness(model, materials)
sol = pyrucast.solver.solve(K, rhs)
```

Le miroir est **sans exception** : aucune fonction libre ne vit au top-level
Python. Les trois consolidations suivent leur module —
`pyrucast.mesh.consolidate`, `pyrucast.node_field.consolidate`,
`pyrucast.element_field.consolidate` — comme leurs homologues Rust.

L'extension compilée `_pyrucast` est, elle, **plate** : les homonymes y
portent un `#[pyo3(name = …)]` distinct (`consolidate_mesh`, …) et la couche
Python pure les ré-exporte sous leur vrai nom dans le bon sous-module. C'est
un détail du namespace privé, pas une entorse au miroir.

## Erreurs

Toute l'API publique renvoie `pyrucast::Result<T>`, alias de `Result<T, PyrucastError>`.

`PyrucastError` est l'unique type d'erreur de la librairie. Côté Python, il est converti automatiquement en `RuntimeError`.

Rust :

```rust,ignore
use pyrucast::coords::Coords;

// Dimension nulle — erreur attendue.
let err = Coords::new(0).unwrap_err();
assert!(err.to_string().contains("dim must be ≥ 1"));

// Pattern-matching sur les variantes.
match Coords::new(0) {
    Ok(_) => unreachable!(),
    Err(pyrucast::PyrucastError::Message(msg)) => println!("erreur : {msg}"),
    Err(e) => println!("autre erreur : {e}"),
}
```

Python :

```python
import pyrucast

try:
    c = pyrucast.Coords(0)  # dimension nulle
except RuntimeError as e:
    print(f"erreur : {e}")  # erreur : dim must be ≥ 1
```

## Affichage : `Debug` vs `Display`

Chaque objet du modèle implémente deux traits :

- `Debug` — vue **structurelle** : utile pour le développement, exposée en Python via `__repr__`.
- `Display` — vue **résumée** orientée utilisateur EF, façon listing cast3m, exposée en Python via `__str__`.

Le binding PyO3 branche ces deux vues sur les dunder methods Python correspondantes.

Rust :

```rust,ignore
use pyrucast::coords::Coords;
use pyrucast::store::{insert, read};

let coords = insert(Coords::new(2).unwrap());
let c = read(&coords).unwrap();
println!("{:?}", &*c);  // vue structurelle (Debug)
println!("{}", &*c);    // vue résumée (Display)
```

Python :

```python
import pyrucast

c = pyrucast.Coords(dim=2)
c.add_node([0.0, 0.0])

print(repr(c))  # vue structurelle — __repr__
print(str(c))  # vue résumée cast3m — __str__
print(c)  # même chose que str(c)
```

## Sérialisation : un seul mécanisme

Le trait `Persist` (implémenté automatiquement pour tout type `serde::Serialize + DeserializeOwned`) produit un format binaire **portable Linux ↔ Windows**. Ce socle unique sert à la fois :

- au **swap disque** (slot par slot, géré par le Store) ;
- à la **sauvegarde fichier** (graphe d'objets d'une `Session`, dans un conteneur versionné).

Rust — sérialisation manuelle d'un type quelconque :

```rust,ignore
use pyrucast::persist::Persist;

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
struct Pt { x: f64, y: f64 }

let original = Pt { x: 1.5, y: -2.0 };
let bytes = original.to_bytes().unwrap();
let restored = Pt::from_bytes(&bytes).unwrap();
assert_eq!(original, restored);
```

Le swap disque des objets pyrucast (Coords, SubMesh, NodeField…) passe par ce même mécanisme sans intervention de l'utilisateur.

> **Python** : `Persist` n'est pas exposé côté Python — c'est une brique interne du store. La sérialisation des objets depuis Python passera par une API `Session::save` / `Session::load` (Phase 5).

## Definition of Done par objet

Un objet n'est considéré comme terminé que lorsque les six points suivants sont verts :

1. Struct Rust vivant dans le Store (`Handle<T>` typé).
2. `Debug` (structure) + `Display` (résumé).
3. Tests unitaires Rust + doctests sur tout l'API public.
4. Binding PyO3 (`__repr__` / `__str__`).
5. Tests Python (pytest).
6. Chapitre de cette documentation.

## Dépendances approuvées

Le socle figé est : `pyo3`, `maturin`, `mdbook`, `serde`, `bincode`. Toute autre dépendance, Rust ou Python, requiert un accord explicite.
