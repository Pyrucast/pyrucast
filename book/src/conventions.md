# Conventions & philosophie

## Méthodes vs fonctions libres

La règle qui décide si une opération est une **méthode** d'un conteneur ou
une **fonction libre** d'un module `ops/` est figée dans le fichier
`CONVENTIONS.md` à la racine du dépôt. En résumé :

- **méthode** — accesseur, mutation préservant l'invariant, ou vue dérivée
  d'**un seul** conteneur (`mesh.cell_count()`, `field.set(...)`,
  `field.components()`) ;
- **fonction libre `ops::<thème>`** — tout opérateur qui croise des
  conteneurs ou appartient à une famille d'opérateurs
  (`ops::field::restrict(field, mesh)`, `ops::assemble::stiffness(model,
  mat)`, `ops::mesher::consolidate(mesh)`).

Le binding Python est un **miroir 1:1** : une fonction Rust devient une
fonction de module Python, une méthode reste une méthode. On vise le style
numpy/scipy (et l'héritage cast3m) — des opérateurs nommés dans des
modules thématiques plutôt que des chaînes de méthodes :

```python
import pyrucast

# fonctions (opérateurs), pas des méthodes :
poi = pyrucast.to_poi1(mesh)
coords = pyrucast.coordinates(mesh)
K = pyrucast.stiffness(model, materials)
sol = pyrucast.solve(K, rhs)
```

## Erreurs

Toute l'API publique renvoie `pyrucast::Result<T>`, alias de `Result<T, PyrucastError>`.

`PyrucastError` est l'unique type d'erreur de la librairie. Côté Python, il est converti automatiquement en `RuntimeError`.

Rust :

```rust,ignore
use pyrucast::mesh::configuration::Configuration;

// Dimension nulle — erreur attendue.
let err = Configuration::new(0).unwrap_err();
assert!(err.to_string().contains("dim must be ≥ 1"));

// Pattern-matching sur les variantes.
match Configuration::new(0) {
    Ok(_) => unreachable!(),
    Err(pyrucast::PyrucastError::Message(msg)) => println!("erreur : {msg}"),
    Err(e) => println!("autre erreur : {e}"),
}
```

Python :

```python
import pyrucast

try:
    c = pyrucast.Configuration(0)  # dimension nulle
except RuntimeError as e:
    print(f"erreur : {e}")         # erreur : dim must be ≥ 1
```

## Affichage : `Debug` vs `Display`

Chaque objet du modèle implémente deux traits :

- `Debug` — vue **structurelle** : utile pour le développement, exposée en Python via `__repr__`.
- `Display` — vue **résumée** orientée utilisateur EF, façon listing cast3m, exposée en Python via `__str__`.

Le binding PyO3 branche ces deux vues sur les dunder methods Python correspondantes.

Rust :

```rust,ignore
use pyrucast::mesh::configuration::Configuration;
use pyrucast::store::{insert, with};

let cfg = insert(Configuration::new(2).unwrap());
with(&cfg, |c| {
    println!("{:?}", c);  // vue structurelle (Debug)
    println!("{}", c);    // vue résumée (Display)
}).unwrap();
```

Python :

```python
import pyrucast

c = pyrucast.Configuration(dim=2)
c.add_node([0.0, 0.0])

print(repr(c))  # vue structurelle — __repr__
print(str(c))   # vue résumée cast3m — __str__
print(c)        # même chose que str(c)
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

Le swap disque des objets pyrucast (Configuration, SubMesh, NodeField…) passe par ce même mécanisme sans intervention de l'utilisateur.

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
