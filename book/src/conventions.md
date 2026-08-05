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

Troisième cas : l'opérateur **générique**, dont le produit est bien un
conteneur mais **pas un conteneur déterminé** — `abs` rend un `NodeField` ou
un `ElementField` selon son argument, la règle ne désigne donc pas *un*
module. Ceux-là se rangent par **domaine** (`field`) et restent des fonctions
libres à part entière.

**Une seule exception, nommée et assumée** : `solver` produit un `NodeField`
et devrait rejoindre `node_field`. Il garde son nom parce que plusieurs
familles distinctes produisent un champ nodal — dérivation, assemblage,
résolution — et que seule la résolution se cherche par son propre nom.

Corollaire pratique : un module ne contient jamais deux opérateurs qui ne
diffèrent que par leur conteneur. Le qualificatif appartient au nom du
module, pas à celui de la fonction — d'où trois `consolidate` homonymes et
sans ambiguïté, au lieu de trois fonctions suffixées au même endroit.

## Le verbe exposé aussi en méthode

Une fonction libre garde sa **forme canonique** — c'est elle qui est
documentée et qui définit l'opération. Elle est **en plus** exposée comme
méthode de son premier argument si, et seulement si, les trois conditions
tiennent :

1. **le premier argument est le sujet** — l'objet qu'on transforme ;
2. **le retour est un conteneur** — sinon il n'y a rien à composer ;
3. **l'opération a un sens pour toute instance du type.**

La troisième est celle qu'on oublie, et c'est la plus coûteuse : une méthode
*promet*, elle apparaît dans l'auto-complétion de chaque objet du type.
`u.deformation(fes)` s'afficherait sur tous les champs nodaux alors qu'elle
exige des composantes `u_x`/`u_y`/`u_z` ; `t.thermal_strain(...)` sur tous les
champs par éléments alors qu'elle exige une température. Ces opérations restent
des fonctions libres seules.

La ligne de partage : une **précondition structurelle** est admise
(`triangulate_surface` veut un contour fermé, `divergence` veut autant de
composantes que d'axes — vérifié par comptage, jamais par nom), une exigence de
**sens porté par les noms de composantes** ne l'est pas.

Pour tester la condition sans se tromper, lire la méthode avec un receveur
**quelconque**, pas avec l'exemple bien nommé : `stresses.internal_forces(model)`
sonne juste, mais c'est le nom de la variable qui fait le travail —
`field.internal_forces()` révèle que le type ne promet rien.

```python
# chaînage, quand les trois conditions tiennent :
peau = maillage.skin().consolidate()
libre = champ.select(ge=0.0)
eps = u.gradient(fes)

# forme canonique seule, sinon :
eps = pyrucast.element_field.deformation(u, fes)  # exige un déplacement
f = pyrucast.node_field.merge(a, b)  # symétrique : `a | b` suffit
```

Deux conséquences à retenir. **Un ordre d'arguments qui ne met pas le sujet en
tête est un défaut à corriger**, pas une raison de renoncer à la méthode.
Et **une opération symétrique n'a pas de méthode** : `a.merge(b)` suggérerait
que l'ordre compte, alors que `merge` est l'alias nommé de `a | b`.

Enfin, le nom peut changer entre les deux formes. Le nom complet est toujours
« qualificatif + verbe » ; la fonction libre reçoit le qualificatif de son
module, la méthode n'en a pas et doit le porter : `matrix.stiffness(model,
mats)` d'un côté, `model.stiffness_matrix(mats)` de l'autre. Quand la sortie
est du type du sujet il n'y a rien à qualifier, et le nom ne bouge pas
(`mesh.consolidate(m)` / `m.consolidate()`).

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
coords = pyrucast.node_field.positions(mesh)
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

> **Python** : `Persist` n'est pas exposé côté Python — c'est une brique interne du store. La sérialisation des objets depuis Python passera par une API de sauvegarde / relecture de la session, encore à écrire.

## Definition of Done par objet

Un objet n'est considéré comme terminé que lorsque les six points suivants sont verts :

1. Struct Rust vivant dans le Store (`Handle<T>` typé).
2. `Debug` (structure) + `Display` (résumé).
3. Tests unitaires Rust + doctests sur tout l'API public.
4. Binding PyO3 (`__repr__` / `__str__`).
5. Tests Python (pytest).
6. Chapitre de cette documentation.

## Dépendances approuvées

Le socle figé est :

- **toujours lié** — `serde` + `bincode` (persistance), `nalgebra` +
  `nalgebra-sparse` (primitives et stockage creux), `faer` (LU creux du
  solveur), `rayon` (parallélisme), `parking_lot` (verrous du store), `paste`
  (macros d'agrégat) ;
- **optionnel, derrière une feature** — `pyo3` et `pyo3-stub-gen` (binding et
  stub), `plotters` / `winit` / `softbuffer` (visualisation) ;
- **outillage** — `maturin`, `mdbook`, `ruff`, `criterion` (bancs).

Chaque dépendance est **confinée** à un étage et ne fuit pas hors de lui — le
[graphe des dépendances](developper/arborescence.md#graphe-des-dépendances-externes)
dit lequel pour chacune. Toute autre dépendance, Rust ou Python, requiert un
accord explicite.
