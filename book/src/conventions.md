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

Un **constructeur nommé** échappe aux deux : il fabrique son propre type, il
reste donc sur le type (`FiniteElementSpace.lagrange1(mesh)`,
`Matrix.block(...)`). Mais l'exception s'arrête au **pluriel** : une famille
qui grossit — le catalogue de physiques d'un `Model`, 28 entrées — se range
comme toute famille d'opérateurs, dans le module du conteneur produit. Deux
raisons, l'une de principe, l'autre très concrète : en Python une
`classmethod` s'atteint aussi depuis une **instance**, si bien que
`m.heat_conduction(fes)` s'exécute en jetant `m` en silence, et
l'auto-complétion d'un objet se remplit de verbes qui ne s'adressent pas à
lui. D'où `pyrucast.model.heat_conduction(fes)` et
non `Model.heat_conduction(fes)` (rupture du 2026-08-25).

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
| `model` | un `Model` (les déclarations de physique) |
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
**quelconque**, pas avec l'exemple bien nommé :
`stresses.internal_forces_continuum(fespace)` sonne juste, mais c'est le nom de
la variable qui fait le travail — `field.internal_forces_continuum(fespace)`
révèle que le type ne promet rien. Son homonyme `internal_forces`, lui, a bien
une méthode : son sujet est le **modèle**, qui promet ses physiques.

```python
{{#include ../../tests/python/test_doc_conventions.py:chainage}}
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
{{#include ../../tests/python/test_doc_conventions.py:miroir}}
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
{{#include ../../tests/doc_conventions.rs:erreurs}}
```

Python :

```python
{{#include ../../tests/python/test_doc_conventions.py:erreurs}}
```

## Affichage : `Debug` vs `Display`

Chaque objet du modèle implémente deux traits :

- `Debug` — vue **structurelle** : utile pour le développement, exposée en Python via `__repr__`.
- `Display` — vue **résumée** orientée utilisateur EF, façon listing cast3m, exposée en Python via `__str__`.

Le binding PyO3 branche ces deux vues sur les dunder methods Python correspondantes.

Rust :

```rust,ignore
{{#include ../../tests/doc_conventions.rs:affichage}}
```

Python :

```python
{{#include ../../tests/python/test_doc_conventions.py:affichage}}
```

## Sérialisation : un seul mécanisme

Le trait `Portable` (implémenté automatiquement pour tout type `serde::Serialize + DeserializeOwned`) produit un format binaire **portable Linux ↔ Windows**. C'est le socle de la **sauvegarde fichier** : un graphe d'objets écrit dans un conteneur versionné, relu en préservant le partage.

Rust — sérialisation manuelle d'un type quelconque :

```rust,ignore
{{#include ../../tests/doc_conventions.rs:serialisation}}
```

> **Python** : `Portable` n'est pas exposé côté Python — c'est une brique interne. La sérialisation depuis Python passe par [`pyrucast.save` / `pyrucast.load`](sauvegarde.md), qui écrivent un graphe entier plutôt qu'un objet isolé.

## Definition of Done par objet

Un objet n'est considéré comme terminé que lorsque les six points suivants sont verts :

1. Struct Rust adressable par un `Handle<T>` typé.
2. `Debug` (structure) + `Display` (résumé).
3. Tests unitaires Rust, **et un doctest portant un exemple exécutable sur
   chaque item public** — `ignore` proscrit, `no_run` si l'exemple ne peut pas
   tourner.
4. Binding PyO3 (`__repr__` / `__str__`).
5. Tests Python (`tests/python/`), la surface pyo3 comprise.
6. Chapitre de cette documentation, dont **le code est inclus depuis un test ou
   un exemple**, jamais recopié dans la page.

Les points 3, 5 et 6 sont détaillés page
[Documentation et tests](developper/documentation-et-tests.md) : quel type
d'exemple vit où, et qui le vérifie.

## Dépendances approuvées

Le socle figé est :

- **toujours lié** — `serde` + `bincode` (persistance), `nalgebra` +
  `nalgebra-sparse` (primitives et stockage creux), `faer` (LU creux du
  solveur), `rayon` (parallélisme), `parking_lot` (verrous des objets), `paste`
  (macros d'agrégat) ;
- **optionnel, derrière une feature** — `pyo3` et `pyo3-stub-gen` (binding et
  stub), `plotters` / `winit` / `softbuffer` (visualisation) ;
- **outillage** — `maturin`, `mdbook`, `ruff`, `criterion` (bancs).

Chaque dépendance est **confinée** à un étage et ne fuit pas hors de lui — le
[graphe des dépendances](developper/arborescence.md#graphe-des-dépendances-externes)
dit lequel pour chacune. Toute autre dépendance, Rust ou Python, requiert un
accord explicite.
