# Correspondance Rust ↔ Python

Cette page liste, par module, **les structures** (exposées en classes
Python) et **les fonctions libres** (exposées en fonctions de module
Python). Elle matérialise la règle de [Conventions](conventions.md) :

- une structure `containers::…::Foo` est exposée sous le **même nom**
  `pyrucast.Foo` (le wrapper PyO3 interne `PyFoo` est masqué) ;
- une fonction libre `ops::<thème>::f` est exposée **à plat** comme
  `pyrucast.f` (le rangement par thème reste une organisation du code
  Rust, pas une hiérarchie de modules Python) ;
- une surcharge d'opérateur Rust devient un dunder Python (`Add` →
  `__add__`, `Index` → `__getitem__`, …) ;
- un constructeur nommé Rust devient un `classmethod` / constructeur
  Python.

> Source de vérité : le `#[pymodule]` de `src/lib.rs` (enregistrement des
> classes et fonctions) et le stub `pyrucast.pyi` (signatures typées).
> Cette page en est un instantané, à régénérer à la main si l'API bouge.

## Structures ↔ classes

Le nom de la classe Python est identique au nom de la structure Rust.

| Module Rust | Structure Rust | Classe Python | Chapitre |
|---|---|---|---|
| `containers::mesh::configuration` | `Configuration` | `pyrucast.Configuration` | [Configuration](configuration.md) |
| `containers::mesh::node` | `Node` | `pyrucast.Node` | [Configuration](configuration.md) |
| `containers::mesh` | `SubMesh` | `pyrucast.SubMesh` *(vue, via `mesh[i]`)* | [Maillage](mesh.md) |
| `containers::mesh` | `Mesh` | `pyrucast.Mesh` | [Maillage](mesh.md) |
| `containers::mesh::cell` | `Cell` | `pyrucast.Cell` | [Maillage](mesh.md) |
| `containers::finite_element_space` | `SubFiniteElementSpace` | `pyrucast.SubFiniteElementSpace` | [Espace EF](fe-space.md) |
| `containers::finite_element_space` | `FiniteElementSpace` | `pyrucast.FiniteElementSpace` | [Espace EF](fe-space.md) |
| `containers::finite_element_space::element` | `Element` | `pyrucast.Element` | [Espace EF](fe-space.md) |
| `containers::node_field` | `NodeField` | `pyrucast.NodeField` | [Champ aux nœuds](node-field.md) |
| `containers::element_field` | `SubElementField` | `pyrucast.SubElementField` *(vue, via `element_field[i]`)* | [Champ aux points de Gauss](element-field.md) |
| `containers::element_field` | `ElementField` | `pyrucast.ElementField` | [Champ aux points de Gauss](element-field.md) |
| `containers::matrix` | `SubMatrix` | `pyrucast.SubMatrix` *(vue, via `matrix[i]`)* | [Matrice creuse](matrix.md) |
| `containers::matrix` | `Matrix` | `pyrucast.Matrix` | [Matrice creuse](matrix.md) |
| `containers::model` | `SubModel` | `pyrucast.SubModel` *(vue, via `model[i]`)* | [Modèle physique](model.md) |
| `containers::model` | `Model` | `pyrucast.Model` | [Modèle physique](model.md) |

Quelques types Rust **ne sont pas** exposés en classes Python : ce sont des
détails d'implémentation (`Physics`, l'énum des physiques sous `SubModel` ;
`DofOrdering`, l'ordonnancement des DOFs d'une `SubMatrix`).

Les sous-objets `Sub*` (`SubMesh`, `SubFiniteElementSpace`, `SubElementField`,
`SubMatrix`, `SubModel`) ne se **construisent pas** directement côté Python :
ce sont des **vues** obtenues par indexation de leur parent (`parent[i]`). On
construit toujours au niveau parent — `Mesh(config, type)`,
`FiniteElementSpace(mesh)`, `ElementField(fes, comps)`,
`Model.heat_conduction(fes)`, `Matrix.block(...)` — et on compose plusieurs
zones avec `+` (merge). Voir la règle « Agrégats : un ou plusieurs » de
`CONVENTIONS.md`.

## Fonctions ↔ fonctions

Toutes les fonctions `ops` prennent leurs conteneurs **par référence** et
renvoient `Result<T>` (converti en exception Python `RuntimeError`). Les
signatures ci-dessous omettent le `&` et le `Result` pour la lisibilité.

### `ops::mesher` — construction et transformation de maillages

| Rust (`ops::mesher::…`) | Python (`pyrucast.…`) |
|---|---|
| `from_live_nodes(config: Handle<Configuration>) -> Mesh` | `from_live_nodes(config) -> Mesh` |
| `line_seg2(a: &Node, b: &Node, n_elems: usize) -> Mesh` | `line_seg2(a, b, n_elems) -> Mesh` |
| `circle_seg2(center: &Node, normal: &[f64], radius: f64, n_elems: usize) -> Mesh` | `circle_seg2(center, normal, radius, n_elems) -> Mesh` |
| `sweep_qua4(mesh_a: &Mesh, mesh_b: &Mesh, n_layers: usize) -> Mesh` | `sweep_qua4(mesh_a, mesh_b, n_layers) -> Mesh` |
| `extrude(mesh: &Mesh, direction: &[f64], n_layers: usize) -> Mesh` | `extrude(mesh, direction, n_layers) -> Mesh` |
| `fill_surface(contour: &Mesh, et: ElementType, refinement: Option<…>) -> Mesh` | `fill_surface(contour, element_type, max_edge_length=None, min_angle_deg=None) -> Mesh` |
| `to_poi1(mesh: &Mesh) -> Mesh` | `to_poi1(mesh) -> Mesh` |
| `consolidate(mesh: &Mesh) -> Mesh` | `consolidate(mesh) -> Mesh` |

### `ops::field` — opérateurs sur les champs aux nœuds

| Rust (`ops::field::…`) | Python (`pyrucast.…`) |
|---|---|
| `coordinates(mesh: &Mesh, components: Option<Vec<String>>) -> NodeField` | `coordinates(mesh, components=None) -> NodeField` |
| `restrict(field: &NodeField, mesh: &Mesh) -> NodeField` | `restrict(field, mesh) -> NodeField` |
| `merge(a: &NodeField, b: &NodeField) -> NodeField` | `merge(a, b) -> NodeField` |

> L'addition champ + champ et champ + scalaire passe par l'**opérateur**
> `+` (Rust `impl Add` → Python `__add__`), pas par une fonction `ops`.

### `ops::build` — construction de champs matériau

| Rust (`ops::build::…`) | Python (`pyrucast.…`) |
|---|---|
| `sub_material_field(sub: &SubModel, pairs: &[(&str, f64)]) -> SubElementField` | `sub_material_field(sub_model, components_and_values) -> SubElementField` |
| `material_field(model: &Model, pairs: &[(&str, f64)]) -> ElementField` | `material_field(model, components_and_values) -> ElementField` |
| `material_field_per_sub_model(model: &Model, per: &[&[(&str, f64)]]) -> ElementField` | `material_field_per_sub_model(model, components_and_values_per_sub_model) -> ElementField` |

### `ops::assemble` — assemblage des matrices

| Rust (`ops::assemble::…`) | Python (`pyrucast.…`) |
|---|---|
| `stiffness(model: &Model, materials: &ElementField) -> Matrix` | `stiffness(model, materials) -> Matrix` |
| `mass(model: &Model) -> Matrix` | `mass(model) -> Matrix` |

### `ops::solver` — résolution

| Rust (`ops::solver::lu::…`) | Python (`pyrucast.…`) |
|---|---|
| `solve(matrix: &Matrix, rhs: &NodeField) -> NodeField` | `solve(matrix, rhs) -> NodeField` |

### Utilitaires de `store` (swap disque)

| Rust (`store::…`) | Python (`pyrucast.…`) |
|---|---|
| `set_swap_dir(path: PathBuf)` | `set_swap_dir(path) -> None` |
| `swap_dir() -> PathBuf` | `swap_dir() -> Path` |

> `ops::geom` (mesures géométriques) est réservé mais encore vide ; aucune
> fonction exposée pour l'instant.

## Opérateurs (dunders ↔ traits Rust)

**Toutes** les classes implémentent `__repr__` (← `Debug`, vue
structurelle) et `__str__` (← `Display`, vue résumée façon cast3m) — voir
[Conventions](conventions.md). Les autres opérateurs, classe par classe :

### Arithmétique sur les champs

| Classe | Opérateurs Python | Sémantique | Trait Rust |
|---|---|---|---|
| `NodeField` | `f + s`, `f - s`, `f * s`, `f / s` | composante par composante avec un scalaire `s` | `Add`/`Sub`/`Mul`/`Div<f64>` |
| `NodeField` | `a + b` | addition de deux champs (union des supports) | `Add<&NodeField>` |
| `SubElementField` | `f + s`, `f - s`, `f * s`, `f / s` | scalaire, composante par composante | `Add`/`Sub`/`Mul`/`Div<f64>` |

> `f + s` et `a + b` renvoient un **nouveau** champ ; `+= ` n'est pas
> surchargé. Côté Python, `NodeField.__add__` accepte un `float` **ou** un
> `NodeField`.

### Indexation par clé

| Classe | Opérateurs Python | Clé | Backing Rust |
|---|---|---|---|
| `NodeField` | `f[nid, "c"]`, `f[nid, "c"] = v` | `(NodeId, composante)` | `Index`/`IndexMut<(NodeId, &str)>` |
| `SubElementField` | `f[cell, g, "c"]`, `f[cell, g, "c"] = v` | `(maille, point de Gauss, composante)` | méthodes `value` / `set_value` (pas de trait `Index`) |

### Protocole séquence — `len(x)`, `x[i]`, `for _ in x`

| Classe | `len(x)` | `x[i]` → | Backing Rust |
|---|---|---|---|
| `Cell` | nombre de nœuds | `Node` | méthodes |
| `SubMesh` | nombre de mailles | `Cell` | méthodes |
| `Mesh` | nombre de sous-maillages | `SubMesh` | `Aggregate` (macro) |
| `SubFiniteElementSpace` | nombre d'éléments | `Element` | méthodes |
| `FiniteElementSpace` | nombre de sous-espaces | `SubFiniteElementSpace` | `Aggregate` (macro) |
| `ElementField` | nombre de sous-champs | `SubElementField` | `Aggregate` (macro) |
| `Model` | nombre de sous-modèles | `SubModel` | `Aggregate` (macro) |
| `Matrix` | nombre de sous-matrices | — (pas de `[i]`) | `Aggregate` (macro pour `len`) |
| `SubMatrix` | nombre d'entrées | — | méthode `entry_count` |

### Fusion

| Classe | Opérateur Python | Sémantique | Trait Rust |
|---|---|---|---|
| `Mesh` | `a + b` | concaténation des sous-maillages (= `merge`) | `Aggregate::merge` / `Add<&Mesh>` |

> Côté Rust, **tous** les agrégats (`Mesh`, `FiniteElementSpace`,
> `ElementField`, `Matrix`, `Model`) ont `&a + &b` (fusion) via le trait
> `Aggregate`. Seul `Mesh` l'expose pour l'instant comme `__add__` en
> Python.
