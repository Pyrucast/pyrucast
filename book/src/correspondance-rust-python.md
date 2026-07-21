# Correspondance Rust ↔ Python

Cette page liste, par module, **les structures** (exposées en classes
Python) et **les fonctions libres** (exposées en fonctions de module
Python). Elle matérialise la règle de [Conventions](conventions.md) :

- une structure `containers::…::Foo` est exposée sous le **même nom**
  `pyrucast.Foo` (le wrapper PyO3 interne `PyFoo` est masqué) ;
- une fonction libre `ops::<thème>::f` est exposée dans le **sous-module du
  thème** : `pyrucast.<thème>.f` (le rangement par thème du code Rust est
  reflété par une hiérarchie de modules Python : `mesher`, `field`,
  `assemble` (dont les forces internes `BSIG`), `behavior`, `solver`,
  `export`, `build`, plus `store`). Seul `pyrucast.consolidate` (dispatch
  mesh/champ) reste au
  top-level, à l'image du niveau racine de `ops` ;
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
| `containers::mesh::coords` | `Coords` | `pyrucast.Coords` | [Coords](coords.md) |
| `containers::mesh::node` | `Node` | `pyrucast.Node` | [Nœud](node.md) |
| `containers::mesh` | `SubMesh` | `pyrucast.SubMesh` *(vue, via `mesh[i]`)* | [Maillage](mesh.md) |
| `containers::mesh` | `Mesh` | `pyrucast.Mesh` | [Maillage](mesh.md) |
| `containers::mesh::cell` | `Cell` | `pyrucast.Cell` | [Maillage](mesh.md) |
| `containers::finite_element_space` | `SubFiniteElementSpace` | `pyrucast.SubFiniteElementSpace` | [Espace EF](fe-space.md) |
| `containers::finite_element_space` | `FiniteElementSpace` | `pyrucast.FiniteElementSpace` | [Espace EF](fe-space.md) |
| `containers::finite_element_space::element` | `Element` | `pyrucast.Element` | [Espace EF](fe-space.md) |
| `containers::node_field` | `SubNodeField` | `pyrucast.SubNodeField` *(vue, via `node_field[i]`)* | [Champ aux nœuds](node-field.md) |
| `containers::node_field` | `NodeField` | `pyrucast.NodeField` | [Champ aux nœuds](node-field.md) |
| `containers::element_field` | `SubElementField` | `pyrucast.SubElementField` *(vue, via `element_field[i]`)* | [Champ aux points de Gauss](element-field.md) |
| `containers::element_field` | `ElementField` | `pyrucast.ElementField` | [Champ aux points de Gauss](element-field.md) |
| `containers::matrix` | `SubMatrix` | `pyrucast.SubMatrix` *(vue, via `matrix[i]`)* | [Matrice creuse](matrix.md) |
| `containers::matrix` | `Matrix` | `pyrucast.Matrix` | [Matrice creuse](matrix.md) |
| `containers::model` | `SubModel` | `pyrucast.SubModel` *(vue, via `model[i]`)* | [Modèle physique](model.md) |
| `containers::model` | `Model` | `pyrucast.Model` | [Modèle physique](model.md) |
| `containers::evolution` | `SubEvolution` | `pyrucast.SubEvolution` *(constructible — voir ci-dessous)* | [Évolution](evolution.md) |
| `containers::evolution` | `Evolution` | `pyrucast.Evolution` | [Évolution](evolution.md) |

Quelques types Rust **ne sont pas** exposés en classes Python : ce sont des
détails d'implémentation (`SubModelKind`, l'énum des physiques sous `SubModel` ;
`DofOrdering`, l'ordonnancement des DOFs d'une `SubMatrix` ; `SubValue`,
`OutOfRange`, `ValueKind` et `Interpolated`, internes à l'`Evolution` — une
valeur tabulée se passe directement en scalaire ou en champ, la politique
hors-plage en chaîne `"error"`/`"clamp"`/`"extrapolate"`).

Les sous-objets `Sub*` (`SubMesh`, `SubFiniteElementSpace`, `SubElementField`,
`SubMatrix`, `SubModel`) ne se **construisent pas** directement côté Python :
ce sont des **vues** obtenues par indexation de leur parent (`parent[i]`). On
construit toujours au niveau parent — `Mesh(coords, type)`,
`FiniteElementSpace(mesh)`, `ElementField(fes, comps)`,
`Model.heat_conduction(fes)`, `Matrix.block(...)` — et on compose plusieurs
zones avec `|` (union — Rust : `union`). Voir la règle « Agrégats : un ou
plusieurs » de `CONVENTIONS.md`.

**Exception : `SubEvolution`.** Seul sous-objet à la fois **vue** (via
`evolution[i]`) **et constructible directement** —
`SubEvolution([(t0, v0), (t1, v1), …])` — car une courbe tabulée n'a pas de
« parent » géométrique qui la définirait. On compose ensuite les courbes par
zone avec `|` (cf. [Évolution](evolution.md)).

## Fonctions ↔ fonctions

Toutes les fonctions `ops` prennent leurs conteneurs **par référence** et
renvoient `Result<T>` (converti en exception Python `RuntimeError`). Les
signatures ci-dessous omettent le `&` et le `Result` pour la lisibilité.

### `ops::mesher` — construction et transformation de maillages

| Rust (`ops::mesher::…`) | Python (`pyrucast.mesher.…`) |
|---|---|
| `from_live_nodes(coords: Handle<Coords>) -> Mesh` | `from_live_nodes(coords) -> Mesh` |
| `SubMesh::poi1_from_nodes(nodes: &[Node]) -> SubMesh` | `poi1_from_nodes(nodes) -> Mesh` |
| `line(a: &Node, b: &Node, n_elems: usize, element_type: ElementType) -> Mesh` | `line(a, b, n_elems, element_type="SEG2") -> Mesh` |
| `circle_seg2(center: &Node, normal: &[f64], radius: f64, n_elems: usize) -> Mesh` | `circle_seg2(center, normal, radius, n_elems) -> Mesh` |
| `sweep(mesh_a: &Mesh, mesh_b: &Mesh, n_layers: usize, element_type: ElementType) -> Mesh` | `sweep(mesh_a, mesh_b, n_layers, element_type="QUA4") -> Mesh` |
| `sweep_solid(mesh_a: &Mesh, mesh_b: &Mesh, n_layers: usize) -> Mesh` | `sweep_solid(mesh_a, mesh_b, n_layers) -> Mesh` |
| `extrude(mesh: &Mesh, direction: &[f64], n_layers: usize) -> Mesh` | `extrude(mesh, direction, n_layers) -> Mesh` |
| `to_quadratic(mesh: &Mesh) -> Mesh` | `to_quadratic(mesh) -> Mesh` |
| `translate(mesh: &Mesh, vector: &[f64]) -> Mesh` | `translate(mesh, vector) -> Mesh` |
| `rotate(mesh: &Mesh, angle: f64, center: &[f64], axis: Option<&[f64]>) -> Mesh` | `rotate(mesh, angle, center, axis=None) -> Mesh` |
| `fill_surface(contour: &Mesh, et: ElementType, refinement: Option<…>) -> Mesh` | `fill_surface(contour, element_type, max_edge_length=None, min_angle_deg=None) -> Mesh` |
| `surface(contour: &Mesh, et: ElementType, size: Option<f64>) -> Mesh` | `surface(contour, element_type, size=None) -> Mesh` |
| `volume(envelope: &Mesh, size: Option<f64>) -> Mesh` | `volume(envelope, size=None) -> Mesh` |
| `contour(mesh: &Mesh) -> Mesh` | `contour(mesh) -> Mesh` |
| `barycenter(mesh: &Mesh) -> Mesh` | `barycenter(mesh) -> Mesh` |
| `to_poi1(mesh: &Mesh) -> Mesh` | `to_poi1(mesh) -> Mesh` |
| `elements_on(mesh: &Mesh, points: &Mesh, strict: bool) -> Mesh` | `elements_on(mesh, points, strict=True) -> Mesh` |
| `merge_nodes(mesh: &Mesh, tol: f64) -> Mesh` | `merge_nodes(mesh, tol) -> Mesh` |
| `read_gmsh(coords: Handle<Coords>, path: &Path) -> Vec<(String, Mesh)>` | `read_gmsh(coords, path) -> dict[str, Mesh]` |
| `read_gmsh_str(coords: Handle<Coords>, text: &str) -> Vec<(String, Mesh)>` | `read_gmsh_str(coords, text) -> dict[str, Mesh]` |
| `consolidate(mesh: &Mesh) -> Mesh` | `pyrucast.consolidate(mesh) -> Mesh` (**top-level** ; dispatch par type, partagé avec `NodeField`) |

### `ops::field` — opérateurs sur les champs

| Rust (`ops::field::…`) | Python (`pyrucast.field.…`) |
|---|---|
| `coordinates(mesh: &Mesh, components: Option<Vec<String>>) -> NodeField` | `coordinates(mesh, components=None) -> NodeField` |
| `set_coordinates(field: &NodeField, components: Option<Vec<String>>) -> ()` | `set_coordinates(field, components=None) -> None` |
| `displace(field: &NodeField, components: Option<Vec<String>>) -> ()` | `displace(field, components=None) -> None` |
| `gradient(field: &NodeField, fespace: &FiniteElementSpace) -> ElementField` | `gradient(field, fespace) -> ElementField` |
| `deformation(u: &NodeField, fespace: &FiniteElementSpace) -> ElementField` | `deformation(u, fespace) -> ElementField` |
| `interp_to_gauss(field: &NodeField, fespace: &FiniteElementSpace) -> ElementField` | `interp_to_gauss(field, fespace) -> ElementField` |
| `thermal_strain(temperature: &ElementField, material: &ElementField, fespace: &FiniteElementSpace, t_ref: f64) -> ElementField` | `thermal_strain(temperature, materials, fespace, t_ref) -> ElementField` |
| `beam_deformation(field: &NodeField, fespace: &FiniteElementSpace) -> ElementField` | `beam_deformation(field, fespace) -> ElementField` |
| `divergence(field: &ElementField) -> NodeField` | `divergence(field) -> NodeField` |
| `integral(field: &NodeField, fespace, component) -> f64` / `integral_element(field: &ElementField, component) -> f64` | `integral(field, component, fespace=None) -> float` (dispatch par type ; `∫ f dΩ`, `fespace` requis pour un `NodeField`) |
| `restrict(field: &NodeField, mesh: &Mesh) -> NodeField` | `restrict(field, mesh) -> NodeField` |
| `restrict_like(field: &NodeField, target: &NodeField) -> NodeField` | `restrict_like(field, target) -> NodeField` |
| `select_nodes(field: &NodeField, band: &Band, …) -> Mesh` / `select_cells(field: &ElementField, …) -> Mesh` | `select(field, ge=None, gt=None, le=None, lt=None, components=None) -> Mesh` (dispatch par type) |
| `mask_nodes(field: &NodeField, band: &Band, …) -> NodeField` / `mask_cells(field: &ElementField, …) -> ElementField` | `mask(field, ge=None, gt=None, le=None, lt=None, components=None) -> field` (dispatch par type ; champ `0/1` de même structure). Sucre : `field >= x` / `> x` / `<= x` / `< x` → masque |
| `Field::filter_components(&self, wanted: &[String]) -> Self` / `SubField::select_components(&self, wanted) -> Self` | `filter_components(field, components) -> field` (dispatch par type ; `components` est un `str` ou une liste — p. ex. `model.primal_vars()` ; `EXCO`) |
| `Field::rename_component(&self, from, to) -> Self` / `SubField::rename_component(&self, from, to) -> Self` | `rename_component(field, old, new) -> field` (dispatch par type ; renommage sans déplacement de valeur) |
| `merge(a: &NodeField, b: &NodeField) -> NodeField` | `merge(a, b) -> NodeField` |
| `consolidate_node(field: &NodeField) -> NodeField` | `pyrucast.consolidate(field) -> NodeField` (**top-level** ; dispatch par type, partagé avec `Mesh`) |
| `consolidate_element(field: &ElementField) -> ElementField` | `pyrucast.consolidate(field) -> ElementField` (**top-level** ; dispatch par type ; fusionne les zones d'une même fespace) |
| `SubField::dot(&self, other) -> f64` / `Field::dot_field(&self, other) -> f64` | `xty(x, y) -> float` (dispatch par type ; produit scalaire **global** de deux champs) |
| `SubField::xtx(&self) -> f64` / `Field::xtx(&self) -> f64` | `xtx(x) -> float` (dispatch par type ; `Σ v²`, norme au carré `XTX`) |
| `SubField::xtx_components(&self, &[&str]) -> Result<f64>` / `Field::xtx_components(&self, &[&str]) -> Result<f64>` | `xtx(x, components=[…]) -> float` (norme au carré restreinte à ces composantes) |
| `SubField::pscal(&self, other) -> Self` / `Field::pscal_field(&self, other) -> Self` | `psca(x, y) -> field` (dispatch par type ; produit scalaire **nœud par nœud**, champ à une composante `"psca"`) |
| `abs` / `sqrt` / `exp` / `log` / `log10` / `cos` / `sin` / `tan` / `sinh` / `cosh` / `tanh` `(field) -> Field` | mêmes noms `pyrucast.field.…(field)` — maths **élément par élément** (style numpy), un champ neuf du même type ; acceptent les quatre saveurs de champ (`NodeField` / `SubNodeField` / `ElementField` / `SubElementField`). Résultats non bornés : `log` de ≤ 0 → `-inf`/`nan` |

> La composition de zones passe par l'**union** (`|` Python / `union` Rust) ;
> l'arithmétique champ + scalaire **et** champ + champ par les **opérateurs**
> `+`, `-`, `*`, `/`, `**` (cf. [Opérateurs](#opérateurs-dunders--traits-rust)).
> Ni l'une ni l'autre n'est une fonction `ops` — `merge(a, b)` est juste un
> alias nommé de `a | b`.

### `ops::build` — construction de champs matériau

| Rust (`ops::build::…`) | Python (`pyrucast.build.…`) |
|---|---|
| `sub_material_field(sub: &SubModel, pairs: &[(&str, f64)]) -> SubElementField` | `sub_material_field(sub_model, components_and_values) -> SubElementField` |
| `material_field(model: &Model, pairs: &[(&str, f64)]) -> ElementField` | `material_field(model, components_and_values) -> ElementField` |
| `material_field_per_sub_model(model: &Model, per: &[&[(&str, f64)]]) -> ElementField` | `material_field_per_sub_model(model, components_and_values_per_sub_model) -> ElementField` |

### `ops::assemble` — assemblage des matrices et forces internes

`internal_forces` (`BSIG`, `∫ Bᵀ σ`) fait partie de la famille assemblage
(`Model` → vecteur nodal, comme `flux`) et vit sous `ops::assemble` /
`pyrucast.assemble`.

| Rust (`ops::assemble::…`) | Python (`pyrucast.assemble.…`) |
|---|---|
| `stiffness(model: &Model, materials: &ElementField) -> Matrix` | `stiffness(model, materials) -> Matrix` |
| `mass(model: &Model, materials: &ElementField) -> Matrix` | `mass(model, materials) -> Matrix` |
| `lump(m: &Matrix) -> Matrix` | `lump(matrix) -> Matrix` |
| `geometric(model: &Model, materials: &ElementField, stress: &ElementField) -> Matrix` | `geometric(model, materials, stress) -> Matrix` |
| `tangent(model: &Model, materials: &ElementField, state: &ElementField) -> Matrix` | `tangent(model, materials, state) -> Matrix` |
| `assemble(k: &mut Matrix) -> Result<()>` | `assemble(matrix) -> None` (mutation en place — (ré)assemble une matrice depuis ses blocs seuls, sans `Model` : chemin de composition, ex. `M/dt + K`) |
| `flux(fespace: &SubFiniteElementSpace, density: FluxDensity, component: &str) -> SubNodeField` | `flux(fespace, density, component) -> NodeField` |
| `internal_forces(model: &Model, stresses: &ElementField) -> NodeField` | `internal_forces(model, stresses) -> NodeField` |
| `internal_forces_continuum(stresses: &ElementField, fespace: &FiniteElementSpace) -> NodeField` | `internal_forces_continuum(stresses, fespace) -> NodeField` |

### `ops::behavior` — intégration du comportement (`COMP`)

| Rust (`ops::behavior::…`) | Python (`pyrucast.behavior.…`) |
|---|---|
| `integrate(model: &Model, deformation: &ElementField, prev: Option<&ElementField>, materials: &ElementField, dt: Option<f64>) -> ElementField` | `integrate_behavior(model, deformation, materials, prev=None, dt=None) -> ElementField` |

### `ops::solver` — résolution

| Rust (`ops::solver::…`) | Python (`pyrucast.solver.…`) |
|---|---|
| `lu::solve(matrix: &Matrix, rhs: &NodeField) -> NodeField` | `solve(matrix, rhs) -> NodeField` |
| `eliminate::solve(model: &Model, matrix: &Matrix, rhs: &NodeField) -> NodeField` | `solve_eliminate(model, matrix, rhs) -> NodeField` |
| `unilateral::solve(model: &Model, matrix: &Matrix, rhs: &NodeField) -> NodeField` | `solve_unilateral(model, matrix, rhs, max_iter=100, tol=1e-10) -> NodeField` |

### `ops::export` — export vers des formats externes

| Rust (`ops::export::…`) | Python (`pyrucast.export.…`) |
|---|---|
| `write_vtk_mesh(mesh: &Mesh, path: &Path)` | `export_vtk(mesh, path) -> None` |
| `write_vtk_node_field(mesh: &Mesh, field: &NodeField, path: &Path)` | `export_vtk(mesh, path, field=node_field) -> None` |
| `write_vtk_element_field(mesh: &Mesh, field: &ElementField, path: &Path)` | `export_vtk(mesh, path, field=element_field) -> None` |

### Utilitaires de `store` (swap disque)

| Rust (`store::…`) | Python (`pyrucast.store.…`) |
|---|---|
| `set_swap_dir(path: PathBuf)` | `set_swap_dir(path) -> None` |
| `swap_dir() -> PathBuf` | `swap_dir() -> Path` |

> `ops::geom` héberge `locate_points` (mapping iso-paramétrique inverse, sous
> le [baignage](contraintes/embedded.md)) et `project_points` (projection au
> point le plus proche sur une surface, sous le [contact](contraintes/contact.md)) ;
> ces deux primitives sont internes (API Rust), pas encore exposées en Python.
> `nearest_node(mesh, point)` (nœud le plus proche d'un point) est en revanche
> exposée comme méthode : `mesh.nearest_node([x, y])`.

## Opérateurs (dunders ↔ traits Rust)

**Toutes** les classes implémentent `__repr__` (← `Debug`, vue
structurelle) et `__str__` (← `Display`, vue résumée façon cast3m) — voir
[Conventions](conventions.md). Les autres opérateurs, classe par classe :

### Arithmétique sur les champs

L'arithmétique (`f + s`, `f + g`, …) existe **au niveau zone et au niveau
agrégat** ; les opérations par composante et entre champs sont portées par les
traits [`SubField` / `Field`](field.md). Les dunders `+`, `-`, `*`, `/`, `**`
**dispatchent selon l'opérande droite** : un `float` déclenche l'arithmétique
scalaire, un champ du même type l'arithmétique **champ + champ** (valeur à
valeur).

| Classe | Opérateurs / méthodes Python | Sémantique | Backing Rust |
|---|---|---|---|
| `SubNodeField` / `SubElementField` | `f + s`, `f - s`, `f * s`, `f / s`, `f ** s` | broadcast scalaire, nouveau champ | `Add`/`Sub`/`Mul`/`Div<f64>`, `map_all` |
| `SubNodeField` / `SubElementField` | `f + g`, `f - g`, `f * g`, `f / g`, `f ** g` | champ + champ **par composante**, union/passthrough (même support), nouveau champ | `SubField::merge_components` |
| `NodeField` / `ElementField` | `f + s`, `f - s`, `f * s`, `f / s`, `f ** s` | broadcast scalaire sur toutes les zones | `Field::combine_scalar` |
| `NodeField` / `ElementField` | `f + g`, `f - g`, `f * g`, `f / g`, `f ** g` | champ + champ **par `(support, composante)`**, union/passthrough | `Field::merge_field` |
| `NodeField` / `ElementField` | `f + sub`, `f - sub`, … (`sub` = sous-champ) | maj **ciblée** de la (des) zone(s) de même support (union/passthrough) | `Field::merge_subfield` |
| zone & agrégat | `add_to_component(c, s)`, `sub_/mul_/div_to_component` | scalaire sur **une** composante, en place | `SubField`/`Field::map_component` |
| zone | `set_uniform(c, v)` | force une composante à `v` | `SubField::set_uniform` |

> `f + s` / `f + g` renvoie un **nouveau** champ ; `+=` n'est pas surchargé. La
> **composition** de zones n'est **pas** sur `+` : c'est l'union `|` (`union`
> en Rust, cf. ci-dessous). L'opérateur `+` est entièrement réservé à
> l'**arithmétique de champ** — scalaire (`f + 1.0`) **et** champ + champ valeur
> à valeur (`f + g` via `merge_components`/`merge_field`) ; p. ex. deux champs
> constants valant 1 s'additionnent en un champ constant valant 2. Pour
> fusionner des zones avec vérification (et non additionner) : `merge(a, b)`
> ≡ `a | b`.

### Facteur scalaire et produit matrice-vecteur sur `Matrix` / `SubMatrix`

`Matrix.__mul__` **dispatche selon l'opérande droite**, comme l'arithmétique de
champ ci-dessus : un `NodeField` déclenche le produit matrice-vecteur
(`mul_field`), un `float` la mise à l'échelle **paresseuse** du facteur (voir
[Matrice creuse](../matrix.md#facteur-scalaire-mulf64--divf64-et-combinaison-de-matrices)).
`/` n'existe que pour le facteur (`Matrix` n'a pas de division matrice-vecteur).
Comme pour les champs, `*`/`/` renvoient une **nouvelle** `Matrix` — jamais de
mutation en place — et ne sont **pas** la composition (`|`, ci-dessous).

| Classe | Opérateurs / méthodes Python | Sémantique | Backing Rust |
|---|---|---|---|
| `Matrix` | `k * field` | produit matrice-vecteur `A·x`, `NodeField` neuf | `Matrix::mul_field`, `Mul<&NodeField>` |
| `Matrix` | `k * s`, `k / s` (`s`: `float`) | facteur scalaire, blocs clonés dans de nouveaux slots (aucune valeur réécrite), `k` inchangée | `Mul`/`Div<f64> for &Matrix` |
| `SubMatrix` | `.factor` (lecture seule) | facteur courant du bloc (`1.0` par défaut) | `SubMatrix::factor` |

### Indexation par clé

| Classe | Opérateurs Python | Clé | Backing Rust |
|---|---|---|---|
| `SubNodeField` | `f[nid, "c"]`, `f[nid, "c"] = v` | `(NodeId, composante)` | `Index`/`IndexMut<(NodeId, &str)>` |
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
| `Evolution` | nombre de sous-évolutions | `SubEvolution` | `Aggregate` (macro) |
| `SubEvolution` | nombre d'échantillons tabulés | — | méthode `__len__` |

### Union `|` (composition d'agrégats)

La composition d'agrégats est l'**union** : côté **Python** elle s'écrit `|`
(comme `set | set`), côté **Rust** ce sont les méthodes nommées `union` /
`union_sub` / `union_subs` (renvoient `Result<…>`). Les sous-objets sont
**partagés** (refcount), jamais copiés ; les contraintes de domaine (même
`Coords` pour `Mesh`, etc.) restent vérifiées.

Sémantique d'union (uniforme pour **tous** les agrégats) :

1. **Déduplication par handle** : un sous-objet dont le `Handle` est déjà
   présent (même slot, cf. `Handle::same_slot`) n'est pas ajouté deux fois.
2. **Finalisation** (`Aggregate::finalize`) : par défaut un no-op ; les
   **champs** la surchargent pour fusionner les zones partageant un même
   support (voir plus bas).

| Python | Rust | Résultat | Sémantique |
|---|---|---|---|
| `agrégat \| agrégat` | `a.union(&b)` | agrégat | union dédupliquée, ordre de 1ʳᵉ apparition |
| `agrégat \| sub` | `a.union_sub(&h)` | agrégat | ajoute un sous-objet (ignoré si déjà présent) |
| `sub \| sub` | `T::union_subs(&a, &b)` | agrégat | union des deux sous-objets |
| `node \| node` | `a.union(&b)` | `Mesh` | maillage POI1 unitaire sur les deux nœuds |
| `mesh \| node` | `m.union_node(&n)` | `Mesh` | ajoute un point (erreur si `Mesh` non unitaire POI1) |

Vaut pour les sept agrégats (`Mesh`, `FiniteElementSpace`, `Model`, `Matrix`,
`NodeField`, `ElementField`, `Evolution`) plus `Node`.

#### Finalisation des champs (fusion par support)

Après l'union par handle, `NodeField` et `ElementField` **fusionnent** les
sous-champs définis sur le **même support** (même `Handle<SubMesh>` pour
`NodeField`, même `Handle<SubFiniteElementSpace>` pour `ElementField`) :

- le sous-champ fusionné porte l'**union des composantes** ;
- une composante définie par plusieurs sous-champs doit y avoir la **même
  valeur** partout (comparaison exacte), sinon `|` lève une erreur ;
- pour `NodeField`, une vérification inter-supports finale impose qu'un nœud
  partagé par des zones de supports différents s'accorde sur toute composante
  commune.

Ces opérations sont aussi exposées en Rust : `ops::field::consolidate_node`
(`NodeField`) et `ops::field::consolidate_element` (`ElementField`).

#### `+` est réservé à l'arithmétique de champ

L'opérateur `+` (et `-`, `*`, `/`) reste l'**arithmétique scalaire** des
sous-champs (`subfield + 2.0` → ajoute la valeur à chaque composante) et a
vocation à porter, à terme, l'**addition réelle de champs** (valeur au nœud
= somme des deux). Il n'est **jamais** utilisé pour composer des agrégats —
c'est `|` qui s'en charge, sans collision.
