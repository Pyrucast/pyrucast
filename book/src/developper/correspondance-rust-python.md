# Correspondance Rust ↔ Python

Cette page liste, par module, **les structures** (exposées en classes
Python) et **les fonctions libres** (exposées en fonctions de module
Python). Elle matérialise la règle de [Conventions](../conventions.md) :

- une structure `containers::…::Foo` est exposée sous le **même nom**
  `pyrucast.Foo` (le wrapper PyO3 interne `PyFoo` est masqué) ;
- une fonction libre `ops::<module>::f` est exposée dans le **sous-module de
  même nom** : `pyrucast.<module>.f`. Le module Rust porte le nom du
  **conteneur produit** (`mesh`, `node_field`, `element_field`, `matrix`,
  `model`, `coords`), ou de l'activité quand il ne produit aucun conteneur
  (`measure`, `geom`, `export`) ; `solver` est l'exception nommée. Seuls trois
  verbes vivent au top-level, faute de conteneur produit : `save` et `load`
  (cf. [`archive`](#archive--sauvegarde-et-relecture-dun-graphe)) et
  `spill_stats` ;
- une surcharge d'opérateur Rust devient un dunder Python (`Add` →
  `__add__`, `Index` → `__getitem__`, …) ;
- un constructeur nommé Rust devient un `classmethod` / constructeur
  Python.

**Les tableaux ci-dessous donnent la forme canonique** — la fonction libre.
Beaucoup de ces opérations sont **aussi** exposées comme méthode de leur
sujet, selon la règle des trois conditions
([Conventions](../conventions.md#le-verbe-exposé-aussi-en-méthode)) : premier
argument sujet, retour conteneur, sens pour toute instance du type. La règle
étant mécanique, la liste n'est pas recopiée ici — elle est **vérifiée par un
test** (`tests/python/test_method_exposure.py`), qui lit le stub et échoue si
une opération éligible perd sa méthode. Ce test porte aussi la liste des
exclusions, chacune avec sa raison.

Deux points à connaître, illustrés plus bas : le nom peut changer entre les
deux formes (`matrix.stiffness(model, mats)` / `model.stiffness_matrix(mats)`,
`element_field.sub_material_field(sub, …)` / `sub_model.material_field(…)`),
et les cinématiques (`deformation`, `beam_deformation`, `shell_deformation`,
`thermal_strain`) n'ont **pas** de méthode : elles exigent des composantes
nommées, elles n'auraient pas de sens sur un champ quelconque.
`internal_forces`, elle, a sa méthode, sur son sujet le modèle :
`model.internal_forces(state, solution, materials)`.

`filter_components` et `rename_component` n'ont **que** la forme méthode
(`f.filter_components(["u_x"])`, `f.rename_component("U", "DX")`) : un seul
conteneur, de petits arguments, une vue dérivée — R1 en fait du vocabulaire du
champ, pas un opérateur.

La **complétude du miroir** est elle aussi vérifiée par un test
(`tests/python/test_mirror_completeness.py`), **dans les deux sens** : aucun
opérateur Rust sans binding Python, et aucune fonction Python sans opérateur
Rust. Les dérogations y vivent avec leur raison.

> Source de vérité : le `#[pymodule]` de `src/lib.rs` (enregistrement des
> classes et fonctions) et le stub `python/pyrucast/_pyrucast/__init__.pyi`
> (signatures typées).
> Cette page en est un instantané, à régénérer à la main si l'API bouge.

## Structures ↔ classes

Le nom de la classe Python est identique au nom de la structure Rust.

| Module Rust | Structure Rust | Classe Python | Chapitre |
|---|---|---|---|
| `coords` | `Coords` | `pyrucast.Coords` | [Coords](../coords.md) |
| `atoms::node` | `Node` | `pyrucast.Node` | [Nœud](../node.md) |
| `containers::mesh` | `SubMesh` | `pyrucast.SubMesh` *(vue, via `mesh[i]`)* | [Maillage](../mesh.md) |
| `containers::mesh` | `Mesh` | `pyrucast.Mesh` | [Maillage](../mesh.md) |
| `atoms::cell` | `Cell` | `pyrucast.Cell` | [Maillage](../mesh.md) |
| `containers::finite_element_space` | `SubFiniteElementSpace` | `pyrucast.SubFiniteElementSpace` | [Espace EF](../fe-space.md) |
| `containers::finite_element_space` | `FiniteElementSpace` | `pyrucast.FiniteElementSpace` | [Espace EF](../fe-space.md) |
| `atoms::element` | `Element` | `pyrucast.Element` | [Espace EF](../fe-space.md) |
| `containers::node_field` | `SubNodeField` | `pyrucast.SubNodeField` *(vue, via `node_field[i]`)* | [Champ aux nœuds](../node-field.md) |
| `containers::node_field` | `NodeField` | `pyrucast.NodeField` | [Champ aux nœuds](../node-field.md) |
| `containers::element_field` | `SubElementField` | `pyrucast.SubElementField` *(vue, via `element_field[i]`)* | [Champ aux points de Gauss](../element-field.md) |
| `containers::element_field` | `ElementField` | `pyrucast.ElementField` | [Champ aux points de Gauss](../element-field.md) |
| `containers::matrix` | `SubMatrix` | `pyrucast.SubMatrix` *(vue, via `matrix[i]`)* | [Matrice creuse](../matrix.md) |
| `containers::matrix` | `Matrix` | `pyrucast.Matrix` | [Matrice creuse](../matrix.md) |
| `containers::model` | `SubModel` | `pyrucast.SubModel` *(vue, via `model[i]`)* | [Modèle physique](../model.md) |
| `containers::model` | `Model` | `pyrucast.Model` | [Modèle physique](../model.md) |
| `containers::evolution` | `SubEvolution` | `pyrucast.SubEvolution` *(constructible — voir ci-dessous)* | [Évolution](../evolution.md) |
| `containers::evolution` | `Evolution` | `pyrucast.Evolution` | [Évolution](../evolution.md) |

Une classe Python n'a pas d'homologue Rust : `pyrucast.Array`, le tableau 1-D
en lecture seule que rendent les fonctions d'échange (`export.to_arrays`…),
lisible sans copie par `numpy.asarray`.

Quelques types Rust **ne sont pas** exposés en classes Python : ce sont des
détails d'implémentation (`SubModelKind`, le trait que chaque physique
implémente derrière `SubModel` ;
`DofOrdering`, l'ordonnancement des DOFs d'une `SubMatrix` ; `SubValue`,
`OutOfRange`, `ValueKind` et `Interpolated`, internes à l'`Evolution` — une
valeur tabulée se passe directement en scalaire ou en champ, la politique
hors-plage en chaîne `"error"`/`"clamp"`/`"extrapolate"`).

Les sous-objets `Sub*` (`SubMesh`, `SubFiniteElementSpace`, `SubNodeField`,
`SubElementField`, `SubMatrix`, `SubModel`) ne se **construisent pas** directement côté Python :
ce sont des **vues** obtenues par indexation de leur parent (`parent[i]`). On
construit toujours au niveau parent — `Mesh(coords, type)`,
`FiniteElementSpace(mesh)`, `ElementField(fes, comps)`, `Matrix.block(...)` —
ou par l'opérateur qui rend ce parent (`model.heat_conduction(fes)`), et on
compose plusieurs
zones avec `|` (union — Rust : `union`). Voir la règle « Agrégats : un ou
plusieurs » de `CONVENTIONS.md`.

**Exception : `SubEvolution`.** Seul sous-objet à la fois **vue** (via
`evolution[i]`) **et constructible directement** —
`SubEvolution([(t0, v0), (t1, v1), …])` — car une courbe tabulée n'a pas de
« parent » géométrique qui la définirait. On compose ensuite les courbes par
zone avec `|` (cf. [Évolution](../evolution.md)).

## Fonctions ↔ fonctions

Les fonctions `ops` prennent leurs conteneurs **par référence** et renvoient
`Result<T>` (converti en exception Python `RuntimeError`), sauf celles qui ne
peuvent pas échouer (`update_contact`, `mask_sub`). Les signatures ci-dessous
omettent souvent le `&` et le `Result` pour la lisibilité.

### `ops::mesh` — construction et transformation de maillages

Les mailleurs prennent `element_type=None` en Python : `"SEG2"` pour `line`,
`circle` et `arc`, `"QUA4"` pour `sweep` et `transfinite`.

| Rust (`ops::mesh::…`) | Python (`pyrucast.mesh.…`) |
|---|---|
| `from_live_nodes(coords: Handle<Coords>) -> Mesh` | `from_live_nodes(coords) -> Mesh` |
| `poi1_from_nodes(nodes: &[Node]) -> Mesh` | `poi1_from_nodes(nodes) -> Mesh` |
| `line(a: &Node, b: &Node, n_elems: usize, element_type: ElementType) -> Mesh` | `line(a, b, n_elems, element_type=None) -> Mesh` |
| `circle(center: &Node, normal: &[f64], radius: f64, n_elems: usize, element_type: ElementType) -> Mesh` | `circle(center, normal, radius, n_elems, element_type=None) -> Mesh` |
| `arc(node_a: &Node, center: &Node, node_b: &Node, n_elems: usize, element_type: ElementType) -> Mesh` | `arc(a, center, b, n_elems, element_type=None) -> Mesh` |
| `sweep(mesh_a: &Mesh, mesh_b: &Mesh, n_layers: usize, element_type: ElementType) -> Mesh` | `sweep(mesh_a, mesh_b, n_layers, element_type=None) -> Mesh` |
| `transfinite(side1: &Mesh, side2: &Mesh, side3: &Mesh, side4: &Mesh, element_type: ElementType) -> Mesh` | `transfinite(side1, side2, side3, side4, element_type=None) -> Mesh` |
| `sweep_solid(mesh_a: &Mesh, mesh_b: &Mesh, n_layers: usize) -> Mesh` | `sweep_solid(mesh_a, mesh_b, n_layers) -> Mesh` |
| `extrude(mesh: &Mesh, direction: &[f64], n_layers: usize) -> Mesh` | `extrude(mesh, direction, n_layers) -> Mesh` |
| `revolve(mesh: &Mesh, angle: f64, n_layers: usize, center: &[f64], axis: Option<&[f64]>) -> Mesh` | `revolve(mesh, angle, n_layers, center, axis=None) -> Mesh` |
| `to_quadratic(mesh: &Mesh) -> Mesh` | `to_quadratic(mesh) -> Mesh` |
| `convert(mesh: &Mesh, target: ElementType) -> Mesh` | `convert(mesh, element_type) -> Mesh` |
| `copy(mesh: &Mesh, new_nodes: bool) -> Mesh` | `copy(mesh, new_nodes=True) -> Mesh` |
| `translate(mesh: &Mesh, vector: &[f64]) -> Mesh` | `translate(mesh, vector) -> Mesh` |
| `rotate(mesh: &Mesh, angle: f64, center: &[f64], axis: Option<&[f64]>) -> Mesh` | `rotate(mesh, angle, center, axis=None) -> Mesh` |
| `symmetry_point(mesh: &Mesh, center: &[f64]) -> Mesh` | `symmetry_point(mesh, center) -> Mesh` |
| `symmetry_line(mesh: &Mesh, a: &[f64], b: &[f64]) -> Mesh` | `symmetry_line(mesh, a, b) -> Mesh` |
| `symmetry_plane(mesh: &Mesh, a: &[f64], b: &[f64], c: &[f64]) -> Mesh` | `symmetry_plane(mesh, a, b, c) -> Mesh` |
| `triangulate_surface(contour: &Mesh, element_type: ElementType, target_size: Option<f64>) -> Mesh` | `triangulate_surface(contour, element_type, size=None) -> Mesh` |
| `pave_surface(contour: &Mesh, element_type: ElementType, target_size: Option<f64>, prefer_quads: bool, relax: FrontRelax) -> Mesh` | `pave_surface(contour, element_type, size=None, prefer_quads=False, relax=None) -> Mesh` |
| `grid_surface(contour: &Mesh, element_type: ElementType, target_size: Option<f64>, band: usize, prefer_quads: bool, relax: FrontRelax) -> Mesh` | `grid_surface(contour, element_type, size=None, band=0, prefer_quads=False, relax=None) -> Mesh` |
| `grid_surface2(contour: &Mesh, element_type: ElementType, target_size: Option<f64>, band: usize, prefer_quads: bool, relax: FrontRelax) -> Mesh` | `grid_surface2(contour, element_type, size=None, band=0, prefer_quads=False, relax=None) -> Mesh` |
| `triangulate_volume(envelope: &Mesh, target_size: Option<f64>, allow_surface_nodes: bool) -> Mesh` | `triangulate_volume(envelope, size=None, allow_surface_nodes=False) -> Mesh` |
| `pave_volume(envelope: &Mesh, layers: usize, thickness: Option<f64>, core_size: Option<f64>) -> Mesh` | `pave_volume(envelope, layers=1, thickness=None, size=None) -> Mesh` |
| `border(mesh: &Mesh, angle_deg: Option<f64>) -> Mesh` | `border(mesh, angle_deg=None) -> Mesh` |
| `skin(mesh: &Mesh, angle_deg: Option<f64>) -> Mesh` | `skin(mesh, angle_deg=None) -> Mesh` |
| `orient(mesh: &Mesh) -> Mesh` | `orient(mesh) -> Mesh` |
| `invert(mesh: &Mesh) -> Mesh` | `invert(mesh) -> Mesh` |
| `chain(mesh: &Mesh) -> Mesh` | `chain(mesh) -> Mesh` |
| `barycenter(mesh: &Mesh) -> Mesh` | `barycenter(mesh) -> Mesh` |
| `to_poi1(mesh: &Mesh) -> Mesh` | `to_poi1(mesh) -> Mesh` |
| `elements_on(mesh: &Mesh, points: &Mesh, strict: bool) -> Mesh` | `elements_on(mesh, points, strict=True) -> Mesh` |
| `points_in_sphere(mesh: &Mesh, center: &[f64], radius: f64, tol: Option<f64>) -> Mesh` | `points_in_sphere(mesh, center, radius, tol=None) -> Mesh` |
| `points_on_sphere(mesh: &Mesh, center: &[f64], radius: f64, tol: Option<f64>) -> Mesh` | `points_on_sphere(mesh, center, radius, tol=None) -> Mesh` |
| `points_on_plane(mesh: &Mesh, origin: &[f64], normal: &[f64], tol: Option<f64>) -> Mesh` | `points_on_plane(mesh, origin, normal, tol=None) -> Mesh` |
| `points_below_plane(mesh: &Mesh, origin: &[f64], normal: &[f64], tol: Option<f64>) -> Mesh` | `points_below_plane(mesh, origin, normal, tol=None) -> Mesh` |
| `points_on_line(mesh: &Mesh, a: &[f64], b: &[f64], tol: Option<f64>) -> Mesh` | `points_on_line(mesh, a, b, tol=None) -> Mesh` |
| `points_in_cylinder(mesh: &Mesh, base: &[f64], top: &[f64], radius: f64, tol: Option<f64>) -> Mesh` | `points_in_cylinder(mesh, base, top, radius, tol=None) -> Mesh` |
| `points_on_cylinder(mesh: &Mesh, base: &[f64], top: &[f64], radius: f64, tol: Option<f64>) -> Mesh` | `points_on_cylinder(mesh, base, top, radius, tol=None) -> Mesh` |
| `points_in_cone(mesh: &Mesh, base: &[f64], top: &[f64], base_radius: f64, top_radius: f64, tol: Option<f64>) -> Mesh` | `points_in_cone(mesh, base, top, base_radius, top_radius=0.0, tol=None) -> Mesh` |
| `points_on_cone(mesh: &Mesh, base: &[f64], top: &[f64], base_radius: f64, top_radius: f64, tol: Option<f64>) -> Mesh` | `points_on_cone(mesh, base, top, base_radius, top_radius=0.0, tol=None) -> Mesh` |
| `points_in_torus(mesh: &Mesh, center: &[f64], axis: &[f64], major_radius: f64, minor_radius: f64, tol: Option<f64>) -> Mesh` | `points_in_torus(mesh, center, axis, major_radius, minor_radius, tol=None) -> Mesh` |
| `points_on_torus(mesh: &Mesh, center: &[f64], axis: &[f64], major_radius: f64, minor_radius: f64, tol: Option<f64>) -> Mesh` | `points_on_torus(mesh, center, axis, major_radius, minor_radius, tol=None) -> Mesh` |
| `merge_nodes(mesh: &Mesh, tol: f64, in_place: bool) -> Mesh` | `merge_nodes(mesh, tol, in_place=False) -> Mesh` |
| `cleanup(mesh: &Mesh) -> Mesh` | `cleanup(mesh) -> Mesh` |
| `merge_triangles(mesh: &Mesh) -> Mesh` | `merge_triangles(mesh) -> Mesh` |
| `regularize(mesh: &Mesh, sweeps: usize, angular: bool, in_place: bool) -> Mesh` | `regularize(mesh, sweeps=20, angular=True, in_place=False) -> Mesh` |
| `read_gmsh(coords: Handle<Coords>, path: &Path) -> Vec<(String, Mesh)>` | `read_gmsh(coords, path) -> dict[str, Mesh]` |
| `read_gmsh_str(coords: Handle<Coords>, text: &str) -> Vec<(String, Mesh)>` | `read_gmsh_str(coords, text) -> dict[str, Mesh]` |
| `from_arrays(coords: Handle<Coords>, node_tags: &[T], node_coords: &[f64], blocks: &[CellBlock<T>], node_values: &[NodeValues<T>], cell_values: &[CellValues<T>], order: NodeOrder) -> Imported` | `from_arrays(coords, node_tags, node_coords, blocks, *, node_fields=[], cell_fields=[], order="pyrucast") -> (dict[str, Mesh], list[NodeField], list[ElementField])` |
| `element_type_from_gmsh(code: u32) -> ElementType` | `element_type_from_gmsh(code) -> str` |
| `gauss_to_external(et: ElementType, order: NodeOrder, ext_ref_nodes: &[f64]) -> (Vec<f64>, Vec<f64>)` | `gauss_to_external(element_type, ref_nodes, order) -> (list, list)` |
| `match_gauss(et: ElementType, order: NodeOrder, ext_ref_nodes: &[f64], ext_xi: &[f64], ext_weights: &[f64]) -> Vec<usize>` | `match_gauss(element_type, ref_nodes, xi, weights, order) -> list[int]` |
| — (exige l'interpréteur) | `from_gmsh(coords, *, dim=-1, tag=-1, views=True) -> (dict[str, Mesh], dict)` |
| — (exige l'interpréteur) | `from_medcoupling(coords, source, *, mesh_name=None) -> (dict[str, Mesh], dict)` |
| `consolidate(mesh: &Mesh) -> Mesh` | `consolidate(mesh) -> Mesh` |
| `select_nodes(field: &NodeField, band: &Band, …) -> Mesh` / `select_cells(field: &ElementField, …) -> Mesh`, et pour une zone `select_sub_nodes` / `select_sub_cells` | `select(field, ge=None, gt=None, le=None, lt=None, components=None) -> Mesh` (dispatch par type ; part d'un champ mais rend un maillage, d'où son rangement ici) |

Les mailleurs longs (`triangulate_surface`, `pave_surface`, `grid_surface`,
`grid_surface2`, `triangulate_volume`, `pave_volume`) ont en Rust une variante
`*_cancellable` qui reçoit un jeton d'interruption ; en Python, l'interruption
(Ctrl+C) est câblée dans la fonction nominale.

### `ops::node_field` — opérateurs produisant un champ aux nœuds

| Rust (`ops::node_field::…`) | Python (`pyrucast.node_field.…`) |
|---|---|
| `positions(mesh: &Mesh, components: Option<Vec<String>>) -> NodeField` | `positions(mesh, components=None) -> NodeField` |
| `divergence(field: &ElementField, prefix: &str) -> NodeField` | `divergence(field, prefix) -> NodeField` |
| `restrict(field: &NodeField, mesh: &Mesh) -> NodeField` | `restrict(field, mesh) -> NodeField` |
| `project(field: &NodeField, mesh: &Mesh, target: &Mesh, tol: f64) -> NodeField` | `project(field, mesh, target, tol=1e-6) -> NodeField`, aussi méthode `field.project(mesh, target, tol=1e-6)` (interpolé par les mailles de `mesh` aux nœuds de `target`, 0 hors de `mesh`) |
| `restrict_like(field: &NodeField, target: &NodeField) -> NodeField` | `restrict_like(field, target) -> NodeField` |
| `merge(a: &NodeField, b: &NodeField) -> NodeField` | `merge(a, b) -> NodeField` |
| `consolidate(field: &NodeField) -> NodeField` | `consolidate(field) -> NodeField` |
| `mask(field: &NodeField, band: &Band, …) -> NodeField` / `mask_sub(sub: &SubNodeField, …) -> SubNodeField` | `mask(field, ge=None, gt=None, le=None, lt=None, components=None) -> NodeField` (champ `0/1` de même structure ; sucre `field >= x`). Accepte aussi un `SubNodeField` |
| `internal_forces(model: &Model, state: &ElementField, solution: &NodeField, materials: &ElementField) -> NodeField` | `internal_forces(model, state, solution, materials) -> NodeField`, aussi `model.internal_forces(state, solution, materials)` |
| `external_forces(model: &Model, materials: &ElementField) -> NodeField` | `external_forces(model, materials) -> NodeField` |
| `sum_to_nodes(field: &ElementField) -> NodeField` | `sum_to_nodes(field) -> NodeField` (champ à un point par maille ; chaque nœud reçoit la somme des mailles qui le touchent) |

`internal_forces` (`BSIG`, `∫ Bᵀ σ`) et `external_forces` sont des **assemblages**,
mais leur résultat est un vecteur nodal et non un opérateur : on se range
par la sortie.

### `ops::coords` — écriture dans le magasin de coordonnées, calcul dans une configuration

| Rust (`ops::coords::…`) | Python (`pyrucast.coords.…`) |
|---|---|
| `set(field: &NodeField, components: Option<Vec<String>>) -> ()` | `set(field, components=None) -> None` |
| `displace(field: &NodeField, components: Option<Vec<String>>) -> ()` | `displace(field, components=None) -> None` |
| `displace_from(field: &NodeField, reference: usize, target: usize, components: Option<Vec<String>>) -> ()` | `displace_from(field, reference, target, components=None) -> None` (pose la configuration `target` en `X + u`, en absolu ; la configuration active ne change pas) |
| `in_config(coords: &Handle<Coords>, config: usize, f: impl FnOnce() -> Result<R>) -> Result<R>` | méthode `Coords.in_config(config) -> InConfig`, gestionnaire de contexte : `with coords.in_config(i):` (le bloc calcule sur la configuration `config`, celle d'avant revient à la sortie, même après une erreur). Chaque langage a sa forme idiomatique : fermeture en Rust, `with` en Python |

### `ops::element_field` — opérateurs produisant un champ aux points de Gauss

| Rust (`ops::element_field::…`) | Python (`pyrucast.element_field.…`) |
|---|---|
| `gradient(field: &NodeField, fespace: &FiniteElementSpace) -> ElementField` | `gradient(field, fespace) -> ElementField` |
| `deformation(u: &NodeField, fespace: &FiniteElementSpace, measure: StrainMeasure) -> ElementField` | `deformation(u, fespace, measure="linear") -> ElementField` (`StrainMeasure` passe par son nom : `"linear"`, `"gradient"`, `"green_lagrange"`, `"right_cauchy_green"`, `"left_cauchy_green"`, `"hencky"`, `"euler_almansi"`, `"jacobian"`) |
| `interp_to_gauss(field: &NodeField, fespace: &FiniteElementSpace) -> ElementField` | `interp_to_gauss(field, fespace) -> ElementField` |
| `thermal_strain(temperature: &ElementField, material: &ElementField, fespace: &FiniteElementSpace, t_ref: f64, measure: StrainMeasure) -> ElementField` | `thermal_strain(temperature, materials, fespace, t_ref, measure="linear") -> ElementField` (la dilatation exacte dans la mesure de la loi) |
| `strain(model: &Model, solution: &NodeField, materials: &ElementField) -> ElementField` | `strain(model, solution, materials) -> ElementField` (`EPSI` : l'entrée de chaque loi ; aussi `model.strain(…)`) |
| `elastic_stress(model: &Model, strain: &ElementField, materials: &ElementField) -> ElementField` | `elastic_stress(model, strain, materials) -> ElementField` (`σ = D:ε` ; aussi `model.elastic_stress(…)`) |
| `elastic_strain(model: &Model, stress: &ElementField, materials: &ElementField) -> ElementField` | `elastic_strain(model, stress, materials) -> ElementField` (`ε = D⁻¹:σ` ; aussi `model.elastic_strain(…)`) |
| `stress_measure(stress: &ElementField, gradient: &ElementField, target: StressMeasure) -> ElementField` | `stress_measure(stress, gradient, target) -> ElementField` (`StressMeasure` par son nom : `"cauchy"`, `"kirchhoff"`, `"pk1"`, `"pk2"`, `"hencky"` ; pas de méthode : elle exige une contrainte et un gradient nommés) |
| `rotate_frame(field: &ElementField, gradient: &ElementField) -> ElementField` | `rotate_frame(field, gradient) -> ElementField` (axes `V1*`, `V2*` tournés par la rotation `R` de `F = R U` ; pas de méthode : elle exige des axes et un gradient nommés) |
| `cell_integral(field: &ElementField, target: &FiniteElementSpace) -> ElementField` | `cell_integral(field, target) -> ElementField` (`target` : un point par maille, quadrature `REDUCED`) |
| `shell_deformation(field: &NodeField, fespace: &FiniteElementSpace, formulation: ShellFormulation) -> ElementField` | `shell_deformation(field, fespace, formulation) -> ElementField` |
| `beam_deformation(field: &NodeField, fespace: &FiniteElementSpace, material: &ElementField) -> ElementField` | `beam_deformation(field, fespace, material) -> ElementField` (1-D, plan ou spatial selon le maillage ; le matériau est requis, l'interpolation dépendant de `Φ`) |
| `consolidate(field: &ElementField) -> ElementField` | `consolidate(field) -> ElementField` (fusionne les zones d'une même fespace) |
| `mask(field: &ElementField, band: &Band, …) -> ElementField` / `mask_sub(sub: &SubElementField, …) -> SubElementField` | `mask(field, ge=None, …) -> ElementField` ; accepte aussi un `SubElementField` |
| `sub_material_field(sub: &SubModel, components_and_values: &[(&str, f64)]) -> SubElementField` | `sub_material_field(sub_model, components_and_values) -> SubElementField` |
| `material_field(model: &Model, components_and_values: &[(&str, f64)]) -> ElementField` | `material_field(model, components_and_values) -> ElementField` |
| `material_field_per_sub_model(model: &Model, per_sub_model: &[&[(&str, f64)]]) -> ElementField` | `material_field_per_sub_model(model, components_and_values_per_sub_model) -> ElementField` |
| `behavior::integrate(model, deformation, prev, materials, dt) -> ElementField` | `integrate_behavior(model, deformation, materials, prev=None, dt=None) -> ElementField` (`COMP`) |

### `ops::measure` — réductions à un nombre

| Rust (`ops::measure::…`) | Python (`pyrucast.measure.…`) |
|---|---|
| `integral(field: &NodeField, fespace, component) -> f64` / `integral_element(field: &ElementField, component) -> f64` | `integral(field, component, fespace=None) -> float` (dispatch par type ; `∫ f dΩ`, `fespace` requis pour un `NodeField`) |
| `SubField::dot(&self, other) -> f64` / `Field::dot_field(&self, other) -> f64` | `xty(x, y) -> float` (dispatch par type ; produit scalaire **global** de deux champs) |
| `SubField::xtx(&self) -> f64` / `Field::xtx(&self) -> f64` | `xtx(x) -> float` (dispatch par type ; `Σ v²`, norme au carré `XTX`) |
| `SubField::xtx_components(&self, &[&str]) -> Result<f64>` | `xtx(x, components=[…]) -> float` (norme au carré restreinte à ces composantes) |

### `ops::field` — opérateurs génériques

Leur produit est un conteneur — toujours — mais **pas un conteneur
déterminé** : il dépend de l'argument. La règle « un module par conteneur
produit » ne désigne donc pas *un* module, et ils se rangent par domaine.

| Rust (`ops::field::…`) | Python (`pyrucast.field.…`) |
|---|---|
| `psca<T: Pscal>(x: &T, y: &T) -> T` | `psca(x, y) -> field` (produit scalaire **nœud par nœud**, champ à une composante `"psca"`). Deux conteneurs en pairs et opération symétrique : fonction libre **seule**, pas de méthode |
| `abs` / `sqrt` / `exp` / `log` / `log10` / `cos` / `sin` / `tan` / `sinh` / `cosh` / `tanh` `<T: MapValues>(field: &T) -> T` | mêmes noms `pyrucast.field.…(field)` — maths **élément par élément** (style numpy), un champ neuf du même type ; acceptent les quatre saveurs de champ (`NodeField` / `SubNodeField` / `ElementField` / `SubElementField`). Résultats non bornés : `log` de ≤ 0 → `-inf`/`nan` |

> La composition de zones passe par l'**union** (`|` Python / `union` Rust) ;
> l'arithmétique champ + scalaire **et** champ + champ par les **opérateurs**
> `+`, `-`, `*`, `/`, `**` (cf. [Opérateurs](#opérateurs-dunders--traits-rust)).
> Ni l'une ni l'autre n'est une fonction `ops` — `merge(a, b)` est juste un
> alias nommé de `a | b`.

### `ops::model` — déclaration des physiques

Chaque opérateur rend un `Model` couvrant **tout** le support reçu (une zone
par sous-espace) ; on compose les physiques hétérogènes avec `|`. Aucun n'a de
forme méthode : le premier argument est le support que le modèle recouvre, pas
un sujet qu'on transforme. Seul `update_contact`, qui transforme un modèle, en
a une.

| Rust (`ops::model::…`) | Python (`pyrucast.model.…`) |
|---|---|
| `heat_conduction(fespace: &FiniteElementSpace, symmetry: MaterialSymmetry) -> Model` | `heat_conduction(fespace, symmetry="isotropic") -> Model` |
| `fick(fespace: &FiniteElementSpace, species: &str, symmetry: MaterialSymmetry) -> Model` | `fick(fespace, species, symmetry="isotropic") -> Model` |
| `radiation(fespace: &FiniteElementSpace, target: &Model) -> Model` | `radiation(fespace, target) -> Model` |
| `flux(fespace: &FiniteElementSpace, target: &Model, dual: &str) -> Model` | `flux(fespace, target, dual) -> Model` (`dual="n"` : le mode normal, une pression `phi_n` le long de la normale de la maille) |
| `boundary_transfer(fespace: &FiniteElementSpace, target: &Model, variables: Vec<String>) -> Model` | `boundary_transfer(fespace, target, variables) -> Model` — la duale de chaque variable est lue dans `target` |
| `interface_transfer(side_a: &FiniteElementSpace, side_b: &FiniteElementSpace, target: &Model, variables: Vec<String>, tol: f64) -> Model` | `interface_transfer(side_a, side_b, target, variables, tol=1e-9) -> Model` — `tol` est une **distance**, dans l'unité du maillage ; défaut `interface_transfer::DEFAULT_TOL` |
| `truss(fespace: &FiniteElementSpace, strain_measure: StrainMeasure) -> Model` | `truss(fespace, strain_measure="linear") -> Model` |
| `elasticity(fespace: &FiniteElementSpace, kinematics: Kinematics, symmetry: MaterialSymmetry, strain_measure: StrainMeasure) -> Model` | `elasticity(fespace, kinematics, symmetry="isotropic", strain_measure="linear") -> Model` |
| `plasticity_perfect(fespace: &FiniteElementSpace, kinematics: Kinematics, strain_measure: StrainMeasure) -> Model` | `plasticity_perfect(fespace, kinematics, strain_measure="linear") -> Model` |
| `plasticity(fespace: &FiniteElementSpace, kinematics: Kinematics, law: PlasticLaw, strain_measure: StrainMeasure) -> Model` | une fonction par loi : `plasticity_isotropic`, `drucker_prager`, `ottosen`, `creep_norton`, `creep_blackburn`, `creep_lemaitre`, `viscoplasticity_chaboche`, `viscoplasticity_lemaitre_chaboche`, `gurson` — toutes `(fespace, kinematics, strain_measure="linear") -> Model`, qui existent aussi en Rust, `(fespace, kinematics, strain_measure)` |
| `mazars(fespace: &FiniteElementSpace, kinematics: Kinematics, strain_measure: StrainMeasure) -> Model` | `mazars(fespace, kinematics, strain_measure="linear") -> Model` |
| `damage(fespace: &FiniteElementSpace, kinematics: Kinematics, law: DamageLaw, strain_measure: StrainMeasure) -> Model` | une fonction par loi : `damage_tc`, `damage_sic_sic` — `(fespace, kinematics, strain_measure="linear") -> Model`, qui existent aussi en Rust, `(fespace, kinematics, strain_measure)` |
| `bernoulli(fespace: &FiniteElementSpace) -> Model` | `bernoulli(fespace) -> Model` |
| `timoshenko(fespace: &FiniteElementSpace) -> Model` | `timoshenko(fespace) -> Model` |
| `shell(fespace: &FiniteElementSpace, formulation: ShellFormulation) -> Model` | `shell(fespace, formulation) -> Model` (`"thick"` ou `"kirchhoff"`) |
| `dirichlet(imposed_mesh: &Mesh, target: &Model, variable: &str, sense: RelationSense) -> Model` | `dirichlet(imposed_mesh, target, variable, sense="=") -> Model` |
| `mpc(terms: &[(&Mesh, &str, f64)], target: &Model, sense: RelationSense) -> Model` | `mpc(terms, target, sense="=") -> Model` — `terms` : liste de `(mesh, variable, coefficient)` |
| `embedded(immersed: &Mesh, host: &Mesh, target: &Model, variables: Vec<String>, tol: f64) -> Model` | `embedded(immersed, host, target, variables, tol=1e-6) -> Model` — `tol` est un jeu dans l'élément de **référence**, sans dimension ; défaut `embedded::DEFAULT_TOL` |
| `contact(slave: &Mesh, master: &Mesh, target: &Model, variables: Vec<String>) -> Model` | `contact(slave, master, target, variables) -> Model` |
| `update_contact(model: &Model, displacement: &NodeField) -> Model` | `update_contact(model, displacement) -> Model`, et la méthode `Model.update_contact(displacement)` |

**Même ordre partout.** Le support d'abord — l'espace EF balayé, ou les
maillages qui portent une contrainte, celui qui porte les relations en tête
(`imposed_mesh`, `immersed`, `slave`, les termes d'une MPC) —, puis le modèle
`target` auquel on se rattache, puis le reste. La duale d'une variable n'est
jamais un argument : elle se lit dans `target` (`Model.dual_of`).

**Les noms, en Rust.** Un nom seul s'emprunte (`&str`) ; une liste de noms se
donne (`Vec<String>`), parce que le modèle la garde — comme les constructeurs
de champs et de matrices. Les termes d'une MPC empruntent leurs maillages : ils
restent à l'appelant.

**Un argument, deux idiomes.** Un argument que Python prend en mot-clé avec
valeur par défaut (`symmetry="isotropic"`, `strain_measure="linear"`), Rust le
reçoit explicitement (`MaterialSymmetry::Isotropic`, `StrainMeasure::Linear`) :
Rust n'a pas d'argument par défaut. Même nom, même ordre ; c'est
`physics_operator!` qui écrit les deux faces. La documentation Rust de chaque
fonction dit la valeur que Python prend par défaut, pour qu'un lecteur qui
n'écrit que du Rust sache ce que l'autre obtient sans rien dire.

**Le pli des lois.** Rust nomme la loi par une **enum** (`PlasticLaw`,
`DamageLaw`) dans `plasticity` et `damage` ; Python déplie
les lois en une fonction chacune, que Rust offre aussi. Les dérogations
correspondantes sont enregistrées, avec leur raison, dans
`tests/python/test_mirror_completeness.py`.

### `ops::matrix` — assemblage des matrices

| Rust (`ops::matrix::…`) | Python (`pyrucast.matrix.…`) |
|---|---|
| `stiffness(model: &Model, materials: &ElementField) -> Matrix` | `stiffness(model, materials) -> Matrix` |
| `mass(model: &Model, materials: &ElementField) -> Matrix` | `mass(model, materials) -> Matrix` |
| `lump(m: &Matrix) -> Matrix` | `lump(matrix) -> Matrix` |
| `geometric(model: &Model, materials: &ElementField, stress: &ElementField) -> Matrix` | `geometric(model, materials, stress) -> Matrix` |
| `tangent(model, materials, deformation, prev: Option<&ElementField>, dt: Option<f64>) -> Matrix` | `tangent(model, materials, deformation, prev=None, dt=None) -> Matrix` |

Les cinq reposent sur `assemble_kind(model, materials, kind: MatrixKind, inputs)`,
le moteur commun des assembleurs, qui n'a pas de binding Python.

### `ops::solver` — résolution

| Rust (`ops::solver::…`) | Python (`pyrucast.solver.…`) |
|---|---|
| `solve(matrix: &Matrix, rhs: &NodeField) -> NodeField` | `solve(matrix, rhs, method="lu", relations="lagrange", active_set="schur", cache=True, max_iter=100, tol=1e-10, verbosity="silent") -> NodeField` |
| `solve_with_options(matrix: &Matrix, rhs: &NodeField, options: &SolveOptions) -> NodeField` | les options sont les arguments nommés de `solve` |
| `solve_cancellable` / `solve_cancellable_with_options` (jeton d'interruption en plus) | l'interruption (Ctrl+C) est câblée dans `solve` |

La matrice se suffit : ses lignes de relation portent leur sens (`SubMatrix::row_sense`),
aucun modèle n'est passé à la résolution. L'élimination est une option
(`relations="eliminate"`), et l'ensemble actif s'enclenche de lui-même sur une
matrice qui porte des inégalités.

### `ops::export` — export vers des formats externes

| Rust (`ops::export::…`) | Python (`pyrucast.export.…`) |
|---|---|
| `write_vtk_mesh(mesh: &Mesh, path: &Path, encoding: VtkEncoding)` | `export_vtk(mesh, path, binary=False) -> None` |
| `write_vtk_node_field(mesh: &Mesh, field: &NodeField, path: &Path, encoding: VtkEncoding)` | `export_vtk(mesh, path, field=node_field, binary=False) -> None` |
| `write_vtk_element_field(mesh: &Mesh, field: &ElementField, path: &Path, encoding: VtkEncoding)` | `export_vtk(mesh, path, field=element_field, binary=False) -> None` |
| `write_vtk_series(mesh: &Mesh, evolution: &Evolution, path: &Path, encoding: VtkEncoding) -> Vec<PathBuf>` | `export_vtk(mesh, path, field=evolution, binary=False) -> None` |
| `to_arrays(groups: &[(String, &Mesh)], node_fields: &[&NodeField], element_fields: &[(&ElementField, ElementLayout)], order: NodeOrder, first_tag: i64) -> Exported` | `to_arrays(groups, *, node_fields=[], element_fields=[], gauss=False, order="pyrucast", first_tag=1) -> dict` |
| `vtk_mesh_string` / `vtk_node_field_string` / `vtk_element_field_string` (le fichier VTK dans une `String`) | — |
| — (exige l'interpréteur) | `to_gmsh(meshes, fields=None, *, model_name="pyrucast") -> str` |
| — (exige l'interpréteur) | `to_medcoupling(meshes, fields=None, *, mesh_name="mesh", gauss=True) -> MEDFileData` |

### `spill` — le swap utilisateur

| Rust (`spill::…`) | Python (`pyrucast.…`) |
|---|---|
| `stats() -> SpillStats` (Unix, feature `spill`) | `spill_stats() -> dict` |

Les clés du dictionnaire sont `threshold`, `count`, `largest`, `live` et `peak`,
en octets ; `threshold` vaut `None` quand rien ne déborde. Sur un build qui ne
peut pas déborder (feature absente, ou hors Unix), `spill_stats` existe quand
même : tous les compteurs y valent 0. Voir
[Calculs plus gros que la RAM](../operateurs/solveur.md).

### `archive` — sauvegarde et relecture d'un graphe

Avec `spill_stats`, les seuls verbes du **niveau racine** : ils ne produisent
aucun conteneur déterminé, mais un dictionnaire de ce qu'on leur a donné.

| Rust (`archive::…`) | Python (`pyrucast.…`) |
|---|---|
| `save(path, roots: &[(&str, &dyn ArchiveRoot)])` | `save(path, objects: dict) -> None` |
| `load(path) -> Objects` | `load(path) -> dict` |

À l'écriture les types sont connus du compilateur, d'où la tranche de paires ;
à la relecture non, d'où la table nommée dont on tire chaque objet avec son
type attendu (`objets.mesh("clef")?`). Voir
[Sauvegarde et relecture](../sauvegarde.md).

> `ops::geom` héberge `locate_points` (mapping iso-paramétrique inverse, sous
> le [baignage](../contraintes/embedded.md)) et `project_points` (projection au
> point le plus proche sur une surface, sous le [contact](../contraintes/contact.md)) ;
> ces deux primitives sont internes (API Rust), pas encore exposées en Python —
> la seule dérogation de module entier enregistrée dans le garde-fou de
> complétude (`tests/python/test_mirror_completeness.py`).
> `ops::coloring` (coloriage glouton des mailles) et `ops::scatter` (scatter
> dans un motif CSR) sont la machinerie de l'assemblage parallèle : API Rust
> interne, ni opérateurs ni bindings.
> Le nœud le plus proche, lui, n'est **pas** un opérateur : c'est la méthode
> `mesh.nearest_node([x, y])`, des deux côtés.

## Opérateurs (dunders ↔ traits Rust)

**Toutes** les classes implémentent `__repr__` (← `Debug`, vue
structurelle) et `__str__` (← `Display`, vue résumée façon Cast3M) — voir
[Conventions](../conventions.md). Les autres opérateurs, classe par classe :

### Arithmétique sur les champs

L'arithmétique (`f + s`, `f + g`, …) existe **au niveau zone et au niveau
agrégat** ; les opérations par composante et entre champs sont portées par les
traits [`SubField` / `Field`](../field.md). Les dunders `+`, `-`, `*`, `/`, `**`
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
| `SubElementField` | `set_uniform(c, v)` | force une composante à `v` | `SubField::set_uniform` |

> `f + s` / `f + g` renvoie un **nouveau** champ ; `+=` n'est pas surchargé. La
> **composition** de zones n'est **pas** sur `+` : c'est l'union `|` (`union`
> en Rust, cf. ci-dessous). L'opérateur `+` est entièrement réservé à
> l'**arithmétique de champ** — scalaire (`f + 1.0`) **et** champ + champ valeur
> à valeur (`f + g` via `merge_components`/`merge_field`) ; p. ex. deux champs
> constants valant 1 s'additionnent en un champ constant valant 2. Pour
> fusionner des zones avec vérification (et non additionner) : `merge(a, b)`
> ≡ `a | b`.

### Algèbre de `Matrix` / `SubMatrix`

`Matrix.__mul__` **dispatche selon l'opérande droite**, comme l'arithmétique de
champ ci-dessus : un `NodeField` déclenche le produit matrice-vecteur
(`mul_field`), un `float` la mise à l'échelle **paresseuse** du facteur (voir
[Matrice creuse](../matrix.md#facteur-scalaire-et-somme-de-matrices)).
`/` n'existe que pour le facteur (`Matrix` n'a pas de division matrice-vecteur),
et refuse un diviseur nul (`ZeroDivisionError`) ou non fini (`ValueError`).
Comme pour les champs, tous ces opérateurs renvoient une **nouvelle** `Matrix` —
jamais de mutation en place.

`+` et `-` additionnent deux opérateurs ; ils ne sont **pas** la composition
(`|`, ci-dessous), qui écarte un bloc dont elle tient déjà l'emplacement : `k | k`
vaut `k`, `k + k` vaut `2k`.

| Classe | Opérateurs / méthodes Python | Sémantique | Backing Rust |
|---|---|---|---|
| `Matrix` | `k * field` | produit matrice-vecteur `A·x`, `NodeField` neuf | `Matrix::mul_field`, `Mul<&NodeField>` |
| `Matrix` | `k * s`, `s * k`, `k / s` (`s`: `float`) | facteur scalaire, blocs clonés dans de nouveaux slots (aucune valeur réécrite), CSR assemblée mise à l'échelle avec, `k` inchangée | `Mul`/`Div<f64> for &Matrix`, `Mul<Matrix> for f64` |
| `Matrix` | `-k` | facteur nié, sucre pour `k * -1.0` | `Neg for &Matrix` |
| `Matrix`, `SubMatrix` | `a + b`, `a - b` | somme : blocs des deux opérandes, partagés, **non** dédoublonnés ; résultat non assemblé | `Add`/`Sub` (toutes combinaisons `Matrix`/`SubMatrix`) |
| `SubMatrix` | `b * s`, `s * b`, `b / s`, `-b` | bloc neuf au facteur ajusté (aucune valeur réécrite) | `Mul`/`Div<f64>`/`Neg for &SubMatrix` |
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
| `NodeField` | nombre de sous-champs | `SubNodeField` | `Aggregate` (macro) |
| `ElementField` | nombre de sous-champs | `SubElementField` | `Aggregate` (macro) |
| `Model` | nombre de sous-modèles | `SubModel` | `Aggregate` (macro) |
| `Matrix` | nombre de sous-matrices | `SubMatrix` | `Aggregate` (macro) |
| `SubMatrix` | nombre d'entrées posées une à une (`add_entry`) ; 0 pour un bloc assemblé par éléments | — | méthode `entry_count` |
| `Evolution` | nombre de sous-évolutions | `SubEvolution` | `Aggregate` (macro) |
| `SubEvolution` | nombre d'échantillons tabulés | — | méthode `__len__` |

### Union `|` (composition d'agrégats)

La composition d'agrégats est l'**union** : côté **Python** elle s'écrit `|`
(comme `set | set`), côté **Rust** ce sont les méthodes nommées `union` /
`union_sub` / `union_sub_first` / `union_subs` (renvoient `Result<…>`). Les
sous-objets sont **partagés** (refcount), jamais copiés ; les contraintes de domaine (même
`Coords` pour `Mesh`, etc.) restent vérifiées.

Sémantique d'union (uniforme pour **tous** les agrégats) :

1. **Déduplication par handle** : un sous-objet dont le `Handle` désigne un
   objet déjà présent (cf. `Handle::same_object`) n'est pas ajouté deux fois.
2. **Finalisation** (`Aggregate::finalize`) : par défaut un no-op ; les
   **champs** la surchargent pour fusionner les zones partageant un même
   support (voir plus bas).

| Python | Rust | Résultat | Sémantique |
|---|---|---|---|
| `agrégat \| agrégat` | `a.union(&b)` | agrégat | union dédupliquée, ordre de 1ʳᵉ apparition |
| `agrégat \| sub` | `a.union_sub(&h)` | agrégat | ajoute un sous-objet **en queue** (ignoré si déjà présent) |
| `sub \| agrégat` | `a.union_sub_first(&h)` | agrégat | la même union, sous-objet **en tête** (via `__ror__`) |
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

Ces opérations sont aussi exposées en Rust : `ops::node_field::consolidate`
(`NodeField`) et `ops::element_field::consolidate` (`ElementField`).

#### `+` est réservé à l'arithmétique de champ

L'opérateur `+` (et `-`, `*`, `/`, `**`) est l'**arithmétique de champ** :
scalaire (`subfield + 2.0` ajoute la valeur à chaque composante) et champ +
champ, valeur à valeur (`f + g`, cf. [plus haut](#arithmétique-sur-les-champs)).
Il n'est **jamais** utilisé pour composer des agrégats — c'est `|` qui s'en
charge, sans collision.
