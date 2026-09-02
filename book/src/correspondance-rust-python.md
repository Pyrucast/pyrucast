# Correspondance Rust ↔ Python

Cette page liste, par module, **les structures** (exposées en classes
Python) et **les fonctions libres** (exposées en fonctions de module
Python). Elle matérialise la règle de [Conventions](conventions.md) :

- une structure `containers::…::Foo` est exposée sous le **même nom**
  `pyrucast.Foo` (le wrapper PyO3 interne `PyFoo` est masqué) ;
- une fonction libre `ops::<module>::f` est exposée dans le **sous-module de
  même nom** : `pyrucast.<module>.f`. Le module Rust porte le nom du
  **conteneur produit** (`mesh`, `node_field`, `element_field`, `matrix`,
  `model`, `coords`), ou de l'activité quand il ne produit aucun conteneur
  (`measure`, `geom`, `export`) ; `solver` est l'exception nommée. Le miroir
  est sans exception : aucune fonction libre ne vit au top-level ;
- une surcharge d'opérateur Rust devient un dunder Python (`Add` →
  `__add__`, `Index` → `__getitem__`, …) ;
- un constructeur nommé Rust devient un `classmethod` / constructeur
  Python.

**Les tableaux ci-dessous donnent la forme canonique** — la fonction libre.
Beaucoup de ces opérations sont **aussi** exposées comme méthode de leur
sujet, selon la règle des trois conditions
([Conventions](conventions.md#le-verbe-exposé-aussi-en-méthode)) : premier
argument sujet, retour conteneur, sens pour toute instance du type. La règle
étant mécanique, la liste n'est pas recopiée ici — elle est **vérifiée par un
test** (`tests/python/test_method_exposure.py`), qui lit le stub et échoue si
une opération éligible perd sa méthode. Ce test porte aussi la liste des
exclusions, chacune avec sa raison.

Deux points à connaître, illustrés plus bas : le nom peut changer entre les
deux formes (`matrix.stiffness(model, mats)` / `model.stiffness_matrix(mats)`,
`element_field.sub_material_field(sub, …)` / `sub_model.material_field(…)`),
et les cinématiques (`deformation`, `beam_deformation`,
`thermal_strain`) n'ont **pas** de méthode : elles exigent des composantes
nommées, elles n'auraient pas de sens sur un champ quelconque. Même raison pour
`internal_forces`, qui lit la contrainte de Voigt par nom.

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
| `coords` | `Coords` | `pyrucast.Coords` | [Coords](coords.md) |
| `atoms::node` | `Node` | `pyrucast.Node` | [Nœud](node.md) |
| `containers::mesh` | `SubMesh` | `pyrucast.SubMesh` *(vue, via `mesh[i]`)* | [Maillage](mesh.md) |
| `containers::mesh` | `Mesh` | `pyrucast.Mesh` | [Maillage](mesh.md) |
| `atoms::cell` | `Cell` | `pyrucast.Cell` | [Maillage](mesh.md) |
| `containers::finite_element_space` | `SubFiniteElementSpace` | `pyrucast.SubFiniteElementSpace` | [Espace EF](fe-space.md) |
| `containers::finite_element_space` | `FiniteElementSpace` | `pyrucast.FiniteElementSpace` | [Espace EF](fe-space.md) |
| `atoms::element` | `Element` | `pyrucast.Element` | [Espace EF](fe-space.md) |
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
`FiniteElementSpace(mesh)`, `ElementField(fes, comps)`, `Matrix.block(...)` —
ou par l'opérateur qui rend ce parent (`model.heat_conduction(fes)`), et on
compose plusieurs
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

### `ops::mesh` — construction et transformation de maillages

| Rust (`ops::mesh::…`) | Python (`pyrucast.mesh.…`) |
|---|---|
| `from_live_nodes(coords: Handle<Coords>) -> Mesh` | `from_live_nodes(coords) -> Mesh` |
| `poi1_from_nodes(nodes: &[Node]) -> Mesh` | `poi1_from_nodes(nodes) -> Mesh` |
| `line(a: &Node, b: &Node, n_elems: usize, element_type: ElementType) -> Mesh` | `line(a, b, n_elems, element_type="SEG2") -> Mesh` |
| `circle(center: &Node, normal: &[f64], radius: f64, n_elems: usize, element_type: ElementType) -> Mesh` | `circle(center, normal, radius, n_elems, element_type="SEG2") -> Mesh` |
| `arc(node_a: &Node, center: &Node, node_b: &Node, n_elems: usize, element_type: ElementType) -> Mesh` | `arc(a, center, b, n_elems, element_type="SEG2") -> Mesh` |
| `sweep(mesh_a: &Mesh, mesh_b: &Mesh, n_layers: usize, element_type: ElementType) -> Mesh` | `sweep(mesh_a, mesh_b, n_layers, element_type="QUA4") -> Mesh` |
| `transfinite(side1: &Mesh, side2: &Mesh, side3: &Mesh, side4: &Mesh, element_type: ElementType) -> Mesh` | `transfinite(side1, side2, side3, side4, element_type="QUA4") -> Mesh` |
| `sweep_solid(mesh_a: &Mesh, mesh_b: &Mesh, n_layers: usize) -> Mesh` | `sweep_solid(mesh_a, mesh_b, n_layers) -> Mesh` |
| `extrude(mesh: &Mesh, direction: &[f64], n_layers: usize) -> Mesh` | `extrude(mesh, direction, n_layers) -> Mesh` |
| `revolve(mesh: &Mesh, angle: f64, n_layers: usize, center: &[f64], axis: Option<&[f64]>) -> Mesh` | `revolve(mesh, angle, n_layers, center, axis=None) -> Mesh` |
| `to_quadratic(mesh: &Mesh) -> Mesh` | `to_quadratic(mesh) -> Mesh` |
| `convert(mesh: &Mesh, element_type: ElementType) -> Mesh` | `convert(mesh, element_type) -> Mesh` |
| `copy(mesh: &Mesh, new_nodes: bool) -> Mesh` | `copy(mesh, new_nodes=True) -> Mesh` |
| `translate(mesh: &Mesh, vector: &[f64]) -> Mesh` | `translate(mesh, vector) -> Mesh` |
| `rotate(mesh: &Mesh, angle: f64, center: &[f64], axis: Option<&[f64]>) -> Mesh` | `rotate(mesh, angle, center, axis=None) -> Mesh` |
| `symmetry_point(mesh: &Mesh, center: &[f64]) -> Mesh` | `symmetry_point(mesh, center) -> Mesh` |
| `symmetry_line(mesh: &Mesh, a: &[f64], b: &[f64]) -> Mesh` | `symmetry_line(mesh, a, b) -> Mesh` |
| `symmetry_plane(mesh: &Mesh, a: &[f64], b: &[f64], c: &[f64]) -> Mesh` | `symmetry_plane(mesh, a, b, c) -> Mesh` |
| `triangulate_surface(contour: &Mesh, et: ElementType, size: Option<f64>) -> Mesh` | `triangulate_surface(contour, element_type, size=None) -> Mesh` |
| `pave_surface(contour: &Mesh, element_type: ElementType, size: Option<f64>, all_quad: bool) -> Mesh` | `pave_surface(contour, element_type, size=None, all_quad=False) -> Mesh` |
| `triangulate_volume(envelope: &Mesh, size: Option<f64>, allow_surface_nodes: bool) -> Mesh` | `triangulate_volume(envelope, size=None, allow_surface_nodes=False) -> Mesh` |
| `pave_volume(envelope: &Mesh, layers: usize, thickness: Option<f64>, size: Option<f64>) -> Mesh` | `pave_volume(envelope, layers=1, thickness=None, size=None) -> Mesh` |
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
| `read_gmsh(coords: Handle<Coords>, path: &Path) -> Vec<(String, Mesh)>` | `read_gmsh(coords, path) -> dict[str, Mesh]` |
| `read_gmsh_str(coords: Handle<Coords>, text: &str) -> Vec<(String, Mesh)>` | `read_gmsh_str(coords, text) -> dict[str, Mesh]` |
| `from_gmsh_arrays(coords: Handle<Coords>, node_tags: &[u64], node_coords: &[f64], blocks: &[GmshBlock]) -> Vec<(String, Mesh)>` | `from_gmsh_arrays(coords, node_tags, node_coords, blocks) -> dict[str, Mesh]` |
| — (exige l'interpréteur) | `from_gmsh(coords, *, dim=-1, tag=-1) -> dict[str, Mesh]` |
| `consolidate(mesh: &Mesh) -> Mesh` | `consolidate(mesh) -> Mesh` |
| `select_nodes(field: &NodeField, band: &Band, …) -> Mesh` / `select_cells(field: &ElementField, …) -> Mesh` | `select(field, ge=None, gt=None, le=None, lt=None, components=None) -> Mesh` (dispatch par type ; part d'un champ mais rend un maillage, d'où son rangement ici) |

### `ops::node_field` — opérateurs produisant un champ aux nœuds

| Rust (`ops::node_field::…`) | Python (`pyrucast.node_field.…`) |
|---|---|
| `positions(mesh: &Mesh, components: Option<Vec<String>>) -> NodeField` | `positions(mesh, components=None) -> NodeField` |
| `divergence(field: &ElementField) -> NodeField` | `divergence(field) -> NodeField` |
| `restrict(field: &NodeField, mesh: &Mesh) -> NodeField` | `restrict(field, mesh) -> NodeField` |
| `restrict_like(field: &NodeField, target: &NodeField) -> NodeField` | `restrict_like(field, target) -> NodeField` |
| `merge(a: &NodeField, b: &NodeField) -> NodeField` | `merge(a, b) -> NodeField` |
| `consolidate(field: &NodeField) -> NodeField` | `consolidate(field) -> NodeField` |
| `mask(field: &NodeField, band: &Band, …) -> NodeField` | `mask(field, ge=None, gt=None, le=None, lt=None, components=None) -> NodeField` (champ `0/1` de même structure ; sucre `field >= x`). Accepte aussi un `SubNodeField` |
| `flux(fespace: &FiniteElementSpace, density: FluxDensity, component: &str) -> NodeField` | `flux(fespace, density, component) -> NodeField` |
| `internal_forces(model: &Model, stresses: &ElementField) -> NodeField` | `internal_forces(stresses, model) -> NodeField` |
| `internal_forces_continuum(stresses: &ElementField, fespace: &FiniteElementSpace) -> NodeField` | `internal_forces_continuum(stresses, fespace) -> NodeField` |

`flux` et `internal_forces` (`BSIG`, `∫ Bᵀ σ`) sont des **assemblages**,
mais leur résultat est un vecteur nodal et non un opérateur : on se range
par la sortie.

### `ops::coords` — écriture dans le magasin de coordonnées

| Rust (`ops::coords::…`) | Python (`pyrucast.coords.…`) |
|---|---|
| `set(field: &NodeField, components: Option<Vec<String>>) -> ()` | `set(field, components=None) -> None` |
| `displace(field: &NodeField, components: Option<Vec<String>>) -> ()` | `displace(field, components=None) -> None` |

### `ops::element_field` — opérateurs produisant un champ aux points de Gauss

| Rust (`ops::element_field::…`) | Python (`pyrucast.element_field.…`) |
|---|---|
| `gradient(field: &NodeField, fespace: &FiniteElementSpace) -> ElementField` | `gradient(field, fespace) -> ElementField` |
| `deformation(u: &NodeField, fespace: &FiniteElementSpace) -> ElementField` | `deformation(u, fespace) -> ElementField` |
| `interp_to_gauss(field: &NodeField, fespace: &FiniteElementSpace) -> ElementField` | `interp_to_gauss(field, fespace) -> ElementField` |
| `thermal_strain(temperature: &ElementField, material: &ElementField, fespace: &FiniteElementSpace, t_ref: f64) -> ElementField` | `thermal_strain(temperature, materials, fespace, t_ref) -> ElementField` |
| `beam_deformation(field: &NodeField, fespace: &FiniteElementSpace, material: &ElementField) -> ElementField` | `beam_deformation(field, fespace, material) -> ElementField` (1-D, plan ou spatial selon le maillage ; le matériau est requis, l'interpolation dépendant de `Φ`) |
| `consolidate(field: &ElementField) -> ElementField` | `consolidate(field) -> ElementField` (fusionne les zones d'une même fespace) |
| `mask(field: &ElementField, band: &Band, …) -> ElementField` | `mask(field, ge=None, …) -> ElementField` ; accepte aussi un `SubElementField` |
| `sub_material_field(sub: &SubModel, pairs: &[(&str, f64)]) -> SubElementField` | `sub_material_field(sub_model, components_and_values) -> SubElementField` |
| `material_field(model: &Model, pairs: &[(&str, f64)]) -> ElementField` | `material_field(model, components_and_values) -> ElementField` |
| `material_field_per_sub_model(model: &Model, per: &[&[(&str, f64)]]) -> ElementField` | `material_field_per_sub_model(model, components_and_values_per_sub_model) -> ElementField` |
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
un sujet qu'on transforme.

| Rust (`ops::model::…`) | Python (`pyrucast.model.…`) |
|---|---|
| `heat_conduction(fes: &FiniteElementSpace) -> Model` | `heat_conduction(fespace, symmetry=None) -> Model` |
| `heat_conduction_with_symmetry(fes, symmetry: MaterialSymmetry) -> Model` | *idem, via `symmetry=`* |
| `fick(fes: &FiniteElementSpace, species: &str) -> Model` | `fick(fespace, species, symmetry=None) -> Model` |
| `fick_with_symmetry(fes, symmetry: MaterialSymmetry, species: &str) -> Model` | *idem, via `symmetry=`* |
| `radiation(fes: &FiniteElementSpace) -> Model` | `radiation(fespace) -> Model` |
| `boundary_transfer(fes, components: Vec<(String, String)>, physics: Physics) -> Model` | `boundary_transfer(fespace, components, physics) -> Model` |
| `interface_transfer(side_a, side_b, components, physics: Physics, tol: f64) -> Model` | `interface_transfer(side_a, side_b, components, physics, tol=None) -> Model` |
| `truss(fes: &FiniteElementSpace) -> Model` | `truss(fespace) -> Model` |
| `elasticity(fes, model: ElasticityModel) -> Model` | `elasticity(fespace, model, symmetry=None) -> Model` |
| `elasticity_with_symmetry(fes, model, symmetry: MaterialSymmetry) -> Model` | *idem, via `symmetry=`* |
| `plasticity_perfect(fes, model: ElasticityModel) -> Model` | `plasticity_perfect(fespace, model) -> Model` |
| `plasticity_with_law(fes, model, law: PlasticLaw) -> Model` | une fonction par loi : `plasticity_isotropic`, `drucker_prager`, `ottosen`, `creep_norton`, `creep_blackburn`, `creep_lemaitre`, `viscoplasticity_chaboche`, `viscoplasticity_lemaitre_chaboche`, `gurson` — toutes `(fespace, model) -> Model` |
| `mazars(fes, model: ElasticityModel) -> Model` | `mazars(fespace, model) -> Model` |
| `damage_with_law(fes, model, law: DamageLaw) -> Model` | une fonction par loi : `damage_tc`, `damage_sic_sic` — `(fespace, model) -> Model` |
| `bernoulli(fes: &FiniteElementSpace) -> Model` | `bernoulli(fespace) -> Model` |
| `timoshenko(fes: &FiniteElementSpace) -> Model` | `timoshenko(fespace) -> Model` |
| `shell(fes, model: ShellModel) -> Model` | `shell(fespace, model) -> Model` |
| `dirichlet(imposed_variable, target_dual, imposed_mesh, multiplier_mesh, multiplier, imposed_value, sense: RelationSense) -> Model` | `dirichlet(imposed_variable, target_dual, imposed_mesh, multiplier_mesh, multiplier=None, imposed_value=None, sense=None) -> Model` |
| `mpc(terms: Vec<MpcTerm>, multiplier_mesh, multiplier, imposed_value, sense) -> Model` | `mpc(terms, multiplier_mesh, multiplier=None, imposed_value=None, sense=None) -> Model` |
| `embedded(immersed, host, components, multipliers, imposed_values, tol) -> Model` | `embedded(immersed, host, components, multipliers=None, imposed_values=None, tol=None) -> Model` |
| `contact(slave, master, components, multiplier, imposed_value) -> Model` | `contact(slave, master, components, multiplier=None, imposed_value=None) -> Model` |

**Les deux plis du catalogue.** Rust nomme la symétrie et la loi par une
**enum** (`MaterialSymmetry`, `ElasticLaw`, `PlasticLaw`, `DamageLaw`) ; Python n'expose pas
ces enums, et replie donc la symétrie en mot-clé `symmetry=` et déplie les lois
en une fonction chacune. Le catalogue est le même des deux côtés — les
dérogations correspondantes sont enregistrées, avec leur raison, dans
`tests/python/test_mirror_completeness.py`.

### `ops::matrix` — assemblage des matrices

| Rust (`ops::matrix::…`) | Python (`pyrucast.matrix.…`) |
|---|---|
| `stiffness(model: &Model, materials: &ElementField) -> Matrix` | `stiffness(model, materials) -> Matrix` |
| `mass(model: &Model, materials: &ElementField) -> Matrix` | `mass(model, materials) -> Matrix` |
| `lump(m: &Matrix) -> Matrix` | `lump(matrix) -> Matrix` |
| `geometric(model: &Model, materials: &ElementField, stress: &ElementField) -> Matrix` | `geometric(model, materials, stress) -> Matrix` |
| `tangent(model: &Model, materials: &ElementField, state: &ElementField) -> Matrix` | `tangent(model, materials, state) -> Matrix` |

### `ops::solver` — résolution

| Rust (`ops::solver::…`) | Python (`pyrucast.solver.…`) |
|---|---|
| `lu::solve(matrix: &Matrix, rhs: &NodeField) -> NodeField` | `solve(matrix, rhs) -> NodeField` |
| `eliminate::solve(model: &Model, matrix: &Matrix, rhs: &NodeField) -> NodeField` | `solve_eliminate(matrix, model, rhs) -> NodeField` |
| `unilateral::solve(model: &Model, matrix: &Matrix, rhs: &NodeField) -> NodeField` | `solve_unilateral(matrix, model, rhs, max_iter=100, tol=1e-10) -> NodeField` |

### `ops::export` — export vers des formats externes

| Rust (`ops::export::…`) | Python (`pyrucast.export.…`) |
|---|---|
| `write_vtk_mesh(mesh: &Mesh, path: &Path)` | `export_vtk(mesh, path) -> None` |
| `write_vtk_node_field(mesh: &Mesh, field: &NodeField, path: &Path)` | `export_vtk(mesh, path, field=node_field) -> None` |
| `write_vtk_element_field(mesh: &Mesh, field: &ElementField, path: &Path)` | `export_vtk(mesh, path, field=element_field) -> None` |

### `archive` — sauvegarde et relecture d'un graphe

Les deux seuls verbes qui restent au **niveau racine** : ils ne produisent
aucun conteneur déterminé, mais un dictionnaire de ce qu'on leur a donné.

| Rust (`archive::…`) | Python (`pyrucast.…`) |
|---|---|
| `save(path, &[(&str, &dyn ArchiveRoot)])` | `save(path, dict) -> None` |
| `load(path) -> Objects` | `load(path) -> dict` |

À l'écriture les types sont connus du compilateur, d'où la tranche de paires ;
à la relecture non, d'où la table nommée dont on tire chaque objet avec son
type attendu (`objets.mesh("clef")?`). Voir
[Sauvegarde et relecture](sauvegarde.md).

> `ops::geom` héberge `locate_points` (mapping iso-paramétrique inverse, sous
> le [baignage](contraintes/embedded.md)) et `project_points` (projection au
> point le plus proche sur une surface, sous le [contact](contraintes/contact.md)) ;
> ces deux primitives sont internes (API Rust), pas encore exposées en Python —
> la seule dérogation de module entier du garde-fou de complétude
> (`tests/python/test_mirror_completeness.py`).
> Le nœud le plus proche, lui, n'est **pas** un opérateur : c'est la méthode
> `mesh.nearest_node([x, y])`, des deux côtés.

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

L'opérateur `+` (et `-`, `*`, `/`) reste l'**arithmétique scalaire** des
sous-champs (`subfield + 2.0` → ajoute la valeur à chaque composante) et a
vocation à porter, à terme, l'**addition réelle de champs** (valeur au nœud
= somme des deux). Il n'est **jamais** utilisé pour composer des agrégats —
c'est `|` qui s'en charge, sans collision.
