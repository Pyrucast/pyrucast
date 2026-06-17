# Modèle physique (`Model`)

Un **`Model`** est l'objet orchestrateur qui décrit un problème physique et **produit les matrices** (raideur, masse) à la demande de l'utilisateur. Il vient se poser sur la couche éléments finis ([`FiniteElementSpace`](fe-space.md)), elle-même posée sur le maillage géométrique.

```text
Géométrie         Mesh / SubMesh                       (purement géométrique)
Formulation EF    FiniteElementSpace / SubFiniteElementSpace      (interpolation, quadrature)
Physique          Model / SubModel / Physics           (loi, matériaux, assemblage)
```

## Architecture

```text
Model
├── sub_models: Vec<SubModel>
├── primal_vars(): Vec<String>      # union — colonnes des matrices
└── dual_vars():   Vec<String>      # union — lignes des matrices

ops::assemble (opérateurs, pas des méthodes de Model)
├── stiffness(model, materials) -> Matrix   # K  (assemblé sur demande)
└── mass(model)                 -> Matrix   # M  (assemblé sur demande)

SubModel  (énum de stockage + dispatch — AUCUNE logique)
├── HeatConduction(HeatConduction)    # chaque variante enveloppe une struct…
├── Dirichlet(Dirichlet)             # …qui porte ses données + impl Physics
└── as_physics(&self) -> &dyn Physics   # l'unique match du module modèle

Physics  (trait — TOUT le comportement, co-localisé par physique)
├── primal_vars / dual_vars / material_components / material_fespace
├── build_stiffness_blocks / build_mass_blocks
└── label / display / render
```

L'énum `SubModel` ne sert qu'au **stockage** et à la **sérialisation**
(`bincode`) ; il délègue chaque appel à l'`impl Physics` de la variante via
`as_physics()`. Tout le code générique (l'agrégat `Model`, l'assembleur,
`Dump`) passe par ce seul point — ajouter une physique ne touche donc aucun
de ces sites. Voir le chapitre [Ajouter une physique](ajouter-une-physique.md).

Le `Model` est **purement orchestrateur** : il énumère les DOFs, dimensionne la `Matrix`, boucle sur les sub-models et accumule. Aucune logique physique ne vit chez lui.

## Identification des DOFs : primal ≠ dual

Chaque physique déclare :

- ses **variables primales** (les inconnues — composantes du vecteur solution, colonnes de la matrice) ;
- ses **variables duales** (les conjuguées énergétiques — composantes du vecteur chargement, lignes de la matrice).

Elles sont presque toujours **différentes** :

| Physique           | Primales (cols)    | Duales (rows)      |
|---|---|---|
| `HeatConduction`   | `T` (température)  | `q` (flux de chaleur) |
| `Truss` / `LinearElasticity` | `u_x`, `u_y`, …   | `f_x`, `f_y`, …    |
| `Dirichlet { imposed_variable: "T" }` | `lambda_T`         | `imposed_T`         |

Les DOFs de la `Matrix` sont identifiés par le couple `(NodeID, nom_de_champ)` (voir [`Matrix`](matrix.md)) : deux SubModels qui utilisent le même nom (`"T"`) sur des nœuds différents ne se collisionnent pas, et la jonction se fait automatiquement quand ils partagent un même `(NodeID, nom)`.

## Chargements complètement séparés du Model

Le `Model` ne porte **aucune** logique de second membre. L'utilisateur :

1. lit `model.dual_vars()` pour connaître les noms de composantes du vecteur force ;
2. construit un `NodeField` avec ces composantes (forces de Neumann, sources de chaleur, valeurs imposées de Dirichlet aux nœuds-multiplicateurs, …) ;
3. compose plusieurs sources avec `|` (union des zones, dédupliquée et fusionnée par support) — le nommé `merge` en est l'alias ;
4. passe `Matrix + NodeField` au solveur.

Cette séparation a deux mérites :
- les chargements sont des données utilisateur, faciles à composer ;
- le `Model` reste une description compacte et indépendante du chargement (le même modèle peut être résolu avec plusieurs chargements en cascade).

## Variantes de `Physics` implémentées en v0

### `HeatConduction` (`models/heat_conduction.rs`)

Conduction thermique linéaire. La forme variationnelle Galerkine donne, pour chaque cellule :

\\[
K_{ij}^{(\text{loc})} = \int_K k(x)\, \nabla N_i \cdot \nabla N_j\, dx
\quad \approx \sum_g k(\xi_g)\, (\nabla N_i \cdot \nabla N_j)|_g\, |J|_g\, w_g
\\]

- **primal** : `"T"`
- **dual** : `"q"`
- **matériau** : l'`ElementField` doit définir une composante nommée `"k"` (conductivité isotrope au point de Gauss).

Le bloc local de la cellule est écrit dans la matrice globale aux positions `row = (NodeID_i, "q")`, `col = (NodeID_j, "T")`. Pour un SEG2 de longueur \\(L\\) avec \\(k\\) uniforme, on retrouve la matrice analytique \\((k/L)\,[[1, -1], [-1, 1]]\\).

### `Dirichlet` (`models/dirichlet.rs`)

Condition de Dirichlet `u(n) = u_d` imposée par multiplicateurs de Lagrange. C'est une **contrainte** : aucun matériau, aucune loi de comportement. Elle ne crée **aucun nœud** et ne mute jamais le `Coords` — l'utilisateur fournit **deux maillages** :

- `imposed_mesh` (POI1 pour l'instant) : les nœuds contraints (partagés avec la physique cible) ;
- `multiplier_mesh` (POI1) : le support des multiplicateurs, apparié élément-par-élément avec `imposed_mesh` (même structure de sous-maillage, même nombre de cellules par paire). On le fabrique typiquement depuis `imposed_mesh` avec le mesher générique `barycenter` (nœuds neufs colocalisés au centre de gravité), mais l'utilisateur reste libre (colocalisés, décalés, ou réutiliser les nœuds contraints eux-mêmes).

Quatre noms de variables, dont deux déduits et **surchargeables** :

| rôle | nom | fourniture |
|---|---|---|
| variable imposée (primale de la **cible**) | `imposed_variable` (ex `"T"`) | requis |
| duale de la **cible** (ligne où atterrit la réaction `Cᵀ`) | `target_dual` (ex `"q"`) | requis |
| primale propre = multiplicateur (inconnue du système) | `multiplier`, défaut `lambda_<imposed_variable>` | déduit |
| duale propre = ligne de contrainte + **slot** où l'utilisateur écrit `u_d` | `imposed_value`, défaut `imposed_<imposed_variable>` | déduit |

À l'assemblage, **une paire de blocs unité par sous-maillage**, chacun marqué **non-symétrique** (seule l'union `C ∪ Cᵀ` l'est — propriété globale du système point-selle) :
  - **bloc C** : `(multiplier_node, imposed_value) × (imposed_node, imposed_variable) = 1`
  - **bloc Cᵀ** : `(imposed_node, target_dual) × (multiplier_node, multiplier) = 1`

Le multiplicateur se retrouve dans la solution sous le nom `multiplier` (`lambda_T`) au nœud-multiplicateur ; sa valeur est la **force de réaction** de la contrainte. La valeur imposée `u_d` n'est **pas** stockée dans le SubModel : l'utilisateur la fournit dans le `NodeField` de chargement à la position `(multiplier_node, imposed_value)`.

Les nœuds-multiplicateurs vivent tant que leur maillage **ou** le SubModel les référence (refcounts) ; quand les deux disparaissent, ils deviennent collectables. Le SubModel ne décrémente que ce qu'il partage — il n'a rien créé.

## Règle invariante : un Model = une Matrice

`assemble::stiffness(model, materials)` et `assemble::mass(model)` produisent chacune **une seule** `Matrix` couvrant l'ensemble des DOFs du Model (primaux ⊕ multiplicateurs). Les conditions limites n'ont pas de statut spécial — ce sont des sub-models comme les autres qui contribuent leurs entrées dans la même matrice globale.

Cette uniformité simplifie tout : le solveur reçoit une seule `Matrix` + un seul `NodeField` ; pas besoin de jongler avec un système saddle-point composé.

## API Rust

```rust,ignore
use pyrucast::containers::mesh::Coords;
use pyrucast::containers::mesh::element_type::ElementType;
use pyrucast::containers::mesh::node::Node;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::model::Model;
use pyrucast::ops::{assemble, build, mesher};
use pyrucast::store::insert;

// 1-D : maillage [0, 1] à un seul SEG2.
let coords = insert(Coords::new(1).unwrap());
let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::SEG2));
mesh.add_cell(&[a.id(), b.id()]).unwrap();
let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

// Modèle : conduction (le matériau est fourni à l'assemblage, pas ici)
// + Dirichlet à gauche. Constructeurs au niveau parent (balaient les
// sous-espaces de `fes`), composés par `|` (union — Rust : `union`) — on
// ne construit jamais de `SubModel` à la main (cf. CONVENTIONS.md).
let hc = Model::heat_conduction(&fes).unwrap();
// Maillage des nœuds imposés + support des multiplicateurs (barycenter
// colocalise des nœuds neufs). Le modèle ne crée aucun nœud lui-même.
let imposed = Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
let multiplier = mesher::barycenter(&imposed).unwrap();
let dir = Model::dirichlet("T".into(), "q".into(), &imposed, &multiplier, None, None).unwrap();
let model = hc.union(&dir).unwrap();

// Matériau k = 1, appliqué aux sous-modèles qui en ont besoin (Dirichlet
// est automatiquement ignoré), puis assemblage.
let materials = build::material_field(&model, &[("k", 1.0)]).unwrap();
let k = assemble::stiffness(&model, &materials).unwrap();
assert_eq!(k.n_rows().unwrap(), 3);  // 2 nœuds physiques + 1 multiplicateur
```

## API Python

```python
import pyrucast

c = pyrucast.Coords(dim=1)
a = c.add_node([0.0])
b = c.add_node([1.0])
mesh = pyrucast.Mesh(c, "SEG2")
mesh.unit().add_cell([a, b])
fes = pyrucast.FiniteElementSpace(mesh)

# Modèle : conduction (matériau fourni à l'assemblage) + Dirichlet à gauche.
# Constructeurs au niveau parent, composés par `|` — pas de SubModel à la main.
# Le maillage des multiplicateurs est fabriqué depuis les nœuds imposés.
imposed = pyrucast.poi1_from_nodes([a])
multiplier = pyrucast.barycenter(imposed)
model = pyrucast.Model.heat_conduction(fes) | pyrucast.Model.dirichlet(
    "T", "q", imposed, multiplier
)

# Matériau k = 1 (les sous-modèles Dirichlet sont ignorés automatiquement).
materials = pyrucast.material_field(model, [("k", 1.0)])

K = pyrucast.stiffness(model, materials)
print("primal_vars =", model.primal_vars())   # ['T', 'lambda_T']
print("dual_vars =",   model.dual_vars())      # ['q', 'imposed_T']
print(K)                                        # Matrix: 3 row(s) × 3 col(s), …
```

## Solveur dense — fonction `solve(matrix, rhs)`

Pour valider l'assemblage de bout en bout, pyrucast embarque un **solveur dense LU minimal** (`pyrucast::ops::solver::lu::solve` / `pyrucast.solve`) qui :

1. lit le `NodeField` de chargement à chacune des **lignes** de la matrice (les entrées absentes valent `0.0` par défaut) ;
2. convertit la `Matrix` en `nalgebra::DMatrix<f64>` ;
3. factorise `A = LU` et résout `A x = b` ;
4. emballe la solution dans un `NodeField` indexé sur les **colonnes** de la matrice.

C'est un harnais de test — pas le solveur final. Un objet `LinearSolver` enfichable (itératif, direct creux, préconditionné) arrivera en **Phase 3** par-dessus `nalgebra-sparse` ; les conversions `Matrix::to_csr` / `to_csc` sont déjà en place pour le brancher sans changement d'API.

**Exemple complet : Poisson 1-D `-u'' = 0` avec `u(0) = 0`, `u(1) = 1`** (solution analytique `u(x) = x`, multiplicateurs aux bords = flux ±1) :

```python
import pyrucast

# 1) Maillage + FE space
c = pyrucast.Coords(dim=1)
nodes = [c.add_node([i / 4.0]) for i in range(5)]
mesh = pyrucast.Mesh(c, "SEG2")
for i in range(4):
    mesh.unit().add_cell([nodes[i], nodes[i + 1]])
fes = pyrucast.FiniteElementSpace(mesh)

# 2) Modèle : conduction + Dirichlet aux deux bouts.
# Chaque physique est un Model au niveau parent ; on les compose par `|`.
# On fabrique le support des multiplicateurs depuis les nœuds imposés
# (`barycenter` colocalise des nœuds neufs), et on y lit le nœud-multiplicateur.
imposed_left = pyrucast.poi1_from_nodes([nodes[0]])
imposed_right = pyrucast.poi1_from_nodes([nodes[-1]])
mult_mesh_left = pyrucast.barycenter(imposed_left)
mult_mesh_right = pyrucast.barycenter(imposed_right)
left = pyrucast.Model.dirichlet("T", "q", imposed_left, mult_mesh_left)
right = pyrucast.Model.dirichlet("T", "q", imposed_right, mult_mesh_right)
mult_left = mult_mesh_left.node(0, 0, 0)
mult_right = mult_mesh_right.node(0, 0, 0)
model = pyrucast.Model.heat_conduction(fes) | left | right

# 3) Matériau k = 1 (appliqué à la conduction, Dirichlet ignoré)
materials = pyrucast.material_field(model, [("k", 1.0)])

# 4) Chargement : valeurs imposées au slot `imposed_T` des nœuds-multiplicateurs.
# NodeField accepte un Mesh comme support (une zone par submesh) ;
# l'écriture passe par la zone (rhs[0]), la lecture par l'agrégat.
rhs_mesh = pyrucast.Mesh(c, "POI1")
rhs_mesh.unit().add_cell([mult_left])
rhs_mesh.unit().add_cell([mult_right])
rhs = pyrucast.NodeField(rhs_mesh, ["imposed_T"])
rhs[0].set_value(mult_left, "imposed_T", 0.0)
rhs[0].set_value(mult_right, "imposed_T", 1.0)

# 5) Assemblage + résolution
K = pyrucast.stiffness(model, materials)
solution = pyrucast.solve(K, rhs)
assert abs(solution.value(nodes[2], "T") - 0.5) < 1e-10  # T au milieu = 0.5
assert abs(solution.value(mult_left, "lambda_T") - 1.0) < 1e-10  # flux à gauche
```

## Limitations actuelles

- **Mass non assemblée** : `assemble::mass(model)` retourne une matrice vide en v0. L'intégrande `∫ ρc_p · N_i N_j dx` (et son équivalent pour les autres physiques) est additif et sera ajouté quand le besoin transient se présentera.
- **Physiques disponibles** : `HeatConduction` (thermique) et `Truss` (barre, cf. [Mécanique](mecanique.md)) ; `LinearElasticity`, `Timoshenko`, `Periodic`, etc. viennent comme nouvelles structs implémentant `Physics` (une variante de l'énum `SubModel` + un bras de `as_physics`, rien d'autre — cf. [Ajouter une physique](ajouter-une-physique.md)). Le coût d'ajout est O(1) fichier, indépendant du nombre de physiques existantes.
- **Pas de check de cohérence pré-assemblage** : la consistance (matériau définit bien `"k"` pour HeatConduction, compatibilité des FE spaces entre sub-models, etc.) est vérifiée au moment de `assemble::stiffness` / `assemble::mass`, pas à l'ajout du sub-model. Si on découvre des cas où ça pose problème, un check eager est facile à ajouter.
- **Solveur dense seulement** : le `solve` fourni est un harnais de test (LU dense via `nalgebra`). Pour les vrais problèmes Phase 3 introduira un trait `LinearSolver` enfichable (itératifs, direct creux, factorisation Cholesky pour les cas symétriques détectés via le drapeau de la `Matrix`).
