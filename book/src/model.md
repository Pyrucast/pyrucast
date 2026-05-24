# Modèle physique (`Model`)

Un **`Model`** est l'objet orchestrateur qui décrit un problème physique et **produit les matrices** (raideur, masse) à la demande de l'utilisateur. Il vient se poser sur la couche éléments finis ([`FiniteElementSpace`](fe-space.md)), elle-même posée sur le maillage géométrique.

```text
Géométrie         Mesh / SubMesh                       (purement géométrique)
Formulation EF    FiniteElementSpace / SubFESpace      (interpolation, quadrature)
Physique          Model / SubModel / Physics           (loi, matériaux, assemblage)
```

## Architecture

```text
Model
├── sub_models: Vec<SubModel>
├── primal_vars(): Vec<String>      # union — colonnes des matrices
├── dual_vars():   Vec<String>      # union — lignes des matrices
├── stiffness() -> Matrix            # K  (assemblé sur demande)
└── mass()      -> Matrix            # M  (assemblé sur demande)

SubModel
└── physics: Physics                  # un SubModel = une physique

Physics  (énum à variantes spécialisées)
├── HeatConduction { fespace, material }
├── Dirichlet      { config, primal_var, primal_dual, constrained_nodes, multiplier_nodes }
└── …  ajouter une physique = ajouter une variante + son match-arm
```

Le `Model` est **purement orchestrateur** : il énumère les DOFs, dimensionne la `Matrix`, boucle sur les sub-models et accumule. Aucune logique physique ne vit chez lui.

## Identification des DOFs : primal ≠ dual

Chaque physique déclare :

- ses **variables primales** (les inconnues — composantes du vecteur solution, colonnes de la matrice) ;
- ses **variables duales** (les conjuguées énergétiques — composantes du vecteur chargement, lignes de la matrice).

Elles sont presque toujours **différentes** :

| Physique           | Primales (cols)    | Duales (rows)      |
|---|---|---|
| `HeatConduction`   | `T` (température)  | `q` (flux de chaleur) |
| `LinearElasticity` | `ux`, `uy`, …      | `fx`, `fy`, …       |
| `Dirichlet { primal_var: "T" }` | `lambda_T`         | `T`                 |

Les DOFs de la `Matrix` sont identifiés par le couple `(NodeID, nom_de_champ)` (voir [`Matrix`](matrix.md)) : deux SubModels qui utilisent le même nom (`"T"`) sur des nœuds différents ne se collisionnent pas, et la jonction se fait automatiquement quand ils partagent un même `(NodeID, nom)`.

## Chargements complètement séparés du Model

Le `Model` ne porte **aucune** logique de second membre. L'utilisateur :

1. lit `model.dual_vars()` pour connaître les noms de composantes du vecteur force ;
2. construit un `NodeField` avec ces composantes (forces de Neumann, sources de chaleur, valeurs imposées de Dirichlet aux nœuds-multiplicateurs, …) ;
3. additionne plusieurs `NodeField` quand il y a plusieurs sources ;
4. passe `Matrix + NodeField` au solveur.

Cette séparation a deux mérites :
- les chargements sont des données utilisateur, faciles à composer ;
- le `Model` reste une description compacte et indépendante du chargement (le même modèle peut être résolu avec plusieurs chargements en cascade).

## Variantes de `Physics` implémentées en v0

### `HeatConduction { fespace, material }`

Conduction thermique linéaire. La forme variationnelle Galerkine donne, pour chaque cellule :

\\[
K_{ij}^{(\text{loc})} = \int_K k(x)\, \nabla N_i \cdot \nabla N_j\, dx
\quad \approx \sum_g k(\xi_g)\, (\nabla N_i \cdot \nabla N_j)|_g\, |J|_g\, w_g
\\]

- **primal** : `"T"`
- **dual** : `"q"`
- **matériau** : l'`ElementField` doit définir une composante nommée `"k"` (conductivité isotrope au point de Gauss).

Le bloc local de la cellule est écrit dans la matrice globale aux positions `row = (NodeID_i, "q")`, `col = (NodeID_j, "T")`. Pour un SEG2 de longueur \\(L\\) avec \\(k\\) uniforme, on retrouve la matrice analytique \\((k/L)\,[[1, -1], [-1, 1]]\\).

### `Dirichlet { config, primal_var, primal_dual, constrained_nodes, multiplier_nodes }`

Condition de Dirichlet `u(n) = u_d` enforcée par multiplicateurs de Lagrange. À la construction :

- l'utilisateur fournit la liste des `constrained_nodes` (nœuds réels à contraindre), le nom de la primale contrainte (`primal_var`, par ex. `"T"`) et le nom de la duale de la physique primaire (`primal_dual`, par ex. `"q"`) ;
- le SubModel crée **un nœud-multiplicateur par contrainte** dans le `Configuration`, au même point que le nœud contraint, et incrémente le refcount des nœuds qu'il protège ;
- à l'assemblage, deux entrées unité sont ajoutées par contrainte :
  - **bloc C** : `(multiplier_node, primal_var) × (constrained_node, primal_var) = 1`
  - **bloc Cᵀ** : `(constrained_node, primal_dual) × (multiplier_node, lambda_<primal_var>) = 1`

Le multiplicateur lui-même se retrouve dans la solution sous le nom `lambda_<primal_var>` au nœud-multiplicateur ; sa valeur est la **force de réaction** de la contrainte. La valeur imposée `u_d` n'est **pas** stockée dans le SubModel : l'utilisateur la fournit dans le `NodeField` de chargement à la position `(multiplier_node, primal_var)`.

À la destruction du SubModel, les refcounts (contraintes + multiplicateurs) sont décrémentés ; les nœuds-multiplicateurs deviennent collectables.

## Règle invariante : un Model = une Matrice

`stiffness()` et `mass()` produisent chacune **une seule** `Matrix` couvrant l'ensemble des DOFs du Model (primaux ⊕ multiplicateurs). Les conditions limites n'ont pas de statut spécial — ce sont des sub-models comme les autres qui contribuent leurs entrées dans la même matrice globale.

Cette uniformité simplifie tout : le solveur reçoit une seule `Matrix` + un seul `NodeField` ; pas besoin de jongler avec un système saddle-point composé.

## API Rust

```rust,ignore
use pyrucast::configuration::Configuration;
use pyrucast::element_field::ElementField;
use pyrucast::element_type::ElementType;
use pyrucast::fe_space::FiniteElementSpace;
use pyrucast::mesh::Mesh;
use pyrucast::model::{Model, SubModel};
use pyrucast::node::Node;
use pyrucast::store::insert;

// 1-D : maillage [0, 1] à un seul SEG2.
let cfg = insert(Configuration::new(1).unwrap());
let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
let mut mesh = Mesh::with_element_type(cfg.clone(), ElementType::SEG2);
mesh.add_cell(&[a.id(), b.id()]).unwrap();
let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
let sub = fes.subspace(0).unwrap();

// Matériau : k = 1 partout.
let mut mat = ElementField::new(sub.clone(), vec!["k".into()]).unwrap();
mat.set_uniform("k", 1.0).unwrap();

// Modèle : conduction + Dirichlet à gauche.
let mut model = Model::new();
model
    .add_sub_model(insert(SubModel::heat_conduction(sub, insert(mat))))
    .unwrap();
model
    .add_sub_model(insert(
        SubModel::dirichlet(cfg.clone(), "T".into(), "q".into(), vec![a.id()]).unwrap(),
    ))
    .unwrap();

let k = model.stiffness().unwrap();
assert_eq!(k.n_rows(), 3);  // 2 nœuds physiques + 1 multiplicateur
```

## API Python

```python
import pyrucast

c = pyrucast.Configuration(dim=1)
a = c.add_node([0.0])
b = c.add_node([1.0])
mesh = pyrucast.Mesh(c, "SEG2")
mesh.add_cell([a.id, b.id])
fes = pyrucast.FiniteElementSpace(mesh)
sub = fes[0]

mat = pyrucast.ElementField(sub, ["k"])
mat.set_uniform("k", 1.0)

model = pyrucast.Model()
model.add_sub_model(pyrucast.SubModel.heat_conduction(sub, mat))
model.add_sub_model(pyrucast.SubModel.dirichlet(c, "T", "q", [a.id]))

K = model.stiffness()
print("primal_vars =", model.primal_vars())   # ['T', 'lambda_T']
print("dual_vars =",   model.dual_vars())     # ['q', 'T']
print(K)                                       # Matrix: 3 row(s) × 3 col(s), …
```

## Solveur dense — fonction `solve(matrix, rhs)`

Pour valider l'assemblage de bout en bout, pyrucast embarque un **solveur dense LU minimal** (`pyrucast::solver::solve` / `pyrucast.solve`) qui :

1. lit le `NodeField` de chargement à chacune des **lignes** de la matrice (les entrées absentes valent `0.0` par défaut) ;
2. convertit la `Matrix` en `nalgebra::DMatrix<f64>` ;
3. factorise `A = LU` et résout `A x = b` ;
4. emballe la solution dans un `NodeField` indexé sur les **colonnes** de la matrice.

C'est un harnais de test — pas le solveur final. Un objet `LinearSolver` enfichable (itératif, direct creux, préconditionné) arrivera en **Phase 3** par-dessus `nalgebra-sparse` ; les conversions `Matrix::to_csr` / `to_csc` sont déjà en place pour le brancher sans changement d'API.

**Exemple complet : Poisson 1-D `-u'' = 0` avec `u(0) = 0`, `u(1) = 1`** (solution analytique `u(x) = x`, multiplicateurs aux bords = flux ±1) :

```python
import pyrucast

# 1) Maillage + FE space + matériau (k = 1)
c = pyrucast.Configuration(dim=1)
nodes = [c.add_node([i / 4.0]) for i in range(5)]
mesh = pyrucast.Mesh(c, "SEG2")
for i in range(4):
    mesh.add_cell([nodes[i].id, nodes[i + 1].id])
fes = pyrucast.FiniteElementSpace(mesh)
sub = fes[0]
mat = pyrucast.ElementField(sub, ["k"])
mat.set_uniform("k", 1.0)

# 2) Modèle : conduction + Dirichlet aux deux bouts
model = pyrucast.Model()
model.add_sub_model(pyrucast.SubModel.heat_conduction(sub, mat))
left = pyrucast.SubModel.dirichlet(c, "T", "q", [nodes[0].id])
right = pyrucast.SubModel.dirichlet(c, "T", "q", [nodes[-1].id])
mult_left = left.multiplier_nodes()[0]
mult_right = right.multiplier_nodes()[0]
model.add_sub_model(left)
model.add_sub_model(right)

# 3) Chargement : valeurs imposées aux nœuds-multiplicateurs
rhs_sm = pyrucast.SubMesh(c, "POI1")
rhs_sm.add_cell([mult_left])
rhs_sm.add_cell([mult_right])
rhs = pyrucast.NodeField(rhs_sm, ["T"])
rhs.set_value(mult_left, "T", 0.0)
rhs.set_value(mult_right, "T", 1.0)

# 4) Assemblage + résolution
K = model.stiffness()
solution = pyrucast.solve(K, rhs)
assert abs(solution.value(nodes[2].id, "T") - 0.5) < 1e-10  # T au milieu = 0.5
assert abs(solution.value(mult_left, "lambda_T") - 1.0) < 1e-10  # flux à gauche
```

## Limitations actuelles

- **Mass non assemblée** : `Model::mass()` retourne une matrice vide en v0. L'intégrande `∫ ρc_p · N_i N_j dx` (et son équivalent pour les autres physiques) est additif et sera ajouté quand le besoin transient se présentera.
- **Une seule physique « réelle »** : `HeatConduction`. `LinearElasticity`, `Timoshenko`, `Periodic`, etc. viendront comme nouvelles variantes de `Physics`, chacune avec son match-arm dans l'assemblage. Le pattern est borné et lisible.
- **Pas de check de cohérence pré-assemblage** : la consistance (matériau définit bien `"k"` pour HeatConduction, compatibilité des FE spaces entre sub-models, etc.) est vérifiée au moment du `stiffness()` / `mass()`, pas à l'ajout du sub-model. Si on découvre des cas où ça pose problème, un check eager est facile à ajouter.
- **Solveur dense seulement** : le `solve` fourni est un harnais de test (LU dense via `nalgebra`). Pour les vrais problèmes Phase 3 introduira un trait `LinearSolver` enfichable (itératifs, direct creux, factorisation Cholesky pour les cas symétriques détectés via le drapeau de la `Matrix`).
