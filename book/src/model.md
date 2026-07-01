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
├── element_matrix          # noyau élémentaire (une cellule) — pur & séquentiel
├── stiffness_layout        # Some ⇒ bloc CALCULÉ (scatter parallèle) ; None ⇒ littéral
├── build_stiffness_blocks  # voie littérale : enveloppe element_matrix via assemble_block
├── build_mass_blocks       # (défaut : vide)
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

## Les variantes de `Physics`

Chaque physique est une struct sous `src/models/` implémentant le trait
`Physics`, enveloppée par une variante de l'énum `SubModel`. Leur **détail**
(équations, matériau, comportement, exemples) est dans la partie
[Détails des physiques](physiques.md) ; on n'en rappelle ici que les
constructeurs et les variables, vus du `Model` :

| Constructeur (`Model::…`) | Primales | Duales | Matériau | Chapitre |
|---|---|---|---|---|
| `heat_conduction(fes)` | `T` | `q` | `k` | [Thermique](thermique.md) |
| `truss(fes)` | `u_x, u_y(, u_z)` | `f_x, f_y(, f_z)` | `E, A` | [Barre](mecanique/truss.md) |
| `elasticity(fes, model)` | `u_x, u_y(, u_z)` | `f_x, f_y(, f_z)` | `E, nu` | [Élasticité](mecanique/elasticite.md) |
| `timoshenko(fes)` | `w, theta` | `f_w, m_theta` | `E, I, G, A_s` | [Timoshenko](mecanique/timoshenko.md) |
| `frame(fes)` | `u_x, u_y, rz` | `f_x, f_y, m_z` | `E, A, I, G, A_s` | [Portique 2D](mecanique/portique.md) |
| `frame3d(fes)` | `u_x…r_z` (6) | `f_x…m_z` (6) | `E, A, I_y, I_z, J, G, A_sy, A_sz` | [Cadre 3D](mecanique/cadre3d.md) |
| `dirichlet(…)` | `lambda_<v>` | `imposed_<v>` | — | [Dirichlet](contraintes/dirichlet.md) |

Toutes balaient **tous** les sous-espaces du `fes` (une zone par sous-espace),
sauf `dirichlet` qui est une [contrainte](contraintes/dirichlet.md) portée par
des maillages fournis par l'utilisateur. Le matériau est toujours fourni **à
l'assemblage**, pas au modèle (cf. ci-dessous).

Pour **ajouter** une physique, voir [Ajouter une physique](ajouter-une-physique.md).

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
print("primal_vars =", model.primal_vars())  # ['T', 'lambda_T']
print("dual_vars =", model.dual_vars())  # ['q', 'imposed_T']
print(K)  # Matrix: 3 row(s) × 3 col(s), …
```

## Assemblage et résolution

L'assemblage (`stiffness` / `mass`) et la résolution (`solve`) sont des
**opérateurs** : ils consomment le `Model` (et le matériau, le chargement) et
sont décrits dans la partie [Détail des opérateurs](operateurs.md) —
[Assemblage](operateurs/assemblage.md) et [Solveur](operateurs/solveur.md). Le
solveur fourni est un harnais LU dense ; un `LinearSolver` enfichable arrivera
en Phase 3.

Des exemples **complets et à solution analytique** (assemblage + contraintes +
lecture des inconnues et des réactions) sont déroulés dans
[Dirichlet](contraintes/dirichlet.md) (Poisson 1-D),
[Conduction thermique](thermique.md) et [Mécanique](mecanique.md).

## Limitations actuelles

- **Mass non assemblée** : `assemble::mass(model)` retourne une matrice vide en v0. L'intégrande `∫ ρc_p · N_i N_j dx` (et son équivalent pour les autres physiques) est additif et sera ajouté quand le besoin transient se présentera.
- **Physiques disponibles** : `HeatConduction` ([thermique](thermique.md)), `Truss`, `LinearElasticity`, `Timoshenko`, `Frame`, `Frame3d` ([mécanique](mecanique.md)) et la contrainte `Dirichlet` ([contraintes](contraintes.md)). Toute nouvelle physique est une struct implémentant `Physics` (une variante de l'énum `SubModel` + un bras de `as_physics`, rien d'autre — cf. [Ajouter une physique](ajouter-une-physique.md)). Le coût d'ajout est O(1) fichier, indépendant du nombre de physiques existantes.
- **Pas de check de cohérence pré-assemblage** : la consistance (matériau définit bien `"k"` pour HeatConduction, compatibilité des FE spaces entre sub-models, etc.) est vérifiée au moment de `assemble::stiffness` / `assemble::mass`, pas à l'ajout du sub-model. Si on découvre des cas où ça pose problème, un check eager est facile à ajouter.
- **Solveur dense seulement** : le `solve` fourni est un harnais de test (LU dense via `nalgebra`). Pour les vrais problèmes Phase 3 introduira un trait `LinearSolver` enfichable (itératifs, direct creux, factorisation Cholesky pour les cas symétriques détectés via le drapeau de la `Matrix`).
