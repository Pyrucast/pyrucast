# Modèle physique (`Model`)

Un **`Model`** est l'objet orchestrateur qui décrit un problème physique et **produit les matrices** (raideur, masse) à la demande de l'utilisateur. Il vient se poser sur la couche éléments finis ([`FiniteElementSpace`](fe-space.md)), elle-même posée sur le maillage géométrique.

```text
Géométrie         Mesh / SubMesh                       (purement géométrique)
Formulation EF    FiniteElementSpace / SubFiniteElementSpace      (interpolation, quadrature)
Physique          Model / SubModel / SubModelKind      (loi, matériaux, assemblage)
```

## Architecture

```text
Model
├── sub_models: Vec<SubModel>
├── primal_vars(): Vec<String>      # union — colonnes des matrices
├── dual_vars():   Vec<String>      # union — lignes des matrices
├── filter(Physics) -> Model        # sous-modèles d'une nature donnée
└── fespace() -> FiniteElementSpace # 1 sous-espace par sous-modèle de domaine
                                     # (contraintes exclues, sans dédup)

ops::matrix (opérateurs, pas des méthodes de Model)
├── stiffness(model, materials) -> Matrix   # K  (assemblé sur demande)
└── mass(model)                 -> Matrix   # M  (assemblé sur demande)

SubModel  (énum de stockage + dispatch — AUCUNE logique)
├── HeatConduction(HeatConduction)    # chaque variante enveloppe une struct…
├── Dirichlet(Dirichlet)             # …qui porte ses données + impl SubModelKind
├── Mpc(Mpc)                         # contrainte multi-points (relations linéaires)
├── fespace() -> Option<SubFiniteElementSpace>  # sous-espace intégré (None si contrainte)
└── as_kind(&self) -> &dyn SubModelKind   # l'unique match du module modèle

SubModelKind  (trait de base — le dénominateur commun, co-localisé par physique)
├── primal_vars / dual_vars
├── physics       # ensemble de natures : &[Physics] (Mechanical|Thermal|Constraint|Other)
├── as_domain / as_constraint  # seams de capacité (None par défaut) — cf. ci-dessous
├── element_matrix          # noyau élémentaire (une cellule) — pur & séquentiel
├── stiffness_layout        # Some ⇒ bloc CALCULÉ (scatter parallèle) ; None ⇒ littéral
├── contributions           # défaut : dérivé du layout ; contraintes rendent leurs C/Cᵀ littéraux
├── build_stiffness_blocks  # défaut : dérivé de stiffness_layout + element_matrix
├── build_mass_blocks       # (défaut : vide)
└── label / display / render

Capacités (sous-traits, miroir des natures ; une struct n'a que la sienne) :
├── Domain      # matériau + comportement (heat, elasticity, poutres, …)
└── Constraint  # multiplicateurs de Lagrange (Dirichlet, MPC, embedded, contact)
```

L'énum `SubModel` ne sert qu'au **stockage** et à la **sérialisation**
(`bincode`) ; il délègue chaque appel à l'`impl SubModelKind` de la variante via
`as_kind()`. Tout le code générique (l'agrégat `Model`, l'assembleur,
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
| `Convection` (film) | `T` (partagée avec `HeatConduction`) | `q` (partagée) |
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

Pour la part **contrainte** du second membre (les valeurs imposées aux
nœuds-multiplicateurs), le helper `model.constraint_rhs([(nœud, g), …])`
construit ce `NodeField` tout seul : on désigne chaque relation par un nœud
contraint (Dirichlet) ou un nœud-terme (MPC) et sa valeur `g`, et le helper
retrouve le nœud-multiplicateur et la composante à renseigner
(`imposed_<v>`, `mpc_rhs`). Voir [Contraintes](contraintes.md).

## Les physiques disponibles

Chaque physique est une struct sous `src/models/` implémentant le trait
`SubModelKind`, enveloppée par une variante de l'énum `SubModel`. Leur **détail**
(équations, matériau, comportement, exemples) est dans la partie
[Détails des physiques](physiques.md) ; on n'en rappelle ici que les
constructeurs et les variables, vus du `Model` :

| Constructeur (`Model::…`) | Primales | Duales | Matériau | Chapitre |
|---|---|---|---|---|
| `heat_conduction(fes)` | `T` | `q` | `k` | [Thermique](thermique.md) |
| `heat_conduction_with_symmetry(fes, sym)` | `T` | `q` | `k_1…` / `k_11…` + repère | [Conduction orientée](thermique.md#conduction-orthotrope-et-anisotrope) |
| `convection(fes)` | `T` | `q` | `h` | [Thermique](thermique.md#convection-de-surface-robin--film) |
| `radiation(fes)` | `T` | `q` | `emis, T_inf` (+ `sigma` facultatif) | [Rayonnement](thermique.md#rayonnement-à-linfini-stefan-boltzmann) |
| `fick(fes, sym)` | `c` | `j` | `D` (ou `D_1…` / `D_11…` + repère) ; `poro` facultatif | [Diffusion](diffusion.md) |
| `interface_transfer(a, b, kind, tol)` | `c` ou `T` | `j` ou `q` | `h` | [Transfert d'interface](diffusion.md#transfert-à-travers-une-interface) |
| `truss(fes)` | `u_x, u_y(, u_z)` | `f_x, f_y(, f_z)` | `E, A` | [Barre](mecanique/truss.md) |
| `elasticity(fes, model)` | `u_x, u_y(, u_z)` | `f_x, f_y(, f_z)` | `E, nu` | [Élasticité](mecanique/elasticite.md) |
| `elasticity_with_symmetry(fes, model, sym)` | `u_x, u_y(, u_z)` | `f_x, f_y(, f_z)` | `E_1…G_23` / `C_11…C_66` + repère | [Orthotropie](mecanique/orthotropie.md) |
| `follower_pressure(fes)` | `u_x, u_y(, u_z)` | `f_x, f_y(, f_z)` | `p` | [Pression suiveuse](mecanique/pression-suiveuse.md) |
| `plasticity_perfect(fes, model)` | `u_x, u_y(, u_z)` | `f_x, f_y(, f_z)` | `E, nu, sigma_y` | [Plasticité](mecanique/plasticite.md) |
| `plasticity_with_law(fes, model, law)` | idem | idem | selon la loi | [Lois d'écoulement](mecanique/lois-plastiques.md), [Fluage](mecanique/fluage.md) |
| `bernoulli(fes, model)` | selon la configuration | idem | `E, I` (+ `A`, `I_y…`) | [Euler-Bernoulli](mecanique/bernoulli.md) |
| `timoshenko(fes)` | `w, theta` | `f_w, m_theta` | `E, I, G, A_s` | [Timoshenko](mecanique/timoshenko.md) |
| `frame(fes)` | `u_x, u_y, rz` | `f_x, f_y, m_z` | `E, A, I, G, A_s` | [Portique 2D](mecanique/portique.md) |
| `frame3d(fes)` | `u_x…r_z` (6) | `f_x…m_z` (6) | `E, A, I_y, I_z, J, G, A_sy, A_sz` | [Cadre 3D](mecanique/cadre3d.md) |
| `shell(fes, model)` | `u_x…r_z` (6) | `f_x…m_z` (6) | `E, nu, h` | [Coques](mecanique/coques.md) |
| `dirichlet(…)` | `lambda_<v>` | `imposed_<v>` | — | [Dirichlet](contraintes/dirichlet.md) |
| `mpc(…)` | `lambda_mpc` | `mpc_rhs` | — | [Multi-points](contraintes/mpc.md) |
| `embedded(…)` | `lambda_<v>` | `imposed_<v>` | — | [Baignage](contraintes/embedded.md) |
| `contact(…)` | `lambda_contact` | `contact_gap` | — | [Contact](contraintes/contact.md) |

Toutes balaient **tous** les sous-espaces du `fes` (une zone par sous-espace),
sauf `dirichlet`, `mpc`, `embedded` et `contact` qui sont des
[contraintes](contraintes.md) portées par des maillages fournis par
l'utilisateur. Le matériau est toujours fourni **à l'assemblage**, pas au
modèle (cf. ci-dessous).

Pour **ajouter** une physique, voir [Ajouter une physique](ajouter-une-physique.md).

## Ce que chaque physique calcule

Le tableau ci-dessus dit *quelles variables* porte chaque physique. Celui-ci dit
**ce qu'elle sait produire** : les genres de matrice qu'elle déclare
([`MatrixKind`](#un-genre-de-matrice--un-layout--un-noyau)), la nature de son
intégration de comportement, et la particularité de calcul qui la distingue.

Une case vide n'est pas un manque : une physique qui ne déclare pas un genre n'y
contribue simplement rien, et l'assembleur l'ignore. `matrix.mass(...)` sur un
modèle qui contient une pression suiveuse ne voit qu'elle n'a pas de masse.

| Physique | `tangent` — et par quelle voie | `geometric` | `mass` | Comportement (`COMP`) | Particularité de calcul |
|---|---|---|---|---|---|
| `heat_conduction` | — | — | capacité `ρ·cp` | flux `K·∇T` | symétrie matériau iso/ortho/aniso ; `k` lu **par point de Gauss** |
| `convection` | — | — | — | `h·T` | aveugle à l'orientation du bord (la normale est déjà dans `q·n`) |
| `radiation` | **analytique** — `4σεT³` | — | — | flux **+** `ktan` | non linéaire ; rigidité **linéarisée autour de `T_∞`** donc constante ; deux natures `[Thermal, Radiation]` |
| `fick` | — | — | stockage `poro` | flux `D·∇c` | nature `Diffusion` propre ; flux nommé `j_*` pour cohabiter avec la thermique |
| `interface_transfer` | — | — | — | `h·(c₁−c₂)` | **quatre blocs**, dont deux `Coupling` inter-maillages ; scatter séquentiel ; conformité vérifiée jusqu'au nœud |
| `truss` | — | oui `N/L·P` | oui `ρA` | `N = E A ε` | forme fermée globale, sans matrice de repère |
| `elasticity` | **analytique** — c'est `K` | oui | oui | `σ = D·ε` | loi linéaire ⇒ la tangente *est* la rigidité ; symétrie matériau |
| `follower_pressure` | — | — | — | traction `−p·n(u)` | **aucune matrice** (`contributions` vide) ; direction recalculée à chaque résidu ; seule physique sensible au sens de parcours du bord |
| `plasticity` | **analytique** (2 lois) / **par perturbation** (8 lois) | oui | oui | σ, `ε_p`, `p` + variables de loi | dix lois en attribut ; tangente toujours **symétrisée** ; les lois visqueuses **exigent `dt`** |
| `damage` | *aucune* | oui | oui | σ, `damage`, histoire | trois lois ; l'opérateur d'itération reste la rigidité **non endommagée** |
| `timoshenko` | — | — | oui | `M`, `V` | **multi-quadrature** : flexion complète, cisaillement réduit |
| `bernoulli` | — | — | — | efforts de section | Hermite, **exact aux nœuds** ; ne demande ni `G` ni `A_s` |
| `frame` / `frame3d` | — | oui | oui | efforts de section | repère local déduit automatiquement de la géométrie |
| `shell` | — | — | — | `N`, `M`, `Q` | multi-quadrature ; six DDL par nœud ; vrillage lié à la rotation de membrane |
| `dirichlet` `mpc` `embedded` `contact` | — | — | — | — | aucun layout : blocs `Literal` C / Cᵀ montés directement |

### « Par perturbation » veut dire différences finies centrées

Pour les huit lois qui n'ont pas de tangente en forme fermée, `D_alg` est obtenu
en **perturbant la déformation** et en relançant le retour, composante par
composante :

\\[
D_{ij} \simeq \frac{\sigma_i(\varepsilon + h\\,e_j) - \sigma_i(\varepsilon - h\\,e_j)}{2h},
\qquad h = 10^{-6}\\,\lVert \varepsilon \rVert_\infty .
\\]

Six composantes de Voigt, deux évaluations chacune : **douze appels** au retour
par point de Gauss. Le pas doit rester bien au-dessus du bruit du retour (celui
d'Ottosen ou de Gurson converge à une tolérance, pas exactement) et bien en
dessous de l'échelle de courbure de la surface ; `1e-6·‖ε‖` tient
confortablement entre les deux. Les colonnes de cisaillement sont divisées par
deux en sortie, ce qui transforme `∂σ/∂ε_ij` en `∂σ/∂γ_ij` — la convention
ingénieur du reste du dépôt.

| voie | lois |
|---|---|
| **analytique** | `plasticity_perfect`, `plasticity_isotropic` (le module algorithmique J2) |
| **par perturbation** | `drucker_prager`, `ottosen`, `gurson`, les trois fluages, les deux Chaboche |

Le partage n'est pas une question de difficulté mais de **vérifiabilité** : seule
la forme fermée de von Mises a été confrontée à une différence finie et validée.
La dérivation analytique de Drucker-Prager, écrite d'abord, était fausse de 24 %
— plausible, et fausse ; seul l'oracle numérique l'a dit. Une tangente obtenue
par perturbation ne peut pas être mal dérivée, coûte douze évaluations d'une mise
à jour bon marché, et laisse Newton converger. C'est un bon échange.

Le rayonnement, lui, a bien une tangente **analytique** (`4σεT³ ∫NᵢNⱼ`) : sa
non-linéarité est une puissance scalaire d'une seule variable, pas une carte de
projection.

### Trois choses que ce tableau ne dit pas

**Les forces internes ne suivent pas toujours `Bᵀσ`.** Le défaut est le noyau de
la mécanique des milieux continus, `f_i = ∫ ∂N_i/∂x · σ`. Une physique dont la
duale n'est pas un vecteur déplacement le redéfinit : la thermique et la
diffusion appliquent `Bᵀ` à un flux scalaire, tandis que la convection, le
rayonnement, le transfert d'interface et la **pression suiveuse** pondèrent par
`N` et non par `Bᵀ` — leur intégrande est une densité surfacique, pas une
grandeur conjuguée d'un gradient.

**La tangente stockée est symétrique.** `D_alg` transite par le champ d'état sous
forme de triangle supérieur (`ktan_i_j`, i ≤ j) et est relue en miroir. Une loi à
écoulement **non associé** (Drucker-Prager) a pourtant une tangente non
symétrique : elle est donc symétrisée, ce qui coûte à Newton son taux quadratique
sur cette loi et rien d'autre. Voir
[Lois d'écoulement plastique](mecanique/lois-plastiques.md#la-tangente-cohérente-et-deux-limites-assumées).

**Une loi peut porter son propre état.** Au-delà de `ε_p` et `p`, une loi déclare
ce qu'elle veut par `internal_names()` : la déformation primaire de Blackburn, la
contrainte de rappel de Chaboche (un tenseur complet), la porosité de Gurson, les
deux histoires de Damage TC. Le premier pas passe un vecteur **vide** — et non
un vecteur de zéros — pour qu'une loi démarrant d'une constante matériau puisse
distinguer « pas encore d'état » de « état nul ».


## Nature physique et filtrage

Chaque physique déclare un **ensemble de natures** — sa classification grossière,
orthogonale à l'axe de capacité `Domain`/`Constraint`. Elle répond à « quel champ
de physique » là où les capacités répondent à « domaine ou contrainte » :

| Nature (`Physics`) | Physiques |
|---|---|
| `Mechanical` | `truss`, `elasticity`, `plasticity`, `mazars`, `timoshenko`, `frame`, `frame3d`, `follower_pressure` |
| `Thermal`    | `heat_conduction`, `convection`, `radiation`, `interface_transfer` (variante `thermal`) |
| `Constraint` | `dirichlet`, `mpc`, `embedded`, `contact` |
| `Other`      | nature « autre / rien » explicite (aucune physique de base ne la déclare) |
| `Diffusion`  | `fick`, `interface_transfer` (variante `mass`) |
| `Radiation`  | `radiation` — portée **en plus** de `Thermal`, donc `filter("thermal")` le rend aussi |

Côté Python, les mêmes natures sont des chaînes : `"mechanical"`, `"thermal"`,
`"constraint"`, `"other"`, `"diffusion"`, `"radiation"`.

`Diffusion` est une nature à part entière bien que la loi de Fick partage
l'opérateur de la conduction : les variables diffèrent (`c`/`j` contre `T`/`q`),
et un problème couplé doit pouvoir sélectionner l'une sans l'autre. Partager un
opérateur n'est pas partager une physique.

La nature d'une physique de base est **entièrement déterminée par la variante** :
c'est une constante par physique, exposée par `SubModelKind::physics()` — un
**slice** `&'static [Physics]` (comme `label()`), pas un champ stocké. Le type est
un **ensemble** pour deux raisons :

- une physique **couplée** (par ex. un futur élément thermo-mécanique) porte
  *plusieurs* natures — `[Mechanical, Thermal]` ;
- un bloc de matrice monté à la main, hors assemblage, n'en porte **aucune** —
  l'ensemble vide, le cas « rien ». `Physics::Other` est la nature « autre »
  *explicite*, pour un bloc qu'on veut classer plutôt que laisser sans étiquette.

L'ensemble voyage avec chaque bloc assemblé jusqu'à la [`SubMatrix`](matrix.md)
(posé par l'assembleur sur les deux chemins, calculé et littéral, donc le couple
C/Cᵀ d'un Dirichlet est étiqueté aussi).

Deux sélecteurs symétriques en découlent — ils gardent les entités dont
l'ensemble **contient** la nature (une physique couplée apparaît donc sous
chacune) — tous deux à partage par compteur de références (pas de copie
profonde) :

- `model.filter(Physics::Mechanical)` → un `Model` ne gardant que les
  sous-modèles au moins mécaniques ;
- `k.filter(Physics::Mechanical)` → une `Matrix` ne gardant que les blocs au moins
  mécaniques (non assemblée — relancer `assemble::assemble` avant de résoudre).

`k.physics()` renvoie l'ensemble des natures **présentes** dans la matrice
(dédupliqué) : une matrice agrégeant plusieurs physiques y expose plusieurs tags
(par ex. `[Thermal, Constraint]`). Un bloc à l'ensemble vide n'est jamais
sélectionné par une nature concrète — l'étiqueter `Physics::Other` le rend
atteignable par `filter(Physics::Other)`.

```rust,ignore
let meca = model.filter(Physics::Mechanical)?;   // sous-modèles au moins mécaniques
let k_meca = k.filter(Physics::Mechanical)?;      // blocs au moins mécaniques (non assemblés)
let natures = k.physics()?;                        // ex. [Thermal, Constraint]
```

## Règle invariante : un Model = une Matrice

`assemble::stiffness(model, materials)` et `assemble::mass(model)` produisent chacune **une seule** `Matrix` couvrant l'ensemble des DOFs du Model (primaux ⊕ multiplicateurs). Les conditions limites n'ont pas de statut spécial — ce sont des sub-models comme les autres qui contribuent leurs entrées dans la même matrice globale.

Cette uniformité simplifie tout : le solveur reçoit une seule `Matrix` + un seul `NodeField` ; pas besoin de jongler avec un système saddle-point composé.

## API Rust

```rust,ignore
use pyrucast::coords::Coords;
use pyrucast::atoms::element_type::ElementType;
use pyrucast::atoms::node::Node;
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
imposed = pyrucast.Mesh.poi1_from_nodes([a])
multiplier = pyrucast.mesh.barycenter(imposed)
model = pyrucast.Model.heat_conduction(fes) | pyrucast.Model.dirichlet(
    "T", "q", imposed, multiplier
)

# Matériau k = 1 (les sous-modèles Dirichlet sont ignorés automatiquement).
materials = pyrucast.element_field.material_field(model, [("k", 1.0)])

K = pyrucast.matrix.stiffness(model, materials)
print("primal_vars =", model.primal_vars())  # ['T', 'lambda_T']
print("dual_vars =", model.dual_vars())  # ['q', 'imposed_T']
print(K)  # Matrix: 3 row(s) × 3 col(s), …
```

## Assemblage et résolution

L'assemblage (`stiffness` / `mass`) et la résolution (`solve`) sont des
**opérateurs** : ils consomment le `Model` (et le matériau, le chargement) et
sont décrits dans la partie [Détail des opérateurs](operateurs.md) —
[Assemblage](operateurs/assemblage.md) et [Solveur](operateurs/solveur.md). Le
solveur est une LU creuse directe (faer), dont la factorisation est mise en
cache sur la `Matrix` — *factoriser une fois, résoudre souvent*.

Des exemples **complets et à solution analytique** (assemblage + contraintes +
lecture des inconnues et des réactions) sont déroulés dans
[Dirichlet](contraintes/dirichlet.md) (Poisson 1-D),
[Conduction thermique](thermique.md) et [Mécanique](mecanique.md).

## Limitations actuelles

- **Physiques disponibles** : `HeatConduction` et `Convection` (échange de
  surface / film) ([thermique](thermique.md)) ; `Truss`, `Elasticity`,
  `Plasticity`, `Mazars`, `Timoshenko`, `Frame`, `Frame3d`
  ([mécanique](mecanique.md)) ; et les contraintes `Dirichlet`, `Mpc`,
  `Embedded`, `Contact` ([contraintes](contraintes.md)). Toute nouvelle physique
  est une struct implémentant `SubModelKind` (une variante de l'énum `SubModel`
  + un bras de `as_kind`, rien d'autre — cf. [Ajouter une
  physique](ajouter-une-physique.md)). Le coût d'ajout est O(1) fichier,
  indépendant du nombre de physiques existantes.
- **Toutes les physiques n'ont pas tous les genres de matrice** : chacune
  déclare les `MatrixKind` qu'elle sait produire (raideur, masse, raideur
  géométrique, tangente cohérente). Assembler un genre qu'une physique n'a pas
  ne casse rien — elle ne contribue simplement pas.
- **Pas de check de cohérence pré-assemblage** : la consistance (matériau
  définit bien `"k"` pour HeatConduction, compatibilité des FE spaces entre
  sub-models, etc.) est vérifiée au moment de `matrix.stiffness` /
  `matrix.mass`, pas à l'ajout du sub-model. Si on découvre des cas où ça pose
  problème, un check eager est facile à ajouter.
- **Un seul back-end de solveur** : la résolution passe par une LU creuse
  directe (faer), avec cache de factorisation. Ni méthode itérative, ni
  factorisation Cholesky exploitant le drapeau de symétrie de la `Matrix` —
  `SolveMethod` est le point d'extension prévu pour cela.
