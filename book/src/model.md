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
├── matrix_element          # pont vers les noyaux élémentaires, qui vivent sur Domain
├── stiffness_layout        # Some ⇒ bloc CALCULÉ (scatter parallèle) ; None ⇒ littéral
├── contributions           # défaut : dérivé du layout ; contraintes rendent leurs C/Cᵀ littéraux
├── build_stiffness_blocks  # défaut : stiffness_layout + Domain::element_matrix
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
| `BoundaryTransfer` (film) | `T` (partagée avec `HeatConduction`) | `q` (partagée) |
| `Truss` / `LinearElasticity` | `u_x`, `u_y`, …   | `f_x`, `f_y`, …    |
| `Dirichlet { imposed_variable: "T" }` | `lambda`           | `imposed`           |

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
retrouve le nœud-multiplicateur et y renseigne la composante `imposed`. Voir
[Contraintes](contraintes.md).

## Les physiques disponibles

Chaque physique est une struct sous `src/models/` implémentant le trait
`SubModelKind`, enveloppée par une variante de l'énum `SubModel`. Leur **détail**
(équations, matériau, comportement, exemples) est dans la partie
[Détails des physiques](physiques.md) ; on n'en rappelle ici que les
opérateurs qui les déclarent et leurs variables, vus du `Model` :

| Opérateur (`ops::model::…`, Python `pyrucast.model.…`) | Primales | Duales | Matériau | Chapitre |
|---|---|---|---|---|
| `heat_conduction(fes, sym)` | `T` | `q` | `k` ; orientée : `k_1…` / `k_11…` + repère | [Thermique](thermique.md), [conduction orientée](thermique.md#conduction-orthotrope-et-anisotrope) |
| `boundary_transfer(fes, cible, comps)` | libres | libres | `h_<primale>`, `a_ext_<primale>` | [Échanges](echanges.md) |
| `radiation(fes, cible)` | `T` | `q` | `emis, T_inf` (+ `sigma` facultatif) | [Rayonnement](thermique.md#rayonnement-à-linfini-stefan-boltzmann) |
| `fick(fes, espèce, sym)` | `c_<espèce>` | `j_<espèce>` | `D_<espèce>` ; orientée : `D_1_<espèce>…` + repère ; `poro` facultatif | [Diffusion](diffusion.md) |
| `interface_transfer(a, b, cible, comps, tol)` | libres | libres | `h_<primale>` | [Échanges](echanges.md) |
| `truss(fes, mesure)` | `u_x, u_y(, u_z)` | `f_x, f_y(, f_z)` | `E, A` | [Barre](mecanique/truss.md) |
| `elasticity(fes, cinématique, sym, mesure)` | `u_x, u_y(, u_z)` | `f_x, f_y(, f_z)` | `E, nu` ; orientée : `E_1…G_23` / `C_11…C_66` + repère | [Élasticité](mecanique/elasticite.md), [orthotropie](mecanique/orthotropie.md) ; en grandes transformations, [mesure de déformation](mecanique/grandes-transformations.md) |
| `plasticity_perfect(fes, cinématique, mesure)` | `u_x, u_y(, u_z)` | `f_x, f_y(, f_z)` | `E, nu, sigma_y` | [Plasticité](mecanique/plasticite.md) |
| `plasticity(fes, cinématique, loi, mesure)` | idem | idem | selon la loi | [Lois d'écoulement](mecanique/lois-plastiques.md), [Fluage](mecanique/fluage.md) |
| `bernoulli(fes)` | selon la configuration | idem | `E, I` (+ `A`, `I_y…`) | [Euler-Bernoulli](mecanique/bernoulli.md) |
| `timoshenko(fes)` | selon la configuration lue sur le maillage : `w, theta` (1-D), `u_x, u_y, r_z` (plan), `u_x…r_z` (6, espace) | `f_w, m_theta` ; `f_x, f_y, m_z` ; `f_x…m_z` | 1-D : `E, I, G, A_s` ; plan : `E, A, I, G, A_s` ; espace : `E, A, I_y, I_z, J, G, A_sy, A_sz` | [Timoshenko](mecanique/timoshenko.md) |
| `shell(fes, model)`<br>`thick`, `kirchhoff` | `u_x…r_z` (6) | `f_x…m_z` (6) | `E, nu, h` | [Coques](mecanique/coques.md) |
| `dirichlet(cible, variable, maillage, sens)` | `lambda` | `imposed` | — | [Dirichlet](contraintes/dirichlet.md) |
| `mpc(cible, termes, sens)` | `lambda` | `imposed` | — | [Multi-points](contraintes/mpc.md) |
| `embedded(cible, immergé, hôte, variables, tol)` | `lambda` | `imposed` | — | [Baignage](contraintes/embedded.md) |
| `contact(cible, esclave, maître, variables)` | `lambda` | `imposed` | — | [Contact](contraintes/contact.md) |

Toutes balaient **tous** les sous-espaces du `fes` (une zone par sous-espace),
sauf `dirichlet`, `mpc`, `embedded` et `contact` qui sont des
[contraintes](contraintes.md) portées par des maillages fournis par
l'utilisateur. Le matériau est toujours fourni **à l'assemblage**, pas au
modèle (cf. ci-dessous).

Pour **ajouter** une physique, voir [Ajouter une physique](ajouter-une-physique.md).

## Ce que chaque physique calcule

Le tableau ci-dessus dit *quelles variables* porte chaque physique. Ce qu'elle
**sait produire** — les genres de matrice qu'elle déclare, la voie par laquelle
elle obtient sa tangente, son intégration de comportement et sa particularité de
calcul — est rassemblé physique par physique dans
[Détails des physiques](physiques.md#ce-que-chacune-sait-produire).


## Nature physique et filtrage

Chaque physique déclare un **ensemble de natures** — sa classification grossière,
orthogonale à l'axe de capacité `Domain`/`Constraint`. Elle répond à « quel champ
de physique » là où les capacités répondent à « domaine ou contrainte » :

| Nature (`Physics`) | Physiques |
|---|---|
| `Mechanical` | `truss`, `elasticity`, `plasticity`, `mazars`, `bernoulli`, `timoshenko`, `shell` |
| `Thermal`    | `heat_conduction`, `radiation` ; `boundary_transfer` et `interface_transfer` quand leur cible est thermique |
| `Constraint` | `dirichlet`, `mpc`, `embedded`, `contact` — portée **avec** la nature de la physique qu'ils contraignent (`[Constraint, Mechanical]` pour un Dirichlet sur `u_x`) |
| `Other`      | nature « autre / rien » explicite (aucune physique de base ne la déclare) |
| `Diffusion`  | `fick` ; `boundary_transfer` et `interface_transfer` quand leur cible est une diffusion |
| `Radiation`  | `radiation` — portée **en plus** de `Thermal`, donc `filter("thermal")` le rend aussi |

Côté Python, les mêmes natures sont des chaînes : `"mechanical"`, `"thermal"`,
`"constraint"`, `"other"`, `"diffusion"`, `"radiation"`.

`Diffusion` est une nature à part entière bien que la loi de Fick partage
l'opérateur de la conduction : les variables diffèrent (`c`/`j` contre `T`/`q`),
et un problème couplé doit pouvoir sélectionner l'une sans l'autre. Partager un
opérateur n'est pas partager une physique.

La nature d'une physique de base est **entièrement déterminée par la variante** :
c'est une constante par physique, exposée par `SubModelKind::physics()` — un
**slice** `&'static [Physics]` (comme `label()`). Les termes qui écrivent dans
les lignes d'une autre physique la lisent sur leur cible à la construction :
`boundary_transfer`, `interface_transfer` et `flux` prennent sa nature, et une
**contrainte** porte `Constraint` **plus** la nature de la physique dont elle
fige les inconnues. Une relation qui lie les inconnues de deux physiques n'en
porte aucune des deux : aucun sous-problème mono-physique ne saurait la
résoudre. Le type est un **ensemble** pour trois raisons :

- une physique **couplée** (par ex. un futur élément thermo-mécanique) porte
  *plusieurs* natures — `[Mechanical, Thermal]` ;
- une contrainte appartient à la fois aux contraintes et à la physique qu'elle
  fige — `[Constraint, Mechanical]` ;
- un bloc de matrice monté à la main, hors assemblage, n'en porte **aucune** —
  l'ensemble vide, le cas « rien ». `Physics::Other` est la nature « autre »
  *explicite*, pour un bloc qu'on veut classer plutôt que laisser sans étiquette.

L'ensemble voyage avec chaque bloc assemblé jusqu'à la [`SubMatrix`](matrix.md)
(posé par l'assembleur sur les deux chemins, calculé et littéral, donc le couple
C/Cᵀ d'un Dirichlet est étiqueté aussi).

Deux sélecteurs en découlent, chacun présent sur `Model` et sur `Matrix`, tous
deux à partage par compteur de références (pas de copie profonde) :

- `filter(nature)` garde les entités dont l'ensemble **contient** la nature (une
  physique couplée apparaît donc sous chacune) ;
- `exclude(nature)`, son complément, garde celles dont l'ensemble **ne la
  contient pas**.

Ils se composent, une condition chacun. Comme une contrainte porte la nature de
ce qu'elle fige, `model.filter(Physics::Mechanical)` rend la mécanique **avec**
ses appuis — un sous-problème solvable tel quel — et
`model.filter(Physics::Mechanical).exclude(Physics::Constraint)` la mécanique
**seule**, par exemple pour lire ses seules inconnues de déplacement. Côté
matrice, `k.filter(...)` rend une `Matrix` non assemblée — relancer
`Matrix::assemble` avant de résoudre.

Les natures disent ce qu'un sous-modèle fige, pas ce qu'il est : un contact et
un Dirichlet sur les déplacements portent tous deux `[Constraint, Mechanical]`.
`label()` nomme la physique (`"Dirichlet"`, `"Contact"`, …) ; c'est par lui que
le [pas à pas](pas-a-pas.md) reconnaît le contact.

`k.physics()` renvoie l'ensemble des natures **présentes** dans la matrice
(dédupliqué) : une matrice agrégeant plusieurs physiques y expose plusieurs tags
(par ex. `[Thermal, Constraint]` pour une conduction et son Dirichlet). Un bloc à l'ensemble vide n'est jamais
sélectionné par une nature concrète — l'étiqueter `Physics::Other` le rend
atteignable par `filter(Physics::Other)`.

```rust,ignore
{{#include ../../tests/doc_conventions.rs:filtrer_par_nature}}
```

## Règle invariante : un Model = une Matrice

`matrix::stiffness(model, materials)` et `matrix::mass(model, materials)` produisent chacune **une seule** `Matrix` couvrant l'ensemble des DOFs du Model (primaux ⊕ multiplicateurs). Les conditions limites n'ont pas de statut spécial — ce sont des sub-models comme les autres qui contribuent leurs entrées dans la même matrice globale.

Cette uniformité simplifie tout : le solveur reçoit une seule `Matrix` + un seul `NodeField` ; pas besoin de jongler avec un système saddle-point composé.

## API Rust

```rust,ignore
{{#include ../../tests/doc_conventions.rs:modele}}
```

## API Python

```python
{{#include ../../tests/python/test_doc_conteneurs.py:model_api}}
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

- **Physiques disponibles** : `HeatConduction` et `BoundaryTransfer` (échange de
  surface / film) ([thermique](thermique.md)) ; `Truss`, `Elasticity`,
  `Plasticity`, `Damage`, `Bernoulli`, `Timoshenko`, `Shell`
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
- **Deux back-ends de solveur, tous deux directs** : LU creuse (défaut) et
  Cholesky, au choix de l'appelant via `method=`, toutes deux en faer et avec
  cache de factorisation. Cholesky exige une matrice qui se **déclare**
  symétrique — un triangle ne se lisant que d'un côté, elle ne se tromperait pas
  bruyamment sur une matrice qui ne l'est pas — et définie positive : un
  point-selle à multiplicateurs, symétrique mais indéfini, est refusé au pivot
  fautif, qu'on l'élimine d'abord ou qu'on le résolve en LU. Pas de méthode
  itérative ; `SolveMethod` reste le point d'extension prévu pour cela.
