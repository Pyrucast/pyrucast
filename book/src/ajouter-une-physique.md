# Ajouter une physique

Ce chapitre liste **tous les points de code à toucher** pour ajouter une
nouvelle physique. L'architecture est conçue pour que ce coût soit **O(1)
fichier**, indépendant du nombre de physiques déjà présentes — voir
*[Pourquoi ça passe l'échelle](#pourquoi-ça-passe-léchelle)* en fin de
chapitre.

## Le principe en une phrase

L'énum [`SubModel`](model.md) ne sert qu'au **stockage** et à la
**sérialisation** ; **tout le comportement** vit dans une struct par
physique (sous `src/models/`) qui implémente le trait `SubModelKind`. Un unique
point de dispatch, `SubModel::as_kind()`, relie les deux. Le code
générique (l'agrégat `Model`, l'assembleur, `Dump`) ne fait **jamais** de
`match` par variante.

```text
SubModel  (enum : stockage + sérialisation bincode)
├── HeatConduction(HeatConduction)
├── Dirichlet(Dirichlet)
├── … une variante par physique
└── as_kind(&self) -> &dyn SubModelKind   ← l'unique match

SubModelKind  (trait de base : le dénominateur commun de tout sous-modèle)
├── primal_vars / dual_vars                      ── les variables
├── physics       -> &'static [Physics]          ── la nature (requise)
├── as_domain     -> Option<&dyn Domain>        (défaut : None)  ── seam capacité
├── as_constraint -> Option<&dyn Constraint>    (défaut : None)  ── seam capacité
│   └── matrix_element(kind, …)                  (le pont vers Domain, fourni)
├── stiffness_layout / mass_layout               (blocs calculés ; défaut : None)
│   geometric_layout / tangent_layout
│   └── matrix_layout(kind)                      (le dispatcher, fourni)
├── contributions(kind, material)                (défaut : dérivé du layout)
├── build_stiffness_blocks                       (défaut : dérivé du layout)
├── internal_force_element                       (défaut : continuum Bᵀσ)
├── build_internal_forces                        (fourni : pilote le précédent)
└── label / display / render

Sous-traits « capacité », miroir des natures de sous-modèle (une struct
n'implémente que celui qui la concerne) :
├── Domain      { material_fespace, material_components,
│                 optional_material_components,
│                 behavior_fespace, behavior_output_components,
│                 integrate_point, integrate_behavior (fourni) }
└── Constraint  { multiplier_mesh, relations }
```

## Les étapes

Ajouter une physique de forme courante — celle qui couvre un espace éléments
finis — coûte **un fichier et une ligne** :

1. **`src/models/<ma_physique>.rs`** (nouveau) — une struct portant ses
   supports, un `impl SubModelKind`, un constructeur `new(...)`, ses tests, et
   une invocation de `physics_operator!` qui déclare l'opérateur public avec sa
   documentation (calque sur `truss.rs`, le cas le plus court).
2. **`src/containers/model.rs`** — **une** variante dans `enum SubModel` et
   **une** ligne dans `SubModel::as_kind()`.

Plus deux lignes de raccordement : `pub mod` dans `src/models/mod.rs`,
`pub use` dans `src/ops/model/mod.rs`, et l'enregistrement dans le `#[pymodule]`
de `src/lib.rs` (le module `_pyrucast` est plat).

**Ce qu'on n'écrit plus.** Le balayage des sous-espaces, le `#[pyfunction]`, son
attribut de stub, le déballage des enveloppes Python : `physics_operator!` les
émet. Un auteur de physique n'ouvre jamais `src/py/`.

```rust
{{#include ../../src/models/truss.rs:operator}}
```

**Deux blocs de documentation, et c'est voulu** : le Rust porte son `///` et son
doctest, le Python un littéral qui atterrit dans le `.pyi`. Les partager
mettrait un doctest Rust dans une docstring Python.

### Les formes que la macro ne couvre pas

Elle sert le balayage — le cas d'une physique nouvelle. Restent écrits à la
main, chacun avec sa raison :

- les **contraintes** (`dirichlet`, `mpc`, `embedded`, `contact`), portées par
  des maillages fournis par l'utilisateur et non par un espace EF ;
- les variantes **à symétrie** (`heat_conduction`, `fick`, `elasticity`), dont
  la face Python replie la symétrie en `symmetry=None` ;
- `interface_transfer`, qui apparie deux espaces avec un contrôle de longueur.

### Ajouter une loi plutôt qu'une physique

Une loi d'écoulement ou d'endommagement n'est **pas** une physique : c'est un
attribut de la plasticité ou de l'endommagement, qui en partagent les DDL,
l'opérateur élastique et l'état. Elle coûte son fichier sous
`src/models/plastic/` ou `src/models/damage/` — struct unitaire, `impl YieldLaw`
ou `impl DamageKind`, façade une façade `physics_operator!` déléguant à l'opérateur générique — plus une
variante d'énuméré et un bras dans `as_law()`.

Tout le reste est générique et **ne change pas**.

## Le trait `SubModelKind`

Défini dans `src/models/mod.rs`. Le trait de base ne porte que le **dénominateur
commun** de tout sous-modèle ; chaque **capacité optionnelle** est un **sous-trait
séparé**, exposé par un *seam* `as_*()` qui rend `None` par défaut. Ces
sous-traits **font miroir des natures** de sous-modèle : `Domain` (physique
définie sur une région : matériau + comportement) et `Constraint`
(multiplicateurs de Lagrange). Une struct n'implémente **que** la capacité qui la
concerne : elle n'a donc jamais de méthode « présente mais qui erronerait ». Un
domaine typique implémente `primal_vars`, `dual_vars`, `physics`, `as_domain` +
`Domain`, le noyau `element_matrix`, `stiffness_layout`, `label` et `render`. Il
**n'écrit pas** `build_stiffness_blocks` : le défaut le dérive de
`stiffness_layout` + `element_matrix`.

Seules trois méthodes sont **sans défaut** : `primal_vars`, `dual_vars` et
`physics` (plus `label` / `render` pour l'affichage). Tout le reste se redéfinit
à la carte.

```text
pub trait SubModelKind: Sync {
    fn primal_vars(&self) -> Vec<String>;
    fn dual_vars(&self) -> Vec<String>;
    // Nature(s) de la physique — slice constante, pendant de `label` ; sert
    // aux sélecteurs `Model::filter` / `Matrix::filter` (match par appartenance) :
    fn physics(&self) -> &'static [Physics];
    // Seams de capacité — None (défaut) ⇒ la struct n'a pas cette capacité.
    // Une struct qui l'a redéfinit le seam pour rendre `Some(self)` :
    fn as_domain(&self)     -> Option<&dyn Domain>     { None }
    fn as_constraint(&self) -> Option<&dyn Constraint> { None }

    // Le pont vers les noyaux de matrice, qui vivent sur `Domain` (voir plus
    // bas) — fourni, on ne l'écrit pas :
    fn matrix_element(&self, kind: MatrixKind, /* … */) -> Result<()> { /* as_domain() puis route */ }

    // ── Déclarations structurelles du bloc *calculé*, une par MatrixKind ;
    // None (défaut) ⇒ pas de terme de ce genre pour cette physique :
    fn stiffness_layout(&self)  -> Option<MatrixLayout> { None }
    fn mass_layout(&self)       -> Option<MatrixLayout> { None }
    fn geometric_layout(&self)  -> Option<MatrixLayout> { None }
    fn tangent_layout(&self)    -> Option<MatrixLayout> { None }
    fn matrix_layout(&self, kind: MatrixKind) -> Option<MatrixLayout> { /* fourni */ }

    // Contributions telles que l'assembleur les consomme (défaut : Computed(layout)
    // si matrix_layout(kind), sinon — pour Stiffness seulement — Literal(build_stiffness_blocks)) :
    fn contributions(&self, kind: MatrixKind, material: Option<&Handle<SubElementField>>)
        -> Result<Vec<Contribution>> { /* défaut : dérivé de matrix_layout */ }
    fn build_stiffness_blocks(&self, material: Option<&Handle<SubElementField>>)
        -> Result<Vec<SubMatrix>> { /* défaut : dérivé de stiffness_layout + element_matrix */ }

    // ── Forces internes f = ∫ Bᵀ σ (Cast3m BSIG) — le transposé de B :
    fn internal_force_element(&self, geoms: &[CellGeom],
        stress: &SubElementField, fe: &mut [f64]) -> Result<()> { /* défaut : continuum */ }
    fn build_internal_forces(&self, stress: &Handle<SubElementField>)
        -> Result<SubNodeField> { /* fourni : pilote le noyau sur le stiffness_layout */ }

    fn label(&self) -> &'static str;
    fn display(&self) -> String { format!("SubModel<{}>", self.label()) }
    fn render(&self, opts: &DumpOptions) -> String;
}

// Capacités optionnelles — implémentées à part, jamais sur le trait de base.
// Un DOMAINE lit un matériau ET intègre un comportement : les deux sont UNE
// capacité, car le matériau paramètre la loi (σ = D(E,ν):ε, M = E·I·κ, …).
pub trait Domain: Sync {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace>;
    // Les constantes exigées, dans l'ordre où le noyau les indexera ;
    // une liste vide dit « aucune composante contrainte » :
    fn material_components(&self) -> Vec<String> { Vec::new() }
    // Composantes acceptées mais non exigées (alpha…) — cf. plus bas :
    fn optional_material_components(&self) -> &'static [&'static str] { &[] }
    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace>;
    fn behavior_output_components(&self) -> Vec<String>;
    // Ce que le noyau lit, dans l'ordre de ses indices :
    fn deformation_reads(&self) -> Vec<String>;
    fn state_reads(&self) -> Vec<String> { Vec::new() }
    // L'état au repos, et la question « cette loi exige-t-elle un pas de
    // temps ? » — posées une fois, pas au point de Gauss :
    fn initial_state(&self, material: &SubElementField) -> Result<SubElementField> { /* fourni : des zéros */ }
    fn requires_dt(&self) -> bool { false }
    // La loi de comportement en UN point de Gauss, montage incrémental A → B.
    // Chaque entrée est LA LIGNE de ce point, empruntée au tampon du champ ;
    // `lay` dit où chaque composante s'y trouve (résolu une fois par zone) :
    fn integrate_point(&self, geom: &CellGeom, g: usize, lay: &ZoneLayout,
        deformation: &[f64], prev: &[f64], material: &[f64],
        dt: f64, out: &mut [f64]) -> Result<()>;
    fn integrate_behavior(&self, deformation: &Handle<SubElementField>,
        prev: &Handle<SubElementField>,
        material: Option<&Handle<SubElementField>>,
        dt: f64) -> Result<SubElementField> { /* fourni : résout la zone, pilote integrate_point */ }

    // ── Voie MATRICE, le miroir exact de la précédente ────────────────────
    // Ce que le noyau de matrice lit dans l'état, par genre (défaut : rien —
    // une raideur ou une masse ne lit que le matériau, déjà déclaré ci-dessus) :
    fn element_state_reads(&self, kind: MatrixKind) -> Vec<String> { Vec::new() }
    fn element_layout(&self, kind: MatrixKind, material: &SubElementField,
        state: Option<&SubElementField>) -> Result<ElementLayout> { /* fourni */ }

    // Les noyaux de matrice élémentaire (une cellule) — purs et séquentiels.
    // `geoms` : un CellGeom par espace EF du layout (geoms[0] pour le cas usuel,
    // plusieurs pour un élément multi-quadrature — poutre/coque).
    // `lay` dit où chaque composante se trouve, résolu une fois par zone.
    fn element_matrix(&self, geoms: &[CellGeom], material: &SubElementField,
        lay: &ElementLayout, ke: &mut [f64]) -> Result<()>;                     // ∫ Bᵀ D B  (requis)
    // Les quatre suivants ont un défaut qui erre : « cette physique n'a pas ce
    // terme » — pas de masse, pas de flambement, pas de tangente, pas de couplage.
    fn element_mass(&self, geoms: &[CellGeom], material: &SubElementField,
        lay: &ElementLayout, ke: &mut [f64]) -> Result<()>;                     // ∫ ρ Nᵀ N
    fn element_geometric(&self, geoms: &[CellGeom], material: &SubElementField,
        lay: &ElementLayout, state: &SubElementField, ke: &mut [f64]) -> Result<()>;   // ∫ Gᵀ σ̂ G
    fn element_tangent(&self, geoms: &[CellGeom], material: &SubElementField,
        lay: &ElementLayout, state: &SubElementField, ke: &mut [f64]) -> Result<()>;   // ∫ Bᵀ D_alg B
    fn coupling_element(&self, kind: MatrixKind, row_geoms: &[CellGeom],
        col_geoms: &[CellGeom], material: &SubElementField,
        lay: &ElementLayout, ke: &mut [f64]) -> Result<()>;                     // interface
}
pub trait Constraint {
    fn multiplier_mesh(&self) -> &Mesh;
    // Les relations linéaires imposées, sous forme neutre vis-à-vis de la méthode
    // d'imposition (Lagrange ou élimination) — cf. « Une contrainte » plus bas :
    fn relations(&self) -> Result<Vec<Relation>>;
}
```

`Domain` porte donc **deux voies**, et elles ont la même forme : la physique
déclare ce qu'elle lit, la zone le traduit en positions une fois, le noyau
indexe. Un noyau ne compare jamais un nom de composante.

| | déclare | résout (1× par zone) | consomme |
|---|---|---|---|
| voie **point de Gauss** | `deformation_reads` / `state_reads` | `zone_layout` → `ZoneLayout` | `integrate_point` |
| voie **matrice** | `material_components` / `element_state_reads` | `element_layout` → `ElementLayout` | `element_matrix` & consorts |

Conséquences pratiques — pour donner une capacité à une physique, implémenter le
sous-trait **et** redéfinir le seam correspondant pour rendre `Some(self)` :

- **Domaine** (physique sur une région) : `impl Domain` + `as_domain()`. Déclarer
  `material_fespace()` (+ `material_components()`) — l'assembleur
  (`src/ops/matrix.rs`) sélectionne et valide le `SubElementField`
  automatiquement ; `behavior_fespace()` + `behavior_output_components()` +
  `deformation_reads()` + `integrate_point(...)`, la loi de constitution **en un
  point de Gauss**. `integrate_behavior` est **fourni** : il résout la zone, puis
  pilote ce noyau en parallèle sur toutes les cellules. Matériau et comportement
  vont **ensemble** : même un élément linéaire (barre, poutre) a un comportement,
  simplement trivial (`N = E·A·ε`).

  **Deux invariants tiennent dans ce noyau**, et ils décident de la forme du
  reste : aucun test que l'amont a déjà tranché — présence d'une composante,
  forme d'un champ, `Option` à déballer — et aucune allocation dynamique. Un
  noyau conforme s'écrit sans un seul `?` sur ses lectures. Ce que l'auteur
  déclare (`deformation_reads`, `state_reads`, `material_components`) est
  exactement ce qui permet cela : la convention est traduite en positions **une
  fois par zone**, dans `zone_layout`, où un champ fabriqué à la main dans un
  autre ordre est refusé avec un message qui nomme le champ et l'écart. Les
  branchements qui restent sont ceux de la physique.
- **Contrainte** (multiplicateurs de Lagrange) : `impl Constraint`
  (`multiplier_mesh()` + `relations()`) + `as_constraint()`.
  `SubModel::multiplier_nodes()` et `multiplier_mesh()` en découlent. C'est le
  foyer de la famille contrainte — `Dirichlet`, `Mpc`, `Embedded`, `Contact`.
- **Terme de masse, raideur géométrique, tangente cohérente** : déclarer le
  `*_layout` correspondant et écrire le noyau `element_*` (voir ci-dessous).

Une **contrainte** comme `Dirichlet` n'implémente *que* `Constraint` (plus
`contributions`, cf. ci-dessous) : elle n'a ni `element_matrix`, ni `Domain` —
leur absence est un **fait de compilation**, pas une erreur à l'exécution.

### La nature : `physics()`

`physics()` rend une **slice constante** de `Physics` (`Mechanical`, `Thermal`,
`Constraint`, `Other`) : la classification grossière de la physique, orthogonale
à l'axe de capacité `Domain`/`Constraint`. Elle est **requise** — chaque physique
déclare sa nature à son site de définition, une physique couplée en déclarant
plusieurs. Cette information voyage avec chaque bloc assemblé jusqu'à la
`SubMatrix`, et alimente les sélecteurs `Model::filter` / `Matrix::filter`, qui
matchent par **appartenance**. Un `HeatConduction` rend
`&[Physics::Thermal]`, un `Dirichlet` `&[Physics::Constraint]`.

### Un genre de matrice = un layout + un noyau

L'assemblage est **agnostique au genre de matrice**. L'énum `MatrixKind`
(`Stiffness`, `Mass`, `Geometric`, `Tangent`) est le discriminant qui fait
tourner **la même machinerie** — recette, scatter colorié, cache de motif creux
par genre — avec un noyau élémentaire différent :

| `MatrixKind` | Cast3m | intégrale | layout | noyau |
|---|---|---|---|---|
| `Stiffness` | `RIGI` / `COND` | `∫ Bᵀ D B` | `stiffness_layout` | `element_matrix` |
| `Mass` | `MASS` / `CAPA` | `∫ ρ Nᵀ N` | `mass_layout` | `element_mass` |
| `Geometric` | `KSIG` | `∫ Gᵀ σ̂ G` | `geometric_layout` | `element_geometric` |
| `Tangent` | `KTAN` | `∫ Bᵀ D_alg B` | `tangent_layout` | `element_tangent` |

Ajouter un terme à une physique, c'est donc **deux méthodes** : le `*_layout`
(souvent le même que celui de la raideur — mêmes espaces EF, même support,
mêmes variables : seul le noyau diffère) et le `element_*`. Une physique sans
terme d'un genre ne redéfinit rien : son layout reste `None`, elle ne contribue
pas, et `ops::matrix::mass(...)` sur un modèle qui la contient l'ignore
simplement. Côté opérateurs, un point d'entrée par genre —
`ops::matrix::{stiffness, mass, geometric, tangent}`, plus `lump` — tous adossés
au même `assemble_kind`.

Les deux genres à état (`Geometric`, `Tangent`) reçoivent en plus un `state` : le
champ produit par `integrate_behavior` (contrainte courante pour la raideur
géométrique, modules tangents algorithmiques `D_alg` pour la tangente cohérente).
C'est le couple producteur/consommateur — le noyau de comportement écrit `D_alg`
dans ses composantes de sortie, `element_tangent` les relit.

### Les forces internes

`build_internal_forces(stress)` est **fourni** : il pilote
`internal_force_element` en parallèle sur les espaces EF du `stiffness_layout` et
disperse aux nœuds de son support. Le noyau élémentaire par défaut est celui de
la mécanique des milieux continus — `f_{i,a} = Σ_g Σ_b (∂N_i/∂x_b) σ_ab`, lu en
nommage Voigt (`sigma_xx`, `sigma_xy`, …), terme de cerceau compris en
axisymétrie. Une physique dont le dual n'est **pas** un vecteur déplacement
(thermique, barre, poutre) redéfinit `internal_force_element`. Pour une loi
linéaire, le résultat vaut `K·u`.

### Le parallélisme est gratuit (et invisible)

Les noyaux qu'une physique écrit — `integrate_point` (un point de Gauss),
`element_matrix` & consorts (la matrice élémentaire d'une cellule),
`internal_force_element` — sont **séquentiels et purs** : ils ne voient ni rayon,
ni un handle, ni un verrou. Les *drivers* de `models::kernel` portent la
parallélisation et le zéro-copie au-dessus d'eux. Voir
[Parallélisme](developper/parallelisme.md).

Concrètement, une physique de continuum déclare son layout (espaces EF, support,
variables, ordering) : le défaut de `contributions()` en tire une
`Contribution::Computed`, et l'assembleur global bâtit un bloc **calculé** puis
disperse le noyau élémentaire directement dans le CSR, en parallèle par
coloration des cellules — sans matérialiser de COO. La voie **littérale**
(`build_stiffness_blocks`) est le second défaut du trait, dérivée du même couple
`stiffness_layout` + `element_matrix` via `kernel::assemble_block` ; elle sert de
référence d'équivalence et de repli, mais une physique volumique **ne l'écrit
plus**.

Une contrainte comme `Dirichlet` (aucun layout, rien d'intégré sur une cellule)
redéfinit directement `contributions()` : elle rend `Vec::new()` pour tout genre
autre que `Stiffness`, et ses blocs C / Cᵀ en `Contribution::Literal` pour
celui-là — l'assembleur reste sans aucun cas particulier « Dirichlet ».

### Un bloc inter-maillages : `Coupling`

Une physique d'**interface** (l'échange `h(c₁ − c₂)` entre deux corps qui ne
partagent pas leurs nœuds) a besoin de blocs dont les **lignes vivent sur un
maillage et les colonnes sur un autre**. C'est la troisième variante,
`Contribution::Coupling(CouplingLayout)`.

Tout ce qui est *sous* ce seam était déjà asymétrique lignes/colonnes :
`SubMatrix::computed` prend deux supports, et le scatter comme les drivers de
noyau les passent séparément. Le seul point qui les confondait était le champ
unique `MatrixLayout.support` — d'où un layout séparé plutôt qu'un champ de plus,
qui aurait touché les treize physiques existantes pour un besoin qu'aucune n'a :

```rust,ignore
{{#include ../../src/models/mod.rs:coupling_layout}}
```

Pas de champ `symmetric` : un bloc de couplage n'est jamais symétrique seul —
seule la réunion des quatre l'est, comme la paire C / Cᵀ.

Le noyau correspondant est `coupling_element(kind, row_geoms, col_geoms,
material, ke)` : il reçoit **deux** `CellGeom`, la maille du côté ligne et la
maille en vis-à-vis du côté colonne. Le driver
`kernel::coupling_block_triplets_per_cell` parcourt les deux connectivités en pas
à pas ; il exige des maillages **conformes** (même type d'élément, même nombre de
mailles, maille `i` face à maille `i`) et le signale sinon.

C'est aussi le noyau qui porte le **signe** — `+h∫NᵢNⱼ` en diagonale via
`element_matrix`, `−h∫NᵢNⱼ` hors diagonale via `coupling_element` — puisque
chaque bloc choisit son noyau depuis sa propre variante de contribution.
L'assembleur n'a rien à savoir des interfaces.

Une réserve de mise en œuvre : le scatter d'un bloc de couplage est **séquentiel**
(ses matrices élémentaires restent, elles, calculées en parallèle). Le coloriage
qui rend le scatter parallèle sûr repose sur *une* connectivité ; avec deux, il ne
prouve plus rien. Une interface porte un maillage de bord — c'est sans effet
mesurable, et cela évite d'inventer un coloriage à deux côtés pour un gain nul.

Voir [`interface_transfer`](diffusion.md#transfert-à-travers-une-interface), son
premier utilisateur.

Le champ `fespaces` du `MatrixLayout` est un **`Vec`** : un seul espace EF
pour une physique de continuum, ou plusieurs — partageant un maillage, ne
différant que par la quadrature — pour un élément **multi-quadrature**. C'est ce
que fait la **coque de Reissner-Mindlin** (`fespaces: vec![full, shear]`, membrane
et flexion en Gauss complet + cisaillement transverse réduit, contre le blocage) :
`element_matrix` reçoit alors deux `CellGeom`, `geoms[0]` pour la membrane et la
flexion, `geoms[1]` pour le cisaillement, et l'élément passe par le **même**
chemin de scatter parallèle que le reste — la sparsité ne dépendant que de la
connectivité, pas de la quadrature.

Le second espace est construit par la physique, pas reçu en argument : il est
entièrement déterminé par le premier, et les deux `CellGeom` doivent désigner
*la même maille*, un invariant qu'on préfère établir plutôt que vérifier. Une
même physique peut d'ailleurs en déclarer un nombre variable selon sa
formulation — la coque en Kirchhoff discret n'en déclare qu'un, n'ayant aucun
cisaillement à intégrer.

### Une contrainte : les relations, forme neutre

`Constraint::relations()` rend une `Relation` par nœud multiplicateur : son
`multiplier_node`, la composante duale `imposed_value` où l'utilisateur écrira le
second membre `g`, la liste des termes `(node, variable, target_dual,
coefficient)`, et un `sense` (`RelationSense::Equality` par défaut,
`GreaterEqual` / `LessEqual` pour l'unilatéral, cf.
[Contact](contraintes/contact.md)).

C'est la **source unique de vérité**, indépendante de la méthode d'imposition :
la voie Lagrange (`contributions()`) en tire ses blocs C / Cᵀ — via le helper
partagé `constraint_block_pair` — et la voie par **élimination**
(`ops::solver::eliminate`) lit les mêmes relations. Ni l'une ni l'autre ne
re-parse le maillage-par-terme fourni par l'utilisateur. Une nouvelle contrainte
n'a donc à décrire ses relations **qu'une fois**.

### Le comportement : le montage incrémental A → B

`integrate_point` intègre le pas **A → B** en un point de Gauss :

- `deformation` — la cinématique de **fin de pas** ε(B), produite par un
  opérateur géométrique (`gradient`, `deformation`, `beam_deformation`) ;
- `prev` — l'**état convergé au début du pas** A : le flux/contrainte σ(A), les
  variables internes `VAR(A)`, et pour les lois incrémentales la cinématique
  ε(A). Vaut `None` au premier pas (configuration de référence) ;
- `material` — les données matériau de la zone, `Some(_)` ssi la physique déclare
  un `material_fespace` ;
- `dt` — l'incrément de temps, `None` pour une loi indépendante du temps (une loi
  visqueuse erronera s'il manque).

Le noyau écrit dans `out` les composantes déclarées par
`behavior_output_components()` : l'état matériau en B — σ(B), `VAR(B)`, et
éventuellement `D_alg` pour une physique qui alimente la tangente cohérente. La
sortie devient le `prev` du pas suivant.

### Une composante matériau facultative

Un coefficient annexe, consommé par un opérateur tiers et non par l'assemblage
(typiquement `alpha`, la dilatation thermique lue par
`ops::element_field::thermal_strain`), se déclare dans
`optional_material_components()` : il traverse le canal matériau s'il est fourni,
mais n'est jamais exigé à l'assemblage — seules les composantes **requises**
discriminent la zone matériau. Ce n'est donc jamais un argument scalaire d'un
opérateur.

## Le dispatch — `src/containers/model.rs`

```text
#[derive(Serialize, Deserialize)]
pub enum SubModel {
    HeatConduction(heat_conduction::HeatConduction),
    Dirichlet(dirichlet::Dirichlet),
    // … une ligne par physique
}

impl SubModel {
    pub fn as_kind(&self) -> &dyn SubModelKind {
        match self {
            SubModel::HeatConduction(p) => p,
            SubModel::Dirichlet(p) => p,
            // … une ligne par physique
        }
    }
}
```

`Debug`, `Display`, `Dump`, les méthodes déléguantes de `SubModel` et
l'assembleur appellent tous `self.as_kind().<méthode>()` — ils sont
écrits une fois pour toutes.

## Ce qui est générique (rien à toucher)

- `src/ops/matrix.rs` : `stiffness()` / `mass()` / `geometric()` / `tangent()`
  délèguent tous à `assemble_kind()`, qui boucle sur `contributions(kind, …)` et
  pilote le matériau via le seam `as_domain()` (`Domain`). Aucun `match` par
  variante.
- `src/ops/element_field/behavior.rs` et `material_field.rs`, avec leur wrapper
  `src/py/ops/element_field.rs`.
- `src/ops/node_field/internal_forces.rs` : passe par `build_internal_forces()`.
- `src/py/ops/matrix.rs` : les assembleurs délèguent à `model.inner`.

## Pour finir

Régénérer le stub `python/pyrucast/_pyrucast/__init__.pyi` (`cargo run --bin
stub_gen --features stub-gen`, venv activé), puis **builder + tester avant de
commiter** — `script/check_all.sh` pour la passe complète, ou le seul bloc
concerné (`check_rust.sh`, `check_doc.sh`…).

---

## Pourquoi ça passe l'échelle

Avec des dizaines de physiques, deux propriétés comptent : le **coût
d'ajout** et la **persistance**.

### Coût d'ajout : O(1) fichier

Le comportement d'une physique est **co-localisé** dans son fichier (struct
+ `impl SubModelKind` + les `impl` de capacité qui la concernent). Ajouter la
physique n°30 ne touche que 4 endroits
(§ Les étapes), dont 2 sont des lignes uniques dans `model.rs`. Aucune des
méthodes génériques n'est modifiée. C'est l'inverse du *« shotgun
surgery »* qu'imposerait un enum où chaque méthode ferait son propre
`match` : là, ajouter une physique forcerait à éditer une dizaine de sites.

La même propriété tient sur l'autre axe : ajouter un **genre de matrice** a
coûté un variant de `MatrixKind`, un `*_layout` et un `element_*` par physique
concernée — l'assembleur, le cache de motif et le scatter n'ont pas bougé.

> Les deux lignes (variante + bras de `as_kind`) pourraient même être
> générées par une macro `physics_enum! { HeatConduction, Dirichlet, … }`
> pour ne laisser qu'une seule déclaration.

### Pourquoi garder l'enum (et pas `Box<dyn SubModelKind>`)

La persistance utilise **`bincode`** sur des `Serialize/Deserialize`
*dérivés* (`src/persist.rs`), un format **non auto-descriptif**. Or :

- un `enum SubModel` se sérialise nativement (indice de variante + payload),
  zéro code manuel ;
- un `Box<dyn SubModelKind>` imposerait `typetag`, qui **ne supporte pas** les
  formats non auto-descriptifs comme `bincode`. On perdrait la persistance.

L'enum donne aussi l'**exhaustivité** : le compilateur refuse d'oublier un
cas dans `as_kind()`. On obtient donc le meilleur des deux mondes —
sérialisation triviale et exhaustivité de l'enum, comportement co-localisé
et coût d'ajout constant du trait.

### Et les données neutres partagées ?

Une donnée commune à *toutes* les physiques (un `name`, un flag `enabled`,
une pondération) se traite selon sa nature :

- **dérivable** (calculable à partir du type/de l'état) → un **défaut dans
  le trait `SubModelKind`** la fournit gratuitement à toutes les physiques, ex.
  `fn weight(&self) -> f64 { 1.0 }`. Le trait « l'impose et l'implémente
  automatiquement ». C'est exactement le statut de `physics()`, à ceci près
  qu'elle est volontairement **sans défaut** : la nature ne se devine pas, on
  veut que chaque physique la déclare.
- **stockée et mutable** (saisie à l'exécution) → un trait **ne peut pas**
  porter de champ ni en générer un par défaut : il imposerait un accesseur
  `fn meta(&self) -> &Meta`, mais chaque struct devrait alors stocker le
  champ (le boilerplate par-physique que la fusion supprime). Le bon foyer
  redevient alors un wrapper `struct SubModel { kind, meta }` — à ré-introduire
  *si et seulement si* ce besoin apparaît (cf. la discussion dans le chapitre
  [Modèle physique](model.md)).

### Bilan

- Format de persistance : enum + `bincode`, **stable**.
- Coût d'ajout : **O(1) fichier**, ~2 lignes de câblage.
- Comportement : **co-localisé** par physique.
- Le seul `match` par variante du module modèle est `as_kind()`.
