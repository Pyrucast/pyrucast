# Ajouter une physique

Ce chapitre liste **tous les points de code à toucher** pour ajouter une
nouvelle physique. L'architecture est conçue pour que ce coût soit
**constant** : il ne dépend pas du nombre de physiques déjà présentes — voir
*[Pourquoi ça passe l'échelle](#pourquoi-ça-passe-léchelle)* en fin de
chapitre.

## Le principe en une phrase

L'énum [`SubModel`](model.md) ne sert qu'au **stockage** et à la
**sérialisation** ; **tout le comportement** vit dans une struct par
physique (sous `src/models/`) qui implémente le trait `SubModelKind`. Un unique
point de dispatch, `SubModel::as_kind()`, relie les deux. Le code
générique (l'agrégat `Model`, l'assembleur, `Dump`) ne fait **jamais** de
`match` sur toutes les variantes ; seul `update_contact` reconnaît la sienne,
`SubModel::Contact`.

```text
SubModel  (enum : stockage + sérialisation bincode)
├── HeatConduction(HeatConduction)
├── BoundaryTransfer(BoundaryTransfer)
├── Dirichlet(Dirichlet)
├── Mpc(Mpc)
├── Embedded(Embedded)
├── Contact(Contact)
├── Truss(Truss)
├── Elasticity(Elasticity)
├── Plasticity(Plasticity)
├── Damage(Damage)
├── Timoshenko(Timoshenko)
├── Fick(Fick)
├── InterfaceTransfer(InterfaceTransfer)
├── Radiation(Radiation)
├── Bernoulli(Bernoulli)
├── Shell(Shell)
├── Flux(Flux)
└── as_kind(&self) -> &dyn SubModelKind   ← le seul match sur toutes les variantes

SubModelKind  (trait de base : le dénominateur commun de tout sous-modèle)
├── primal_vars / dual_vars                      ── les variables (requises)
├── physics       -> &'static [Physics]          ── la nature (requise)
├── as_domain     -> Option<&dyn Domain>         (défaut : None)
├── as_behavior   -> Option<&dyn Behavior>       (défaut : None)
├── as_constraint -> Option<&dyn Constraint>     (défaut : None)
├── matrix_element(kind, …)                      (le pont vers les noyaux, fourni)
├── stiffness_layout / mass_layout /             (blocs calculés ; défaut : None)
│   geometric_layout / tangent_layout
│   └── matrix_layout(kind)                      (le choix par genre, fourni)
├── tangent_is_stiffness                         (défaut : false)
├── contributions(kind, material)                (défaut : dérivé du layout)
├── build_stiffness_blocks(material)             (défaut : dérivé du layout)
├── internal_force_reads / internal_force_element (défaut : continuum Bᵀσ)
├── residual_input                               (défaut : le primal interpolé)
├── internal_force_contribution                  (défaut : Computed(layout))
├── external_force_element                       (défaut : erreur)
├── external_force_contribution                  (défaut : rien)
└── label / render (requis), display (fourni)

Sous-traits « capacité » (une struct n'implémente que ceux qui la concernent) :
├── Domain      { material_fespace (requis), material_components,
│                 optional_material_components, element_state_reads,
│                 element_layout, element_matrix, element_mass,
│                 element_geometric, coupling_element }
├── Behavior: Domain
│               { behavior_fespace, behavior_output_components,
│                 deformation_reads, strain_operator, integrate_point (requis) ;
│                 state_reads, zone_layout, element_tangent, initial_state,
│                 requires_dt, elastic_operator, tangent_source,
│                 tangent_point, integrate_behavior (fournis) }
└── Constraint  { multiplier_mesh, relations }
```

## Les étapes

Ajouter une physique de forme courante — celle qui couvre un espace éléments
finis — coûte **un fichier** et quelques lignes de raccordement :

1. **`src/models/<ma_physique>.rs`** (nouveau) — une struct portant ses
   supports, un `impl SubModelKind` (et des capacités qui la concernent), un
   constructeur `new(...)`, ses tests, et une invocation de
   `physics_operator!` qui déclare l'opérateur public avec sa documentation
   (calque sur `truss.rs`, le cas le plus court).
2. **`src/containers/model.rs`** — **une** variante dans `enum SubModel`, en fin
   de liste ; **une** ligne dans `SubModel::as_kind()` ; et le constructeur
   `SubModel::<ma_physique>(…)`, qui enveloppe `new(...)` dans sa variante :
   c'est lui que `physics_operator!` appelle (`via SubModel::<ma_physique>`).

Puis le raccordement : `pub mod` dans `src/models/mod.rs`, `pub use` dans
`src/ops/model/mod.rs`, l'enregistrement `add_function` dans le `#[pymodule]`
de `src/lib.rs` (le module `_pyrucast` est plat), et l'import, avec son entrée
dans `__all__`, dans `python/pyrucast/model.py`.

**Ce qu'on n'écrit pas.** Le balayage des sous-espaces, le `#[pyfunction]`, son
attribut de stub, le déballage des enveloppes Python : `physics_operator!` les
émet. Un auteur de physique n'ouvre jamais `src/py/`.

**Un argument peut porter une valeur par défaut**, `arg: T = défaut`, comme la
mesure de la barre ci-dessous. Rust n'en a pas : son appelant passe
l'argument ; la face Python le prend en mot-clé, avec cette valeur. Même nom,
l'idiome de chaque langage. La macro écrit cette valeur dans la documentation
Rust, pour qu'un lecteur qui n'écrit que du Rust sache ce que Python obtient
sans rien dire.

```rust
{{#include ../../src/models/truss.rs:operator}}
```

**Un terme qui écrit dans les lignes d'une autre physique** — une charge
répartie (`flux`), un échange de bord (`boundary_transfer`), un rayonnement
(`radiation`) — déclare la forme `(fespace, target, …)`. Il a besoin du modèle
qu'il charge ou refroidit : pour y vérifier que ses lignes sont assemblées, y
lire la duale de chaque variable et, quand ses noms de variables sont libres, en
hériter la nature. La macro fait suivre la cible au constructeur de chaque zone
et l'ajoute à la face Python ; l'auteur appelle `transfer::target_pairs` (des
variables, dont la cible donne les duales), `transfer::target_physics` (un
couple primale/duale fixé, comme le rayonnement) ou `transfer::owner_physics`
(une ligne seule) dans son `new`, une fois, à la construction.

**Deux blocs de documentation, et c'est voulu** : le Rust porte son `///` et son
doctest, le Python un littéral qui atterrit dans le `.pyi`. Les partager
mettrait un doctest Rust dans une docstring Python.

Côté documentation : la page de la physique dans le book, son rangement dans un
thème de [`operateurs.md`](operateurs.md) (le garde-fou `reference` refuse un
oubli), un exemple à l'ancre `model-<ma_physique>` dans un
`tests/python/test_doc_*.py`, et sa ligne dans la
[correspondance Rust ↔ Python](developper/correspondance-rust-python.md).

### Les formes que la macro ne couvre pas

`physics_operator!` a trois formes : la physique sur son espace EF seul, la
physique qui porte sur un modèle cible (`fespace, target, …`), et la façade
nommée d'une loi (voir plus bas). Restent écrits à la main, dans
`src/ops/model/`, chacun avec sa raison :

- les **contraintes** (`dirichlet`, `mpc`, `embedded`, `contact`), portées par
  des maillages fournis par l'utilisateur et non par un espace EF, et
  `update_contact`, qui transforme un modèle existant ;
- `interface_transfer`, qui apparie deux espaces avec un contrôle de longueur ;
- les deux opérateurs génériques `plasticity` et `damage`, qui reçoivent la loi
  en argument et vers lesquels délèguent les façades.

### Ajouter une loi plutôt qu'une physique

Une loi de comportement n'est **pas** une physique : c'est un attribut d'une
physique existante, qui en partage les DDL, la modélisation et les layouts. Elle
coûte son fichier sous `src/models/<physique>/` — struct unitaire, `impl` du
trait de sa famille, façade `physics_operator!` déléguant à l'opérateur
générique — plus une variante d'énuméré, en fin de liste, et un bras dans
`as_law()`.

Deux niveaux, et ils ne disent pas la même chose : l'**énuméré** porte l'identité
physique — c'est lui que `bincode` archive, et une nouvelle variante va donc
toujours en fin de liste — tandis que le **trait** porte la *structure
d'intégration*, celle qui fixe sa signature. D'où trois familles, nommées d'après
cette structure et non d'après la physique :

| trait | ce que le sous-modèle fait avant d'appeler la loi | signature | énuméré |
|---|---|---|---|
| `StatelessLawKind` | rien : la loi ne voit ni état, ni `dt` | `stress(strain, mat, continuum, symmetry, out)` | `ElasticLaw` |
| `ReturnMapLawKind` | le prédicteur élastique | `return_map(trial, prev, mat, dt)` | `PlasticLaw` |
| `DirectUpdateLawKind` | rien : la loi reçoit ε et l'état de A | `update(eps, prev, mat, space_dim)` | `DamageLaw` |

C'est bien la structure qui décide, pas le nom : `ViscoplasticLemaitreChaboche`
est physiquement « viscoplasticité **+ endommagement** » et siège du côté
plastique, parce qu'elle est de forme retour radial. Une loi hyperélastique
relève de `StatelessLawKind` ; une loi viscoélastique, de
`DirectUpdateLawKind`.

Tout le reste est générique et **ne change pas**.

## Le trait `SubModelKind`

Défini dans `src/models/mod.rs`. Le trait de base ne porte que le **dénominateur
commun** de tout sous-modèle ; chaque **capacité optionnelle** est un **sous-trait
séparé**, que rend une méthode d'accès `as_domain()`, `as_behavior()` ou
`as_constraint()`, `None` par défaut. Ces sous-traits nomment chacun **un**
fait, et un seul : `Domain` — *j'intègre sur un espace EF, avec du matériau* —,
`Behavior` — *et j'ai une loi évaluée en chaque point* —, `Constraint` — *mes
relations existent sous forme neutre, on peut m'imposer autrement que par mes
blocs*. `Behavior` a `Domain` pour supertrait : toute loi s'intègre, toute
intégration n'a pas de loi — un transfert de bord intègre `∫ h NᵀN` et n'a
aucune loi. Une struct n'implémente **que** la capacité qui la concerne : elle
n'a donc jamais de méthode « présente mais qui échouerait ». Un domaine typique
implémente `primal_vars`, `dual_vars`, `physics`, `as_domain` + `Domain`, le
noyau `element_matrix`, `stiffness_layout`, `label` et `render`. Il **n'écrit
pas** `build_stiffness_blocks` : le défaut le dérive de `stiffness_layout` +
`element_matrix`.

Seules cinq méthodes du trait de base sont **sans défaut** : `primal_vars`,
`dual_vars`, `physics`, `label` et `render`. Tout le reste se redéfinit à la
carte.

```text
pub trait SubModelKind: Sync {
    fn primal_vars(&self) -> Vec<String>;
    fn dual_vars(&self) -> Vec<String>;
    // Nature(s) de la physique — slice constante, pendant de `label` ; sert
    // aux sélecteurs `Model::filter` / `Matrix::filter` (match par appartenance) :
    fn physics(&self) -> &'static [Physics];

    // Méthodes d'accès aux capacités — None (défaut) ⇒ la struct n'a pas cette
    // capacité. Une struct qui l'a les redéfinit pour rendre `Some(self)` :
    fn as_domain(&self)     -> Option<&dyn Domain>     { None }
    fn as_constraint(&self) -> Option<&dyn Constraint> { None }
    fn as_behavior(&self)   -> Option<&dyn Behavior>   { None }

    // Le pont vers les noyaux de matrice, qui vivent sur `Domain` et `Behavior`
    // (voir plus bas) — fourni, on ne l'écrit pas :
    fn matrix_element(&self, kind: MatrixKind, geoms: &[CellGeom],
        material: &SubElementField, lay: &ElementLayout,
        inputs: &KernelState, ke: &mut [f64]) -> Result<()> { /* as_domain() puis route */ }

    // Contributions telles que l'assembleur les consomme (défaut : Computed(layout)
    // si matrix_layout(kind), sinon — pour Stiffness seulement —
    // Literal(build_stiffness_blocks), une erreur pour Tangent, rien sinon) :
    fn contributions(&self, kind: MatrixKind, material: Option<&Handle<SubElementField>>)
        -> Result<Vec<Contribution>> { /* défaut : dérivé de matrix_layout */ }
    fn build_stiffness_blocks(&self, material: &Handle<SubElementField>)
        -> Result<Vec<SubMatrix>> { /* défaut : dérivé de stiffness_layout + element_matrix */ }

    // Déclarations structurelles du bloc *calculé*, une par MatrixKind ;
    // None (défaut) ⇒ pas de terme de ce genre pour cette physique :
    fn stiffness_layout(&self)  -> Option<MatrixLayout> { None }
    fn mass_layout(&self)       -> Option<MatrixLayout> { None }
    fn geometric_layout(&self)  -> Option<MatrixLayout> { None }
    fn tangent_layout(&self)    -> Option<MatrixLayout> { None }
    // Physique linéaire sans noyau de tangente (ou contrainte) : sa tangente EST
    // sa raideur. Défaut false ⇒ ni tangent_layout ni ceci : la tangente échoue.
    fn tangent_is_stiffness(&self) -> bool { false }
    fn matrix_layout(&self, kind: MatrixKind) -> Option<MatrixLayout> { /* fourni */ }

    // Forces internes f = ∫ Bᵀ σ — le transposé de B. `lay` et `mat` disent où
    // chaque composante lue se trouve dans l'état et dans le matériau :
    fn internal_force_reads(&self) -> Vec<String> { /* défaut : σ du continuum */ }
    fn internal_force_element(&self, geoms: &[CellGeom], stress: &SubElementField,
        lay: &[u32], material: &SubElementField, mat: &[u32],
        fe: &mut [f64]) -> Result<()> { /* défaut : continuum */ }
    // Ce à quoi le terme applique son Bᵀ quand il n'a pas de loi (défaut : le
    // primal interpolé aux points) :
    fn residual_input(&self, fespace: &Handle<SubFiniteElementSpace>,
        solution: &NodeField) -> Result<Handle<SubElementField>> { /* fourni */ }
    // Les recettes du résidu, de part et d'autre de Σ f_int = Σ f_ext :
    fn internal_force_contribution(&self) -> Vec<ResidualContribution> { /* Computed(layout) */ }
    fn external_force_element(&self, geoms: &[CellGeom], material: &SubElementField,
        lay: &ElementLayout, fe: &mut [f64]) -> Result<()> { /* défaut : erreur */ }
    fn external_force_contribution(&self) -> Vec<ResidualContribution> { Vec::new() }

    fn label(&self) -> &'static str;
    fn display(&self) -> String { format!("SubModel<{}>", self.label()) }
    fn render(&self, opts: &DumpOptions) -> String;
}

// Capacités optionnelles — implémentées à part, jamais sur le trait de base.
// Un DOMAINE lit un matériau et intègre sur un espace EF. Rien de plus :
// avoir une loi est l'affaire de `Behavior`, juste en dessous.
pub trait Domain: Sync {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace>;
    // Les constantes exigées, dans l'ordre où le noyau les indexera ;
    // une liste vide dit « aucune composante contrainte » :
    fn material_components(&self) -> Vec<String> { Vec::new() }
    // Composantes acceptées mais non exigées (alpha) — cf. plus bas :
    fn optional_material_components(&self) -> &'static [&'static str] { &[] }

    // Ce que le noyau de matrice lit dans l'état, par genre (défaut : rien —
    // une raideur ou une masse ne lit que le matériau, déjà déclaré ci-dessus) :
    fn element_state_reads(&self, kind: MatrixKind) -> Vec<String> { Vec::new() }
    fn element_layout(&self, kind: MatrixKind, material: &SubElementField,
        state: Option<&SubElementField>) -> Result<ElementLayout> { /* fourni */ }

    // Les noyaux de matrice élémentaire (une cellule) — purs et séquentiels.
    // `geoms` : un CellGeom par espace EF du layout (geoms[0] pour le cas usuel,
    // plusieurs pour un élément multi-quadrature — la coque épaisse).
    // `lay` dit où chaque composante se trouve, résolu une fois par zone.
    // Tous ont un défaut qui échoue : « cette physique n'a pas ce terme ».
    fn element_matrix(&self, geoms: &[CellGeom], material: &SubElementField,
        lay: &ElementLayout, ke: &mut [f64]) -> Result<()>;                // ∫ Bᵀ D B
    fn element_mass(&self, geoms: &[CellGeom], material: &SubElementField,
        lay: &ElementLayout, ke: &mut [f64]) -> Result<()>;                // ∫ ρ Nᵀ N
    fn element_geometric(&self, geoms: &[CellGeom], material: &SubElementField,
        lay: &ElementLayout, state: &SubElementField, ke: &mut [f64]) -> Result<()>;  // ∫ Gᵀ σ̂ G
    fn coupling_element(&self, kind: MatrixKind, row_geoms: &[CellGeom],
        col_geoms: &[CellGeom], material: &SubElementField,
        lay: &ElementLayout, ke: &mut [f64]) -> Result<()>;                // interface
}

// Une LOI évaluée en chaque point de Gauss.
pub trait Behavior: Domain {
    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace>;
    fn behavior_output_components(&self) -> Vec<String>;
    // Ce que le noyau lit, dans l'ordre de ses indices :
    fn deformation_reads(&self) -> Vec<String>;
    // Et le `B` qui le produit depuis la solution — `Deformation(mesure)`,
    // `Gradient`, `Value`, `Beam`, `Shell(formulation)` ; `strain` l'exécute :
    fn strain_operator(&self) -> StrainOperator;
    fn state_reads(&self) -> Vec<String> { Vec::new() }
    fn zone_layout(&self, deformation: &SubElementField, prev: &SubElementField,
        material: &SubElementField) -> Result<ZoneLayout> { /* fourni */ }
    // L'état au repos, et la question « cette loi exige-t-elle un pas de
    // temps ? » — posées une fois, pas au point de Gauss :
    fn initial_state(&self, material: &SubElementField) -> Result<SubElementField> { /* des zéros */ }
    fn requires_dt(&self) -> bool { false }
    fn elastic_operator(&self) -> Option<(&Continuum, MaterialSymmetry)> { None }
    // Ce que coûtera la tangente : forme fermée ou différences finies :
    fn tangent_source(&self) -> TangentSource { TangentSource::Analytic }
    // La loi de comportement en UN point de Gauss, montage incrémental A → B.
    // Chaque entrée est LA LIGNE de ce point, empruntée au tampon du champ ;
    // `lay` dit où chaque composante s'y trouve (résolu une fois par zone) :
    fn integrate_point(&self, geom: &CellGeom, g: usize, lay: &ZoneLayout,
        deformation: &[f64], prev: &[f64], material: &[f64],
        dt: f64, out: &mut [f64]) -> Result<()>;
    fn integrate_behavior(&self, deformation: &Handle<SubElementField>,
        prev: &Handle<SubElementField>, material: &Handle<SubElementField>,
        dt: f64) -> Result<SubElementField> { /* fourni : résout la zone, pilote integrate_point */ }
    // La tangente cohérente — défaut qui échoue :
    fn tangent_point(&self, geom: &CellGeom, g: usize, lay: &ZoneLayout,
        deformation: &[f64], prev: &[f64], material: &[f64], dt: f64,
        d: &mut [[f64; 6]; 6]) -> Result<()>;                                // D_alg en un point
    fn element_tangent(&self, geoms: &[CellGeom], lay: &ZoneLayout,
        deformation: &SubElementField, prev: &SubElementField,
        material: &SubElementField, dt: f64, ke: &mut [f64]) -> Result<()>;  // ∫ Bᵀ D_alg B
}

pub trait Constraint {
    fn multiplier_mesh(&self) -> &Mesh;
    // Les relations linéaires imposées, sous forme neutre vis-à-vis de la méthode
    // d'imposition (Lagrange ou élimination) — cf. « Une contrainte » plus bas :
    fn relations(&self) -> Result<Vec<Relation>>;
}
```

Les deux capacités `Domain` et `Behavior` portent chacune une voie, et elles ont
la même forme : la physique déclare ce qu'elle lit, la zone le traduit en
positions une fois, le noyau indexe. Un noyau ne compare jamais un nom de
composante.

| | déclare | résout (1× par zone) | consomme |
|---|---|---|---|
| voie **point de Gauss** (`Behavior`) | `deformation_reads` / `state_reads` / `material_components` | `zone_layout` → `ZoneLayout` | `integrate_point`, `tangent_point` |
| voie **matrice** (`Domain`) | `material_components` / `element_state_reads` | `element_layout` → `ElementLayout` | `element_matrix`, `element_mass`, `element_geometric`, `coupling_element` |

Conséquences pratiques — pour donner une capacité à une physique, implémenter le
sous-trait **et** redéfinir la méthode d'accès correspondante pour rendre
`Some(self)` :

- **Domaine** (physique sur une région) : `impl Domain` + `as_domain()`. Déclarer
  `material_fespace()` (+ `material_components()`) — l'assembleur
  (`src/ops/matrix.rs`) sélectionne et valide le `SubElementField`
  automatiquement — et les noyaux de matrice qu'on a. Tous sont **fournis** et
  échouent par défaut : une physique peut intégrer sans produire de matrice d'un
  genre donné, voire aucune.
- **Comportement** (une loi au point) : `impl Behavior` + `as_behavior()`.
  Déclarer `behavior_fespace()` + `behavior_output_components()` +
  `deformation_reads()` + `strain_operator()` (quel producteur géométrique
  fabrique cette entrée depuis la solution : c'est ce qu'exécute
  `element_field::strain`) + `integrate_point(...)`, la loi de constitution **en
  un point de Gauss**. `integrate_behavior` est **fourni** : il résout la zone, puis
  pilote ce noyau en parallèle sur toutes les cellules. Un élément linéaire en a
  bien une, simplement triviale (`N = E·A·ε`) ; un transfert de bord, lui, n'en a
  pas — son `h·a` est le coefficient de son propre opérateur appliqué en un
  point, et son résidu suit de l'opérateur.

  **Une convention fixe la forme de la loi** : la contrainte en B est le
  matériau de B appliqué à la déformation élastique de B,
  `σ(B) = C(B) : ε_e(B)` — jamais `σ(A) + C:Δε`. C'est ce qui la garde juste
  quand le matériau dépend de la température, sans que le pas-à-pas ait à
  corriger quoi que ce soit (détail dans
  [comportement](algorithmes/comportement.md)).

  **Deux règles tiennent dans ce noyau**, et elles décident de la forme du
  reste : aucun test que l'amont a déjà tranché — présence d'une composante,
  forme d'un champ, `Option` à déballer — et aucune allocation dynamique. Un
  noyau conforme s'écrit sans un seul `?` sur ses lectures. Ce que l'auteur
  déclare (`deformation_reads`, `state_reads`, `material_components`) est
  exactement ce qui permet cela : la convention est traduite en positions **une
  fois par zone**, dans `zone_layout`, où un champ fabriqué à la main dans un
  autre ordre est refusé avec un message qui nomme le champ et l'écart. Les
  branchements qui restent sont ceux de la physique.

  **La modélisation, elle, ne s'écrit pas ici.** Une physique du continu tient
  un `Continuum` (`src/models/continuum/`) au lieu de redéclarer sa géométrie :
  il porte le sous-espace EF, le support POI1, la dimension et la cinématique,
  valide les trois cohérences à la construction (élément solide et non variété
  plongée, cinématique possible dans cet espace, accord géométrie ↔
  axisymétrie), et fournit les noyaux `element_stiffness`, `element_mass`,
  `element_geometric` et `element_tangent_of` ainsi que la nomenclature Voigt.
  Élasticité, plasticité et endommagement s'en servent toutes trois — c'est ce
  qui rend le produit *modélisation × loi* réel plutôt qu'une délégation d'une
  physique vers sa voisine.
- **Contrainte** (multiplicateurs de Lagrange) : `impl Constraint`
  (`multiplier_mesh()` + `relations()`) + `as_constraint()`.
  `SubModel::multiplier_nodes()` et `SubModel::multiplier_mesh()` en découlent.
  C'est le foyer de la famille contrainte — `Dirichlet`, `Mpc`, `Embedded`,
  `Contact`.
- **Terme de masse, raideur géométrique, tangente cohérente** : déclarer le
  `*_layout` correspondant et écrire le noyau `element_*` (voir ci-dessous).
- **Tangente d'une physique linéaire** : pas de noyau à écrire. Déclarer
  `tangent_is_stiffness()` à `true` suffit, et une demande de tangente assemble
  alors la raideur de ce sous-modèle, exactement. C'est le cas de la conduction,
  de Fick, des échanges, des barres, poutres et coques, et de toutes les
  contraintes, dont les relations sont linéaires. Une physique qui ne déclare ni
  ceci ni `tangent_layout` fait **échouer** la tangente : un bloc nul ferait
  diverger le Newton loin de la cause.

Une **contrainte** comme `Dirichlet` n'implémente *que* `Constraint` (plus
`contributions`, cf. ci-dessous) : elle n'a ni `element_matrix`, ni `Domain` —
leur absence est un **fait de compilation**, pas une erreur à l'exécution.

### La nature : `physics()`

`physics()` rend une **slice constante** de `Physics` (`Mechanical`, `Thermal`,
`Constraint`, `Other`, `Diffusion`, `Radiation`, `Contact`) : la classification
grossière de la physique, indépendante des capacités `Domain` / `Behavior` /
`Constraint`. Elle est **requise** — chaque physique déclare sa nature à son
site de définition, une physique couplée en déclarant plusieurs. Cette
information voyage avec chaque bloc assemblé jusqu'à la `SubMatrix`, et
alimente les sélecteurs `Model::filter` / `Matrix::filter`, qui retiennent un
bloc par **appartenance**. Un `HeatConduction` rend `&[Physics::Thermal]`, un
`Dirichlet` `&[Physics::Constraint]`.

### Un genre de matrice = un layout + un noyau

{{#pyalgo matrix.stiffness matrix.mass matrix.geometric matrix.tangent}}

L'assemblage ne dépend pas du genre de matrice. L'énum `MatrixKind`
(`Stiffness`, `Mass`, `Geometric`, `Tangent`) est le discriminant qui fait
tourner **la même machinerie** — recette, dispersion par coloration, cache de
motif creux par genre — avec un noyau élémentaire différent :

| `MatrixKind` | intégrale | layout | noyau |
|---|---|---|---|
| `Stiffness` | `∫ Bᵀ D B` | `stiffness_layout` | `element_matrix` |
| `Mass` | `∫ ρ Nᵀ N` | `mass_layout` | `element_mass` |
| `Geometric` | `∫ Gᵀ σ̂ G` | `geometric_layout` | `element_geometric` |
| `Tangent` | `∫ Bᵀ D_alg B` | `tangent_layout` | `element_tangent` |

Ajouter un terme à une physique, c'est donc **deux méthodes** : le `*_layout`
(souvent le même que celui de la raideur — mêmes espaces EF, même support,
mêmes variables : seul le noyau diffère) et le `element_*`. Une physique sans
terme d'un genre ne redéfinit rien : son layout reste `None`, elle ne contribue
pas, et `ops::matrix::mass(...)` sur un modèle qui la contient l'ignore
simplement. Côté opérateurs, un point d'entrée par genre —
`ops::matrix::{stiffness, mass, geometric, tangent}`, plus `lump` qui condense
une masse — les quatre premiers adossés au même `assemble_kind`.

La raideur **géométrique** reçoit en plus un `state` : la contrainte courante,
produite par `integrate_behavior`. C'est le couple producteur/consommateur.

La **tangente**, elle, ne reçoit aucun champ d'état préparé : `D_alg` est évalué
**au point de Gauss** par `tangent_point`, à partir des mêmes entrées que
`integrate_point`, et seulement quand on appelle `ops::matrix::tangent`.
L'intégration du comportement ne paie donc jamais une tangente qu'on ne
demande pas, et aucun champ de modules (6 à 21 réels par point) n'est stocké.
`Behavior::tangent_source()` dit d'avance ce qu'elle coûtera : `Analytic` pour
une forme fermée, `Numerical` pour une dérivée par différences centrées, soit
douze retours radiaux par point de Gauss.

### Les forces internes

`ops::node_field::internal_forces` lit la recette de chaque sous-modèle,
`internal_force_contribution()` : par défaut un
`ResidualContribution::Computed` sur le `stiffness_layout`, dont le noyau est
`internal_force_element`, piloté en parallèle sur les espaces EF du layout et
dispersé aux nœuds de son support. Le noyau élémentaire par défaut est celui de
la mécanique des milieux continus — `f_{i,a} = Σ_g Σ_b (∂N_i/∂x_b) σ_ab`, lu en
nommage Voigt (`sigma_xx`, `sigma_xy`, …), terme de cerceau compris en
axisymétrie. Une physique dont le dual n'est **pas** un vecteur déplacement
(thermique, barre, poutre, coque, échange) redéfinit `internal_force_element`.
Une contrainte déclare `ResidualContribution::Relations` : ses relations,
appliquées à la solution courante. Pour une loi linéaire, le résultat vaut
`K·u`.

Redéfinir, c'est écrire le transposé du `B` que la physique intègre déjà dans sa
rigidité — non en dériver un second. Les structurels rendent donc ce `B`
explicite et le partagent entre les deux sens : `models::beam::b_into` pour les
poutres, `models::shell::b_into` pour les coques. `tests/internal_forces.rs`
mesure l'égalité `f_int == K·u` qui en découle, et elle est **exacte** : la loi
d'un élément structurel est linéaire, il n'y a pas de tolérance physique à
choisir.

Le noyau reçoit la géométrie, l'**état** et le matériau. Le `B` d'un continuum
est le gradient symétrique, il ignore tout module ; un échange, lui, lit son
coefficient `h` dans le matériau. Une physique dont le `B` dépend d'une
grandeur dérivée du matériau la fait porter par son état :
[`Timoshenko`](mecanique/timoshenko.md) ajoute `Φ` (`phi`, ou `phi_y` et `phi_z`
en 3-D) aux efforts de son comportement, parce que le résidu le relit à
**chaque** itération de Newton.

### Le parallélisme est gratuit (et invisible)

Les noyaux qu'une physique écrit — `integrate_point` (un point de Gauss),
`element_matrix`, `element_mass`, `element_geometric` (la matrice élémentaire
d'une cellule), `internal_force_element` — sont **séquentiels et purs** : ils ne
voient ni rayon, ni un handle, ni un verrou. Les *drivers* de `models::kernel`
portent la parallélisation et le zéro-copie au-dessus d'eux. Voir
[Parallélisme](developper/parallelisme.md).

Concrètement, une physique de continuum déclare son layout (espaces EF, support,
variables, ordre des DDL) : le défaut de `contributions()` en tire une
`Contribution::Computed`, et l'assembleur global bâtit un bloc **calculé** puis
disperse le noyau élémentaire directement dans le CSR, en parallèle par
coloration des cellules — sans construire de matrice COO intermédiaire. La voie
**littérale** (`build_stiffness_blocks`) est le second défaut du trait, dérivée
du même couple `stiffness_layout` + `element_matrix` via `kernel::assemble_block`
; elle sert de référence d'équivalence dans les tests et de voie de secours pour
un sous-modèle sans layout, mais une physique volumique ne l'écrit pas.

Une contrainte comme `Dirichlet` (aucun layout, rien d'intégré sur une cellule)
redéfinit directement `contributions()` : elle rend `Vec::new()` pour tout genre
autre que `Stiffness`, et ses blocs C / Cᵀ en `Contribution::Literal` pour
celui-là — l'assembleur reste sans aucun cas particulier « Dirichlet ».

### Un bloc inter-maillages : `Coupling`

Une physique d'**interface** (l'échange `h(c₁ − c₂)` entre deux corps qui ne
partagent pas leurs nœuds) a besoin de blocs dont les **lignes vivent sur un
maillage et les colonnes sur un autre**. C'est la troisième variante,
`Contribution::Coupling(CouplingLayout)`.

Sous cette variante, tout est déjà asymétrique lignes/colonnes :
`SubMatrix::computed` prend deux supports, et la dispersion comme les drivers de
noyau les passent séparément. Seul `MatrixLayout.support` est unique ; un layout
séparé évite d'y ajouter un champ dont aucune autre physique n'a besoin :

```rust,ignore
{{#include ../../src/models/mod.rs:coupling_layout}}
```

Pas de champ `symmetric` : un bloc de couplage n'est jamais symétrique seul —
seule la réunion des quatre l'est, comme la paire C / Cᵀ.

Le noyau correspondant est `coupling_element(kind, row_geoms, col_geoms,
material, lay, ke)` : il reçoit **deux** jeux de `CellGeom`, la maille du côté
ligne et la maille en vis-à-vis du côté colonne. Le driver
`kernel::coupling_block_values_per_cell` parcourt les deux connectivités en pas
à pas ; il exige des maillages **conformes** (même type d'élément, même nombre de
mailles, maille `i` face à maille `i`) et le signale sinon.

C'est aussi le noyau qui porte le **signe** — `+h∫NᵢNⱼ` en diagonale via
`element_matrix`, `−h∫NᵢNⱼ` hors diagonale via `coupling_element` — puisque
chaque bloc choisit son noyau depuis sa propre variante de contribution.
L'assembleur n'a rien à savoir des interfaces.

Une réserve de mise en œuvre : la dispersion d'un bloc de couplage est
**séquentielle** (ses matrices élémentaires sont, elles, calculées en
parallèle). La coloration qui rend la dispersion parallèle sûre repose sur *une*
connectivité ; avec deux, elle ne prouve plus rien. Une interface porte un
maillage de bord : l'effet n'est pas mesurable, et une coloration à deux côtés
ne rapporterait rien.

Voir [`interface_transfer`](echanges.md#transfert-dinterface), son seul
utilisateur.

Le champ `fespaces` du `MatrixLayout` est un **`Vec`** : un seul espace EF
pour une physique de continuum, ou plusieurs — partageant un maillage, ne
différant que par la quadrature — pour un élément **multi-quadrature**. C'est ce
que fait la **coque de Reissner-Mindlin** (l'espace complet, puis l'espace de
cisaillement : membrane et flexion en Gauss complet, cisaillement transverse en
quadrature réduite, contre le verrouillage) : `element_matrix` reçoit alors deux
`CellGeom`, `geoms[0]` pour la membrane et la flexion, `geoms[1]` pour le
cisaillement, et l'élément passe par le **même** chemin de dispersion parallèle
que le reste — la sparsité ne dépendant que de la connectivité, pas de la
quadrature.

Le second espace est construit par la physique, pas reçu en argument : il est
entièrement déterminé par le premier, et les deux `CellGeom` doivent désigner
*la même maille*, une propriété qu'on préfère établir plutôt que vérifier. Une
même physique peut d'ailleurs en déclarer un nombre variable selon sa
formulation — la coque en Kirchhoff discret n'en déclare qu'un, n'ayant aucun
cisaillement à intégrer.

### Une contrainte : les relations, forme neutre

`Constraint::relations()` rend une `Relation` par nœud multiplicateur : son
`multiplier_node` (créé par la contrainte, un par relation), la liste des termes
`ConstraintTerm { node, variable, target_dual, coefficient }`, et un `sense`
(`RelationSense::Equality` par défaut, `GreaterEqual` / `LessEqual` pour
l'unilatéral, cf. [Contact](contraintes/contact.md)).

C'est la **source unique de vérité** : `contributions()` en tire les blocs
C / Cᵀ — via `constraint_block_pair`, qui pose aussi le sens des relations sur
le bloc C (`SubMatrix::set_row_sense`) — et les forces internes (`Cᵀλ`, `C·u`)
les appliquent. La **résolution**, elle, ne lit que la matrice : la ligne
`imposed` est la relation, ses coefficients sont la ligne, son sens est celui
du bloc ; multiplicateurs, ensemble actif ou élimination s'en servent tous
trois. Une nouvelle contrainte n'a donc à décrire ses relations **qu'une
fois**, et à poser leur sens sur son bloc C si elles sont unilatérales.

### Le comportement : le montage incrémental A → B

`integrate_point` intègre le pas **A → B** en un point de Gauss :

- `deformation` — la cinématique de **fin de pas** ε(B), produite par
  l'opérateur que désigne `strain_operator` : `deformation`, `gradient`,
  `interp_to_gauss`, `beam_deformation` ou `shell_deformation` ;
- `prev` — l'**état convergé au début du pas** A : le flux/contrainte σ(A), les
  variables internes en A, et pour les lois incrémentales la cinématique
  ε(A). Il existe toujours : au premier pas, c'est l'état de repos que fabrique
  `initial_state` (zéro partout, sauf ce que la loi fait partir d'une constante
  du matériau) ;
- `material` — les données matériau du point, vides pour une physique qui n'en
  déclare pas ;
- `dt` — l'incrément de temps, `0.0` pour une loi indépendante du temps ; une
  loi qui en a besoin le reçoit toujours, l'intégration refusant de démarrer
  sinon.

Le noyau écrit dans `out` les composantes déclarées par
`behavior_output_components()` : l'état matériau en B — σ(B), les variables
internes en B, et l'éventuelle ε(B) recopiée. La sortie devient le `prev` du
pas suivant. La tangente cohérente n'y est pas : `tangent_point` l'évalue au
point, à la demande de `matrix.tangent`.

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
    BoundaryTransfer(boundary_transfer::BoundaryTransfer),
    Dirichlet(dirichlet::Dirichlet),
    Mpc(mpc::Mpc),
    Embedded(embedded::Embedded),
    Contact(contact::Contact),
    Truss(truss::Truss),
    Elasticity(elasticity::Elasticity),
    Plasticity(plasticity::Plasticity),
    Damage(damage::Damage),
    Timoshenko(timoshenko::Timoshenko),
    // Les nouvelles variantes vont en fin de liste : bincode écrit l'indice.
    Fick(fick::Fick),
    InterfaceTransfer(interface_transfer::InterfaceTransfer),
    Radiation(radiation::Radiation),
    Bernoulli(bernoulli::Bernoulli),
    Shell(shell::Shell),
    Flux(flux::Flux),
}

impl SubModel {
    pub fn as_kind(&self) -> &dyn SubModelKind {
        match self {
            SubModel::HeatConduction(p) => p,
            SubModel::BoundaryTransfer(p) => p,
            // une ligne par variante, dans le même ordre
            SubModel::Flux(p) => p,
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
  pilote le matériau via `as_domain()` (`Domain`). Aucun `match` par variante.
- `src/ops/element_field/behavior.rs` et `material_field.rs`, avec leur wrapper
  `src/py/ops/element_field.rs`.
- `src/ops/node_field/internal_forces.rs` et `external_forces.rs` : passent par
  `internal_force_contribution()` et `external_force_contribution()`.
- `src/py/ops/matrix.rs` : les assembleurs délèguent à `model.inner`.

## Pour finir

Régénérer le stub `python/pyrucast/_pyrucast/__init__.pyi` (`cargo run --bin
stub_gen --features stub-gen`, venv activé), puis **compiler et tester avant de
commiter** — `script/check_all.sh` pour la passe complète, ou le seul bloc
concerné (`check_rust.sh`, `check_doc.sh`…).

---

## Pourquoi ça passe l'échelle

Avec des dizaines de physiques, deux propriétés comptent : le **coût
d'ajout** et la **persistance**.

### Coût d'ajout : constant

Le comportement d'une physique est **co-localisé** dans son fichier (struct
+ `impl SubModelKind` + les `impl` de capacité qui la concernent). Ajouter une
physique ne touche, en plus de ce fichier, que les lignes de raccordement de
[Les étapes](#les-étapes) — trois dans `model.rs`, une dans chacun de
`models/mod.rs`, `ops/model/mod.rs`, `lib.rs` et `python/pyrucast/model.py` —,
quel que soit le nombre de physiques présentes. Aucune des méthodes génériques
n'est modifiée. Un enum où chaque méthode ferait son propre `match` forcerait,
lui, à éditer chacune de ces méthodes à chaque physique ajoutée.

La même propriété tient sur l'autre axe : un **genre de matrice** est une
variante de `MatrixKind`, avec sa méthode de layout et son noyau dans les
traits, son point d'entrée dans `ops::matrix`, et un `*_layout` et un
`element_*` par physique concernée — le cache de motif et la dispersion ne font
pas de cas par genre.

### Pourquoi garder l'enum (et pas `Box<dyn SubModelKind>`)

La persistance utilise **`bincode`** sur des `Serialize/Deserialize`
*dérivés* (`src/archive/portable.rs`), un format **non auto-descriptif**. Or :

- un `enum SubModel` se sérialise nativement (indice de variante + contenu),
  zéro code manuel ;
- un `Box<dyn SubModelKind>` imposerait `typetag`, qui **ne supporte pas** les
  formats non auto-descriptifs comme `bincode` : la persistance serait perdue.

L'enum donne aussi l'**exhaustivité** : le compilateur refuse d'oublier un
cas dans `as_kind()`. On obtient donc le meilleur des deux mondes —
sérialisation triviale et exhaustivité de l'enum, comportement co-localisé
et coût d'ajout constant du trait.

### Bilan

- Format de persistance : enum + `bincode`, nouvelle variante en fin de liste.
- Coût d'ajout : **un fichier** et des lignes de raccordement en nombre fixe.
- Comportement : **co-localisé** par physique.
- Le seul `match` sur toutes les variantes de la couche modèle est
  `as_kind()` ; `update_contact` ne distingue que `SubModel::Contact` du
  reste.
