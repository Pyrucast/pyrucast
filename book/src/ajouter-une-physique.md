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
└── as_kind(&self) -> &dyn SubModelKind   ← l'unique match

SubModelKind  (trait de base : le dénominateur commun de tout sous-modèle)
├── primal_vars / dual_vars
├── as_domain     -> Option<&dyn Domain>        (défaut : None)  ── seam capacité
├── as_constraint -> Option<&dyn Constraint>    (défaut : None)  ── seam capacité
├── element_matrix                              (noyau cellule ; défaut : erreur)
├── stiffness_layout                            (bloc calculé ; défaut : None)
├── contributions                               (défaut : dérivé du layout)
├── build_stiffness_blocks                      (défaut : dérivé du layout)
├── build_mass_blocks                           (défaut : vide)
└── label / display / render

Sous-traits « capacité », miroir des natures de sous-modèle (une struct
n'implémente que celui qui la concerne) :
├── Domain      { material_fespace, material_components,
│                 behavior_fespace, behavior_output_components,
│                 integrate_point, integrate_behavior (fourni) }
└── Constraint  { multiplier_mesh }
```

## Les étapes

Ajouter une physique se réduit à **quatre** gestes :

1. **`src/models/<ma_physique>.rs`** (nouveau) — une struct portant ses
   supports + un `impl SubModelKind` + un constructeur `new(...)` faisant le
   travail de construction (calque sur `heat_conduction.rs`, cas simple à
   1 bloc, ou `dirichlet.rs`, contrainte de Lagrange à 2 blocs portée par des
   maillages fournis par l'utilisateur). La struct dérive `Serialize,
   Deserialize` (et `Clone` si ses champs le permettent).
2. **`src/models/mod.rs`** — `pub mod <ma_physique>;`.
3. **`src/containers/model.rs`** — **une** variante dans `enum SubModel` et
   **une** ligne dans `SubModel::as_kind()`. Plus le constructeur public
   `Model::<ma_physique>(...)` (l'API parent).
4. **`src/py/model.rs`** — un `#[classmethod]` `PyModel::<ma_physique>(...)`.
   Étant dans `#[pymethods]`, aucun enregistrement n'est nécessaire.

Tout le reste est générique et **ne change pas**.

## Le trait `SubModelKind`

Défini dans `src/models/mod.rs`. Le trait de base ne porte que le **dénominateur
commun** de tout sous-modèle ; chaque **capacité optionnelle** est un **sous-trait
séparé**, exposé par un *seam* `as_*()` qui rend `None` par défaut. Ces
sous-traits **font miroir des natures** de sous-modèle : `Domain` (physique
définie sur une région : matériau + comportement) et `Constraint` (multiplicateurs
de Lagrange). Une struct n'implémente **que** la capacité qui la concerne : elle
n'a donc jamais de méthode « présente mais qui erronerait ». Un domaine typique
implémente `primal_vars`, `dual_vars`, `as_domain` + `Domain`, le noyau
`element_matrix`, `stiffness_layout`, `label` et `render`. Il **n'écrit pas**
`build_stiffness_blocks` : le défaut le dérive de `stiffness_layout` +
`element_matrix`.

```rust,ignore
pub trait SubModelKind: Sync {
    fn primal_vars(&self) -> Vec<String>;
    fn dual_vars(&self) -> Vec<String>;
    // Seams de capacité — None (défaut) ⇒ la struct n'a pas cette capacité.
    // Une struct qui l'a redéfinit le seam pour rendre `Some(self)` :
    fn as_domain(&self)     -> Option<&dyn Domain>     { None }
    fn as_constraint(&self) -> Option<&dyn Constraint> { None }
    // Noyau de matrice élémentaire (une cellule) — pur et séquentiel ;
    // `geoms` : un CellGeom par espace EF du layout (geoms[0] pour le cas usuel,
    // plusieurs pour un élément multi-quadrature — poutre/coque) :
    fn element_matrix(&self, geoms: &[CellGeom],
        material: Option<&SubElementField>, ke: &mut [f64]) -> Result<()> { /* défaut : erreur */ }
    // Déclare le bloc *calculé* (assemblage global par scatter colorié parallèle) ;
    // None (défaut) ⇒ physique assemblée en littéral (Dirichlet, blocs à la main) :
    fn stiffness_layout(&self) -> Option<StiffnessLayout> { None }
    // Contributions de raideur, telles que l'assembleur les consomme (défaut :
    // Computed(layout) si stiffness_layout, sinon Literal(build_stiffness_blocks)) :
    fn contributions(&self, material: Option<&Handle<SubElementField>>)
        -> Result<Vec<Contribution>> { /* défaut : dérivé de stiffness_layout */ }
    fn build_stiffness_blocks(&self, material: Option<&Handle<SubElementField>>)
        -> Result<Vec<SubMatrix>> { /* défaut : dérivé de stiffness_layout + element_matrix */ }
    fn build_mass_blocks(&self, _material: Option<&Handle<SubElementField>>)
        -> Result<Vec<SubMatrix>> { Ok(Vec::new()) }
    fn label(&self) -> &'static str;
    fn display(&self) -> String { format!("SubModel<{}>", self.label()) }
    fn render(&self, opts: &DumpOptions) -> String;
}

// Capacités optionnelles — implémentées à part, jamais sur le trait de base.
// Un DOMAINE lit un matériau ET intègre un comportement : les deux sont UNE
// capacité, car le matériau paramètre la loi (σ = D(E,ν):ε, M = E·I·κ, …).
pub trait Domain: Sync {
    fn material_fespace(&self) -> Handle<SubFiniteElementSpace>;
    fn material_components(&self) -> Option<&'static [&'static str]> { None }
    fn behavior_fespace(&self) -> Handle<SubFiniteElementSpace>;
    fn behavior_output_components(&self) -> Result<Vec<String>>;
    fn integrate_point(&self, geom: &CellGeom, input: &SubElementField,
        material: Option<&SubElementField>, g: usize, out: &mut [f64]) -> Result<()>;
    fn integrate_behavior(&self, input: &Handle<SubElementField>,
        material: Option<&Handle<SubElementField>>) -> Result<SubElementField> { /* fourni : pilote integrate_point */ }
}
pub trait Constraint {
    fn multiplier_mesh(&self) -> &Mesh;
}
```

Conséquences pratiques — pour donner une capacité à une physique, implémenter le
sous-trait **et** redéfinir le seam correspondant pour rendre `Some(self)` :

- **Domaine** (physique sur une région) : `impl Domain` + `as_domain()`. Déclarer
  `material_fespace()` (+ `material_components()`) — l'assembleur
  (`src/ops/assemble/mod.rs`) sélectionne et valide le `SubElementField`
  automatiquement ; `behavior_fespace()` + `behavior_output_components()` +
  `integrate_point(...)`, la loi de constitution **en un point de Gauss**.
  `integrate_behavior` est **fourni** : il pilote ce noyau en parallèle sur toutes
  les cellules. Matériau et comportement vont **ensemble** : même un élément
  linéaire (barre, poutre) a un comportement, simplement trivial (`N = E·A·ε`).
- **Contrainte** (multiplicateurs de Lagrange) : `impl Constraint`
  (`multiplier_mesh()`) + `as_constraint()`. `SubModel::multiplier_nodes()` et
  `multiplier_mesh()` en découlent. C'est le foyer de la famille contrainte
  (Dirichlet, à venir MPC / contact fort).
- **Terme de masse** : redéfinir `build_mass_blocks()` (sinon : pas de masse).

Une **contrainte** comme `Dirichlet` n'implémente *que* `Constraint` (plus
`contributions`, cf. ci-dessous) : elle n'a ni `element_matrix`, ni `Domain` —
leur absence est un **fait de compilation**, pas une erreur à l'exécution.

### Le parallélisme est gratuit (et invisible)

Les noyaux qu'une physique écrit — `integrate_point` (un point de Gauss) et
`element_matrix` (la matrice élémentaire d'une cellule) — sont **séquentiels et
purs** : ils ne voient ni rayon, ni le store, ni un verrou. Les *drivers* de
`models::kernel` portent la parallélisation et le zéro-copie au-dessus d'eux.
Voir [Parallélisme](developper/parallelisme.md).

Concrètement, une physique de continuum déclare `stiffness_layout()` (espaces EF,
support, variables, ordering) : le défaut de `contributions()` en tire une
`Contribution::Computed`, et l'assembleur global bâtit un bloc **calculé** puis
disperse `element_matrix` directement dans le CSR, en parallèle par coloration des
cellules — sans matérialiser de COO. La voie **littérale**
(`build_stiffness_blocks`) est le second défaut du trait, dérivée du même couple
`stiffness_layout` + `element_matrix` via `kernel::assemble_block` ; elle sert de
référence d'équivalence et de repli, mais une physique volumique **ne l'écrit
plus**. Une contrainte comme `Dirichlet` (aucun `stiffness_layout`, rien
d'intégré sur une cellule) redéfinit directement `contributions()` pour rendre ses
blocs C / Cᵀ en `Contribution::Literal` — l'assembleur reste sans aucun cas
particulier « Dirichlet ».

Le champ `fespaces` du `stiffness_layout` est un **`Vec`** : un seul espace EF
pour une physique de continuum, ou plusieurs — partageant un maillage, ne
différant que par la quadrature — pour un élément **multi-quadrature**. C'est ce
que fait la **poutre de Timoshenko** (`fespaces: vec![bending, shear]`, flexion en
Gauss complet + cisaillement réduit) : `element_matrix` reçoit alors deux
`CellGeom`, `geoms[0]` pour la flexion et `geoms[1]` pour le cisaillement, et
l'élément passe par le **même** chemin de scatter parallèle que le reste — la
sparsité ne dépendant que de la connectivité, pas de la quadrature.

## Le dispatch — `src/containers/model.rs`

```rust,ignore
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

- `src/ops/assemble/mod.rs` : `stiffness()` boucle sur `contributions()` et
  pilote le matériau via le seam `as_domain()` (`Domain`) ; `mass()` via
  `build_mass_blocks()`. Aucun `match` par variante.
- `src/ops/build/material_field.rs` et son wrapper `src/py/ops/build.rs`.
- `src/py/ops/assemble.rs` : `stiffness` / `mass` délèguent à `model.inner`.

## Pour finir

Régénérer le stub `pyrucast.pyi` via `src/bin/stub_gen.rs`, puis **builder
+ tester avant de commiter** (`PYO3_PYTHON=/usr/bin/python3.13`, ou
`script/check.sh` pour la passe complète).

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
  automatiquement ».
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
