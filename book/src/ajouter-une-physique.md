# Ajouter une physique

Ce chapitre liste **tous les points de code à toucher** pour ajouter une
nouvelle physique. L'architecture est conçue pour que ce coût soit **O(1)
fichier**, indépendant du nombre de physiques déjà présentes — voir
*[Pourquoi ça passe l'échelle](#pourquoi-ça-passe-léchelle)* en fin de
chapitre.

## Le principe en une phrase

L'énum [`SubModel`](model.md) ne sert qu'au **stockage** et à la
**sérialisation** ; **tout le comportement** vit dans une struct par
physique (sous `src/models/`) qui implémente le trait `Physics`. Un unique
point de dispatch, `SubModel::as_physics()`, relie les deux. Le code
générique (l'agrégat `Model`, l'assembleur, `Dump`) ne fait **jamais** de
`match` par variante.

```text
SubModel  (enum : stockage + sérialisation bincode)
├── HeatConduction(HeatConduction)
├── Dirichlet(Dirichlet)
└── as_physics(&self) -> &dyn Physics   ← l'unique match

Physics  (trait : tout le comportement)
├── primal_vars / dual_vars
├── material_components / material_fespace   (défaut : None)
├── multiplier_mesh                          (défaut : None)
├── element_matrix                           (noyau cellule ; défaut : erreur)
├── stiffness_layout                         (bloc calculé ; défaut : None)
├── build_stiffness_blocks
├── build_mass_blocks                        (défaut : vide)
└── label / display / render
```

## Les étapes

Ajouter une physique se réduit à **quatre** gestes :

1. **`src/models/<ma_physique>.rs`** (nouveau) — une struct portant ses
   supports + un `impl Physics` + un constructeur `new(...)` faisant le
   travail de construction (calque sur `heat_conduction.rs`, cas simple à
   1 bloc, ou `dirichlet.rs`, contrainte de Lagrange à 2 blocs portée par des
   maillages fournis par l'utilisateur). La struct dérive `Serialize,
   Deserialize` (et `Clone` si ses champs le permettent).
2. **`src/models/mod.rs`** — `pub mod <ma_physique>;`.
3. **`src/containers/model.rs`** — **une** variante dans `enum SubModel` et
   **une** ligne dans `SubModel::as_physics()`. Plus le constructeur public
   `Model::<ma_physique>(...)` (l'API parent).
4. **`src/py/model.rs`** — un `#[classmethod]` `PyModel::<ma_physique>(...)`.
   Étant dans `#[pymethods]`, aucun enregistrement n'est nécessaire.

Tout le reste est générique et **ne change pas**.

## Le trait `Physics`

Défini dans `src/models/mod.rs`. La plupart des méthodes ont une valeur par
défaut : une physique volumique typique n'implémente que `primal_vars`,
`dual_vars`, `material_*`, le noyau `element_matrix`, `stiffness_layout`, `label`
et `render`. Elle **n'écrit pas** `build_stiffness_blocks` : le défaut le dérive
de `stiffness_layout` + `element_matrix`.

```rust,ignore
pub trait Physics: Sync {
    fn primal_vars(&self) -> Vec<String>;
    fn dual_vars(&self) -> Vec<String>;
    fn material_components(&self) -> Option<&'static [&'static str]> { None }
    fn material_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> { None }
    fn multiplier_mesh(&self) -> Option<&Mesh> { None }
    // Noyau de matrice élémentaire (une cellule) — pur et séquentiel :
    fn element_matrix(&self, geom: &CellGeom,
        material: Option<&SubElementField>, ke: &mut [f64]) -> Result<()> { /* défaut : erreur */ }
    // Déclare le bloc *calculé* (assemblage global par scatter colorié parallèle) ;
    // None (défaut) ⇒ physique assemblée en littéral (Dirichlet, blocs à la main) :
    fn stiffness_layout(&self) -> Option<StiffnessLayout> { None }
    fn build_stiffness_blocks(&self, material: Option<&Handle<SubElementField>>)
        -> Result<Vec<SubMatrix>> { /* défaut : dérivé de stiffness_layout + element_matrix */ }
    fn build_mass_blocks(&self, _material: Option<&Handle<SubElementField>>)
        -> Result<Vec<SubMatrix>> { Ok(Vec::new()) }
    // Comportement (loi de constitution) — noyau point-local pur :
    fn behavior_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> { None }
    fn behavior_output_components(&self) -> Result<Vec<String>> { /* défaut : erreur */ }
    fn integrate_point(&self, geom: &CellGeom, input: &SubElementField,
        material: Option<&SubElementField>, g: usize, out: &mut [f64]) -> Result<()> { /* défaut : erreur */ }
    fn integrate_behavior(&self, input: &Handle<SubElementField>,
        material: Option<&Handle<SubElementField>>) -> Result<SubElementField> { /* fourni : pilote integrate_point */ }
    fn label(&self) -> &'static str;
    fn display(&self) -> String { format!("SubModel<{}>", self.label()) }
    fn render(&self, opts: &DumpOptions) -> String;
}
```

Conséquences pratiques :

- **Matériau** : déclarer `material_fespace()` + `material_components()`
  suffit ; l'assembleur (`src/ops/assemble/mod.rs`) sélectionne et valide le
  `SubElementField` automatiquement, sans connaître la physique.
- **Multiplicateurs de Lagrange** : redéfinir `multiplier_mesh()` suffit ;
  `SubModel::multiplier_nodes()` et `multiplier_mesh()` en découlent.
- **Terme de masse** : redéfinir `build_mass_blocks()` (sinon : pas de masse).
- **Comportement** : déclarer `behavior_fespace()` + `behavior_output_components()`
  et écrire `integrate_point(...)` — la loi de constitution **en un point de
  Gauss**. `integrate_behavior` est **fourni** : il pilote ce noyau en parallèle
  sur toutes les cellules.

### Le parallélisme est gratuit (et invisible)

Les noyaux qu'une physique écrit — `integrate_point` (un point de Gauss) et
`element_matrix` (la matrice élémentaire d'une cellule) — sont **séquentiels et
purs** : ils ne voient ni rayon, ni le store, ni un verrou. Les *drivers* de
`models::kernel` portent la parallélisation et le zéro-copie au-dessus d'eux.
Voir [Parallélisme](developper/parallelisme.md).

Concrètement, une physique de continuum déclare `stiffness_layout()` (support,
variables, ordering) : l'assembleur global bâtit alors un bloc **calculé** et
disperse `element_matrix` directement dans le CSR, en parallèle par coloration
des cellules — sans matérialiser de COO. La voie **littérale**
(`build_stiffness_blocks`) en est le **défaut du trait**, dérivé du même couple
`stiffness_layout` + `element_matrix` via `kernel::assemble_block` ; elle sert de
référence d'équivalence et de repli, mais une physique volumique **ne l'écrit
plus**. (Seules les physiques *sans* `stiffness_layout` — `Dirichlet`, Timoshenko
à deux quadratures — redéfinissent `build_stiffness_blocks`.)

## Le dispatch — `src/containers/model.rs`

```rust,ignore
#[derive(Serialize, Deserialize)]
pub enum SubModel {
    HeatConduction(heat_conduction::HeatConduction),
    Dirichlet(dirichlet::Dirichlet),
    // … une ligne par physique
}

impl SubModel {
    pub fn as_physics(&self) -> &dyn Physics {
        match self {
            SubModel::HeatConduction(p) => p,
            SubModel::Dirichlet(p) => p,
            // … une ligne par physique
        }
    }
}
```

`Debug`, `Display`, `Dump`, les méthodes déléguantes de `SubModel` et
l'assembleur appellent tous `self.as_physics().<méthode>()` — ils sont
écrits une fois pour toutes.

## Ce qui est générique (rien à toucher)

- `src/ops/assemble/mod.rs` : `stiffness()` pilote le matériau via
  `material_fespace()` / `material_components()` ; `mass()` via
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
+ `impl Physics`). Ajouter la physique n°30 ne touche que 4 endroits
(§ Les étapes), dont 2 sont des lignes uniques dans `model.rs`. Aucune des
méthodes génériques n'est modifiée. C'est l'inverse du *« shotgun
surgery »* qu'imposerait un enum où chaque méthode ferait son propre
`match` : là, ajouter une physique forcerait à éditer une dizaine de sites.

> Les deux lignes (variante + bras de `as_physics`) pourraient même être
> générées par une macro `physics_enum! { HeatConduction, Dirichlet, … }`
> pour ne laisser qu'une seule déclaration.

### Pourquoi garder l'enum (et pas `Box<dyn Physics>`)

La persistance utilise **`bincode`** sur des `Serialize/Deserialize`
*dérivés* (`src/persist.rs`), un format **non auto-descriptif**. Or :

- un `enum SubModel` se sérialise nativement (indice de variante + payload),
  zéro code manuel ;
- un `Box<dyn Physics>` imposerait `typetag`, qui **ne supporte pas** les
  formats non auto-descriptifs comme `bincode`. On perdrait la persistance.

L'enum donne aussi l'**exhaustivité** : le compilateur refuse d'oublier un
cas dans `as_physics()`. On obtient donc le meilleur des deux mondes —
sérialisation triviale et exhaustivité de l'enum, comportement co-localisé
et coût d'ajout constant du trait.

### Et les données neutres partagées ?

Une donnée commune à *toutes* les physiques (un `name`, un flag `enabled`,
une pondération) se traite selon sa nature :

- **dérivable** (calculable à partir du type/de l'état) → un **défaut dans
  le trait `Physics`** la fournit gratuitement à toutes les physiques, ex.
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
- Le seul `match` par variante du module modèle est `as_physics()`.
