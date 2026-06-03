# Ajouter une physique

Ce chapitre liste **tous les points de code à toucher** pour ajouter une
nouvelle physique. L'architecture est conçue pour que ce coût soit **O(1)
fichier**, indépendant du nombre de physiques déjà présentes — voir
*[Pourquoi ça passe l'échelle](#pourquoi-ça-passe-léchelle)* en fin de
chapitre.

## Le principe en une phrase

L'enum [`Physics`](model.md) ne sert qu'au **stockage** et à la
**sérialisation** ; **tout le comportement** vit dans une struct par
physique (sous `src/models/`) qui implémente le trait `PhysicsKind`. Un
unique point de dispatch, `Physics::as_kind()`, relie les deux. Le code
générique (`SubModel`, l'assembleur, `Dump`) ne fait **jamais** de `match`
par variante.

```text
SubModel ── physics: Physics
                       │
        Physics  (enum : stockage + sérialisation bincode)
        ├── HeatConduction(HeatConduction)
        ├── Dirichlet(Dirichlet)
        └── as_kind(&self) -> &dyn PhysicsKind   ← l'unique match

        PhysicsKind  (trait : tout le comportement)
        ├── primal_vars / dual_vars
        ├── material_components / material_fespace   (défaut : None)
        ├── multiplier_support                       (défaut : None)
        ├── build_stiffness_blocks
        ├── build_mass_blocks                        (défaut : vide)
        └── label / display / render
```

## Les étapes

Ajouter une physique se réduit à **quatre** gestes :

1. **`src/models/<ma_physique>.rs`** (nouveau) — une struct portant ses
   supports + un `impl PhysicsKind` + un constructeur `new(...)` faisant le
   travail de construction (calque sur `heat_conduction.rs`, cas simple à
   1 bloc, ou `dirichlet.rs`, cas Lagrange à 2 blocs + nœuds créés à la
   volée). La struct dérive `Clone, Serialize, Deserialize`.
2. **`src/models/mod.rs`** — `pub mod <ma_physique>;`.
3. **`src/containers/model.rs`** — **une** variante dans `enum Physics` et
   **une** ligne dans `Physics::as_kind()`. Plus le constructeur public
   `Model::<ma_physique>(...)` (l'API parent).
4. **`src/py/model.rs`** — un `#[classmethod]` `PyModel::<ma_physique>(...)`.
   Étant dans `#[pymethods]`, aucun enregistrement n'est nécessaire.

Tout le reste est générique et **ne change pas**.

## Le trait `PhysicsKind`

Défini dans `src/models/mod.rs`. La plupart des méthodes ont une valeur par
défaut : une physique volumique typique n'implémente que `primal_vars`,
`dual_vars`, `material_*`, `build_stiffness_blocks`, `label` et `render`.

```rust,ignore
pub trait PhysicsKind {
    fn primal_vars(&self) -> Vec<String>;
    fn dual_vars(&self) -> Vec<String>;
    fn material_components(&self) -> Option<&'static [&'static str]> { None }
    fn material_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> { None }
    fn multiplier_support(&self) -> Option<&Handle<SubMesh>> { None }
    fn build_stiffness_blocks(&self, material: Option<&Handle<SubElementField>>)
        -> Result<Vec<SubMatrix>>;
    fn build_mass_blocks(&self, _material: Option<&Handle<SubElementField>>)
        -> Result<Vec<SubMatrix>> { Ok(Vec::new()) }
    fn label(&self) -> &'static str;
    fn display(&self) -> String { format!("SubModel<{}>", self.label()) }
    fn render(&self, opts: &DumpOptions) -> String;
}
```

Conséquences pratiques :

- **Matériau** : déclarer `material_fespace()` + `material_components()`
  suffit ; l'assembleur (`src/ops/assemble/mod.rs`) sélectionne et valide le
  `SubElementField` automatiquement, sans connaître la physique.
- **Multiplicateurs de Lagrange** : redéfinir `multiplier_support()` suffit ;
  `SubModel::multiplier_nodes()` et `multiplier_mesh()` en découlent.
- **Terme de masse** : redéfinir `build_mass_blocks()` (sinon : pas de masse).

## Le dispatch — `src/containers/model.rs`

```rust,ignore
#[derive(Clone, Serialize, Deserialize)]
pub enum Physics {
    HeatConduction(heat_conduction::HeatConduction),
    Dirichlet(dirichlet::Dirichlet),
    // … une ligne par physique
}

impl Physics {
    pub fn as_kind(&self) -> &dyn PhysicsKind {
        match self {
            Physics::HeatConduction(p) => p,
            Physics::Dirichlet(p) => p,
            // … une ligne par physique
        }
    }
}
```

`SubModel`, `Debug`, `Display`, `Dump` et l'assembleur appellent tous
`self.physics.as_kind().<méthode>()` — ils sont écrits une fois pour toutes.

## Ce qui est générique (rien à toucher)

- `src/ops/assemble/mod.rs` : `stiffness()` pilote le matériau via
  `material_fespace()` / `material_components()` ; `mass()` via
  `build_mass_blocks()`. Aucun `match` par variante.
- `src/ops/build/material_field.rs` et son wrapper `src/py/ops/build.rs`.
- `src/py/ops/assemble.rs` : `stiffness` / `mass` délèguent à `model.inner`.

## Pour finir

Régénérer le stub `pyrucast.pyi` via `src/bin/stub_gen.rs`, puis **builder
+ tester avant de commiter** (`PYO3_PYTHON=/usr/bin/python3.13`, ou
`scripts/check.sh` pour la passe complète).

---

## Pourquoi ça passe l'échelle

Avec des dizaines de physiques, deux propriétés comptent : le **coût
d'ajout** et la **persistance**.

### Coût d'ajout : O(1) fichier

Le comportement d'une physique est **co-localisé** dans son fichier (struct
+ `impl PhysicsKind`). Ajouter la physique n°30 ne touche que 4 endroits
(§ Les étapes), dont 2 sont des lignes uniques dans `model.rs`. Aucune des
méthodes génériques n'est modifiée. C'est l'inverse du *« shotgun
surgery »* qu'imposerait un enum où chaque méthode ferait son propre
`match` : là, ajouter une physique forcerait à éditer une dizaine de sites.

> Les deux lignes (variante + bras de `as_kind`) pourraient même être
> générées par une macro `physics_enum! { HeatConduction, Dirichlet, … }`
> pour ne laisser qu'une seule déclaration.

### Pourquoi garder l'enum (et pas `Box<dyn PhysicsKind>`)

La persistance utilise **`bincode`** sur des `Serialize/Deserialize`
*dérivés* (`src/persist.rs`), un format **non auto-descriptif**. Or :

- un `enum Physics` se sérialise nativement (indice de variante + payload),
  zéro code manuel ;
- un `Box<dyn PhysicsKind>` imposerait `typetag`, qui **ne supporte pas**
  les formats non auto-descriptifs comme `bincode`. On perdrait la
  persistance.

L'enum donne aussi l'**exhaustivité** : le compilateur refuse d'oublier un
cas dans `as_kind()`. On obtient donc le meilleur des deux mondes —
sérialisation triviale et exhaustivité de l'enum, comportement co-localisé
et coût d'ajout constant du trait.

### Bilan

- Format de persistance : enum + `bincode`, **stable**.
- Coût d'ajout : **O(1) fichier**, ~2 lignes de câblage.
- Comportement : **co-localisé** par physique.
- Le seul `match` par variante du module modèle est `as_kind()`.
