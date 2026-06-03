# Ajouter une physique

Ce chapitre liste **tous les points de code à toucher** pour ajouter une
nouvelle physique, puis évalue la **viabilité de l'architecture** quand le
nombre de physiques se compte en dizaines.

Le point d'entrée central est l'enum [`Physics`] dans
`src/containers/model.rs` : tout le reste en découle. Le résumé canonique
est déjà dans `src/models/mod.rs` :

> *Étendre `Physics` avec une variante, ajouter `models/<nom>.rs` avec ses
> fonctions `build` + assemblage, et câbler le dispatch dans `SubModel`.*

## 1. Le code de la physique

**Nouveau fichier `src/models/<ma_physique>.rs`** — à calquer sur
`models/heat_conduction.rs` (cas simple, 1 bloc) ou `models/dirichlet.rs`
(Lagrange, 2 blocs + nœuds créés à la volée). Il contient :

- les constantes de noms (`PRIMAL_VAR`, `DUAL_VAR`, `MATERIAL_COMPONENT`…) ;
- une éventuelle fonction `build(...)` si la construction fait plus que
  ranger ses arguments ;
- la/les fonction(s) d'assemblage (`assemble_stiffness` / `assemble_blocks`).

**`src/models/mod.rs`** — déclarer `pub mod <ma_physique>;`.

## 2. Le dispatch central — `src/containers/model.rs`

C'est ici que se trouve l'essentiel du câblage. Chaque `match` sur `Physics`
doit recevoir un bras :

| Endroit | Rôle |
|---|---|
| variante de `enum Physics` | déclarer la variante + ses supports |
| `Physics::primal_vars` | noms des colonnes |
| `Physics::dual_vars` | noms des lignes |
| `Physics::material_components` | composants matériau requis (ou `None`) |
| `SubModel::<ma_physique>(...)` | constructeur sub-modèle |
| `SubModel::material_fespace` | FE subspace du matériau (ou `None`) |
| **`SubModel::build_stiffness_blocks`** | **le vrai dispatch d'assemblage** |
| `impl Debug for SubModel` | étiquette de debug |
| `impl Display for SubModel` | rendu une ligne |
| `impl Dump for SubModel` | rendu détaillé |
| `Model::<ma_physique>(...)` | constructeur parent (l'API publique) |

Si la physique est de type Lagrange (crée des nœuds multiplicateurs comme
Dirichlet), ajouter aussi un bras à `multiplier_nodes` et `multiplier_mesh`.

## 3. Assemblage — `src/ops/assemble/mod.rs`

- `stiffness()` : bras du `match sub.physics()` qui sélectionne/valide le
  matériau.
- `mass()` : aujourd'hui un *stub* renvoyant une matrice vide. **Seulement
  si la physique a un terme de masse**, prévoir un `build_mass_blocks`
  parallèle à `build_stiffness_blocks` et le brancher ici.

## 4. Couche Python (PyO3)

- **`src/py/model.rs`** : ajouter le `#[classmethod]`
  `PyModel::<ma_physique>(...)` (modèles : `heat_conduction` / `dirichlet`).
  Étant dans `#[pymethods]`, **aucun enregistrement** n'est nécessaire.
  Ajouter des accesseurs sur `PySubModel` seulement si la physique en
  expose (ex. `multiplier_mesh`).

## 5. Ce qui est générique (rien à toucher en principe)

- `src/ops/build/material_field.rs` et son wrapper `src/py/ops/build.rs` :
  pilotés par `material_fespace()` / `material_components()`, donc
  automatiques.
- `src/py/ops/assemble.rs` : `stiffness`/`mass` délèguent à `model.inner`,
  génériques.

## 6. Pour finir

Régénérer le stub `pyrucast.pyi` via le binaire `src/bin/stub_gen.rs`, puis
**builder + tester avant de commiter** (`PYO3_PYTHON=/usr/bin/python3.13`).

---

## Viabilité à l'échelle (dizaines de physiques)

**Verdict : le *modèle de données* (enum) est le bon choix, mais les ~12
`match` parallèles ne passeront pas l'échelle en l'état. Un refactor ciblé,
sans changer le format de persistance, supprime le problème.**

### Pourquoi garder l'enum (et pas `Box<dyn Trait>`)

La persistance utilise **`bincode`** sur des `Serialize/Deserialize`
*dérivés* (`src/persist.rs`). `bincode` est un format **non
auto-descriptif**. Conséquence dure :

- un `enum Physics` se sérialise nativement (indice de variante + payload) :
  zéro code manuel, robuste ;
- un `Box<dyn PhysicsKind>` imposerait `typetag` (ou un ser/de manuel).
  **`typetag` ne supporte pas les formats non auto-descriptifs comme
  `bincode`.** On perdrait donc la persistance, ou il faudrait réécrire
  `ser/de` à la main.

L'enum donne aussi l'**exhaustivité** : le compilateur refuse d'oublier un
cas. Avec des dizaines de physiques, c'est une assurance correctness — mais
c'est aussi la source de la douleur ci-dessous.

### Le vrai problème : la chirurgie en fusil à pompe

Ajouter la physique n°30 force aujourd'hui à éditer ~12 endroits dispersés
(§2 + §3 + §4). C'est le *smell* « shotgun surgery » : le code d'**une**
physique est éclaté dans dix `match` au lieu d'être co-localisé.

### Le refactor recommandé : enum pour le stockage, trait pour le comportement

Garder l'enum **uniquement** pour les données + la sérialisation, et faire
transiter **tout le comportement** par un trait, via un seul point de
dispatch :

```rust,ignore
// Chaque physique : sa propre struct, porte ses données, implémente le trait.
trait PhysicsKind {
    fn primal_vars(&self) -> Vec<String>;
    fn dual_vars(&self) -> Vec<String>;
    fn material_components(&self) -> Option<&'static [&'static str]>;
    fn build_stiffness_blocks(&self, m: Option<&Handle<SubElementField>>)
        -> Result<Vec<SubMatrix>>;
    // valeurs par défaut : la plupart des physiques n'ont rien à redéfinir
    fn build_mass_blocks(&self, _m: Option<&Handle<SubElementField>>)
        -> Result<Vec<SubMatrix>> { Ok(vec![]) }
    fn material_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> { None }
    fn multiplier_support(&self) -> Option<&Handle<SubMesh>> { None }
    fn label(&self) -> &'static str;
    fn render(&self, opts: &DumpOptions) -> String;
}

// Enum conservé SEULEMENT pour la sérialisation + un dispatch unique.
#[derive(Serialize, Deserialize)]
enum Physics {
    HeatConduction(HeatConduction),
    Dirichlet(Dirichlet),
    // … une ligne par physique
}

impl Physics {
    fn as_kind(&self) -> &dyn PhysicsKind {
        match self {
            Physics::HeatConduction(p) => p,
            Physics::Dirichlet(p) => p,
            // … une ligne par physique
        }
    }
}
```

Désormais, **toutes** les méthodes génériques (`primal_vars`, `dual_vars`,
`material_components`, `Debug`, `Display`, `Dump`, le dispatch de
`build_stiffness_blocks`, l'assemblage dans `ops/assemble`) appellent
`self.physics.as_kind()` et **ne sont plus jamais touchées** quand on ajoute
une physique.

Ajouter une physique se réduit alors à :

1. un nouveau fichier `models/<nom>.rs` (struct + `impl PhysicsKind`) ;
2. **une** variante dans `enum Physics` ;
3. **une** ligne dans `as_kind()` ;
4. le constructeur parent `Model::<nom>` + son `#[classmethod]` Python.

Les points 2 et 3 peuvent même être générés par une macro
(`physics_enum! { HeatConduction, Dirichlet, … }`) pour ne laisser qu'**une
seule** déclaration. On passe de ~12 points d'édition à 1–2, **sans rien
changer au format `bincode`** ni perdre l'exhaustivité.

### Bilan

- Format de persistance : **inchangé** (toujours enum + `bincode`).
- Coût d'ajout d'une physique : **O(1) fichier**, ~2 lignes de câblage.
- Comportement co-localisé : tout le code d'une physique vit dans son
  fichier.
- À faire tant que le nombre de variantes est petit : plus le refactor
  tarde, plus il y a de `match` à migrer.
