# Agrégat

La plupart des conteneurs de pyrucast viennent par **paires** : un objet
**zone** (`SubMesh`, `SubFiniteElementSpace`, `SubNodeField`,
`SubElementField`, `SubModel`, `SubMatrix`) et son **agrégat** (`Mesh`,
`FiniteElementSpace`, `NodeField`, `ElementField`, `Model`, `Matrix`). Tous les
agrégats partagent **exactement la même grammaire d'accès** et la même
**composition par union** — factorisées dans le trait Rust `Aggregate`
(`src/aggregate.rs`). Ce chapitre décrit ce contrat commun une fois pour
toutes ; chaque chapitre d'objet n'en redonne que les spécificités.

## Principe : une liste de handles

Un agrégat est, au fond, **un `Vec<Handle<Sub>>`** : une liste de handles vers
des sous-objets vivant dans le store. Il ne copie jamais ses zones — il en
**partage** les handles (refcount). Le trait `Aggregate` dérive toute la
mécanique d'accès de deux accesseurs seulement (`items()` / `items_mut()`), si
bien qu'ajouter un nouvel agrégat ne duplique aucun code d'indexation.

```text
   Mesh                FiniteElementSpace        Model
   ├── Handle<SubMesh> ├── Handle<SubFES>        ├── Handle<SubModel>
   ├── Handle<SubMesh> ├── Handle<SubFES>        └── Handle<SubModel>
   └── …               └── …
```

## Interface uniforme

Tout agrégat expose, côté **Python** :

| Opération | Sens |
|---|---|
| `len(agg)` | nombre de zones |
| `agg[i]` | **vue** typée sur la zone `i` (un `Sub…`) — jamais une copie |
| `for sub in agg:` | itère les zones (via le protocole séquence) |
| `agg.unit()` | la **seule** zone d'un agrégat unitaire, sinon une erreur claire |
| `agg.add_sub(sub)` | ajoute une zone en place |
| `agg \| other` | **union** (voir plus bas) |
| `repr` / `str` / `dump()` | les trois niveaux d'affichage |

Côté **Rust**, le trait `Aggregate` fournit les mêmes : `len`, `is_empty`,
`get(i)`, `iter`, `unit`, `push`/`add_sub`, plus `Index<usize>` et
`IntoIterator` via la macro `impl_aggregate_std_traits!`.

> **`unit()` vs `[0]`.** `agg[0]` prend silencieusement la première de
> plusieurs zones ; `agg.unit()` **exige** qu'il n'y en ait qu'une et lève
> sinon. Aux frontières qui n'ont de sens que mono-zone (par exemple ajouter
> une cellule à un maillage qu'on vient de créer avec un seul sous-maillage),
> préférer `unit()` — plus honnête (cf. `CONVENTIONS.md`).

### Les sous-objets ne se construisent pas seuls

Un `Sub…` obtenu par `agg[i]` est une **vue** : on ne le construit pas
directement côté Python. On construit toujours au **niveau parent**
(`Mesh(coords, type)`, `FiniteElementSpace(mesh)`, `ElementField(fes, comps)`,
`Model.heat_conduction(fes)`…) puis on indexe pour atteindre une zone.

## Composition : l'union `|`

Composer deux agrégats, c'est l'**union** : `a | b` côté Python, `a.union(&b)`
côté Rust. La sémantique est **uniforme pour les six agrégats** :

1. **Déduplication par handle.** Une zone dont le `Handle` est déjà présent
   (même slot du store, `Handle::same_slot`) n'est pas ajoutée deux fois.
   L'ordre est celui de première apparition.
2. **Partage, pas copie.** Les zones retenues sont partagées (refcount), jamais
   dupliquées en mémoire.
3. **Contraintes de domaine.** Les invariants (même `Coords` pour un `Mesh`,
   etc.) restent vérifiés au moment de l'ajout (`check_push`).
4. **Finalisation.** Un crochet `finalize()` tourne en fin d'union : **no-op**
   pour la plupart des agrégats, mais les **champs** le surchargent pour fusionner
   les zones partageant un même support (voir [Champ](field.md),
   [Champ aux nœuds](node-field.md)).

Les trois formes de l'union :

| Python | Rust | Résultat |
|---|---|---|
| `agg \| agg` | `a.union(&b)` | union dédupliquée des deux listes |
| `agg \| sub` | `a.union_sub(&h)` | ajoute une zone (ignorée si déjà présente) |
| `sub \| sub` | `T::union_subs(&a, &b)` | un agrégat neuf portant les deux zones |

Plus, pour les nœuds (cf. [Nœud](node.md)) :

| Python | Résultat |
|---|---|
| `node \| node` | `Mesh` POI1 unitaire sur les deux nœuds |
| `mesh \| node` | ajoute un point (erreur si `Mesh` non unitaire POI1) |

> **`|` compose, `+` calcule.** L'union est **toujours** `|`. L'opérateur `+`
> (et `-`, `*`, `/`) est réservé à l'**arithmétique des champs**
> (`field + 2.0`, addition de champs…) — il n'est **jamais** utilisé pour
> composer des agrégats. Les deux ne se télescopent donc jamais. C'est un
> changement par rapport aux toutes premières versions, où la composition
> passait par `+` ; aujourd'hui `+` est entièrement libéré pour le calcul.

### Exemple

```python
import pyrucast

c = pyrucast.Coords(dim=2)
ns = [c.add_node(p) for p in [(0,0), (1,0), (1,1), (0,1)]]

tri = pyrucast.Mesh(c, "TRI3")
tri.unit().add_cell([ns[0], ns[1], ns[2]])

qua = pyrucast.Mesh(c, "QUA4")
qua.unit().add_cell([ns[0], ns[1], ns[2], ns[3]])

# Union de deux maillages (zones partagées par handle).
mesh = tri | qua
print(len(mesh))          # 2 sous-maillages
print(mesh)               # Mesh: 2 submesh(es), 2 cell(s) total
```

## Pourquoi un trait commun ?

Factoriser l'accès et l'union dans `Aggregate` garantit qu'un sous-maillage,
un sous-espace EF ou un sous-modèle s'indexent, s'itèrent et se composent
**strictement de la même façon**. Aucun `__getitem__` n'est réécrit par type,
aucun comportement d'union ne diverge d'un agrégat à l'autre, et un nouvel
agrégat se câble en deux macros (`impl_aggregate!` + `impl_aggregate_pymethods!`).
Le seul point de personnalisation est `finalize()` (et `check_push()`), que les
champs exploitent pour leur fusion par support.
