# Conventions d'API — méthodes vs fonctions libres

Ce document fige **une seule règle** pour décider si une opération est une
méthode inhérente d'un conteneur ou une fonction libre des modules `ops/`,
et **une seule règle** pour projeter cette décision vers l'API Python.

L'objectif : ne plus jamais trancher « méthode ou fonction ? » au cas par
cas. La réponse doit tomber de l'arbre de décision ci-dessous, et la forme
Python doit se déduire mécaniquement de la forme Rust.

## Vocabulaire

- **Conteneur** : un type de `containers/` (`Mesh`, `SubMesh`, `NodeField`,
  `ElementField`, `Matrix`, `Model`, `FiniteElementSpace`, …). On les
  qualifie de *lourds* par opposition aux scalaires, noms de composantes,
  `NodeId`, slices `&[f64]`, etc.
- **Opérateur** : une fonction qui consomme un ou plusieurs conteneurs et
  produit un nouveau conteneur (ou une donnée dérivée). Les opérateurs
  vivent dans `ops/`, rangés **par thème** (`build`, `geom`, `field`,
  `assemble`, `mesher`, `solver`) et non par conteneur — voir `ops/mod.rs`.

## Règle Rust : méthode vs fonction libre

Une opération est une **méthode inhérente** si, et seulement si, les trois
conditions tiennent :

1. il existe **un** conteneur qui est le `self` évident ;
2. l'opération lit/écrit *essentiellement* ce conteneur — les autres
   arguments sont des scalaires, des noms de composantes, des `NodeId`,
   des slices `&[f64]`, … (jamais un second conteneur lourd traité en
   pair) ;
3. c'est l'un de :
   - un **accesseur** (`node_count`, `components`, `get`, …) ;
   - une **mutation qui préserve l'invariant** du conteneur (`set`,
     `add_cell`, `add_to_component`, …) ;
   - une **vue dérivée bon marché** de ce seul conteneur
     (`to_poi1_submesh` = le support du field vu comme POI1).

Sinon, c'est une **fonction libre dans `ops/<thème>`**.

### Départage des cas limites

> **Deux conteneurs lourds entrent-ils comme pairs ?**
> — Oui → fonction libre (`ops::field::restrict(field, mesh)`).
> — Non, ça ne lit que `self` (+ petits args) → méthode.

Et un repère de cohérence : si une opération mono-conteneur appartient à
une **famille** déjà installée dans `ops/` (p. ex. les transformations
mesh→mesh de `mesher`, ou les assembleurs de `assemble`), elle rejoint sa
famille même si elle pourrait techniquement être une méthode. Une famille
d'opérateurs ne se scinde pas entre `ops/` et les `impl` de conteneur.

### Exceptions assumées

- **Surcharges d'opérateurs** (`Add`, `Sub`, `Mul`, `Index`, …) : toujours
  des `impl` de trait sur le conteneur, jamais des fonctions `ops::`. C'est
  la forme idiomatique dans les deux langages. L'arithmétique field+scalaire
  et field+field passe par là (`a + b`), pas par une `ops::field::add`.
- **Constructeurs nommés** (`from_poi1`, `lagrange1`, `heat_conduction`,
  `dirichlet`) : fonctions associées / `classmethod`. Elles fabriquent
  *leur propre* type → elles restent sur le type. Quand ce type est un
  **agrégat** (`Mesh`, `FiniteElementSpace`, `Model`, `ElementField`), le
  constructeur vit au niveau du **parent** et renvoie un parent — voir
  « Agrégats : un ou plusieurs, de manière transparente » ci-dessous.

## Règle Rust → Python : miroir 1:1

- fonction libre `ops::<thème>::f` → fonction **top-level** Python
  `pyrucast.f(...)` ;
- méthode Rust `Type::m` → méthode Python `obj.m(...)` ;
- surcharge d'opérateur Rust → dunder Python (`__add__`, `__getitem__`, …) ;
- constructeur nommé Rust → `classmethod` Python.

Aucune op n'a le droit d'être une fonction d'un côté et une méthode de
l'autre, ni de changer de sémantique entre les deux langages. Le wrapper
`py/` ne **redessine** pas l'API ; il peut seulement la **restreindre** —
voir l'exception ci-dessous.

### Exception assumée : Rust bas niveau, Python curé

Une seule asymétrie est tolérée entre les deux surfaces : **Python peut
masquer les constructeurs directs de sous-objets `Sub*`** que Rust, lui,
expose en `pub`.

- **Côté Rust (couche bas niveau).** `SubMesh::new`, `SubElementField::new`,
  `SubMatrix::new`, `SubModel::heat_conduction` / `dirichlet`, … restent
  `pub`. La couche qui écrit les mailleurs, les assembleurs et les
  constructeurs parent *doit* pouvoir fabriquer des `Sub*` et les insérer
  dans le store ; le contrôle total est le rôle de l'API Rust.
- **Côté Python (surface curée).** Les `Sub*` sont des **vues** obtenues par
  indexation du parent (`parent[i]`) ; ils ne se **construisent pas**
  directement. On construit au niveau parent et on compose par `|` (union,
  voir « Agrégats : un ou plusieurs » ci-dessous). La coercition parent→sub
  unitaire (`Aggregate::unit`) est l'autre face de cette restriction : là où
  une op a besoin d'un seul sous-objet, on lui passe un parent unitaire.

C'est une **restriction de surface**, pas un redesign : Python n'invente
aucune op, n'en renomme aucune, n'en change pas la sémantique ; il **n'expose
pas** certains constructeurs. Tout ce qui est exposé des deux côtés reste un
miroir 1:1. Si une nouvelle op apparaît, elle suit la règle 1:1 par défaut ;
masquer un constructeur `Sub*` est le **seul** écart permis, et il doit rester
limité à ce cas.

Le style visé côté Python est celui de **numpy / scipy** (et l'héritage
**cast3m**) : des opérateurs **nommés** (`pyrucast.to_poi1(mesh)`,
`pyrucast.stiffness(model, mat)`) plutôt que des chaînes de méthodes, et
des méthodes réservées aux accesseurs, mutations et vues dérivées.

### Le thème vit côté Rust, pas dans la hiérarchie de modules Python

Le rangement par thème (`mesher`, `field`, `assemble`, `build`, `solver`)
est une organisation **du code Rust** (`src/ops/<thème>/`). Côté Python,
toutes les fonctions sont exposées **à plat** au top-level
(`pyrucast.to_poi1`, `pyrucast.coordinates`, `pyrucast.stiffness`,
`pyrucast.solve`, …), **pas** dans des sous-modules `pyrucast.mesher.*`.

Pourquoi : de vrais sous-modules Python typés imposeraient de passer le
projet en *layout mixte* maturin (dossier `python/pyrucast/`, stubs en
package) — `pyo3-stub-gen` en layout « Pure Rust » refuse explicitement
plusieurs modules. On garde donc un seul `pyrucast.pyi` et un namespace
plat. Si le besoin de sous-modules se confirme, c'est une migration
packaging à part entière (voir l'historique de cette décision).

### Les fichiers wrappers reflètent l'arborescence Rust

Le namespace Python reste plat (ci-dessus), mais les **fichiers** de la
couche FFI suivent le même découpage que Rust — `containers/` (data) vs
`ops/` (algos) :

- **wrappers de type** → `src/py/<type>.rs`, en miroir de
  `src/containers/<type>` ; n'y vivent que des `#[pyclass]` + `#[pymethods]`
  (méthodes, vues, dunders, `classmethod` constructeurs du type) ;
- **wrappers d'opération** → `src/py/ops/<famille>.rs`, en miroir de
  `src/ops/<famille>/` ; n'y vivent que des `#[pyfunction]` libres.

C'est un **repère de navigation**, pas un changement de surface : depuis un
wrapper on retrouve l'impl Rust par parité de chemin —
`py/ops/mesher.rs` ↔ `ops/mesher/`, et `line_seg2` ↔ `ops/mesher/line_seg2.rs`.
Corollaire : une fonction libre se range par sa **famille `ops`** (cf. table
de projection et cas tranchés ci-dessous), jamais par son type d'entrée ni
de sortie.

## Agrégats : un ou plusieurs, de manière transparente

Les conteneurs `Mesh`, `FiniteElementSpace`, `Model` et `ElementField` sont
des **agrégats** : chacun est un `Vec<Handle<Sub>>` (voir `aggregate.rs`).
Le but de l'agrégat est de manipuler **1 ou plusieurs** sous-objets d'un
geste, de façon transparente. Pour que ce soit *réellement* transparent à
l'usage, l'utilisateur ne doit jamais avoir à construire un sous-objet puis
à l'attacher à la main, ni à « plonger » dans l'agrégat avec `parent[0]`
pour le cas courant (un seul sous-objet).

D'où **une seule règle** :

> **Les constructeurs nommés vivent au niveau du parent et renvoient un
> parent ; on compose des parents avec `|` (union — Rust : `union`) ; le
> `Sub*` est une vue indexée, jamais un objet qu'on construit-puis-attache.**

Trois conséquences mécaniques :

1. **Construire = un parent prêt à l'emploi.** Un constructeur nommé qui
   produit un agrégat renvoie le parent, pas le sous-objet. Quand il a
   besoin d'un support, il consomme le **parent** correspondant et balaie
   ses sous-objets : un support unitaire → agrégat unitaire, un support à N
   zones → agrégat à N zones. C'est *là* qu'est la transparence « 1 ou
   plusieurs ». Précédents déjà en place : `FiniteElementSpace(mesh)`
   fabrique un sous-espace par sous-maillage ; `Mesh(config, element_type)`
   crée un maillage à un sous-maillage. Cible : `Model::heat_conduction(&fes)`
   crée une zone par sous-espace.

2. **Composer = union (`|` Python, `union` Rust), jamais `add_sub` à la
   main.** Pour assembler des physiques / zones hétérogènes, on unit des
   parents : Python `Model.heat_conduction(fes) | Model.dirichlet(...)`,
   Rust `model.union(&dirichlet)?`. L'union clone les `Handle` (bump de
   refcount, pas de copie profonde) et **déduplique par handle**, donc les
   sous-objets sont **partagés** entre parents. `add_sub` reste un primitif
   bas niveau (et le chemin interne des constructeurs), pas l'API d'usage.

3. **Le `Sub*` est une vue, pas un point de construction (surface Python).**
   On y accède par indexation (`parent[i]`), exactement comme
   `submesh[j] → Cell` et `cell[k] → Node` sont déjà des vues. Le `Sub*`
   garde une identité dans le store (refcount, partage de `Handle`), mais il
   sort du chemin de construction **côté Python** : les constructeurs
   `Sub*` n'y sont pas exposés (c'est l'« Exception assumée : Rust bas
   niveau, Python curé » ci-dessus). **Côté Rust**, `SubMesh::new` & co.
   restent `pub` — la couche bas niveau en a besoin.

   **Cas unitaire : `.unit()`, pas de re-codage au parent.** Pour atteindre
   une méthode du sous-objet quand l'agrégat est unitaire, on **n'expose
   pas** la méthode du `Sub` sur le parent (zéro re-codage) : on expose
   `.unit()` — la **vue de l'unique sous-objet**, erreur claire si l'agrégat
   n'est pas exactement unitaire — et on écrit `parent.unit().méthode(...)`
   (`mesh.unit().add_cell(...)`, `ef.unit().set_uniform(...)`,
   `K.unit().add_entry(...)`). C'est plus honnête que `parent[0]` (qui prend
   silencieusement le premier de plusieurs) et garde l'utilisateur conscient
   qu'il manipule un agrégat unitaire. Attention à ne **pas** confondre avec
   les méthodes parent qui *agrègent réellement* — somme/union/liste/global
   (`Mesh::cell_count`, `Model::dual_vars`, `Matrix::n_rows`, …) : celles-là
   ne sont pas des délégations et restent au parent. Côté Rust, la couche bas
   niveau garde par commodité ses quelques délégations (`Mesh::add_cell`).

**Coercition aux frontières.** Là où une opération exige *vraiment* un seul
sous-objet (p. ex. le support d'un `NodeField`, ou `Matrix.block`), elle
accepte le **parent** et déballe son unique sous-objet via `Aggregate::unit`
(erreur explicite si l'agrégat n'est pas unitaire). On ne demande jamais à
l'utilisateur de fournir le `Sub*` lui-même.

Ce qui est **projeté mécaniquement** vers Python (miroir 1:1) : le
constructeur nommé du parent (Rust `Model::heat_conduction` → `classmethod`
Python), l'union (`union` → `__or__`), l'indexation → `__getitem__`. La seule
asymétrie est la **non-exposition** des constructeurs `Sub*` côté Python (et
la coercition parent→sub unitaire qui l'accompagne) — l'exception décrite
plus haut. La règle s'applique uniformément aux quatre agrégats et a vocation
à être portée par les macros `impl_aggregate!` / `impl_aggregate_pymethods!`.

## Trois niveaux d'affichage

Tout objet expose trois niveaux d'affichage, en couches, tous reliés à Python.
Chaque niveau a un rôle distinct et **ne déborde jamais** sur le suivant :

| Niveau | Rust | Python | Rôle | Borne |
|---|---|---|---|---|
| résumé | `Display` | `__str__` | une ligne : identité + dimensions clés | O(1), jamais de contenu |
| structure | `Debug` | `__repr__` | compteurs, dimensions, noms, handles, métadonnées (`{:#?}` indenté) | borné, jamais de contenu en masse |
| contenu | `dump::Dump` | `dump(precision=3, max_rows=20, max_cols=12)` | contenu complet : grilles de matrices, tables de valeurs, connectivité | borné par `DumpOptions` (élision `… (N de plus)`) |

Règles :

- `Display`/`Debug` ne déversent **jamais** le contenu en masse (valeurs,
  connectivité, grille). Un `repr` reste borné quelle que soit la taille de
  l'objet.
- `dump()` **imprime directement au terminal** et ne renvoie rien (`()` en
  Rust, `None` en Python). Côté Python l'impression passe par le `print` de
  Python (respecte `sys.stdout`, redirections, capture des tests). Le cœur
  `Dump::render(&self, opts) -> String` produit la chaîne (composition des
  agrégats) ; il n'est pas exposé à Python.
- Les matrices se dumpent en **grille dense labellisée** : les DOF `(node, var)`
  étiquettent directement lignes et colonnes.
- Les agrégats génériques (`Mesh`, `FiniteElementSpace`, `Model`,
  `ElementField`) dumpent le résumé puis le `dump` indenté de chaque
  sous-objet ; `Matrix` dumpe une seule grille globale.

Côté implémentation : trait + helpers partagés dans `src/dump.rs` (chaque type
implémente `render`, `dump` est fourni par défaut) ; macro
`impl_aggregate_dump!` pour les agrégats génériques ; `impl_dump_pymethod!`
pour le câblage Python des wrappers non-agrégats.

## Table de projection (état cible)

Côté Python, toutes les fonctions des thèmes ci-dessous sont **à plat**
(`pyrucast.f(...)`) — voir la note sur le namespace plat plus haut.

| Opération | Rust | Python (top-level) |
|---|---|---|
| accesseur / mutation mono-conteneur | méthode | méthode |
| vue dérivée d'un seul conteneur | méthode | méthode |
| transformation mesh→mesh | `ops::mesher::*` | `pyrucast.to_poi1`, `pyrucast.consolidate`, … |
| construction de conteneur | `ops::build::*` | `pyrucast.material_field`, … |
| opérateur sur field croisant un mesh/field | `ops::field::*` | `pyrucast.coordinates`, `pyrucast.restrict`, `pyrucast.merge` |
| assemblage `Model` → `Matrix` | `ops::assemble::*` | `pyrucast.stiffness`, `pyrucast.mass` |
| résolution `A·x = b` | `ops::solver::*` | `pyrucast.solve` |
| arithmétique (`+ - * /`, indexation) | `impl` d'opérateur | dunder |
| constructeur nommé | fn associée | `classmethod` |

## Cas tranchés explicitement

- `restrict(field, mesh)` → **`ops::field`** (field + mesh en pairs).
- `merge(a, b)` → **`ops::field`** (deux fields en pairs) ; alias nommé de
  l'union `a | b` (`Aggregate::union`), fusion non arithmétique.
- addition field+field → **opérateur `+`** (arithmétique de valeurs, pas la
  composition de zones), pas une `ops::field::add`. Les opérateurs `+ - * /`
  (zone à zone **et** agrégat à agrégat) combinent **par `(support, composante)`**
  en union/passthrough (composante ou support d'un seul côté = valeur brute
  inchangée) ; les opérandes n'ont pas besoin du même jeu de composantes ni de
  la même décomposition. Primitives : `SubField::merge_components` (zone) /
  `Field::merge_field` (agrégat), `Field::merge_subfield` (maj ciblée d'une zone).
  Là où un écart de composantes doit être une erreur (interpolation `Evolution`),
  `SubField::check_same_components` garde `merge_components` en amont.
- `stiffness(model, mat)`, `mass(model)` → **`ops::assemble`** (famille
  assembleur ; `mass` suit `stiffness`, elles ne se séparent pas).
- `consolidate(mesh)`, `to_poi1(mesh)` → **`ops::mesher`** (famille des
  transformations mesh→mesh ; mono-conteneur mais rattachées à leur
  famille).
- `material_field(model, …)` → **`ops::build`** (produit un `ElementField`
  dans la chaîne d'assemblage). *C'est le cas le plus limite — il ne lit que
  le `Model` + une spec scalaire ; on le rattache à sa famille `build` pour
  garder la chaîne `build → assemble → solve` uniforme.*
- `mul_dense(self, x: &[f64])` → **méthode** (`x` est un slice, pas un
  conteneur lourd : produit matrice-vecteur mono-conteneur).
- `support_submesh` / `support_mesh` sur `NodeField` → **méthodes** (vue du
  support d'un seul field). Renommées depuis `to_poi1_submesh` /
  `to_poi1_mesh` pour ne pas se confondre avec l'opérateur
  `ops::mesher::to_poi1(mesh)`.
