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
> — Oui → fonction libre (`ops::node_field::restrict(field, mesh)`).
> — Non, ça ne lit que `self` (+ petits args) → méthode.

Et un repère de cohérence : si une opération mono-conteneur appartient à
une **famille** déjà installée dans `ops/` (les mailleurs, les assembleurs),
elle rejoint sa famille même si elle pourrait techniquement être une méthode.
Une famille d'opérateurs ne se scinde pas entre `ops/` et les `impl` de
conteneur.

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

## Où vit une fonction libre : le conteneur produit

**Un module d'`ops/` rassemble les opérateurs qui produisent le même
conteneur, et porte son nom.** Une opération se range par sa **sortie**,
jamais par son entrée : `gradient(field, fespace)` produit un
`ElementField`, donc il vit dans `ops::element_field`, à côté de
`deformation` et `interp_to_gauss` — pas dans `mesh` ni dans `node_field`.

| module | produit |
|---|---|
| `ops::mesh` | un `Mesh` |
| `ops::node_field` | un `NodeField` |
| `ops::element_field` | un `ElementField` |
| `ops::matrix` | une `Matrix` |
| `ops::coords` | écrit dans le magasin |

Les opérateurs qui ne produisent **aucun** conteneur échappent à la règle par
construction ; on les range par activité : `ops::measure` (réductions à un
nombre), `ops::geom` (requêtes géométriques), `ops::export` (effets de bord).

Un troisième cas existe : l'opérateur **générique**, dont le produit est un
conteneur — toujours — mais **pas un conteneur déterminé**. `abs` rend un
`NodeField` ou un `ElementField` selon ce qu'on lui donne : la règle ne
désigne donc pas *un* module. Attention à ne pas confondre avec deux fonctions
monomorphes de même famille (`mask_nodes` / `mask_cells`), qui ont chacune un
produit déterminé et se rangent normalement. Ces opérateurs polymorphes se rangent par **domaine** :
`ops::field` (masque de bande, filtrage et renommage de composantes, maths
élément par élément). Ils restent des fonctions libres à part entière, avec
leur méthode sur chacune des quatre saveurs de champ.

### L'exception unique et nommée : `solver`

`ops::solver` produit un `NodeField` et devrait rejoindre `ops::node_field`.
Il garde son nom parce que **plusieurs familles distinctes produisent un
champ nodal** — dérivation, assemblage, résolution — et que seule la
résolution se cherche par son propre nom. C'est le seul module nommé d'après
une activité tout en produisant un conteneur, et il doit le rester.

### Corollaire : pas de qualificatif dans le nom d'une fonction

Un module ne contient **jamais** deux opérateurs qui ne diffèrent que par le
conteneur sur lequel ils portent. Si le cas se présente, le qualificatif
appartient au **nom du module**, pas au nom de la fonction, et le module doit
être scindé. C'est ce qui donne trois fusions homonymes et sans ambiguïté —
`mesh::consolidate`, `node_field::consolidate`, `element_field::consolidate` —
au lieu de trois fonctions suffixées au même endroit. De même `coords::set`
face à `node_field::positions` qui lit.

### Une limite de R3, à connaître

R3 dit que le qualificatif d'une fonction doit passer dans le nom du module.
Le remède ne s'applique que si le qualificatif distingue la **sortie** : c'est
le cas des trois `consolidate`, qui produisent trois conteneurs différents.

`select_nodes` et `select_cells` sont hors de portée : elles produisent toutes
deux un `Mesh` et vivent donc toutes deux dans `ops::mesh`, leur qualificatif
distinguant l'**entrée**. On ne peut pas le déplacer dans le nom du module,
puisque le module est fixé par la sortie. Le suffixe reste donc légitime, et
c'est côté Python que l'ambiguïté disparaît — une seule fonction `select` qui
répartit selon le type reçu. Même situation pour `integral` /
`integral_element`.

### Ce qui ne se construit pas

Rien dans `ops/` ne produit un `Model`, un `FiniteElementSpace` ou une
`Evolution`, et ce n'est pas un oubli : ces conteneurs se **déclarent** par
constructeur nommé sur le type lui-même, ils ne se fabriquent pas par
transformation.

## Le verbe exposé aussi en méthode

Une fonction libre garde sa **forme canonique** — c'est elle qui est
documentée et qui définit l'opération. Elle est **en plus** exposée comme
méthode de son premier argument si, et seulement si, les **trois** conditions
tiennent :

1. **le premier argument est le sujet** — l'objet qu'on transforme, pas un
   paramètre ni un support ;
2. **le retour est un conteneur** — sinon il n'y a rien à composer, et
   `f.integral(comp, fes) -> float` n'apporte rien sur `integral(f, comp, fes)` ;
3. **l'opération a un sens pour toute instance du type.** C'est la condition
   qui coûte le plus et qu'on oublie le plus vite : une méthode *promet*, elle
   apparaît dans l'auto-complétion de chaque objet du type. `u.deformation(fes)`
   apparaîtrait sur tous les champs nodaux alors qu'elle exige des composantes
   `u_x`/`u_y`/`u_z` ; `t.thermal_strain(...)` sur tous les champs par éléments
   alors qu'elle exige une température. Ces opérations restent fonctions libres
   seules.

La ligne de partage de la condition 3 : une **précondition structurelle** est
admise (`triangulate_surface` veut un contour fermé, `divergence` veut autant de
composantes que d'axes — vérifié par comptage, jamais par nom), une exigence de
**sens porté par les noms de composantes** ne l'est pas (`sigma_xx`, `u_x`,
« une température »).

**Comment tester la condition 3 sans se tromper** : lire la méthode avec un
receveur *quelconque*, pas avec l'exemple bien nommé.
`stresses.internal_forces(model)` sonne juste — mais c'est le nom de la
variable qui fait le travail. `field.internal_forces()` révèle que le *type*
ne promet rien : n'importe quel champ par éléments porterait la méthode, alors
qu'elle exige la contrainte de Voigt. Comparer avec `field.sqrt()` ou
`field.mask(ge=0.0)`, qui gardent leur sens sur n'importe quel champ.

Deux conséquences pratiques :

- **Un ordre d'arguments qui ne met pas le sujet en tête est un défaut à
  corriger, pas une raison de renoncer à la méthode.** C'est ce qui a fait
  passer `internal_forces(model, stresses)` à `(stresses, model)` et
  `solve_eliminate(model, matrix, rhs)` à `(matrix, model, rhs)`.
- **Une opération symétrique n'a pas de méthode** : `a.merge(b)` suggérerait
  que l'ordre compte. `merge` est l'alias nommé de `a | b` — l'opérateur donne
  déjà la forme symétrique, il suffit.

### Le nom peut changer entre les deux formes

Le nom complet est toujours « qualificatif + verbe » ; ce qui change, c'est où
se loge le qualificatif. La fonction libre le reçoit de son **module**, la
méthode n'en a pas et doit donc le **porter** :

| fonction libre | méthode |
|---|---|
| `matrix::stiffness(model, materials)` | `model.stiffness_matrix(materials)` |
| `matrix::mass(model, materials)` | `model.mass_matrix(materials)` |
| `matrix::tangent(...)` | `model.tangent_matrix(...)` |
| `matrix::geometric(...)` | `model.geometric_matrix(...)` |

Quand la sortie est du type du sujet, il n'y a rien à qualifier et le nom ne
bouge pas : `mesh::consolidate(m)` et `m.consolidate()`.

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
  constructeurs parent *doit* pouvoir fabriquer des `Sub*` et les placer
  derrière un `Handle` ; le contrôle total est le rôle de l'API Rust.
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
**cast3m**) : des opérateurs **nommés** (`pyrucast.mesh.to_poi1(mesh)`,
`pyrucast.matrix.stiffness(model, mat)`) plutôt que des chaînes de
méthodes, et des méthodes réservées aux accesseurs, mutations et vues
dérivées.

### Le module de production est reflété par un sous-module Python

Le rangement par conteneur produit organise le code Rust
(`src/ops/<module>/`) **et** l'API Python : une fonction libre
`ops::<module>::f` est exposée comme `pyrucast.<module>.f`
(`pyrucast.mesh.to_poi1`, `pyrucast.node_field.positions`,
`pyrucast.matrix.stiffness`, `pyrucast.solver.solve`, …). Les conteneurs
(`containers::…`) et les atomes (`atoms::…`) restent des classes au top-level
(`pyrucast.Coords`, `pyrucast.Mesh`, `pyrucast.Node`, …). Le miroir est
**sans exception** : aucune fonction libre ne vit au top-level Python.

L'extension compilée `_pyrucast` est en revanche **plate** : deux opérateurs
homonymes dans deux modules (les trois `consolidate`, `coords::set`) y portent
un `#[pyo3(name = "…")]` distinct, et la couche Python pure les ré-exporte
sous leur vrai nom dans le bon sous-module. C'est un détail d'implémentation
du namespace privé, pas une entorse au miroir.

C'est le passage au *layout mixte* maturin (dossier `python/pyrucast/`,
extension privée `_pyrucast` + couche Python pure) qui débloque ce rangement :
chaque module est un vrai fichier `.py` ré-exportant, de façon typée, les
symboles de l'extension plate. Un seul stub `_pyrucast/__init__.pyi` reste
généré pour l'extension ; les sous-modules n'étant que de la ré-exportation,
les types les suivent sans stub dédié.

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
`py/ops/mesh.rs` ↔ `ops/mesh/`, et `line` ↔ `ops/mesh/line.rs`.
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
   garde son identité propre (partage de `Handle`, comptage de références),
   mais il sort du chemin de construction **côté Python** : les constructeurs
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

Côté Python, les fonctions vivent dans le **sous-module du conteneur
produit** (`pyrucast.<module>.f(...)`) — voir la note sur les sous-modules
plus haut, sans exception.

| Opération | Rust | Python |
|---|---|---|
| accesseur / mutation mono-conteneur | méthode | méthode |
| vue dérivée d'un seul conteneur | méthode | méthode |
| transformation mesh→mesh | `ops::mesh::*` | `pyrucast.mesh.to_poi1`, `pyrucast.mesh.consolidate`, … |
| production d'un champ nodal | `ops::node_field::*` | `pyrucast.node_field.positions`, `pyrucast.node_field.restrict`, `pyrucast.node_field.merge` |
| production d'un champ par éléments | `ops::element_field::*` | `pyrucast.element_field.gradient`, `pyrucast.element_field.material_field` |
| réduction à un nombre | `ops::measure::*` | `pyrucast.measure.integral`, `pyrucast.measure.xty` |
| écriture dans le magasin | `ops::coords::*` | `pyrucast.coords.set`, `pyrucast.coords.displace` |
| champ → **même** champ (polymorphe) | `ops::field::*` | `pyrucast.field.mask`, `pyrucast.field.sqrt`, `pyrucast.field.filter_components` |
| écriture d'un format externe | `ops::export::*` | `pyrucast.export.export_vtk` |
| assemblage `Model` → `Matrix` | `ops::matrix::*` | `pyrucast.matrix.stiffness`, `pyrucast.matrix.mass` |
| résolution `A·x = b` | `ops::solver::*` | `pyrucast.solver.solve` |
| arithmétique (`+ - * /`, indexation) | `impl` d'opérateur | dunder |
| constructeur nommé | fn associée | `classmethod` |
| verbe éligible aux trois conditions | fonction libre **et** méthode | idem — voir « Le verbe exposé aussi en méthode » |

`ops::geom` n'apparaît pas : ses deux fonctions (`locate_points`,
`project_points`) sont les primitives internes de `Model.embedded` et
`Model.contact`, et ne sont pas exposées à Python. C'est la seule dérogation
de module entier, enregistrée dans `tests/python/test_mirror_completeness.py`.

## Cas tranchés explicitement

- `restrict(field, mesh)` → **`ops::node_field`** (field + mesh en pairs ;
  produit un champ nodal).
- `merge(a, b)` → **`ops::node_field`** (deux fields en pairs) ; alias nommé de
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
- `stiffness(model, mat)`, `mass(model)` → **`ops::matrix`** (famille
  assembleur ; `mass` suit `stiffness`, elles ne se séparent pas).
- `consolidate(mesh)`, `to_poi1(mesh)` → **`ops::mesh`** (mono-conteneur,
  mais elles produisent un `Mesh` et appartiennent à la famille des
  mailleurs).
- `select(field, ge=…)` → **`ops::mesh`** : elle part d'un champ mais rend un
  `Mesh`, et on se range par la sortie. Sa jumelle `mask`, qui réécrit les
  valeurs sans changer la structure, rend un champ de la sorte reçue et
  reste donc dans `ops::field`.
- `material_field(model, …)` → **`ops::element_field`** (produit un
  `ElementField`). L'ancien module `build`, qui ne désignait aucune famille,
  disparaît.
- `flux`, `internal_forces` → **`ops::node_field`** : ce sont des
  assemblages, mais leur résultat est un vecteur, pas un opérateur. La
  machinerie qu'ils partagent avec `ops::matrix` (`ops::coloring`,
  `ops::scatter`) vit à la racine d'`ops` — ce ne sont pas des opérateurs.
- `mul_dense(self, x: &[f64])` → **méthode** (`x` est un slice, pas un
  conteneur lourd : produit matrice-vecteur mono-conteneur).
- `support_submesh` / `support_mesh` sur `NodeField` → **méthodes** (vue du
  support d'un seul field). Renommées depuis `to_poi1_submesh` /
  `to_poi1_mesh` pour ne pas se confondre avec l'opérateur
  `ops::mesh::to_poi1(mesh)`.
- `coords()` est réservé au **retour au conteneur** : sur `Mesh`, `SubMesh`,
  `NodeField`, `Matrix` et `Node`, il rend le `Coords` porté. Les *valeurs*
  s'appellent donc **position** partout : `position()` / `set_position(…)`
  sur `Node` comme sur `Coords`, et `node_field::positions(mesh)` pour le
  champ qui les lit toutes. Les anciens noms — `coord()` sur un nœud,
  `coordinates` pour l'opérateur — plaçaient sur un même objet deux méthodes
  sans argument que rien ne départageait (`mesh.coords()` face à
  `mesh.coordinates()`), l'une rendant le conteneur, l'autre les valeurs.
