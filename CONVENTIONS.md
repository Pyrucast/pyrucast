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
  *leur propre* type → elles restent sur le type.

## Règle Rust → Python : miroir 1:1

- fonction libre `ops::<thème>::f` → fonction de module Python
  `pyrucast.<thème>.f(...)` ;
- méthode Rust `Type::m` → méthode Python `obj.m(...)` ;
- surcharge d'opérateur Rust → dunder Python (`__add__`, `__getitem__`, …) ;
- constructeur nommé Rust → `classmethod` Python.

Aucune op n'a le droit d'être une fonction d'un côté et une méthode de
l'autre. Le wrapper `py/` est une projection mécanique, pas un lieu de
redesign de l'API.

Le style visé côté Python est celui de **numpy / scipy** (et l'héritage
**cast3m**) : des opérateurs nommés dans des sous-modules thématiques
(`pyrucast.mesher.to_poi1(mesh)`, `pyrucast.assemble.stiffness(model, mat)`),
et des méthodes réservées aux accesseurs, mutations et vues dérivées.

## Table de projection (état cible)

| Opération | Rust | Python |
|---|---|---|
| accesseur / mutation mono-conteneur | méthode | méthode |
| vue dérivée d'un seul conteneur | méthode | méthode |
| transformation mesh→mesh | `ops::mesher::*` | `pyrucast.mesher.*` |
| construction de conteneur | `ops::build::*` | `pyrucast.build.*` |
| opérateur sur field croisant un mesh/field | `ops::field::*` | `pyrucast.field.*` |
| assemblage `Model` → `Matrix` | `ops::assemble::*` | `pyrucast.assemble.*` |
| résolution `A·x = b` | `ops::solver::*` | `pyrucast.solver.*` |
| arithmétique (`+ - * /`, indexation) | `impl` d'opérateur | dunder |
| constructeur nommé | fn associée | `classmethod` |

## Cas tranchés explicitement

- `restrict(field, mesh)` → **`ops::field`** (field + mesh en pairs).
- `merge(a, b)` → **`ops::field`** (deux fields en pairs ; fusion nommée,
  non arithmétique).
- addition field+field → **opérateur `+`**, pas une `ops::field::add`.
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
- `to_poi1_submesh` / `to_poi1_mesh` sur `NodeField` → **méthodes** (vue du
  support d'un seul field). À renommer `support_submesh` / `support_mesh`
  pour ne pas se confondre avec l'opérateur `ops::mesher::to_poi1(mesh)`.
