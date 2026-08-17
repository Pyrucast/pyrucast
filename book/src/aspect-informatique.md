# Aspect informatique

Avant de détailler les objets un par un, ce chapitre donne la **vue
informatique** de pyrucast : comment le code est organisé, comment les objets
vivent en mémoire, et les quelques motifs transverses qu'on retrouve partout.
Chaque section renvoie au chapitre qui en donne le détail.

## Deux langages, un seul cœur

pyrucast est une librairie **Rust** exposée à **Python**.

- Le **cœur de calcul** (structures de données, maillage, champs, assemblage,
  solveur) est écrit en Rust : typage fort, pas de ramasse-miettes runtime,
  performances natives.
- Le **binding Python** est une fine couche [`pyo3`](https://pyo3.rs),
  compilée par [`maturin`](https://www.maturin.rs) en un module
  d'extension `.so` / `.pyd`. Chaque classe Python (`pyrucast.Coords`,
  `pyrucast.Mesh`, …) enveloppe un type Rust ; chaque fonction de module
  (`pyrucast.matrix.stiffness`, `pyrucast.solver.solve`, …) appelle une fonction Rust.

Le binding est un **miroir 1:1** du Rust, sans logique propre : une méthode
Rust reste une méthode, une fonction libre reste une fonction. La table de
correspondance exhaustive est dans [Correspondance Rust ↔
Python](correspondance-rust-python.md) ; la règle qui décide *méthode vs
fonction libre* est dans [Conventions](conventions.md).

```text
   Script Python  ──►  module pyrucast (.so, pyo3)  ──►  crate Rust pyrucast
        |                      |                                |
   pyrucast.Coords        wrapper PyCoords                 struct Coords
   pyrucast.matrix.stiffness     fonction de module              ops::matrix::stiffness
```

## Les objets se désignent par des handles

Les objets ne se relient pas entre eux par des références Rust directes, mais
par un **`Handle<T>`** : une référence comptée munie de son propre verrou
(`Arc<RwLock<T>>`). Pas de store à interroger, pas de session à passer — le
handle *est* l'adresse de l'objet.

Trois propriétés en découlent, et elles structurent tout le reste de la
librairie :

1. **Libération automatique.** `Clone` partage, `Drop` relâche. Quand le
   dernier handle d'un objet disparaît, l'objet est détruit — aucune fonction
   `remove()` à appeler.
2. **Toujours valide.** Détenir un handle maintient l'objet en vie : il n'y a
   pas de référence périmée, et `read` / `write` ne peuvent pas échouer.
3. **L'identité, c'est le pointeur.** `same_object` dit si deux références
   désignent le même objet — la base de l'union des agrégats.

L'accès passe par un **guard** (`read` / `write`) qui verrouille ce seul objet
le temps de l'opération (RAII). Le détail — guards possédés, granularité du
verrouillage, compteur par nœud — est dans
[Modèle mémoire](memory-model.md).

## Refcount à deux niveaux

La gestion de durée de vie opère à **deux échelles** indépendantes :

- au niveau **objet** : le `Handle<T>` décide si un objet entier (un `Coords`,
  un `SubMesh`…) est vivant ;
- au niveau **interne** : à l'intérieur d'un `Coords`, un second compteur par
  **nœud** décide si tel nœud est vivant. Le ramasse-miettes manuel
  `Coords::gc()` opère sur ce niveau-là.

C'est pourquoi un nœud reste protégé tant qu'un maillage ou un champ le
référence, même si tous les `Node` utilisateurs ont disparu. Voir
[Coordonnées](coords.md) et [Nœud](node.md).

## Le motif agrégat / sous-objet

La plupart des conteneurs viennent par **paires** : un objet **zone**
(`Sub…`) et son **agrégat** (une liste de zones partageant la même grammaire
d'accès) :

| Zone | Agrégat |
|---|---|
| `SubMesh` | `Mesh` |
| `SubFiniteElementSpace` | `FiniteElementSpace` |
| `SubNodeField` | `NodeField` |
| `SubElementField` | `ElementField` |
| `SubModel` | `Model` |
| `SubMatrix` | `Matrix` |
| `SubEvolution` | `Evolution` |

Tous les agrégats exposent la même interface (`len`, `[i]`, itération,
`unit()`) et la même **composition par union `|`** (côté Rust : `union`). Les
sous-objets ne se construisent pas directement : on construit au niveau
parent, et on indexe (`parent[i]`) pour obtenir une **vue** sur une zone. Ce
motif est factorisé dans le trait `Aggregate` — voir [Agrégat](aggregate.md).

> **Union `|`, pas `+`.** Composer deux zones, c'est l'**union** (`mesh_a |
> mesh_b`), avec partage des sous-objets (refcount) et déduplication par
> handle. L'opérateur `+` est réservé à l'**arithmétique des champs** (cf.
> [Champ](field.md)).

## Trois niveaux d'affichage

Chaque objet implémente trois vues, du plus court au plus complet :

- `__str__` (Rust `Display`) — **résumé** une ligne, façon listing cast3m ;
- `__repr__` (Rust `Debug`) — vue **structurelle** bornée, pour le
  développement ;
- `dump()` — **contenu intégral** (valeurs, topologie) imprimé sur la sortie
  standard, au-delà de ce que `repr` montre.

Détail dans [Conventions](conventions.md).

## Erreurs

Toute l'API publique renvoie `Result<T, PyrucastError>`. Côté Python,
`PyrucastError` est converti automatiquement en `RuntimeError`. Il n'y a qu'un
seul type d'erreur dans la librairie — voir [Conventions](conventions.md).

## Persistance portable

Un trait unique, `Persist` (`serde` + `bincode`), produit un format binaire
**portable Linux ↔ Windows**. C'est le socle de la future
**sauvegarde / reprise** d'une session (graphe complet d'objets). Voir
[Conventions](conventions.md) et [Modèle mémoire](memory-model.md).

## Pour le développeur

L'organisation des fichiers Rust (où vit chaque morceau) est décrite dans
[Arborescence](developper/arborescence.md).
