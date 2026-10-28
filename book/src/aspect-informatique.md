# Aspect informatique

Avant de détailler les objets un par un, ce chapitre donne la **vue
informatique** de pyrucast, vue de Python : comment les objets vivent, et les
quelques motifs qu'on retrouve partout. Quand cela éclaire l'usage, un encart
dit ce qu'il y a dessous, côté Rust. Chaque section renvoie au chapitre qui en
donne le détail.

## Deux langages, un seul cœur

pyrucast est une librairie **Rust** exposée à **Python**.

- Le **cœur de calcul** (structures de données, maillage, champs, assemblage,
  solveur) est écrit en Rust : typage fort, pas de ramasse-miettes runtime,
  performances natives.
- Le **binding Python** est une fine couche [`pyo3`](https://pyo3.rs),
  compilée par [`maturin`](https://www.maturin.rs) en un module
  d'extension `.so` / `.pyd`. Les classes Python (`pyrucast.Coords`,
  `pyrucast.Mesh`, …) enveloppent des types Rust, et les fonctions de module
  (`pyrucast.matrix.stiffness`, `pyrucast.solver.solve`, …) appellent des
  fonctions Rust.

Le binding est un **miroir 1:1** du Rust, sans logique propre : une méthode
Rust reste une méthode, une fonction libre reste une fonction. Seules quelques
parties sont écrites en Python, parce qu'elles n'ont rien à faire en Rust : les
échanges avec gmsh et MED (`mesh.from_gmsh`, `export.to_medcoupling`…), qui
appellent ces modules Python, et la [couche pas à pas](pas-a-pas.md)
(`pyrucast.stepping`), qui orchestre les opérateurs. La table de
correspondance exhaustive est dans [Correspondance Rust ↔
Python](developper/correspondance-rust-python.md) ; la règle qui décide *méthode vs
fonction libre* est dans [Conventions](conventions.md).

```text
   Script Python  ──►  module pyrucast (.so, pyo3)  ──►  crate Rust pyrucast
        |                      |                                |
   pyrucast.Coords        wrapper PyCoords                 struct Coords
   pyrucast.matrix.stiffness     fonction de module              ops::matrix::stiffness
```

## Durée de vie des objets

En Python, rien à libérer à la main : un objet pyrucast vit tant qu'une
variable ou un autre objet le référence — un maillage garde ses nœuds, un champ
garde son support.

Les nœuds ont en plus leur propre compteur, dans les `Coords` : un nœud reste
vivant tant qu'un maillage ou un champ l'utilise, même quand plus aucune
variable ne le désigne. `coords.gc()` collecte ceux qui ne servent plus. Voir
[Coordonnées](coords.md) et [Nœud](node.md).

> **Côté Rust**, chaque objet vit derrière un `Handle<T>` — une référence
> comptée munie de son propre verrou — et la `Coords` porte un compteur par
> nœud. Le détail est dans [Modèle mémoire](memory-model.md).

## Agrégats et zones

La plupart des conteneurs viennent par **paires** : une **zone** (`Sub…`) et
son **agrégat**, une liste de zones qui s'utilise toujours de la même façon :

| Zone | Agrégat |
|---|---|
| `SubMesh` | `Mesh` |
| `SubFiniteElementSpace` | `FiniteElementSpace` |
| `SubNodeField` | `NodeField` |
| `SubElementField` | `ElementField` |
| `SubModel` | `Model` |
| `SubMatrix` | `Matrix` |
| `SubEvolution` | `Evolution` |

Un agrégat s'utilise comme une liste de zones : `len(mesh)`, `mesh[i]`,
`for zone in mesh`, et `mesh.unit()` quand il n'en a qu'une. Les zones ne se
construisent pas directement (sauf `SubEvolution`, une courbe tabulée) : on
construit l'agrégat, et `mesh[i]` donne une **vue** sur une zone.

Deux agrégats se composent par **union** : `mesh_a | mesh_b`. Les zones y sont
partagées, pas copiées, et une zone déjà présente n'est pas ajoutée deux fois.
L'opérateur `+`, lui, **additionne** des valeurs — champs ou matrices — et ne
compose jamais (cf. [Champ](field.md)).

> **Côté Rust**, ce motif est factorisé dans le trait `Aggregate`, et l'union
> s'y écrit `a.union(&b)` — voir [Agrégat](aggregate.md).

## Trois niveaux de listing

Tout objet pyrucast se liste à trois niveaux, du plus court au plus complet :

- `print(x)` ou `str(x)` — le **résumé**, une ligne : le listing résumé de
  Cast3M ;
- `repr(x)` — la **structure** : compteurs, dimensions, noms, identifiants,
  sans jamais le contenu ;
- `x.dump()` — le **contenu** : valeurs, coordonnées, connectivité, comme le
  listing complet de Cast3M. Il reste borné : au-delà de `max_rows` lignes
  (20 par défaut), la table est coupée et la coupure annoncée.

Sur des coordonnées qui portent les 11 nœuds d'une ligne de 10 éléments :

```python
{{#include ../../tests/python/test_doc_aspect_informatique.py:listing}}
```

Seule exception, `Array` — le tableau que rendent les fonctions d'échange — n'a
pas de `dump()` : simple tampon sans homologue Rust, ce n'est pas un objet
pyrucast à proprement parler.

> **Côté Rust**, les trois niveaux sont les traits `Display`, `Debug` et
> `Dump`.

## Erreurs

En Python, une erreur signalée par pyrucast lève une `RuntimeError`, dont le
message dit ce qui ne va pas. Les cas que Python nomme déjà gardent leur
exception : `IndexError` pour un indice hors bornes, `ZeroDivisionError` pour
une matrice divisée par zéro, `KeyboardInterrupt` sur un Ctrl+C.

> **Côté Rust**, toute l'API publique renvoie `Result<T, PyrucastError>` : un
> seul type d'erreur pour toute la librairie — voir [Conventions](conventions.md).

## Persistance portable

`pyrucast.save` / `pyrucast.load` écrivent un **graphe** d'objets et le
relisent en préservant le partage — deux champs sur un support restent deux
champs sur un support. Le fichier est **identique Linux ↔ Windows**. Voir
[Sauvegarde et relecture](sauvegarde.md).

> **Côté Rust**, ce format repose sur un trait unique, `Portable` (`serde` +
> `bincode`) — voir [Conventions](conventions.md) et
> [Modèle mémoire](memory-model.md).

## Pour le développeur Rust

L'organisation des fichiers Rust (où vit chaque morceau) est décrite dans
[Arborescence](developper/arborescence.md).
