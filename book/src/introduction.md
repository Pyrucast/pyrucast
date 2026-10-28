# Introduction

**pyrucast** est une librairie d'éléments finis dont le cœur est écrit en Rust et qui expose une API Python. Elle s'inspire des principes de Cast3M : un noyau d'objets typés, accompagné de fonctions opérant sur ces objets.

> 📖 **Référence API Rust (rustdoc).** Cette documentation couvre les principes,
> l'architecture et les exemples ; la référence par item (signatures, types,
> modules) est générée par rustdoc et publiée à côté de ce book :
> <https://pyrucast.github.io/pyrucast/rust/>.

## Philosophie

- **Simplicité avant tout.** Le code doit rester maintenable et éditable par un humain non expert ; on évite la sophistication gratuite.
- **Dépendances minimales.** Tout ajout de dépendance externe (Rust ou Python) requiert un accord explicite.
- **Vérification continue.** Chaque objet est livré avec des tests unitaires Rust, des doctests, des tests Python et un chapitre de cette documentation.

## Modèle d'objets (arbre de dépendances)

Chaque structure ne dépend que des structures qui la précèdent. La plupart
viennent par **paire** zone / agrégat (`SubMesh`/`Mesh`…) — c'est le motif
[Agrégat](aggregate.md), avec sa composition par union `|`.

```text
Coords ── Node                          (NodeId u32 stable ; Node = accesseur RAII)
   │
   └── Mesh  (agrège des SubMesh, + ElementType)   géométrie
          ├── NodeField  (agrège des SubNodeField, support POI1)   valeurs par nœud × composante
          └── FiniteElementSpace  (agrège des SubFiniteElementSpace)
                 │                  (+ Interpolation, QuadratureRule)
                 ├── ElementField  (agrège des SubElementField)   valeurs aux Gauss
                 └── Model  (agrège des SubModel : physiques + contraintes)
                        └── ops::matrix ──► Matrix   (matrice creuse, DOFs nommés)

Evolution  (agrège des SubEvolution)    scalaire, NodeField ou ElementField
                                        tabulé vs une variable

Model + ElementField (matériau) ──► stiffness ──► Matrix
Matrix + NodeField (second membre) ──► solve ──► NodeField (solution)
```

Deux **traits transverses** factorisent le comportement commun :
[`Aggregate`](aggregate.md) (accès `len`/`[i]`, union — `|` en Python, `union`
en Rust) et
[`Field`/`SubField`](field.md) (composantes nommées, `min`/`max`, arithmétique),
partagés entre `NodeField` et `ElementField`.

Résumé des rôles :

| Structure | Rôle |
|---|---|
| [`Coords`](coords.md) | Référentiel de coordonnées de nœuds (plusieurs configurations) |
| [`Node`](node.md) | Accesseur utilisateur d'un nœud, avec protection GC automatique |
| [`SubMesh`](mesh.md) / `Mesh` | Cellules d'un même `ElementType` / union de sous-maillages |
| [`SubNodeField`](node-field.md) / `NodeField` | Valeurs par nœud × composante (zone / agrégat) |
| [`SubFiniteElementSpace`](fe-space.md) / `FiniteElementSpace` | Formulation EF (interpolation + quadrature) / union |
| [`SubElementField`](element-field.md) / `ElementField` | Valeurs par cellule × point de Gauss × composante |
| [`SubModel`](model.md) / `Model` | Physique ou contrainte locale / problème complet |
| [`SubMatrix`](matrix.md) / `Matrix` | Matrice creuse dont les lignes/colonnes sont des DOFs nommés `(NodeId, inconnue)`, par ex. l'inconnue `"UX"` d'un nœud |
| [`SubEvolution`](evolution.md) / `Evolution` | Valeur (scalaire ou champ) tabulée vs une variable, interpolée linéairement (zone / agrégat) |

## Premiers pas

Exemple minimal en Rust :

```rust,ignore
{{#include ../../tests/doc_conteneurs.rs:premiers_pas}}
```

En Python, il suffit d'installer le paquet :

```bash
pip install pyrucast
```

Exemple équivalent en Python :

```python
{{#include ../../tests/python/test_doc_conteneurs.py:premiers_pas}}
```
