# Maillage (Mesh / SubMesh)

Le maillage de pyrucast se compose à deux niveaux, selon le motif
[agrégat / zone](aggregate.md) :

- **`SubMesh`** : regroupe toutes les cellules d'un même `ElementType`. Stocke
  la connectivité à plat (un `Vec<NodeId>` de longueur `cell_count ×
  nodes_per_cell`).
- **`Mesh`** : agrège plusieurs `SubMesh` liés à la même `Coords`.

Le cas POI1 est volontairement dégénéré : un sous-maillage POI1 est exactement
une **liste de nœuds** (un nœud par cellule), ce qui sert de support naturel
aux [`NodeField`](node-field.md).

> Ce chapitre décrit l'**objet** maillage (structure, types d'éléments,
> refcount). La **construction** de maillages par des générateurs
> (`line`, `triangulate_surface`, `extrude`…) relève des
> [opérateurs de maillage](operateurs/maillage.md).

## Types d'éléments

L'enum `ElementType` liste les types supportés ; chaque variante porte sa
propre méta-information (nombre de nœuds, dimension topologique, nom court
cast3m).

| Variante | Nœuds | Dim. topo. | Cas usuel |
|---|---:|---:|---|
| `POI1` | 1 | 0 | liste de nœuds |
| `SEG2` | 2 | 1 | segment linéaire |
| `TRI3` | 3 | 2 | triangle linéaire |
| `QUA4` | 4 | 2 | quadrangle linéaire |
| `TET4` | 4 | 3 | tétraèdre linéaire |
| `PYRA5` | 5 | 3 | pyramide linéaire (raccord hexaèdre ↔ tétraèdre) |
| `PENTA6` | 6 | 3 | prisme linéaire (extrusion d'un TRI3) |
| `HEX8` | 8 | 3 | hexaèdre linéaire |
| `SEG3` | 3 | 1 | segment quadratique |
| `TRI6` | 6 | 2 | triangle quadratique |
| `QUA8` | 8 | 2 | quadrangle quadratique (sérendipité) |
| `QUA9` | 9 | 2 | quadrangle biquadratique (Lagrange complet, nœud central) |
| `TET10` | 10 | 3 | tétraèdre quadratique |
| `PENTA15` | 15 | 3 | prisme quadratique (sérendipité) |
| `HEX20` | 20 | 3 | hexaèdre quadratique (sérendipité) |
| `HEX27` | 27 | 3 | hexaèdre tri-quadratique (Lagrange complet, centres de face + nœud central) |

Les huit derniers types sont **quadratiques** (Lagrange-2) : ils reprennent la
numérotation des sommets de leur parent linéaire puis ajoutent les nœuds de
**milieu d'arête**, dans l'ordre d'arêtes de la convention VTK (voir le rustdoc
d'`ElementType`). `QUA8`, `HEX20` et `PENTA15` sont **sérendipité** (nœuds
d'arête seulement) ; `SEG3`, `TRI6`, `TET10`, `QUA9` et `HEX27` sont des
Lagrange complets (`QUA9`/`HEX27` = quadrangle/hexaèdre bi-/tri-quadratiques,
avec nœuds de face et central). Ils se posent avec l'interpolation `LAGRANGE2`
(cf. [Espace éléments finis](fe-space.md)).

Ajouter un nouveau type d'élément est purement additif : un fichier
`src/atoms/element_kind/<nom>.rs` et une variante — voir
[Ajouter un élément fini](developper/ajouter-un-element-fini.md).

## Cellule (`Cell`)

`mesh.cell(submesh_idx, cell_idx)` (ou `mesh[submesh_idx][cell_idx]`) renvoie
une **vue** `Cell` sur une cellule : `len(cell)` donne son nombre de nœuds et
`cell[k]` le `k`-ième `Node`. C'est l'accès lecture à la connectivité ;
l'ajout passe par `submesh.add_cell([...])`.

## Refcount sur les nœuds — interaction avec le GC

Chaque appel à `SubMesh::add_cell` **incrémente** le refcount interne de chaque
nœud dans la `Coords` (cf. [Coordonnées](coords.md)). Le `Drop` du `SubMesh`
les **décrémente**. Tant qu'un `SubMesh` référence un nœud, le ramasse-miettes
le protège — même si tous les [`Node`](node.md) utilisateurs ont disparu.

```text
   Coords             ◀── refcount par NodeId
        │                          (le nœud est-il vivant ?)
        ├── Node(s) utilisateur(s) ── chacun +1
        └── SubMesh(s)             ── chacun +1 par cellule incidente
```

En cas d'échec partiel d'`add_cell` (par exemple un nœud déjà ramassé), les
incréments déjà effectués pour la cellule courante sont **annulés** (rollback
transactionnel à l'échelle d'une cellule).

**Construire en bloc.** Un `add_cell` prend le verrou d'écriture de la `Coords`
et lâche les caches dérivés du sous-maillage : c'est le bon grain pour une
maille posée à la main, mais sur un million de mailles c'est *ce* prix-là qu'on
paie, pas la connectivité. Un opérateur qui produit un gros maillage passe donc
par `SubMesh::from_connectivity(coords, type, connectivité)` (côté Rust), qui
valide et incrémente tout le tableau en une seule prise de verrou — une unité
par **occurrence**, comme toujours — et par `Coords::add_nodes` pour créer ses
nœuds d'un coup. C'est la couture qu'empruntent `translate`, `rotate`, les
symétries, `copy` et `merge_nodes`.

## Scellement (connectivité figée après consommation)

Par convention, un maillage n'est plus modifié une fois construit : un
[espace éléments finis](fe-space.md), un [champ](node-field.md) ou une
[matrice](matrix.md) indexent ses cellules, et lui ajouter une maille par la
suite les laisserait dans un état incohérent.

Cette convention est désormais **imposée**. Dès qu'un objet autre qu'un
maillage capture un `SubMesh` et **indexe ses cellules** (construction d'un
`SubFiniteElementSpace`, d'un support de `SubMatrix`…), ce sous-maillage est
**scellé** : sa connectivité est figée pour toujours. `add_cell` /
`add_cell_taking` renvoient alors l'erreur `MeshSealed`. Un `Mesh` qui se
contente de contenir le sous-maillage ne le scelle **pas** — il peut continuer
à grossir tant qu'aucun consommateur ne s'y attache. On teste l'état avec
`is_sealed`.

Un [champ aux nœuds](node-field.md) fait exception, parce qu'il n'indexe pas
les cellules du maillage : il se pose sur le **compagnon POI1** de la zone, et
c'est ce compagnon-là qui est scellé. Le maillage donné en argument, lui, reste
modifiable ; le modifier lâche son compagnon, si bien qu'un champ construit
ensuite se retrouve sur un **autre** support — les champs déjà construits ne
sont pas invalidés pour autant, ils restent définis sur le nuage de nœuds
d'avant (voir [Champ aux nœuds](node-field.md#support--un-sous-maillage-poi1-par-zone)).

Pour repartir d'un maillage scellé et le modifier à nouveau, on en prend une
**copie profonde** avec `duplicate()` : un `SubMesh` (ou `Mesh`) neuf, **non
scellé**, avec la même connectivité (les nœuds sont partagés — même `Coords` —,
seuls leurs refcounts augmentent). L'opérateur
[`mesh.copy(m, new_nodes=…)`](operateurs/maillage.md#copie-sur-place--copy) est
la même chose au niveau `Mesh`, avec le choix supplémentaire de recréer des
nœuds neufs aux mêmes endroits plutôt que de partager ceux de l'original.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:scellement}}
```

## API Rust

```rust,ignore
{{#include ../../tests/doc_conteneurs.rs:maillage}}
```

## API Python

```python
{{#include ../../tests/python/test_doc_conteneurs.py:mesh_api}}
```

## Durée de vie et refcount

Les `SubMesh` et `Mesh` portent un effet de bord dans leur `Drop` : ils
décrémentent le refcount des nœuds qu'ils référencent dans la `Coords`. Cet
effet a lieu **exactement une fois**, quand le dernier `Handle` sur le maillage
disparaît. Le détail est dans le chapitre
[Modèle mémoire](memory-model.md).

## Visualisation

`mesh.plot(...)` trace le maillage (chaque sous-maillage avec sa propre couleur,
ou coloré par un champ). Voir [Visualisation](visualization.md).
