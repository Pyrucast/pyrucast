# Maillage (Mesh / SubMesh)

Un maillage est un [agrégat](aggregate.md) de **zones** :

- une **zone** (`SubMesh`) regroupe des mailles d'un **seul type d'élément** :
  que des `TRI3`, ou que des `QUA4`… Chaque maille y est une liste de nœuds
  d'un [`Coords`](coords.md) ;
- un **maillage** (`Mesh`) est une liste de zones, toutes sur le **même**
  `Coords`. C'est ce qui lui permet de mêler triangles et quadrangles : une
  zone par type.

Le type `POI1` est le cas dégénéré : une maille, un nœud. Une zone POI1 est
donc exactement une **liste de nœuds**, et c'est elle qui sert de support aux
[champs aux nœuds](node-field.md).

Cette page décrit l'**objet** maillage. Pour **construire** un maillage, on
passe en général par les [opérateurs de maillage](operateurs/maillage.md)
(`line`, `extrude`, `pave_surface`…, cf. [Mailler une géométrie](mailler.md)) ;
la construction à la main, maille par maille, sert surtout aux petits cas et
aux tests.

## Créer et lire un maillage

`pyrucast.Mesh(coords, "TRI3")` crée un maillage d'**une** zone vide de ce type
(`pyrucast.Mesh(coords)`, sans type, un maillage sans zone). `mesh.unit()`
atteint cette zone, et `zone.add_cell([n0, n1, n2])` y pose une maille et rend
son indice. Les nœuds se donnent dans l'**ordre local** de l'élément, celui
que décrit sa fiche (section « Repère de référence » des
[éléments finis supportés](elements/index.md)) ; un nombre de nœuds qui ne
correspond pas au type est refusé, comme un nœud d'un autre `Coords`.
Plusieurs types se composent par l'union `|`.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:mesh_api}}
```

| `Mesh` | |
|---|---|
| `len(mesh)`, `mesh[i]`, `for zone in mesh`, `unit()`, `\|` | l'interface des [agrégats](aggregate.md) |
| `element_types()`, `cell_counts()` | le type et le nombre de mailles de chaque zone |
| `cell_count()` | le nombre total de mailles |
| `cell(i, j)` | la maille `j` de la zone `i` (= `mesh[i][j]`) |
| `node(i, j, k)` | le nœud `k` de cette maille, un [`Node`](node.md) |
| `nearest_node(point)` | le nœud du maillage le plus proche d'un point |
| `coords()` | le `Coords` des zones |
| `duplicate()` | une copie non scellée (voir [Scellement](#scellement)) |
| `set_face_color((r, g, b))` | une couleur pour toutes les zones ; rend le maillage |
| `plot(…)` | le tracé, cf. [Visualisation](visualization.md) |

| `SubMesh` (zone) | |
|---|---|
| `element_type` | le type d'élément, `"TRI3"` |
| `cell_count()`, `len(zone)` | le nombre de mailles |
| `zone[j]` | la maille `j` |
| `add_cell([nœuds])` | pose une maille, rend son indice |
| `is_sealed` | la zone est-elle scellée ? |
| `duplicate()` | une copie non scellée |
| `face_color` | la couleur de tracé `(r, g, b)`, en lecture et en écriture |

Une **maille** (`Cell`) est une **vue** : elle tient sa zone et son indice, et
lit ses nœuds à la demande. `len(cell)` donne son nombre de nœuds, `cell[k]` le
`k`-ième `Node`, `cell.nodes()` la liste ; `cell.index` et `cell.element_type`
la situent. `cell.dump()` liste ses nœuds avec leurs coordonnées.

Les opérateurs de maillage s'appellent aussi en méthode : `mesh.skin()` est
`pyrucast.mesh.skin(mesh)`. Maillage, zone et maille s'affichent sur les
[trois niveaux de listing](aspect-informatique.md#trois-niveaux-de-listing) ;
le `dump()` d'une zone donne sa table de connectivité.

## Types d'éléments

Le type d'une zone se désigne par son nom.

| Type | Nœuds | Dim. topo. | Cas usuel |
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
| `HEX27` | 27 | 3 | hexaèdre tri-quadratique (Lagrange complet, centres de face et nœud central) |

Les huit derniers types sont **quadratiques**. Ils reprennent les sommets de
leur parent linéaire, dans le même ordre, puis ajoutent les nœuds de **milieu
d'arête**, dans l'ordre d'arêtes de la convention VTK ; `QUA9` et `HEX27`
ajoutent encore leurs nœuds de face et leur nœud central. `QUA8`, `PENTA15` et
`HEX20` sont **sérendipité** (nœuds d'arête seulement), les autres sont des
Lagrange complets. Un espace éléments finis les accepte avec l'interpolation
`LAGRANGE2`, qu'il faut demander : `pyrucast.FiniteElementSpace(mesh,
"LAGRANGE2")` — l'interpolation par défaut, `LAGRANGE1`, les refuse (cf.
[Espace éléments finis](fe-space.md)).

## Un maillage compte ses nœuds

Chaque fois qu'un nœud apparaît dans une maille, la zone prend une
**référence** sur lui dans le `Coords` : un nœud partagé par quatre mailles en
reçoit quatre, que `coords.refcount(id)` additionne avec celles des `Node` et
des autres objets (cf. [Coordonnées](coords.md#identité-et-durée-de-vie-des-nœuds)).
La zone les rend toutes à sa destruction, quand plus rien ne la tient — ni
variable Python, ni maillage, ni espace éléments finis, ni champ. Un
`add_cell` refusé ne prend aucune référence.

## Scellement

Un espace éléments finis, un champ ou une matrice **indexent** les mailles ou
les nœuds de la zone sur laquelle ils sont posés. Ajouter une maille ensuite
les laisserait sur une numérotation qui n'est plus la bonne. La zone est donc
**scellée** dès qu'un tel objet la prend : sa connectivité est figée pour de
bon, `add_cell` lève une `RuntimeError` (« submesh is sealed »), et
`zone.is_sealed` vaut `True`. Ce qui scelle :

- la construction d'un **espace éléments finis** scelle ses zones ;
- une **matrice** scelle les zones POI1 qui portent ses lignes et ses
  colonnes ;
- un **champ aux nœuds** scelle sa zone **support**, une zone POI1.

Contenir une zone ne la scelle pas : un maillage, une union `|`, une tranche
la partagent sans la figer, et la zone peut grossir tant qu'aucun de ces objets
ne s'y est posé. De même, `pyrucast.mesh.merge_nodes(…, in_place=True)`, qui
réécrit la connectivité en place, refuse une zone scellée.

Le champ aux nœuds mérite un mot. Posé sur une zone POI1, il prend cette zone
pour support et la scelle. Posé sur une autre zone, il n'en indexe pas les
mailles : il se pose sur son **nuage de nœuds**, une zone POI1 tirée d'elle
(voir [ci-dessous](#ce-quune-zone-garde-en-cache)), et c'est ce nuage-là qui
est scellé. La zone de départ reste modifiable ; si on la modifie, les champs
déjà posés restent valides sur le nuage d'avant, et les suivants se posent sur
un nouveau nuage.

Pour repartir d'une zone scellée, on en prend une **copie** :
`mesh.duplicate()` rend un maillage neuf, non scellé, de même connectivité,
sur les **mêmes** nœuds (seules leurs références augmentent). L'opérateur
[`pyrucast.mesh.copy(mesh, new_nodes=…)`](operateurs/maillage.md#copie-sur-place--copy)
fait de même, avec en plus le choix de créer des nœuds neufs aux mêmes
positions.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:scellement}}
```

## Ce qu'une zone garde en cache

Une zone garde deux résultats tirés de sa connectivité. Chacun se calcule à la
première demande, puis resert tel quel.

**Son nuage de nœuds.** `pyrucast.mesh.to_poi1(mesh)` rend, pour chaque zone,
la zone POI1 de ses nœuds distincts, dans leur ordre de première apparition.
Ce nuage est calculé une fois par zone, puis **le même** est rendu à chaque
appel. Tout ce qui pose un champ aux nœuds sur un maillage passe par lui :
`NodeField(mesh, …)`, les supports des matrices d'un modèle — donc le résultat
de `solve(K, f)` et le produit `K * x` —, `pyrucast.node_field.restrict(champ,
mesh)`, les sorties de `divergence` et de `sum_to_nodes`.

C'est ce qui permet à ces champs de se combiner. Deux champs aux nœuds ne
s'additionnent ou ne se soustraient **nœud à nœud** que s'ils sont posés sur
la **même** zone support ; sur deux supports distincts, même portant les mêmes
nœuds, ils restent côte à côte, en deux zones (cf.
[Champ aux nœuds](node-field.md)). Grâce au cache, une différence
`solve(K, f) - restrict(g, mesh)` tombe sur le même support et se calcule
directement. Le cache épargne aussi le dédoublonnage des nœuds, que chaque
construction de modèle referait sinon.

**L'index de ses nœuds.** Une table « identifiant de nœud → rang dans la
zone ». C'est par elle qu'un champ aux nœuds trouve la ligne d'un nœud
(`champ.value(nœud, "T")`, `set_value`) et qu'une matrice trouve la ligne et
la colonne d'un degré de liberté. Construite à la première recherche, elle sert
ensuite à tous les champs et à toutes les matrices posés sur cette zone ; en
pratique, c'est sur les zones POI1 support qu'elle se construit.

Les deux caches sont **oubliés dès que la connectivité change** (`add_cell`,
`merge_nodes(…, in_place=True)`), et recalculés à la demande suivante : ils
répondent toujours pour l'état courant de la zone. Un champ déjà posé garde son
nuage, et `restrict` le porte sur le nouveau :

```python
{{#include ../../tests/python/test_doc_conteneurs.py:caches}}
```

Les caches ne sont pas écrits par [`pyrucast.save`](sauvegarde.md) : après une
relecture, le nuage se recalcule, et un champ relu ne partage donc plus son
support avec les champs qu'on pose ensuite sur le maillage relu.
`restrict(champ, mesh)` l'y ramène.

## Visualisation

`mesh.plot(…)` trace le maillage, chaque zone dans sa couleur (`face_color`),
ou coloré par un champ. Voir [Visualisation](visualization.md).

## Côté Rust

Une zone stocke sa connectivité **à plat** : un `Vec<NodeId>` où la maille `i`
occupe les entrées `[i·n, (i+1)·n)`, `n` étant le nombre de nœuds du type.
Elle porte aussi son `Handle<Coords>`, sa couleur, son drapeau de scellement et
ses deux caches. `Mesh` est l'agrégat de `Handle<SubMesh>` ; il ne porte pas de
`Coords` lui-même et le retrouve par sa première zone, et son `check_push`
refuse une zone d'un autre `Coords`.

```rust,ignore
{{#include ../../tests/doc_conteneurs.rs:maillage}}
```

- **Références.** `SubMesh::add_cell` incrémente le compteur de chaque nœud et
  annule les incréments déjà faits si l'un échoue ; `add_cell_taking` adopte
  des références que l'appelant possède déjà (un nœud tout juste créé) ; le
  `Drop` de la zone les rend toutes, une fois, quand le dernier `Handle`
  disparaît (cf. [Modèle mémoire](memory-model.md)).
- **Construction en bloc.** `add_cell` prend le verrou d'écriture du `Coords`
  et vide les caches à chaque maille. `SubMesh::from_connectivity(coords, type,
  connectivité)` reçoit tout le tableau, le valide et incrémente ses nœuds en
  une seule prise de verrou, et `Coords::add_nodes` crée des nœuds d'un coup :
  c'est par là que les opérateurs de maillage produisent leurs zones.
- **Scellement.** Les consommateurs passent par la fonction libre
  `mesh::seal(&handle)`, qui ne prend le verrou d'écriture que si la zone
  n'est pas déjà scellée : sceller une zone qu'un lecteur tient déjà ne bloque
  pas. `add_cell`, `add_cell_taking` et `remap_nodes` rendent alors
  `PyrucastError::MeshSealed`.
- **Caches.** Deux `OnceLock`, lus par `node_index()` et `to_poi1()`, vidés
  par chaque mutation de la connectivité et marqués `#[serde(skip)]`.
- **Types d'éléments.** `ElementType` porte le nom, le nombre de nœuds et la
  dimension topologique ; ce que l'élément sait de lui-même (repère, fonctions
  de forme, quadrature) vit dans `src/atoms/element_kind/`, un fichier par
  type — cf. [Ajouter un élément fini](developper/ajouter-un-element-fini.md).
