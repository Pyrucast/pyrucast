# Champ aux nœuds (NodeField/SubNodeField)

Un champ aux nœuds porte des valeurs **par nœud** : une température, un
déplacement, des forces nodales, les positions des nœuds. Là où un [champ aux
points de Gauss](element-field.md) donne une valeur à chaque point
d'intégration de chaque maille, celui-ci en donne une à chaque nœud, que les
mailles voisines partagent. C'est un [agrégat](aggregate.md) de **zones** :

- une **zone** (`SubNodeField`) range les valeurs des nœuds d'un **nuage de
  nœuds**, une zone de maillage POI1 ;
- un **champ** (`NodeField`) est une liste de zones, qui n'ont pas forcément
  les mêmes composantes : un champ multiphysique porte `T` sur tout le domaine
  et `ux`, `uy` sur la partie solide seulement.

Les règles communes aux deux familles de champs (composantes, règles de
cohérence, statistiques, arithmétique) sont décrites dans [Champ](field.md) ;
cette page en donne ce qui est propre aux nœuds.

## Créer et lire un champ

`pyrucast.NodeField(maillage, ["T"])` crée une zone par zone du maillage, avec
ces composantes, toutes les valeurs à zéro ; sur une seule zone de maillage
(`SubMesh`), le champ a une zone.
`NodeField.with_components_per_submesh(maillage, [["T", "ux"], ["T"]])` donne
une liste de composantes **par** zone. Une liste vide, un nom en double, un
nombre de listes qui ne correspond pas aux zones ou un maillage qui liste deux
fois la même zone sont refusés.

Le champ se lit nœud par nœud : `champ.value(nœud, "T")` lit la première zone
qui porte ce nœud et cette composante, et lève une erreur si aucune ne les
porte. `champ.values(nœuds, "T")` lit plusieurs nœuds d'un coup et rend leurs
valeurs dans le même ordre ; `nœuds` est une liste de `Node`, ou une zone ou un
maillage dont la connectivité est lue dans l'ordre (pour une zone POI1, ses
points). Un nœud d'un autre `Coords` est refusé.

On écrit **dans une zone** : `champ[i]` ou `champ.unit()` la rendent, et ce
sont des vues (écrire dans la zone, c'est écrire dans le champ). Dans une
zone, un nœud se désigne par lui-même (`zone[nœud, "T"]`, `set_value`) ou par
sa **ligne**, son rang dans le support de la zone (`get(ligne, rang)`, `set`,
`node_values`). Les opérations par composante, comme `add_to_component`,
s'appliquent au champ entier, en place, dans toutes les zones qui portent la
composante.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:node_field_api}}
```

| `NodeField` | |
|---|---|
| `len(champ)`, `champ[i]`, `for zone in champ`, `unit()`, `\|`, `add_sub` | l'interface des [agrégats](aggregate.md) |
| `support()` | le support de chaque zone, dans l'ordre, partagé |
| `components()` | l'union des composantes des zones |
| `value(nœud, "c")`, `values(nœuds, "c")` | une valeur, ou une liste dans l'ordre des nœuds |
| `node_count` | le nombre de nœuds distincts, toutes zones confondues |
| `min(c)`, `max(c)`, `sum(c)` | sur toutes les zones qui portent `c` ; `sum` compte une fois un nœud partagé |
| `champ["c"]`, `filter_components([…])`, `rename_component(a, b)` | un champ réduit à ces composantes, ou une composante renommée |
| `add_to_component(c, s)`, `mul_to_component(c, s)`… | une opération sur une composante, en place |
| `check()` | vérifie que les zones s'accordent sur les nœuds qu'elles partagent |
| `consolidate()` | fond les zones posées sur un même support (voir [Combiner](#combiner-des-champs)) |
| `plot(…)` | le tracé, voir [Visualisation](#visualisation) |

| `SubNodeField` (zone) | |
|---|---|
| `support()` | son nuage de nœuds, partagé |
| `node_count`, `component_count` | ses deux dimensions |
| `value(nœud, "c")`, `zone[nœud, "c"]` | une valeur, par nœud et nom de composante |
| `set_value(nœud, "c", v)`, `zone[nœud, "c"] = v` | écrire une valeur |
| `get(ligne, rang)`, `set(ligne, rang, v)` | par ligne et rang de composante |
| `get_by_node(nœud, rang)`, `set_by_node(nœud, rang, v)` | par nœud et rang de composante |
| `node_values(ligne)` | toutes les composantes d'une ligne, dans l'ordre |
| `set_uniform("c", v)` | la composante `c` à `v` en tout nœud |
| `components()`, `component_index("c")` | les composantes, le rang de l'une |

## Le support : un nuage de nœuds

Chaque zone est posée sur une zone de maillage **POI1** : une liste de nœuds,
un par maille. La ligne `i` de la zone est le `i`-ème nœud de ce support ; le
champ ne range que ses valeurs et lit ses nœuds dans le support.

- **Sur une zone POI1**, le champ prend cette zone pour support, telle quelle.
- **Sur une zone d'un autre type** (TRI3, HEX8…), il prend son **nuage de
  nœuds** : la zone POI1 de ses nœuds distincts, dans leur ordre de première
  apparition, celle que rend `pyrucast.mesh.to_poi1`. La zone de maillage le
  calcule une fois et le garde en [cache](#ce-qui-est-mis-en-cache) : deux
  champs construits sur le même maillage reçoivent le même support.

Le support est [scellé](mesh.md#scellement), puisque le champ range ses lignes
dans l'ordre de ses nœuds. La zone TRI3 d'où vient le nuage, elle, reste
modifiable. Si on lui ajoute une maille, elle oublie son nuage : les champs
déjà posés restent valides sur l'ancien, et un champ construit ensuite se pose
sur un nouveau nuage. Les deux ne se combinent plus nœud à nœud ;
`pyrucast.node_field.restrict(champ, maillage)` porte l'ancien sur le nouveau
(voir [Maillage](mesh.md#ce-quune-zone-garde-en-cache)).

`champ.support()` rend le support de chaque zone, dans l'ordre, et
`zone.support()` celui d'une zone. C'est le même objet, pas une copie : un
champ construit dessus se combine nœud à nœud avec le premier. Le champ garde
son support en vie, mais pas la zone de maillage dont il a été tiré : il n'a
plus besoin d'elle.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:node_field_support}}
```

## Nœuds partagés entre zones

Deux zones de maillage qui se touchent partagent leurs nœuds de frontière, et
un champ posé sur elles range ces nœuds **dans chaque zone**, comme `a` et `d`
dans le premier exemple. La [deuxième règle de
cohérence](field.md#les-règles-de-cohérence) veut qu'un tel nœud ait la même
valeur dans toutes les zones qui le portent, pour chaque composante qu'elles
ont en commun. Le champ s'appuie sur elle :

- `value` et `values` lisent la première zone qui porte le nœud ;
- `sum`, `node_count`, `pyrucast.measure.xtx` et `xty` comptent un nœud
  partagé une fois ;
- `add_to_component` et les autres opérations par composante modifient toutes
  les zones à la fois, et gardent donc la règle.

L'union et les opérations entre deux champs la vérifient quand c'est
nécessaire. Écrire dans une zone, en revanche, ne touche pas les autres :
`T[0][d, "T"] = 30.0` laisse l'ancienne valeur de `d` dans la zone du haut.
`champ.check()` compare alors les valeurs exactement et lève une erreur au
premier désaccord.

## Combiner des champs

Les opérations par un scalaire et entre deux champs suivent les règles de
[Champ](field.md#calculer-avec-des-champs) : les valeurs s'associent par support
et par composante, et deux zones s'associent quand leurs supports sont le même
objet.

L'**union** `a | b` réunit les zones sans toucher aux valeurs :

- les zones posées sur un **même** support sont fondues en une, qui porte
  l'union de leurs composantes ; une composante portée des deux côtés doit y
  avoir les mêmes valeurs ;
- les zones posées sur des supports **distincts** restent côte à côte, et
  doivent s'accorder sur les nœuds qu'elles partagent.

Un désaccord lève une erreur. `pyrucast.node_field.merge(a, b)` est cette même
union. `add_sub` ajoute une zone telle quelle, sans rien fondre, et refuse une
zone qui porterait une composante déjà portée sur son support ;
`consolidate()` fond ensuite les zones qui partagent un support, après la même
vérification.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:node_field_combiner}}
```

## D'où viennent ces champs, et où ils vont

Les opérateurs qui **produisent** un champ aux nœuds sont rangés dans
`pyrucast.node_field` (détail dans [Opérateurs sur les
champs](operateurs/champs.md) et [Assemblage](operateurs/assemblage.md)) :

| Famille | Opérateurs |
|---|---|
| géométrie | `positions(maillage)`, les coordonnées des nœuds, une composante par axe |
| d'un champ aux points de Gauss | `divergence`, `sum_to_nodes` |
| assemblage | `internal_forces`, `external_forces` |
| d'un champ aux nœuds | `restrict`, `restrict_like`, `project`, `merge`, `consolidate`, `mask` |

La résolution en produit un aussi : `pyrucast.solver.solve(K, f)` pose sa
solution sur les supports des blocs de la matrice, ceux que
`restrict(…, maillage)` donne sur le même maillage (voir
[Solveur](operateurs/solveur.md)).

Ces champs sont ensuite **consommés** :

- portés aux points de Gauss : `interp_to_gauss`, `gradient`, `deformation`,
  `beam_deformation`, `shell_deformation`, `strain` (dans
  `pyrucast.element_field`) ;
- comme second membre de `solve`, où un degré de liberté qu'aucune zone ne
  porte vaut zéro ;
- pour déplacer les nœuds : `pyrucast.coords.set`, `displace`,
  `displace_from`, qui traitent une fois un nœud partagé ;
- par `pyrucast.model.update_contact`, les réductions de `pyrucast.measure`
  (`integral`, `xtx`, `xty`), l'export (`pyrucast.export.to_arrays(…,
  node_fields=[…])`, `export_vtk`) et le tracé.

## Ce qui est mis en cache

Un champ aux nœuds ne garde **aucun cache** : une zone ne contient que ses
valeurs, les noms de ses composantes et son support. Il s'appuie sur deux
caches des zones de maillage (détaillés dans
[Maillage](mesh.md#ce-quune-zone-garde-en-cache)) :

- **le nuage de nœuds**, gardé par la zone de maillage d'origine, calculé à la
  première demande puis rendu tel quel. C'est lui qui donne le même support à
  `NodeField(maillage, …)`, à `restrict(…, maillage)`, à la solution de `solve`
  et aux sorties de `divergence` et de `sum_to_nodes` sur ce maillage : ils se
  combinent nœud à nœud. Il épargne aussi le dédoublonnage des nœuds de la zone
  à chaque nouveau champ ;
- **l'index des nœuds**, gardé par le support : la table « nœud → ligne »,
  construite à la première lecture par nœud (`value`, `set_value`, `values`,
  les opérateurs qui lisent le champ nœud par nœud). Elle sert ensuite à tous
  les champs posés sur ce support : trouver la ligne d'un nœud ne parcourt plus
  la liste.

Le support étant scellé, son index ne change plus. Le nuage, lui, est oublié
dès que la zone de maillage d'origine change.

## Visualisation

`champ.plot(component="T")` trace un **nuage de points colorés** : le support
POI1 n'a pas de mailles, aucune surface ne s'en déduit. Pour colorer les
mailles, on passe le champ au maillage : `maillage.plot(field=champ,
component="T")`. Voir [Visualisation](visualization.md).

## Sauvegarde

[`pyrucast.save`](sauvegarde.md) écrit le champ avec son support et son
`Coords`. Des champs sauvegardés dans un même fichier partagent leur support à
la relecture et se combinent nœud à nœud. Le nuage mis en cache n'est pas
écrit : un champ construit sur le maillage relu se pose sur un nouveau nuage,
et `restrict(champ, maillage)` y ramène un champ relu. Les nœuds relus
appartiennent au `Coords` relu : `value` refuse un `Node` du script qui a écrit
le fichier, et on lit avec le nœud relu, `o["maillage"].coords().acquire(id)`.

## Côté Rust

`SubNodeField` porte le `Handle<SubMesh>` de son support, ses noms de
composantes et un `Vec<f64>` à plat, rangé **ligne → composante** :

```text
values[ligne * component_count + c]
```

Les composantes d'un nœud sont donc contiguës : `node_values(ligne)` en rend
la tranche sans copie.

Le `Handle<SubMesh>` du support est le **compagnon POI1 lui-même**, sans objet
intermédiaire. `SubMesh::to_poi1` garde ce compagnon dans un `OnceLock` et en
rend un clone du `Handle` ; le champ garde ce clone. Un `Handle` étant un
`Arc<RwLock<SubMesh>>`, le champ et la zone d'origine tiennent deux clones du
même `Arc`, et le champ ne référence pas la zone d'origine.

`NodeField` est l'agrégat de `Handle<SubNodeField>`. Son `check_push` refuse
une zone d'un autre `Coords`, son `check_add` une composante déjà portée sur le
support de la zone (première règle), et son `finalize`, appelé à la fin de
chaque union, est `ops::node_field::consolidate`.

```rust,ignore
{{#include ../../tests/doc_conteneurs.rs:champ_nodal}}
```

- **Contrat commun.** Les noms de composantes, `min`/`max`/`sum`, les
  opérations par composante et l'arithmétique viennent des traits `SubField`
  et `Field` (`src/containers/field.rs`), partagés avec le champ aux points de
  Gauss.
- **Ligne d'un nœud.** `SubNodeField::index_of` passe par
  `SubMesh::node_index()`, un `OnceLock<HashMap<NodeId, usize>>` du support ;
  `index_of_with` prend un verrou de lecture déjà tenu, pour une boucle qui
  cherche beaucoup de nœuds.
- **Lecture par les opérateurs.** `NodeFieldView` tient un verrou de lecture
  sur chaque zone et sur son support pendant tout un opérateur (`gradient`,
  `restrict`, `project`, le tracé…), et lit les valeurs en place.
- **Un nœud une fois.** `first_rows` parcourt les zones dans l'ordre et donne
  chaque nœud une fois, à la première zone qui le porte : c'est ce
  qu'utilisent `node_ids`, `node_count` et le déplacement des nœuds.
