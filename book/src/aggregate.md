# Agrégat

La plupart des conteneurs de pyrucast viennent par **paires** : une **zone**
(`SubMesh`, `SubFiniteElementSpace`, `SubNodeField`, `SubElementField`,
`SubModel`, `SubMatrix`, `SubEvolution`) et son **agrégat** (`Mesh`,
`FiniteElementSpace`, `NodeField`, `ElementField`, `Model`, `Matrix`,
`Evolution`), une liste de zones. Les sept agrégats s'utilisent **exactement de
la même façon** et se composent de la même manière, par l'union `|`. Ce chapitre
décrit ce contrat commun une fois pour toutes ; chaque chapitre d'objet n'en
redonne que les spécificités.

Un agrégat ne copie jamais ses zones : il les **partage**. Deux agrégats peuvent
porter la même zone, et la modifier par l'un se voit par l'autre.

## Interface commune

| Opération | Sens |
|---|---|
| `len(agg)` | nombre de zones |
| `agg[i]` | **vue** sur la zone `i` (un `Sub…`) — jamais une copie |
| `agg[i:j:k]` | **nouvel agrégat** du même type portant ces zones (slicing Python : pas, bornes négatives) |
| `for zone in agg:` | parcourt les zones |
| `agg.unit()` | la **seule** zone d'un agrégat qui n'en a qu'une, sinon une erreur |
| `agg.add_sub(zone)` | ajoute une zone, en place |
| `agg.add_subs(other)` | ajoute en place **toutes** les zones d'un autre agrégat, sans dédoublonner |
| `agg \| other` | **union** (voir plus bas) |
| `print` / `repr` / `dump()` | les [trois niveaux de listing](aspect-informatique.md#trois-niveaux-de-listing) |

> **`agg[i]` est une vue, `agg[i:j]` un agrégat.** L'indice entier rend la
> zone elle-même ; le slicing rend un agrégat du même type, qui partage les
> zones retenues.

> **Passer une seule zone à un opérateur.** Les opérateurs prennent des
> **agrégats**, pas des vues : `pyrucast.mesh.invert(maillage[1])` lève un
> `TypeError` (« 'SubMesh' object is not an instance of 'Mesh' »). Pour n'en
> traiter qu'une, passer par le slicing, qui rend un agrégat d'une seule zone :
> `pyrucast.mesh.invert(maillage[1:2])`.

> **`unit()` plutôt que `[0]`.** `agg[0]` prend sans rien dire la première de
> plusieurs zones ; `agg.unit()` **exige** qu'il n'y en ait qu'une et lève
> sinon. Là où une seule zone a un sens — ajouter une maille à un maillage
> qu'on vient de créer —, `unit()` dit ce qu'on suppose.

### Les zones ne se construisent pas seules

Une zone s'obtient par `agg[i]` ; on ne la construit pas directement. On
construit l'**agrégat** — `Mesh(coords, type)`, `FiniteElementSpace(mesh)`,
`ElementField(fes, composantes)` — ou on le reçoit d'un opérateur
(`pyrucast.model.heat_conduction(fes)`…), puis on indexe pour atteindre une
zone. Seule exception, `SubEvolution` : une courbe tabulée n'a pas de parent
qui la définirait, elle se construit directement (cf.
[Évolution](evolution.md)).

## Composition : l'union `|`

Composer deux agrégats, c'est l'**union**, `a | b`. Elle rend un agrégat
**neuf** et laisse ses deux opérandes intacts, avec la même règle pour les
sept agrégats :

1. **Pas de doublon.** Une zone déjà présente n'est pas ajoutée une seconde
   fois ; les zones gardent l'ordre de leur première apparition.
2. **Partage, pas copie.** Les zones retenues sont celles des opérandes, pas
   des copies.
3. **Cohérence vérifiée.** Ce qui doit être commun l'est, à l'ajout : par
   exemple, deux maillages sur des `Coords` différents se refusent
   (« mismatched Coords »).
4. **Fusion des champs.** Pour `NodeField` et `ElementField`, les zones
   posées sur le même support sont fusionnées en fin d'union (voir
   [Champ](field.md), [Champ aux nœuds](node-field.md)).

L'union accepte agrégats et zones, dans les deux sens :

| Écriture | Résultat |
|---|---|
| `agg \| agg` | l'union des deux listes |
| `agg \| zone` | la zone ajoutée **en queue** (ignorée si déjà présente) |
| `zone \| agg` | la même union, la zone **en tête** (déplacée si elle y était) |
| `zone \| zone` | un agrégat neuf portant les deux zones |

> **En place : `add_sub` / `add_subs`.** Pour modifier un agrégat plutôt que
> d'en créer un, `agg.add_sub(zone)` ajoute une zone et `agg.add_subs(other)`
> toutes celles d'un autre agrégat, dans l'ordre. C'est de la concaténation :
> la cohérence est vérifiée, mais rien n'est dédoublonné ni fusionné. Pour
> composer, `|` reste la forme à préférer.

Les nœuds s'unissent aussi (cf. [Nœud](node.md)) :

| Écriture | Résultat |
|---|---|
| `node \| node` | un `Mesh` POI1 d'une zone, sur les deux nœuds |
| `mesh \| node` | ajoute un point à un `Mesh` POI1 d'une seule zone, sinon une erreur |

> **`|` compose, `+` calcule.** L'union s'écrit **toujours** `|`. L'opérateur
> `+` (et `-`, `*`, `/`) est l'**arithmétique** — des champs (`champ + 2.0`,
> `champ + champ`) et des matrices (`k + m`) — et ne compose jamais : un
> `Mesh + Mesh` lève un `TypeError`.

### Exemple

```python
{{#include ../../tests/python/test_doc_conteneurs.py:aggregat}}
```

## Côté Rust

Un agrégat est une liste de `Handle<Sub>` : il partage ses zones par le compte
de références de leur `Handle`, ce qui explique qu'il ne les copie jamais.

```text
   Mesh                FiniteElementSpace        Model
   ├── Handle<SubMesh> ├── Handle<SubFES>        ├── Handle<SubModel>
   ├── Handle<SubMesh> ├── Handle<SubFES>        └── Handle<SubModel>
   └── …               └── …
```

Tout le contrat est porté par le trait `Aggregate` (`src/aggregate.rs`), qui
le dérive de deux accesseurs, `items()` et `items_mut()` : `len`, `is_empty`,
`get(i)`, `subset(indices)` (sur lequel s'appuie le slicing Python), `iter`,
`unit`, `push`, `add_sub` / `add_subs`, et l'union sous des noms explicites —
`a.union(&b)`, `a.union_sub(&h)` (zone en queue), `a.union_sub_first(&h)`
(zone en tête), `T::union_subs(&a, &b)`. « Pas de doublon » y veut dire : pas
deux `Handle` sur le même objet (`Handle::same_object`). Deux points de
personnalisation seulement : `check_push`, qui vérifie la cohérence à chaque
ajout, et `finalize`, qui tourne en fin d'union et que les champs utilisent pour
fusionner leurs zones.

C'est ce qui garantit qu'un sous-maillage, un sous-espace EF ou un sous-modèle
s'indexent, se parcourent et se composent **strictement de la même façon** :
aucun accès n'est réécrit par type. Un agrégat se câble par des macros :
`impl_aggregate!`, qui pose aussi `Index<usize>` et `IntoIterator`, le plus
souvent `impl_aggregate_dump!` pour le listing, et côté Python
`impl_aggregate_pymethods!`.
