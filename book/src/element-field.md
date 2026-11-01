# Champ aux points de Gauss (ElementField/SubElementField)

Un champ aux points de Gauss porte des valeurs **par maille et par point de
Gauss** : là où un [champ aux nœuds](node-field.md) donne une valeur à chaque
nœud, celui-ci en donne une à chaque point d'intégration de chaque maille. Deux
mailles voisines ne partagent donc rien, et le champ peut sauter d'une maille à
l'autre. C'est un [agrégat](aggregate.md) de **zones** :

- une **zone** (`SubElementField`) porte les valeurs d'un **sous-espace** d'un
  [espace éléments finis](fe-space.md), c'est-à-dire d'une zone de maillage
  munie de sa quadrature ;
- un **champ** (`ElementField`) est une liste de zones, qui n'ont pas forcément
  les mêmes composantes.

On y range ce qui vit aux points où se calculent les intégrales :

- les **propriétés matériau** (module d'Young, coefficient de Poisson,
  conductivité, masse volumique…) ;
- les **variables internes** d'une loi de comportement (déformation plastique,
  endommagement…), gardées d'un pas de calcul au suivant ;
- les **grandeurs dérivées** d'une solution (déformations, contraintes,
  gradients, flux…).

On en construit rarement à la main : la plupart sortent des
[opérateurs](#doù-viennent-ces-champs-et-où-ils-vont) (`material_field`,
`gradient`, `integrate_behavior`…). Les règles communes aux deux familles de
champs (composantes, statistiques, arithmétique) sont décrites dans
[Champ](field.md) ; cette page en donne ce qui est propre aux points de Gauss.

## Créer et lire un champ

`pyrucast.ElementField(fes, ["E", "nu"])` crée une zone par sous-espace de
`fes`, avec ces composantes, toutes les valeurs à zéro.
`ElementField.with_components_per_subspace(fes, [["E", "nu"], ["k"]])` donne
une liste de composantes **par** sous-espace. Une liste vide, un nom en double
ou un nombre de listes qui ne correspond pas aux sous-espaces sont refusés.

On lit et on écrit **dans une zone** : `champ[i]` ou `champ.unit()` la rendent,
et ce sont des vues (écrire dans la zone, c'est écrire dans le champ). Un point
s'y désigne par trois indices : la **maille**, dans l'ordre des mailles de la
zone de maillage ; le **point de Gauss** dans la maille, dans l'ordre de la
quadrature ; la **composante**, par son nom ou par son rang. Un indice hors
bornes ou une composante inconnue lèvent une erreur.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:element_field_api}}
```

| `ElementField` | |
|---|---|
| `len(champ)`, `champ[i]`, `for zone in champ`, `unit()`, `\|` | l'interface des [agrégats](aggregate.md) |
| `support()` | l'espace de ses zones, partagé |
| `components()` | l'union des composantes des zones |
| `min(c)`, `max(c)`, `sum(c)` | sur toutes les zones qui portent `c` |
| `champ["c"]`, `filter_components([…])`, `rename_component(a, b)` | un champ réduit à ces composantes, ou une composante renommée |
| `add_to_component(c, s)`, `mul_to_component(c, s)`… | une opération sur une composante, en place |
| `consolidate()` | fond les zones posées sur un même sous-espace (voir [Combiner](#combiner-des-champs)) |
| `plot(…)` | le tracé, voir [Visualisation](#visualisation) |

| `SubElementField` (zone) | |
|---|---|
| `support()` | son sous-espace, partagé |
| `cell_count`, `gauss_count`, `component_count` | les trois dimensions de la zone |
| `value(maille, g, "c")`, `zone[maille, g, "c"]` | une valeur, par nom de composante |
| `get(maille, g, rang)` | une valeur, par rang de composante |
| `point_values(maille, g)` | toutes les composantes d'un point, dans l'ordre |
| `set_value`, `zone[…] = v`, `set` | écrire une valeur |
| `set_uniform("c", v)` | la composante `c` à `v` en tout point |
| `set_cell_uniform(maille, "c", v)` | la composante `c` à `v` en tout point d'une maille |
| `components()`, `component_index("c")` | les composantes, le rang de l'une |

Les **positions** des points de Gauss ne sont pas dans le champ : leurs
coordonnées dans l'élément de référence et leurs poids sont sur l'espace
(`fes[i].gauss_xi(g)`, `fes[i].gauss_weight(g)`), et leur position réelle se
calcule à partir des nœuds.

## Composantes

Chaque zone a ses propres composantes, au moins une, chacune nommée une seule
fois. Au niveau du champ, `components()` rend leur **union**, dans l'ordre de
première apparition : une composante peut n'exister que sur certaines zones.
C'est le cas d'un champ matériau multiphysique, qui porte `E` et `nu` sur la
zone de l'élasticité et `k` sur celle de la thermique.

`champ["E"]` et `champ.filter_components(["E", "nu"])` rendent un nouveau champ
réduit à ces composantes. Une zone qui n'en porte aucune disparaît ; une zone
qui n'en porte pas d'autres est **partagée** avec le champ d'origine, sans
copie, si bien qu'écrire dans l'une modifie l'autre ; une zone dont il faut
retirer des composantes est recopiée.

## Le support : un espace éléments finis

Chaque zone est posée sur **un** sous-espace de l'[espace éléments
finis](fe-space.md), qui fixe ses mailles (celles de sa zone de maillage) et
son nombre de points de Gauss par maille (celui de sa quadrature). Ces deux
dimensions sont lues à la création de la zone et ne changent plus : l'espace a
[scellé](mesh.md#scellement) sa zone de maillage, aucune maille ne peut y être
ajoutée. Les nœuds, eux, peuvent bouger : les valeurs du champ restent ce
qu'elles sont, et les points de Gauss suivent les mailles.

Deux champs se combinent **point à point** seulement s'ils sont posés sur le
**même** espace, au sens du même objet. Deux appels à
`FiniteElementSpace(mesh)` sur un même maillage créent deux espaces distincts,
et leurs champs restent côte à côte au lieu de s'additionner (voir l'exemple
ci-dessous). On construit donc un espace une fois et on le passe à tous les
opérateurs ; `material_field(modele, …)` pose ses zones sur les sous-espaces
des sous-modèles, ceux-là mêmes.

Un champ rend aussi le sien : `champ.support()` donne l'espace de ses zones, et
`zone.support()` le sous-espace d'une zone. C'est le même objet, pas une copie :
un champ construit dessus se combine point à point avec le premier.

Un champ reste sur sa quadrature : aucun opérateur ne transporte des valeurs
d'une quadrature à une autre. Pour obtenir un champ sur un autre espace, on
l'interpole depuis les nœuds (`interp_to_gauss`), ou on l'intègre maille par
maille vers un espace à un point par maille (`cell_integral`).

Un champ garde son espace en vie, et l'espace garde son maillage et son
`Coords` : qu'un script ne tienne plus de variable sur l'espace ou le maillage
ne gêne pas le champ. Il ne compte pas les nœuds pour autant ; c'est la zone de
maillage qui les compte (voir [Maillage](mesh.md#un-maillage-compte-ses-nœuds)).

## Combiner des champs

Les opérations suivent les règles de [Champ](field.md#opération-arithmétique) :

- une opération par un scalaire (`champ * 1.1`, `champ + 2.0`…) s'applique à
  toutes les valeurs et rend un nouveau champ ;
- `a + b`, `a - b`, `a * b`, `a / b`, `a ** b` entre deux champs combinent,
  sur chaque espace porté des deux côtés, les composantes présentes des deux
  côtés ; une composante ou une zone présente d'un seul côté passe telle
  quelle ;
- `champ + zone` ne touche que la zone du champ posée sur le même sous-espace ;
- entre deux zones, les sous-espaces doivent être les mêmes, sinon c'est une
  erreur ;
- `abs`, `sqrt`, `exp`, `cos`… s'appliquent valeur par valeur, et `champ >= s`
  rend un masque de 0 et de 1 (`mask`).

L'**union** `a | b` compose des zones sans toucher aux valeurs. Elle garde côte
à côte deux zones posées sur le même sous-espace si leurs composantes sont
disjointes, et refuse qu'une même composante y soit portée deux fois.
`consolidate()` fond en une seule zone celles qui partagent un sous-espace.
Contrairement au champ aux nœuds, l'union ne les fond pas d'elle-même.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:element_field_combiner}}
```

`sum(c)` additionne les valeurs de tous les points de Gauss, sans poids : ce
n'est pas une intégrale. L'intégrale d'une composante sur chaque maille est
`cell_integral` (exemple de la section suivante).

## D'où viennent ces champs, et où ils vont

Les opérateurs qui **produisent** un champ aux points de Gauss sont rangés
dans `pyrucast.element_field` (détail dans [Opérateurs sur les
champs](operateurs/champs.md), [Construction](operateurs/construction.md) et
[Comportement](operateurs/comportement.md)) :

| Famille | Opérateurs |
|---|---|
| matériau | `material_field(modele, [(nom, valeur)…])`, `material_field_per_sub_model`, `sub_material_field` |
| d'un champ aux nœuds | `interp_to_gauss`, `gradient`, `deformation`, `beam_deformation`, `shell_deformation`, `strain` |
| comportement | `integrate_behavior`, `elastic_stress`, `elastic_strain`, `thermal_strain`, `stress_measure`, `rotate_frame` |
| d'un champ aux points de Gauss | `cell_integral`, `mask`, `consolidate` |

`material_field` crée une zone par sous-modèle qui consomme du matériau, sur
l'espace de ce sous-modèle, et ignore les autres (Dirichlet…). Il refuse une
liste à laquelle manque une composante qu'une physique attend.

Ces champs sont ensuite **consommés** par l'assemblage (`stiffness`, `mass`,
`tangent`, `internal_forces`, `external_forces`, cf.
[Assemblage](operateurs/assemblage.md)), ramenés aux nœuds (`divergence`,
`sum_to_nodes`), tracés (`plot`) ou exportés (`pyrucast.export.to_arrays(…,
element_fields=[…])`).

```python
{{#include ../../tests/python/test_doc_conteneurs.py:element_field_operateurs}}
```

## Ce qui est mis en cache

Un champ aux points de Gauss ne garde **aucun cache** : il ne contient que ses
valeurs, les noms de ses composantes et ses deux dimensions. Ce qui ne se
calcule qu'une fois vit sur l'espace éléments finis qui le porte :

- les **tables de référence** (coordonnées et poids des points de Gauss,
  fonctions de forme et leurs dérivées en ces points), calculées à la création
  de l'espace ;
- la **coloration des mailles** : des groupes de mailles qui ne partagent aucun
  nœud. Elle sert aux opérateurs qui accumulent des contributions de mailles
  sur les nœuds (l'assemblage d'une matrice, `divergence`, `sum_to_nodes`,
  `internal_forces`, `external_forces`) :
  les mailles d'un même groupe s'écrivent en parallèle sans conflit. Elle est
  calculée au premier de ces calculs, puis resert à tous les suivants sur ce
  sous-espace.

C'est une raison de plus de réutiliser un même espace : un second espace sur
le même maillage refait ses tables et sa coloration.

## Visualisation

`champ.plot(component="E")` trace le champ sur son propre support : chaque zone
retrouve sa zone de maillage par son sous-espace, sans copie. Le rendu reste
**par maille** : dans chaque maille, des valeurs aux nœuds sont ajustées aux
valeurs de Gauss par moindres carrés, sans moyenne avec les mailles voisines,
si bien que les sauts d'une maille à l'autre (contraintes, flux) restent
visibles. `smooth` règle la subdivision des mailles (`0` : une couleur par
maille). Voir [Visualisation](visualization.md).

## Sauvegarde

[`pyrucast.save`](sauvegarde.md) écrit le champ avec son espace, son maillage
et son `Coords`. Des champs sauvegardés dans un **même** fichier partagent
toujours leur espace à la relecture et se combinent point à point ; un champ
relu ne partage pas son espace avec ceux du script qui l'a écrit, ni avec ceux
d'un autre fichier. Le scellement de la zone de maillage est relu avec elle ;
la coloration, elle, n'est pas écrite et se recalcule au premier usage.

## Côté Rust

`SubElementField` porte un `Handle<SubFiniteElementSpace>`, ses noms de
composantes, ses deux dimensions et un `Vec<f64>` à plat, rangé **maille →
point de Gauss → composante** :

```text
values[maille * gauss_count * component_count
       + g * component_count
       + c]
```

Les composantes d'un point sont donc contiguës : `point_values(maille, g)` en
rend la tranche sans copie, et c'est ce que lit un noyau de calcul.
`ElementField` est l'agrégat de `Handle<SubElementField>` ; son `finalize`
refuse une composante portée deux fois sur un même sous-espace.

```rust,ignore
{{#include ../../tests/doc_conteneurs.rs:champ_gauss}}
```

- **Contrat commun.** Les noms de composantes, `min`/`max`/`sum`, les
  opérations par composante et l'arithmétique viennent des traits `SubField`
  et `Field` (`src/containers/field.rs`), partagés avec le champ aux nœuds.
- **Support.** `SubField::same_support` compare les `Handle` par identité
  (`same_object`) : c'est ce qui fait de deux espaces sur un même maillage deux
  supports distincts.
- **Durée de vie.** Le `Handle` de l'espace est un `Arc` : la zone le garde en
  vie, l'espace garde son `SubMesh`, qui garde son `Coords` et compte ses
  nœuds (cf. [Modèle mémoire](memory-model.md)).
- **Coloration.** Un `OnceLock` sur `SubFiniteElementSpace`, rempli par
  `coloring(…)` et marqué `#[serde(skip)]`.
