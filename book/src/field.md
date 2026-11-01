# Champ (Field/SubField)

pyrucast a deux familles de champs : le [champ aux nœuds](node-field.md)
(`NodeField`), qui donne une valeur à chaque nœud, et le [champ aux points de
Gauss](element-field.md) (`ElementField`), qui en donne une à chaque point
d'intégration de chaque maille. Les deux se lisent et se calculent de la même
façon, et cette page décrit ce qu'elles ont en commun ; les deux pages
suivantes donnent ce qui est propre à chacune.

En Python, il n'y a pas de classe `Field` : tout ce qui suit s'écrit à
l'identique sur `NodeField`, sur `ElementField` et sur leurs zones
(`SubNodeField`, `SubElementField`). Côté Rust, ce socle commun est écrit une
seule fois, dans deux traits (voir [Côté Rust](#côté-rust)).

## Zones, supports et composantes

Un champ est un [agrégat](aggregate.md) de **zones**. Chaque zone :

- est posée sur un **support** : un nuage de nœuds (une zone de maillage `POI1`)
  pour un champ aux nœuds, un sous-espace d'[espace éléments finis](fe-space.md)
  pour un champ aux points de Gauss. `zone.support()` le rend, et
  `champ.support()` rend celui de chaque zone, dans l'ordre ;
- porte des **composantes nommées** (`"T"`, `"u_x"`, `"E"`, `"sigma_xx"`…) :
  au moins une, chaque nom une seule fois ;
- range une valeur par composante et par **ligne** de son support : un nœud,
  ou un couple (maille, point de Gauss).

Les zones d'un même champ n'ont pas forcément les mêmes composantes. L'exemple
coupe un carré en deux zones de maillage, `gauche` et `droite`, qui partagent
la diagonale : la température `T` est connue sur les deux, le flux `q` sur
celle de droite seulement.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:champ_lire}}
```

| | |
|---|---|
| `components()` | toutes les composantes du champ, dans l'ordre où les zones les présentent |
| `min(c)`, `max(c)` | la plus petite ou la plus grande valeur de `c`, cherchée dans toutes les zones qui portent `c` |
| `min()`, `max()` | la même chose sur toutes les valeurs, toutes composantes confondues : un nombre, pas une grandeur physique quand les unités diffèrent |
| `sum(c)` | la somme des valeurs de `c`, zone par zone, sans poids |
| `champ["c"]`, `filter_components([…])` | un champ réduit à ces composantes |
| `rename_component(a, b)` | un champ où `a` s'appelle `b` |

Une composante qu'aucune zone ne porte lève une erreur.

`champ["c"]` laisse de côté une zone qui ne porte aucune des composantes
demandées. Une zone qui n'en porte pas d'autres est **partagée** avec le champ
d'origine, sans copie : écrire dans l'une modifie l'autre. Une zone dont il faut
retirer des composantes est recopiée. `rename_component` partage de même les
zones qui ne portent pas `a`.

`sum(c)` ajoute les valeurs de chaque zone : un nœud que deux zones partagent
compte une fois par zone. Ce n'est pas une intégrale, qui pèse chaque valeur
par la taille des mailles : c'est `pyrucast.measure.integral`. Deux autres
réductions s'appliquent aux deux familles : `pyrucast.measure.xtx(champ)`, la
somme des carrés des valeurs, et `pyrucast.measure.xty(a, b)`, la somme des
produits des valeurs que `a` et `b` portent sur les mêmes supports et sous les
mêmes noms (voir [Opérateurs sur les champs](operateurs/champs.md#réduction)).

## Calculer avec des champs

### Par un scalaire

`champ + s`, `champ - s`, `champ * s` et `champ / s` appliquent l'opération à
chaque valeur, de chaque composante, de chaque zone, et rendent un **nouveau**
champ ; le scalaire se place d'un côté ou de l'autre. `champ ** s` élève
chaque valeur à la puissance `s`. Il en va de même sur une zone.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:arithmetique}}
```

`champ += 1.0` construit lui aussi un nouveau champ, et le range sous le même
nom : une autre variable qui désignait l'ancien le garde tel quel. Pour
modifier un champ lui-même, on passe par les opérations par composante.

Aucune valeur n'est protégée : une division par zéro donne `inf`, la puissance
fractionnaire d'un nombre négatif donne `nan`, comme avec numpy. La forme
`pow(champ, s, m)`, avec un modulo, est refusée.

### Sur une composante, en place

`add_to_component(c, s)`, `sub_to_component`, `mul_to_component` et
`div_to_component` modifient le champ lui-même : seule la composante `c`
change, sur toutes les zones qui la portent. Elles lèvent une erreur si aucune
zone ne porte `c`, et `div_to_component` refuse une division par zéro.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:par_composante}}
```

Sur une zone de champ aux points de Gauss, `set_uniform(c, v)` donne la valeur
`v` à tous les points (voir [Champ aux points de Gauss](element-field.md)).

### Entre deux champs

`a + b`, `a - b`, `a * b`, `a / b` et `a ** b` combinent les valeurs de deux
champs de la même famille. Les zones s'associent par support :

- sur un support que les deux champs portent, les composantes présentes des
  deux côtés se combinent valeur à valeur. Une composante présente d'un seul
  côté passe telle quelle, quelle que soit l'opération : `a - b` rend `b`, et
  non `-b`, sur une composante que seul `b` porte ;
- une zone dont le support n'existe que d'un côté passe telle quelle.

Les deux champs n'ont donc besoin ni des mêmes composantes, ni des mêmes
zones :

```python
{{#include ../../tests/python/test_doc_conteneurs.py:champ_binaire}}
```

`champ + zone` ne combine que la zone du champ posée sur le même support que
`zone`, et recopie les autres ; il lève une erreur si aucune ne l'est. Entre
deux zones, les supports doivent être les mêmes, sinon c'est une erreur.

Ces règles supposent qu'un support porte chaque composante dans une seule zone
du champ : c'est ce que vérifie l'union `|` ([plus bas](#lunion-)).

### Même support veut dire même objet

Deux zones s'associent quand leurs supports sont **le même objet**, jamais en
comparant leurs nœuds. Deux régions qui se touchent, construites chacune sur
sa zone de maillage, ont des supports distincts, même là où elles partagent
des nœuds : `g + dr` les met côte à côte sans rien additionner à la frontière.
À la lecture d'un nœud que les deux zones portent, `value` rend la valeur de la
première ; l'autre ne compte pas, et rien ne le signale.

Pour additionner deux régions, on les ramène d'abord sur les mêmes supports.
`restrict(champ, maillage)` pose un champ aux nœuds sur les zones de
`maillage`, et deux champs restreints au même maillage ont les mêmes supports ;
`restrict_like(b, a)` pose `b` sur les supports de `a` (voir [Opérateurs sur
les champs](operateurs/champs.md#restriction-fusion-consolidation)). Un nœud
que le champ restreint ne portait pas y vaut 0.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:meme_support}}
```

Pour un champ aux points de Gauss, le support est le sous-espace : deux espaces
construits sur le même maillage sont deux supports (voir [Champ aux points de
Gauss](element-field.md#le-support--un-espace-éléments-finis)).

### Fonctions mathématiques et masques

`pyrucast.field` réunit les fonctions qui s'appliquent valeur par valeur et
rendent un nouveau champ de la même famille : `abs`, `sqrt`, `exp`, `log`
(népérien), `log10`, `cos`, `sin`, `tan`, `cosh`, `sinh` et `tanh`. Chacune
existe aussi en méthode : `champ.sqrt()` donne `pyrucast.field.sqrt(champ)`.
Elles ne protègent rien non plus : `log` d'un nombre négatif donne `nan`.

`champ >= s`, `champ > s`, `champ <= s` et `champ < s` rendent un **masque** :
un champ de mêmes zones et de mêmes composantes, qui vaut 1 où la condition
tient et 0 ailleurs (c'est `mask`, voir [Opérateurs sur les
champs](operateurs/champs.md#masque-par-valeur)). `==` et `!=` ne comparent pas
les valeurs : `champ == 0.0` vaut `False`.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:maths_champ}}
```

## L'union `|`

L'union compose des zones sans toucher aux valeurs ; c'est l'arithmétique qui
combine des valeurs. Elle traite les deux familles différemment :

- **champ aux nœuds** : les zones posées sur un même support sont fondues en
  une. Un nœud doit avoir la même valeur dans toutes les zones qui le portent,
  pour chaque composante qu'elles ont en commun, sinon l'union lève une
  erreur ;
- **champ aux points de Gauss** : deux zones posées sur un même sous-espace
  restent côte à côte si leurs composantes sont différentes ; une même
  composante portée deux fois sur un sous-espace est refusée.

`consolidate()` fond, à la demande, les zones qui partagent un support (voir
[Champ aux nœuds](node-field.md) et [Champ aux points de
Gauss](element-field.md#combiner-des-champs)).

## Ce qui est mis en cache

Un champ ne garde aucun cache : une zone ne contient que le nom de ses
composantes, ses valeurs et une référence à son support. Ce qui ne se calcule
qu'une fois vit sur les supports, et c'est ce qui rend les opérations
ci-dessus possibles ou rapides :

- **le nuage de nœuds d'une zone de maillage.** `NodeField(maillage, …)`,
  `restrict` et les opérateurs qui produisent un champ aux nœuds posent leurs
  zones sur le nuage des nœuds de chaque zone de maillage. Ce nuage est calculé
  une fois, puis rendu tel quel à chaque demande, tant que la zone de maillage
  ne change pas. Deux champs construits sur le même maillage reçoivent donc le
  même support et se combinent nœud à nœud : c'est ce qui fait marcher `a - b`
  et `restrict(…, carre)` dans les exemples ci-dessus ;
- **la table qui donne la ligne d'un nœud** dans ce nuage, construite à la
  première lecture d'un nœud : `value(nœud, c)`, `set_value` et les opérateurs
  trouvent la ligne d'un nœud sans parcourir la liste ;
- **les tables de l'espace éléments finis** (points et poids de Gauss,
  fonctions de forme et leurs dérivées) et la coloration de ses mailles, pour
  un champ aux points de Gauss (voir [Champ aux points de
  Gauss](element-field.md#ce-qui-est-mis-en-cache)).

## Côté Rust

Le socle commun est écrit une seule fois, dans deux traits de
`src/containers/field.rs`. Python n'en voit que le résultat : les méthodes et
les opérateurs décrits plus haut, présents sur les quatre classes.

- **`SubField`**, la zone. Un type n'a que cinq choses à fournir : `support()`,
  `components()`, `values()`, `values_mut()` et `same_support_with(…)`, une
  zone neuve, à zéro, sur le même support. Tout le reste en découle :
  `component_index`, `min`, `max`, `sum`, `xtx`, `dot`, `pscal`, `map_all`,
  `add_to_component`…, `merge_components`, `select_components`,
  `rename_component`. `SubNodeField` et `SubElementField` l'implémentent.
- **`Field`**, l'agrégat. Il est implémenté d'office pour tout
  [`Aggregate`](aggregate.md) dont la zone est un `SubField` : `NodeField` et
  `ElementField` le reçoivent sans une ligne de code. Chaque opération y est
  faite zone par zone, puis les résultats sont réunis : `components`, `min`,
  `max`, `sum`, `map_subs`, `combine_scalar`, `map_all`, `merge_field`,
  `merge_subfield`, `dot_field`, `pscal_field`, `filter_components`,
  `rename_component`.

Le contrat ne dit pas ce qu'est une ligne — un nœud pour `SubNodeField`, un
couple (maille, point de Gauss) pour `SubElementField`. Il fixe seulement la
disposition des valeurs dans un tableau plat, où la composante varie le plus
vite :

```text
   zone aux composantes [c0, c1]
   values = [ ligne0.c0, ligne0.c1, ligne1.c0, ligne1.c1, … ]
```

```rust,ignore
{{#include ../../tests/doc_conteneurs.rs:champ_contrat}}
```

- **Opérateurs.** Trois macros écrivent `+ - * /` :
  `impl_subfield_scalar_ops!` (une zone et un scalaire, d'un côté ou de
  l'autre), `impl_subfield_field_ops!` (deux zones, par `merge_components`) et
  `impl_field_ops!` (deux agrégats par `merge_field`, un agrégat et un scalaire
  par `combine_scalar`). Combiner peut échouer, sauf une zone avec un
  scalaire : ces opérateurs rendent un `Result`. Une zone prise par valeur est
  modifiée sur place, sans copie ; prise par référence, elle est d'abord
  copiée.
- **Puissance.** Rust n'a pas d'opérateur `**` : `combine_scalar(f64::powf, s)`
  et `merge_field(&autre, f64::powf)` font ce que fait `**` en Python.
- **Support.** `same_support` compare les `Handle` par identité
  (`Handle::same_object`).
- **Mêmes composantes exigées.** `check_same_components` refuse deux zones
  dont les supports ou les composantes diffèrent. L'interpolation d'une
  [`Evolution`](evolution.md) l'appelle avant `merge_components` : deux valeurs
  tabulées d'un même champ doivent porter les mêmes composantes.
- **Lecture par rang.** Un noyau de calcul ne cherche pas une composante par
  son nom à chaque point : `resolve_components` traduit, une fois par zone, une
  liste de noms en rangs dans la zone, et `resolve_optional_components` marque
  `ABSENT_COMPONENT` les noms que la zone ne porte pas.
- **Fonctions mathématiques.** Les traits `MapValues` et `Pscal` donnent un
  nom commun aux quatre types (zone ou agrégat, nœuds ou points de Gauss) :
  `ops::field::cos`, `ops::field::psca`… sont écrits une seule fois.
- **Vue de lecture.** `Field::view` prend un verrou de lecture par zone et le
  garde le temps de la vue. Les opérateurs qui lisent beaucoup (`gradient`,
  `deformation`, `divergence`, `restrict`, `integral`, le tracé…) lisent ainsi
  les valeurs sur place, sans copie ; d'autres lectures peuvent avoir lieu en
  même temps, une écriture attend la fin de la vue.
- **Parallélisme.** `min`, `max`, `sum`, `xtx`, `map_all`… parcourent les
  valeurs en parallèle. `min` et `max` donnent le même résultat quel que soit
  le nombre de threads ; `sum`, `xtx` et `dot` peuvent différer au dernier
  chiffre.
