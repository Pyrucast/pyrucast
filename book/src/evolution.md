# Évolution (Evolution/SubEvolution)

Une **évolution** associe des **valeurs** à des **abscisses** et **interpole
linéairement** entre elles. L'abscisse est souvent le temps, mais pas
forcément : une température, une déformation, ou plusieurs variables à la fois.
La valeur est un nombre, un [champ aux nœuds](node-field.md) ou un
[champ aux points de Gauss](element-field.md).

C'est un [agrégat](aggregate.md) :

- une **`SubEvolution`** est une courbe : la liste de ses abscisses et la liste
  des valeurs en regard. L'interpoler en `x` rend **une** valeur ;
- une **`Evolution`** réunit des `SubEvolution`, **une par zone**, comme un
  `NodeField` réunit ses zones. L'interpoler interpole chaque courbe et
  regroupe les résultats : un `NodeField` ou un `ElementField` pour des
  champs, une **liste de flottants** pour des nombres.

## Une courbe

`SubEvolution(échantillons)` prend une liste de couples `(abscisse, valeur)`.
Les échantillons sont triés à la construction ; deux à la même abscisse sont
refusés. Entre deux échantillons `x₀ ≤ x ≤ x₁`, la valeur est
`v₀ + (v₁ − v₀)·(x − x₀)/(x₁ − x₀)` ; sur un échantillon, c'est sa valeur.

```python
{{#include ../../tests/python/test_doc_evolution.py:courbe}}
```

## Hors plage

Une abscisse en dehors de la plage tabulée suit la **politique hors plage** de
l'évolution :

| Politique | Hors plage |
|---|---|
| `"error"` (par défaut) | lève une erreur |
| `"clamp"` | rend la valeur de l'extrémité la plus proche |
| `"extrapolate"` | prolonge le premier ou le dernier segment |

La politique se donne à la construction (`out_of_range="clamp"`), se lit par
`out_of_range`, et s'impose pour un seul appel :
`interpolate(x, out_of_range="clamp")`.

```python
{{#include ../../tests/python/test_doc_evolution.py:hors_plage}}
```

## Des champs comme valeurs

`Evolution([(t₀, champ₀), (t₁, champ₁), …])` prend un champ entier par
abscisse, un `NodeField` ou un `ElementField`. Elle en fait une courbe par
zone : les zones sont celles du premier champ, et chaque champ suivant doit
porter une zone sur chacun de ces supports ; une zone qu'il aurait en plus est
ignorée. Toutes les valeurs d'une courbe ont le même support et les mêmes
composantes.

L'évolution **copie** les valeurs des champs ; le support, lui, est partagé.
Modifier un champ ensuite ne change pas l'évolution, et le champ interpolé est
posé sur le même support que les champs d'origine : il se combine avec eux
zone à zone.

`shared_abscissas()` rend les abscisses, communes à toutes les zones, et
`frames()` les champs tabulés, un par abscisse.

```python
{{#include ../../tests/python/test_doc_evolution.py:champ}}
```

## Une abscisse à plusieurs variables

Une abscisse peut être un **tuple** de deux à quatre nombres : la courbe
devient une **table**. Chaque échantillon est alors `((T, φ), valeur)`, et
`interpolate((T, φ))` rend la valeur en ce point.

```python
{{#include ../../tests/python/test_doc_evolution.py:table}}
```

La table se lit **variable après variable**. Les échantillons sont rangés par
leur première variable : ici, un groupe à 20 °C (φ = 0 et 1), un autre à
500 °C (φ = 0, 0,5 et 1). Chaque groupe est une courbe le long de la variable
suivante. En `(300, 0,4)` :

```text
   φ
   1  ● 450                               ● 320
      │                                   │
 0,5  │                                   ● 260
 0,4  ◆ 360 ─────────────── ★ ─────────── ◆ 248
      │                   294,7           │
   0  ● 300                               ● 200
      └────────────────────┬──────────────┴───── T
      20                  300            500

  ●  échantillons de la table
  ◆  étape 1 : chaque groupe est lu le long de φ, en φ = 0,4
  ★  étape 2 : les deux résultats sont pesés le long de T, en T = 300
```

1. On cherche les deux groupes qui encadrent `T = 300` : 20 °C et 500 °C.
2. Chacun est interpolé le long de φ en `0,4` : entre 300 et 450 à 20 °C, ce
   qui donne 360 ; entre 200 et 260 à 500 °C (φ de 0 à 0,5), ce qui donne 248.
3. On interpole entre ces deux résultats le long de T :
   `t = (300 − 20)/(500 − 20) ≈ 0,583`, et `360 + 0,583 × (248 − 360) ≈ 294,7`.

Pour deux variables, avec `T_a ≤ T ≤ T_b` les groupes encadrants et `f_a`,
`f_b` leurs courbes en φ :

\\[
f(T, \varphi) = (1-t)\\, f_a(\varphi) + t\\, f_b(\varphi),
\qquad t = \frac{T - T_a}{T_b - T_a}.
\\]

Avec trois ou quatre variables, chaque groupe se lit de la même façon le long
de la variable suivante. Ce que cela donne :

- les groupes **n'ont pas à partager leurs abscisses** : la table ci-dessus
  mesure deux valeurs de φ à 20 °C et trois à 500 °C ;
- sur une **grille complète**, le résultat est l'interpolation
  **multilinéaire** (bilinéaire pour deux variables). Une fonction linéaire en
  chaque variable séparément, comme `a + bT + cφ + dTφ`, est retrouvée
  exactement, grille complète ou non ;
- un groupe où la variable suivante ne prend **qu'une valeur** est **constant**
  le long d'elle : une propriété mesurée pour une seule valeur de φ à une
  température donnée ne dépend pas de φ à cette température ;
- l'**ordre des variables** compte : la première range les groupes, et c'est
  le long d'elle que se fait le dernier mélange. Sur une table dont les groupes
  ne partagent pas leurs abscisses, échanger l'ordre des variables change le
  résultat ;
- hors de la plage d'une variable, la politique hors plage s'applique à cette
  variable, dans le groupe où elle est lue.

`abscissas()` rend les abscisses triées : des nombres pour une courbe, des
tuples pour une table, rangés par première variable puis par la suivante.

## Types d'abscisse et d'ordonnée

Une courbe peut porter le **nom** de ce que mesurent ses axes :

- `abscissa_type`, le nom de l'abscisse (`"T"`, `"t"`), un tuple de noms pour
  une table (`("T", "phi")`), un par variable ;
- `ordinate_type`, le nom de la valeur (`"E"`), pour une courbe de nombres
  seulement : un champ a déjà ses composantes, et le donner à une évolution de
  champs est une erreur.

Ils se donnent à la construction et se lisent par `abscissa_type()` et
`ordinate_type()` ; ceux d'une `Evolution` sont ceux de sa première courbe.
Ils étiquettent les tracés et relient une courbe à un champ (section
suivante).

```python
{{#include ../../tests/python/test_doc_evolution.py:types}}
```

## Appliquer une courbe à un champ

Une courbe de nombres s'applique à un **champ** comme `cos` : `interpolate`
reçoit un champ au lieu d'une abscisse, et rend un champ sur le même support,
dont chaque nœud ou point de Gauss est la courbe évaluée en ses propres
valeurs.

- La courbe lit les composantes **nommées comme son `abscissa_type`**, une par
  variable. Sans `abscissa_type`, ou si le champ n'a pas ces composantes,
  c'est une erreur.
- Elle écrit **une** composante, nommée d'après son `ordinate_type`, ou
  `"value"` à défaut.
- La politique hors plage s'applique point par point.

```python
{{#include ../../tests/python/test_doc_evolution.py:transfert}}
```

Une `Evolution` applique **toutes ses courbes** : `k` courbes donnent un champ
à `k` composantes, chacune lisant ses propres variables. Deux courbes qui
écriraient la même composante sont refusées. C'est ainsi qu'on calcule des
propriétés matériau qui dépendent de l'état, par exemple `E(T)` et `σ_y(T, φ)`
aux points de Gauss en un seul appel, avant de les réunir au besoin aux
propriétés constantes ; il n'y a pas d'opérateur matériau à part.

```python
{{#include ../../tests/python/test_doc_evolution.py:lois}}
```

## Composer des courbes

`|` réunit des `SubEvolution` en une `Evolution`, une courbe par zone. Toutes
portent le même genre de valeur : des nombres, ou des champs aux nœuds, ou des
champs aux points de Gauss. `evolution[i]` rend la courbe `i`.

Une `Evolution` interpolée applique **sa** politique hors plage à toutes ses
courbes. Celle que rendent `|` et une tranche `evolution[i:j]` est `"error"` ;
chaque courbe garde la sienne, qui sert quand on l'interpole seule.

```python
{{#include ../../tests/python/test_doc_evolution.py:composer}}
```

## Tracer

`plot()` dessine une évolution de nombres en courbe, une ligne par zone, et
une évolution de champs en champ sur le maillage, avec un curseur pour choisir
l'abscisse dans la fenêtre interactive (voir
[Visualisation](visualization.md#tracé-dune-évolution)). Sans `x_label` ni
`y_label`, les axes et le curseur prennent l'`abscissa_type` et
l'`ordinate_type`. Une table ne se dessine pas en courbe.

```python
{{#include ../../tests/python/test_doc_evolution.py:trace}}
```

## Sauvegarde

Une évolution se [sauvegarde](sauvegarde.md) comme tout objet, avec sa
politique hors plage et ses types. Les champs qu'elle tabule gardent leur
support, partagé avec les autres objets du même fichier.

## Côté Rust

`SubEvolution` porte :

- `abscissas`, un `Vec<f64>` plat : `dims` nombres par échantillon, les
  échantillons triés dans l'ordre de leurs tuples (première variable, puis
  deuxième…) ;
- `dims`, le nombre de variables, de 1 à `MAX_ABSCISSAS` (4) ;
- `values`, un `Vec<SubValue>`, une valeur par échantillon. `SubValue` vaut
  `Scalar(f64)`, `Node(SubNodeField)` ou `Element(SubElementField)` : les
  champs sont rangés dans la courbe, et leur support est un `Handle` partagé ;
- `out_of_range` (`OutOfRange`), `abscissa_type` (`Vec<String>`) et
  `ordinate_type` (`Option<String>`).

`SubEvolution::new` prend des échantillons dont l'abscisse implémente
`Abscissa` : un `f64`, un `[f64; N]` ou un `Vec<f64>`.

`Evolution` est l'agrégat de `Handle<SubEvolution>`, plus sa politique hors
plage. `check_push` refuse une courbe dont les valeurs ne sont pas du même
genre que celles des autres. `Evolution::interpolate_at` rend un
`Interpolated` : `Scalars(Vec<f64>)`, `Node(NodeField)` ou
`Element(ElementField)`. `from_scalars`, `from_node_fields` et
`from_element_fields` sont les constructeurs que Python appelle.

**Interpolation.** Une interrogation pèse au plus deux échantillons par
variable, donc au plus 16. `gather` les trouve par recherche dichotomique,
variable après variable, et range leurs poids dans un tableau de taille fixe,
sur la pile : les poids ne demandent aucune allocation. Pour des champs, la
somme pondérée reprend l'arithmétique des champs (`map_all`, puis
`merge_components` après `check_same_components`).

**Appliquer à un champ.** `SubEvolution::interpolate_field` et
`Evolution::interpolate_node_field` / `interpolate_element_field` résolvent
une fois, avant de parcourir le champ, quelles composantes chaque courbe lit
et quelle politique elle applique. Les lignes du champ sont ensuite traitées
en parallèle, sans allocation ; chaque ligne du résultat est écrite une fois,
depuis sa propre ligne d'entrée, si bien que le résultat ne dépend pas du
nombre de fils.

```rust,ignore
{{#include ../../tests/doc_conteneurs.rs:evolution}}
```
