# Évolution (`Evolution` / `SubEvolution`)

Une **évolution** associe une série de **valeurs** à une **variable** (souvent
le temps, mais pas nécessairement) et **interpole linéairement** entre les
échantillons tabulés. C'est l'analogue de l'`EVOLUTIO` de Cast3M, généralisé :
l'abscisse n'est pas forcément le temps et la valeur peut être un **champ**
entier, pas seulement un réel.

Elle suit la même grammaire d'agrégat que tous les conteneurs de pyrucast (cf.
[Agrégat](aggregate.md)) :

- **`SubEvolution`** — **une courbe tabulée** : une liste d'abscisses (triées,
  strictement croissantes) et la liste des valeurs en regard. Une valeur est un
  **scalaire**, un **`SubNodeField`** ([champ aux nœuds](node-field.md)) ou un
  **`SubElementField`** ([champ aux points de Gauss](element-field.md)) — toutes
  du même type, et pour les champs sur le **même support**. Son interpolation en
  `x` rend **une valeur**.
- **`Evolution`** — l'**agrégat** : une liste de `SubEvolution`, **une par
  zone**, exactement comme un `NodeField` agrège des `SubNodeField`. Son
  interpolation en `x` interpole chaque courbe, puis **regroupe** les
  sous-champs résultants en un `NodeField` / `ElementField`. Pour des scalaires,
  elle rend une **liste de flottants** (il n'existe pas d'agrégat de flottant).

```text
   Evolution (agrégat)
   ├── SubEvolution zone 0 ── abscisses [t₀, t₁, …] × valeurs [v₀, v₁, …]
   ├── SubEvolution zone 1 ── …
   └── …
```

## Interpolation linéaire

Entre deux échantillons encadrants `x_lo ≤ x ≤ x_hi`, le résultat est le mélange
`v_lo·(1−t) + v_hi·t` avec `t = (x − x_lo) / (x_hi − x_lo)`. Pour les champs, le
mélange **réutilise l'arithmétique de champs** (`map_all` + `merge_components`
précédé de `check_same_components`, cf.
[Champ](field.md)) : aucune logique numérique n'est dupliquée, et la
compatibilité des supports/composantes des deux champs encadrants est vérifiée à
ce moment-là. Une abscisse tombant exactement sur un échantillon rend la valeur
telle quelle.

## Types d'abscisse et d'ordonnée

Une évolution peut porter le **type physique** de ses axes :

- **`abscissa_type`** — le type de l'abscisse (p. ex. `"T"`, `"time"`). Valable
  pour toutes les évolutions. Il sert à **étiqueter** les tracés (axe X d'une
  courbe, slider d'un champ) et, lorsqu'on interpole un **champ**, à **choisir la
  composante** du champ à lire (voir ci-dessous).
- **`ordinate_type`** — le type de la valeur, pour les évolutions **scalaires
  uniquement** (p. ex. `"young"`). Il étiquette l'axe Y et **nomme la
  composante** produite quand on interpole un champ. Le donner sur une évolution
  de champs est une erreur (un champ a déjà ses propres composantes).

```python
{{#include ../../tests/python/test_doc_sauvegarde_evolution.py:subevolution}}
```

## Interpoler un champ (courbe de transfert)

Une évolution **scalaire** à une seule courbe s'utilise comme une **fonction de
transfert** `y = f(x)` : au lieu d'un scalaire, on lui passe un **champ** et elle
rend un **autre champ** de même support, où **chaque nœud / point de Gauss** est
l'interpolation de la valeur d'entrée sur la courbe.

- La composante lue dans le champ d'entrée est celle **nommée comme
  l'`abscissa_type`** — la **correspondance de type** est vérifiée : si le champ
  n'a pas de composante de ce nom, c'est une erreur.
- Le champ de sortie a **une seule composante**, nommée d'après l'`ordinate_type`
  (à défaut `"value"`).
- La politique hors-plage s'applique valeur par valeur, comme pour un scalaire.

```python
{{#include ../../tests/python/test_doc_sauvegarde_evolution.py:loi_materiau}}
```

Côté `Evolution` (agrégat), l'appel exige **une seule courbe scalaire** (sans
quoi le choix de la courbe serait ambigu) ; une `SubEvolution` s'interpole
directement.

## Politique hors plage

Chaque évolution **porte** une politique appliquée quand l'abscisse demandée
sort de l'intervalle tabulé `[x_min, x_max]` :

| Politique | Effet hors plage |
|---|---|
| `"error"` (défaut) | lève une erreur |
| `"clamp"` | renvoie la valeur de l'extrémité la plus proche (pas d'extrapolation) |
| `"extrapolate"` | prolonge linéairement avec le segment extrême |

La politique stockée peut être **surchargée à l'appel** :
`evol.interpolate(x, out_of_range="clamp")`.

## Construction

Deux voies, le constructeur haut niveau n'étant que du sucre au-dessus du
primitif bas niveau (motif `model.heat_conduction(fes)` / `SubModel` + `|`) :

- **temps-major (haut niveau)** — `Evolution([(t0, champ0), (t1, champ1), …])`
  avec un `NodeField` / `ElementField` / flottant **complet** par pas ; les
  champs entiers sont **transposés** en une courbe par zone (zones appariées
  entre pas par leur support, identique d'un pas à l'autre) ;
- **zone-major (bas niveau)** — construire chaque `SubEvolution` depuis sa liste
  `(abscisse, sous-champ)`, puis agréger avec `|`.

L'union `|` et le slicing **réinitialisent** la politique hors-plage de
l'agrégat à `"error"`.

## API Rust

```rust,ignore
{{#include ../../tests/doc_conteneurs.rs:evolution}}
```

## API Python

```python
{{#include ../../tests/python/test_doc_sauvegarde_evolution.py:interpolate}}
```

## Tracé

`evolution.plot(...)` visualise l'évolution : **courbe X-Y** pour des scalaires, **champ + slider** de
valeur tabulée pour des champs. Voir [Visualisation › Tracé d'une évolution](visualization.md#tracé-dune-évolution).

À défaut de `x_label` / `y_label` explicites, les étiquettes reprennent
l'`abscissa_type` (axe X d'une courbe, **slider** d'un champ) et l'`ordinate_type`
(axe Y d'une courbe).

```python
{{#include ../../tests/python/test_doc_sauvegarde_evolution.py:plot}}
```

## Place dans le modèle

`SubValue` est un enum de stockage **inline** (scalaire / `SubNodeField` /
`SubElementField`), comme `SubModel` l'est pour les physiques. `SubEvolution`
s'adresse par un `Handle<SubEvolution>` et sérialise ses valeurs en ligne
via le [trait `Portable`](memory-model.md) ; les courbes sont donc portables
comme tout autre objet. L'homogénéité du type de valeur est garantie
à la construction (au sein d'une courbe) et par `check_push` (entre zones d'un
même agrégat).
