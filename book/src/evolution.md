# Évolution (`Evolution` / `SubEvolution`)

Une **évolution** associe une série de **valeurs** à une **variable** (souvent
le temps, mais pas nécessairement) et **interpole linéairement** entre les
échantillons tabulés. C'est l'analogue de l'`EVOLUTIO` de Cast3M, généralisé :
l'abscisse n'est pas forcément le temps, elle peut compter **plusieurs
variables** (le `NUAGE` de Cast3M), et la valeur peut être un **champ** entier,
pas seulement un réel.

Mathématiquement, une courbe est une **application tabulée**
`f : ℝᵐ → ℝⁿ`, linéaire par morceaux : `m` est le nombre de variables de
l'abscisse (1 pour une courbe, davantage pour une table), `n` vaut 1 pour une
valeur réelle et le nombre de valeurs du champ pour un champ. L'appliquer à un
argument de `ℝ^(m·c)` — `c` points portant chacun ses `m` variables — donne un
résultat de `ℝ^(c·n)`. Les deux usages courants en sont des cas particuliers :
l'interpolation en temps (`m = 1`, `c = 1`, on obtient un champ) et la loi
matériau appliquée à un champ (`n = 1`, on obtient un champ, exactement comme
`cos` appliqué à un champ).

Elle suit la même grammaire d'agrégat que tous les conteneurs de pyrucast (cf.
[Agrégat](aggregate.md)) :

- **`SubEvolution`** — **une courbe tabulée** : une liste d'abscisses (triées,
  strictement croissantes ; des **tuples** pour une table, triés dans l'ordre
  lexicographique) et la liste des valeurs en regard. Une valeur est un
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

## Tables à plusieurs variables (`NUAGE`)

Une abscisse peut être un **tuple** : `((T, φ), valeur)`. La table se lit
**hiérarchiquement** — c'est la forme `FLOTTANT-EVOLUTION` du `NUAGE` de
Cast3M, mais saisie à plat, sans composer d'évolutions à la main :

1. les échantillons sont **groupés** par leur première variable (valeurs
   égales), puis chaque groupe par la deuxième, et ainsi de suite ; chaque
   groupe le plus profond est une courbe ;
2. en un point `(T, φ)`, on cherche les deux groupes de la première variable
   qui encadrent `T` ; chacun est lu de la même façon le long de `φ` ; puis on
   interpole linéairement entre les deux résultats.

Pour deux variables, avec `Tₐ ≤ T ≤ T_b` les groupes encadrants et `f_a`, `f_b`
leurs courbes en `φ` :

\[
f(T, \varphi) = (1-t)\, f_a(\varphi) + t\, f_b(\varphi),
\qquad t = \frac{T - T_a}{T_b - T_a}.
\]

Conséquences :

- les groupes **n'ont pas à partager leurs abscisses** : une table irrégulière,
  telle que viennent les données matériau (trois valeurs de `φ` mesurées à
  500 °C, deux à 20 °C), est acceptée ;
- sur une **grille complète**, on retrouve exactement l'interpolation
  **multilinéaire** ; une fonction linéaire en chaque variable séparément
  (`a + bT + cφ + dTφ`) est reproduite exactement, grille complète ou non ;
- un groupe où la variable suivante ne prend **qu'une valeur** est **constant**
  le long d'elle : une propriété mesurée en une seule valeur de `φ` à une
  température donnée ne dépend pas de `φ` à cette température ;
- la **première variable est le niveau extérieur** : sur une table irrégulière,
  l'ordre des variables change le résultat — c'est le prix de l'irrégularité
  (une triangulation du nuage le lèverait, au prix d'une triangulation en
  dimension `m` et d'un écart à l'interpolation bilinéaire sur grille) ;
- hors de la plage d'un niveau, la politique hors plage s'applique à ce niveau
  (voir plus bas) ; au plus quatre variables.

```python
{{#include ../../tests/python/test_doc_sauvegarde_evolution.py:table}}
```

## Types d'abscisse et d'ordonnée

Une évolution peut porter le **type physique** de ses axes :

- **`abscissa_type`** — le type de l'abscisse (p. ex. `"T"`, `"time"`), un
  **tuple de noms** pour une table (`("T", "phi")`), un par variable. Valable
  pour toutes les évolutions. Il sert à **étiqueter** les tracés (axe X d'une
  courbe, slider d'un champ) et, lorsqu'on interpole un **champ**, à **choisir
  les composantes** du champ à lire (voir ci-dessous). Chaque courbe d'un
  agrégat garde le sien.
- **`ordinate_type`** — le type de la valeur, pour les évolutions **scalaires
  uniquement** (p. ex. `"young"`). Il étiquette l'axe Y et **nomme la
  composante** produite quand on interpole un champ. Le donner sur une évolution
  de champs est une erreur (un champ a déjà ses propres composantes).

```python
{{#include ../../tests/python/test_doc_sauvegarde_evolution.py:subevolution}}
```

## Interpoler un champ (courbe de transfert)

Une évolution **scalaire** s'utilise comme une **fonction** appliquée à un
champ, à la manière de `cos` : au lieu d'un scalaire, on lui passe un **champ**
et elle rend un **autre champ** de même support, où **chaque nœud / point de
Gauss** est l'évaluation de la courbe en ses propres variables.

- Les composantes lues dans le champ d'entrée sont celles **nommées comme
  l'`abscissa_type`** (une par variable pour une table) — la **correspondance de
  type** est vérifiée : si le champ n'a pas de composante de ce nom, c'est une
  erreur.
- Chaque courbe écrit **une composante**, nommée d'après son `ordinate_type`
  (à défaut `"value"`) ; deux courbes qui écriraient la même sont refusées.
- La politique hors-plage s'applique valeur par valeur, comme pour un scalaire.

```python
{{#include ../../tests/python/test_doc_sauvegarde_evolution.py:loi_materiau}}
```

Côté `Evolution` (agrégat), **toutes les courbes** sont appliquées : `k`
courbes donnent un champ à `k` composantes, chacune lisant ses propres
variables. C'est ainsi qu'on évalue des **lois matériau à l'état** — l'analogue
du `VARI 'NUAG'` de Cast3M : `E(T)` et `σ_y(T, φ)` appliquées en un appel aux
points de Gauss, puis réunies au matériau constant. Il n'y a pas d'opérateur
« matériau » dédié : c'est une interpolation générique.

```python
{{#include ../../tests/python/test_doc_sauvegarde_evolution.py:lois_etat}}
```

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
  avec un `NodeField` / `ElementField` / flottant **complet** par pas (l'abscisse
  pouvant être un tuple pour une table) ; les
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
