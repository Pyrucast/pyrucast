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
se = pc.SubEvolution(
    [(0.0, 0.0), (100.0, 210e9)], abscissa_type="T", ordinate_type="young"
)
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
# Loi matériau E(T) : module d'Young fonction de la température.
loi = pc.Evolution(
    [(0.0, 0.0), (100.0, 210e9)], abscissa_type="T", ordinate_type="young"
)
young = loi.interpolate(temperature)  # temperature : NodeField de composante "T"
# young : NodeField de composante "young"
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
primitif bas niveau (motif `Model.heat_conduction(fes)` / `SubModel` + `|`) :

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
use pyrucast::containers::evolution::{SubEvolution, SubValue, OutOfRange, Evolution, Interpolated};

// Courbe scalaire X→Y : 0→10, 1→20.
let se = SubEvolution::new(
    vec![(0.0, SubValue::Scalar(10.0)), (1.0, SubValue::Scalar(20.0))],
    OutOfRange::Error,
).unwrap();
match se.interpolate(0.5, None).unwrap() {
    SubValue::Scalar(v) => assert_eq!(v, 15.0),
    _ => unreachable!(),
}
// Hors plage : Error (défaut) lève ; surcharge Clamp → extrémité.
assert!(se.interpolate(2.0, None).is_err());

// Agrégat scalaire → liste de flottants.
let e = Evolution::from_scalars(vec![(0.0, 10.0), (1.0, 20.0)], OutOfRange::Error).unwrap();
match e.interpolate(0.5, None).unwrap() {
    Interpolated::Scalars(v) => assert_eq!(v, vec![15.0]),
    _ => unreachable!(),
}

// Courbe de transfert typée : mapper un champ nœud par nœud.
let mut loi = Evolution::from_scalars(vec![(0.0, 0.0), (100.0, 210e9)], OutOfRange::Error).unwrap();
loi.set_abscissa_type(Some("T".into())).unwrap();     // composante lue
loi.set_ordinate_type(Some("young".into())).unwrap(); // composante produite
let young = loi.interpolate_node_field(&temperature, None).unwrap();
```

## API Python

```python
import pyrucast as pc

# Courbe scalaire (une SubEvolution).
se = pc.SubEvolution([(0.0, 10.0), (1.0, 20.0)])
print(se.interpolate(0.5))  # 15.0
print(se.interpolate(2.0, out_of_range="clamp"))  # 20.0 (sinon : erreur)

# Agrégat scalaire → liste de flottants.
e = pc.Evolution([(0.0, 10.0), (1.0, 20.0)])
print(e.interpolate(0.5))  # [15.0]

# Bas niveau : composition de courbes par zone avec `|`.
agg = pc.SubEvolution([(0.0, 1.0), (1.0, 2.0)]) | pc.SubEvolution(
    [(0.0, 3.0), (1.0, 4.0)]
)
print(agg.interpolate(0.5))  # [1.5, 3.5]

# Haut niveau temps-major : un NodeField complet par pas → NodeField interpolé.
ev = pc.Evolution([(0.0, champ_t0), (2.0, champ_t1)])
champ = ev.interpolate(1.0)  # NodeField à mi-chemin

# Courbe de transfert : passer un champ → champ (loi matériau E(T)).
loi = pc.Evolution(
    [(0.0, 0.0), (100.0, 210e9)], abscissa_type="T", ordinate_type="young"
)
young = loi.interpolate(temperature)  # composante "T" lue → composante "young"
```

## Tracé

`evolution.plot(...)` visualise l'évolution : **courbe X-Y** pour des scalaires, **champ + slider** de
valeur tabulée pour des champs. Voir [Visualisation › Tracé d'une évolution](visualization.md#tracé-dune-évolution).

À défaut de `x_label` / `y_label` explicites, les étiquettes reprennent
l'`abscissa_type` (axe X d'une courbe, **slider** d'un champ) et l'`ordinate_type`
(axe Y d'une courbe).

```python
e = pc.Evolution([(0.0, 10.0), (1.0, 20.0), (2.0, 5.0)])
e.plot(save="courbe.svg", x_label="temps", y_label="T")  # courbe scalaire
ev.plot(save="frame.png", frame=2)  # champ tabulé (un pas)
```

## Place dans le modèle

`SubValue` est un enum de stockage **inline** (scalaire / `SubNodeField` /
`SubElementField`), comme `SubModel` l'est pour les physiques. `SubEvolution`
réside dans le store (`Handle<SubEvolution>`) et sérialise ses valeurs en ligne
via le [trait `Persist`](memory-model.md) ; les courbes sont donc portables
comme tout autre objet. L'homogénéité du type de valeur est garantie
à la construction (au sein d'une courbe) et par `check_push` (entre zones d'un
même agrégat).
