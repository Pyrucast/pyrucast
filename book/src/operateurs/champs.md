# Opérateurs sur les champs

Le module `ops::field` **dérive et transforme** les [champs](../field.md) :
coordonnées, restriction, fusion, et dérivations géométriques vers les points
de Gauss. Ces opérateurs croisent des conteneurs (maillage + champ, espace EF
+ champ) — ce sont donc des **fonctions libres**. L'arithmétique scalaire et
par composante, elle, reste **sur les types de champ** (cf. [Champ](../field.md)).

Les autres thèmes d'opérateurs ont leur propre page : construction du matériau
([Construction](construction.md)), [Assemblage](assemblage.md) (dont le
chargement réparti `flux`), [Comportement](comportement.md), [Solveur](solveur.md).

## Coordonnées et déplacement

| Python | Effet |
|---|---|
| `coordinates(mesh, components=None)` | un `NodeField` portant les coordonnées des nœuds (`"X"`, `"Y"`, `"Z"`), une zone par sous-maillage. `None` ⇒ tous les axes présents dans la dimension du `Coords`. |
| `set_coordinates(field, components=None)` | **écrit** les coordonnées du `Coords` actif depuis un champ `"X"/"Y"/"Z"`. |
| `displace(field, components=None)` | **ajoute** un champ de déplacement aux coordonnées (chaque nœud distinct traité **une seule fois**). |

`coordinates` est le pont géométrie → champ : on en tire un `NodeField` qu'on
peut tracer, dériver, ou réinjecter après calcul (`displace` pour passer à la
configuration déformée).

## Restriction, fusion, consolidation

| Python | Effet |
|---|---|
| `restrict(field, mesh)` | restreint un `NodeField` aux nœuds de `mesh` (une zone par sous-maillage cible ; `0.0` pour les nœuds non couverts ; nœuds hors de `mesh` abandonnés). Erreur si `mesh` n'est pas sur le même `Coords`. |
| `merge(a, b)` | union structurelle de deux `NodeField`, consolidée — c'est l'**alias nommé** de `a \| b`. |
| `consolidate(obj)` | dispatch par type : sur un `NodeField`, fusionne les zones de **même support** (handle identique) en vérifiant la cohérence des valeurs partagées ; sur un `Mesh`, fusionne les sous-maillages de même type. |

La consolidation d'un `NodeField` est exactement la **finalisation** de l'union
`|` : après déduplication par handle, les zones définies sur le même `SubMesh`
deviennent une seule zone portant l'union de leurs composantes — une composante
définie par plusieurs zones doit y avoir la **même valeur** partout (sinon
erreur). Une vérification inter-supports finale impose qu'un nœud partagé par
des zones de supports différents s'accorde sur toute composante commune.
(L'équivalent côté `ElementField` est `consolidate_element`.)

## Sélection par valeur

`select` extrait, **zone par zone**, la partie du support d'un champ dont les
valeurs tombent dans une bande `[min, max]`. C'est un filtre par valeur qui
renvoie un `Mesh` — un sous-maillage par zone traitée (les zones restent
séparées, rien n'est moyenné ni fusionné).

| Python | Effet |
|---|---|
| `select(field, min=None, max=None, components=None)` | sous-ensemble du support du champ respectant la bande, une zone à la fois. |

- **Type de champ.** Sur un `NodeField` / `SubNodeField`, on sélectionne les
  **nœuds** : chaque zone donne un sous-maillage **POI1** des nœuds retenus.
  Sur un `ElementField` / `SubElementField`, on sélectionne les **cellules** :
  chaque zone donne un sous-maillage de **son propre type d'élément**, et une
  cellule n'est retenue que si **tous** ses points de Gauss passent (la bande
  doit tenir tout le long de la cellule).
- **Bornes.** Au moins une de `min` / `max` doit être donnée (bornes
  **inclusives**) ; une borne absente laisse ce côté ouvert. Erreur si les deux
  sont `None`, ou si `min > max`.
- **Composantes.** `components=None` teste **toutes** les composantes de chaque
  zone. Une liste `components` ne teste **que** ces composantes, et seulement
  sur les zones qui les portent **toutes** — une zone à laquelle il manque une
  composante demandée est **ignorée** (aucun sous-maillage produit).
- **Combinaison (ET).** Quand plusieurs composantes sont testées, elles sont
  combinées en **ET** : un nœud / une cellule n'est retenu que si **chaque**
  composante testée est dans la bande.

```python
# Nœuds dont la température est entre 20 et 80 °C.
chauds = pyrucast.select(temperature, min=20.0, max=80.0)

# Cellules dont la contrainte de von Mises dépasse un seuil (borne basse seule).
critiques = pyrucast.select(sigma, min=250e6, components=["vm"])
```

## Dérivation géométrique (vers les points de Gauss)

Ces opérateurs ne dépendent **que** de l'espace EF et du champ — aucune
physique. Ils produisent l'`ElementField` que le
[comportement](comportement.md) (`integrate_behavior`) consomme ensuite.

### `gradient(field, fespace)` → `ElementField`

Gradient d'un champ nodal aux points de Gauss, cellule par cellule :

\\[
\nabla f = \sum_i f_i\, \nabla N_i \quad \text{évalué en chaque } \xi_g.
\\]

Une composante de sortie `<comp>_<axe>` par couple (composante d'entrée, axe).

### `deformation(u, fespace)` → `ElementField`

Déformation **linearisée** (petites déformations) d'un champ de déplacement :

\\[
\varepsilon = \tfrac{1}{2}\big(\nabla u + \nabla u^\top\big).
\\]

`u` doit porter exactement `space_dim` composantes (déplacement selon x, y, z).
Le résultat est le tenseur **symétrique** en convention **tenseur**
(`eps_xy = ½(∂u_x/∂y + ∂u_y/∂x)`, **pas** le cisaillement ingénieur `γ`), une
composante `eps_<ai><aj>` par entrée indépendante `i ≤ j`. C'est l'entrée du
comportement de l'[élasticité](../mecanique/elasticite.md).

### `divergence(field)` → `NodeField`

Divergence **faible** (consistante) d'un champ vectoriel par éléments — l'adjoint
de `gradient` :

\\[
d_i = \int_\Omega \nabla N_i \cdot F\, d\Omega
\approx \sum_{\text{cell}} \sum_g (\nabla N_i \cdot F)\big|_g\, |J|_g\, w_g,
\\]

accumulé par nœud. C'est l'opérateur `Bᵀ`, transposé du gradient : il vérifie
`⟨∇f, F⟩ = ⟨f, div F⟩`. Le champ d'entrée doit porter exactement `space_dim`
composantes (`F_x, F_y, F_z`) ; chaque sous-espace donne une zone de sortie à
une composante `"div"`. Ce sont des **quantités intégrées**, pas les valeurs
ponctuelles de `∇·F` (pas de projection L²).

### `beam_deformation(field, fespace)` → `ElementField`

Déformations de section d'une poutre de [Timoshenko](../mecanique/timoshenko.md)
à partir d'un champ `(w, theta)` : la **courbure** `κ = θ'` et la **distorsion
de cisaillement** `γ = w' − θ`. Les deux sont pris **constants par élément** —
`γ` échantillonné au **centre** (point réduit), ce qui élimine le cisaillement
parasite. Le champ doit porter les composantes `"w"` et `"theta"`. Le résultat
se donne au [comportement](comportement.md) pour obtenir les efforts de section.

## Maths élément par élément

Onze fonctions appliquent une fonction scalaire à **chaque valeur** d'un champ
et renvoient un **nouveau** champ du même type (style numpy). Elles acceptent
indifféremment les quatre saveurs de champ — `NodeField`, `SubNodeField`,
`ElementField`, `SubElementField` — et dispatchent par type.

| Python | Effet |
|---|---|
| `abs(field)` | valeur absolue |
| `sqrt(field)` | racine carrée (`nan` pour les négatifs) |
| `exp(field)` | exponentielle `eˣ` |
| `log(field)` | logarithme népérien (`-inf`/`nan` pour ≤ 0) |
| `log10(field)` | logarithme base 10 |
| `cos(field)` / `sin(field)` / `tan(field)` | trigonométrie (radians) |
| `sinh(field)` / `cosh(field)` / `tanh(field)` | trigonométrie hyperbolique |

Les résultats sont **non bornés**, comme en numpy : aucune protection sur le
domaine (`log` de ≤ 0 donne `-inf`/`nan`, `sqrt` d'un négatif donne `nan`). Ces
fonctions se combinent à l'arithmétique scalaire des champs (`f + s`, `f * s`,
cf. [Champ](../field.md)) pour bâtir des expressions par composante.

```python
import pyrucast

# Atténuation exponentielle d'un champ de température.
attenue = pyrucast.exp(temperature * -0.1)

# Magnitude d'un champ (combiné à l'arithmétique scalaire de champ).
amplitude = pyrucast.abs(signal)
```

## Réduction

### `xty(x, y)` → `float`

Produit scalaire de deux champs — l'opérateur `XTY`/`PSCA` de Cast3M :

\\[
x \cdot y = \sum_i x_i\, y_i,
\\]

la somme parcourant **toutes** les valeurs (nœuds/points × composantes). Les
deux opérandes doivent être de la **même saveur** (`NodeField`, `SubNodeField`,
`ElementField`, `SubElementField`), posés sur le **même support** (même
décomposition en zones), et porter le **même jeu de composantes** — alignées
**par nom**, l'ordre pouvant différer (sinon erreur, comme l'arithmétique
stricte de [Champ](../field.md)). Le résultat est un unique `float` : le produit
scalaire qui sert au calcul d'énergie (`F·u`), aux normes de résidu, etc.

L'addition flottante n'étant pas associative, le total dépend du nombre de
threads jusqu'au dernier ULP — comme le solveur, ce n'est pas reproductible
bit à bit.

```python
import pyrucast

# Énergie de déformation externe : travail des efforts nodaux dans le champ
# de déplacement (mêmes composantes, même maillage).
energie = pyrucast.xty(forces, deplacements)
```

## À venir dans `ops::field`

Le module est conçu pour accueillir d'autres dérivations sur le même patron
`(champ, espace EF) → champ` : interpolation vers les Gauss (`interp_to_gauss`),
projection L² vers les nœuds (`project_to_nodes`), mesures non linéaires de
déformation (Green-Lagrange). Elles arriveront avec les premiers besoins.
