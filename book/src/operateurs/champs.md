# Opérateurs sur les champs

Les modules `ops::node_field`, `ops::element_field`, `ops::coords`,
`ops::measure` et `ops::field` **dérivent et transforment** les
[champs](../field.md) — chacun nommé d'après ce qu'il produit :
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
| `positions(mesh, components=None)` | un `NodeField` portant les coordonnées des nœuds (`"X"`, `"Y"`, `"Z"`), une zone par sous-maillage. `None` ⇒ tous les axes présents dans la dimension du `Coords`. |
| `coords.set(field, components=None)` | **écrit** les coordonnées du `Coords` actif depuis un champ `"X"/"Y"/"Z"`. |
| `displace(field, components=None)` | **ajoute** un champ de déplacement aux coordonnées (chaque nœud distinct traité **une seule fois**). |

`positions` est le pont géométrie → champ : on en tire un `NodeField` qu'on
peut tracer, dériver, ou réinjecter après calcul (`displace` pour passer à la
configuration déformée).

## Restriction, fusion, consolidation

| Python | Effet |
|---|---|
| `restrict(field, mesh)` | restreint un `NodeField` aux nœuds de `mesh` (une zone par sous-maillage cible). Le support est le **nuage POI1 canonique** du sous-maillage (`to_poi1`, matérialisé une fois et **mis en cache**) : deux restrictions sur le **même** `mesh` partagent le support ⇒ `restrict(a,mesh) - restrict(b,mesh)` se soustrait directement, et s'aligne avec `K·restrict(f,mesh)` / `solve(K,f)`. Pour les ops élément (`gradient`, `integral`, …), repasser `mesh` à côté. `0.0` pour les nœuds non couverts ; nœuds hors de `mesh` abandonnés. Erreur si `mesh` n'est pas sur le même `Coords`. |
| `restrict_like(field, target)` | reprojette `field` sur le support **et** les composantes de `target`, zone par zone (mêmes slots que `target`) ⇒ le résultat se combine directement avec `target` par les opérateurs `+ - * /`. Nœuds/composantes de `field` absents de `target` abandonnés ; `0.0` si non couverts. Typiquement pour replier un incrément de `solve` (qui porte aussi les multiplicateurs) dans une solution courante. Erreur si `Coords` différents. |
| `merge(a, b)` | union structurelle de deux `NodeField`, consolidée — c'est l'**alias nommé** de `a \| b`. |
| `node_field.consolidate(field)` | fusionne les zones de **même support** (handle identique) en vérifiant la cohérence des valeurs partagées. |
| `element_field.consolidate(field)` | fusionne les zones d'une même `FiniteElementSpace` (union des composantes) — le pendant de `|`, qui les laisse côte à côte. |

La consolidation d'un `NodeField` est exactement la **finalisation** de l'union
`|` : après déduplication par handle, les zones définies sur le même `SubMesh`
deviennent une seule zone portant l'union de leurs composantes — une composante
définie par plusieurs zones doit y avoir la **même valeur** partout (sinon
erreur). Une vérification inter-supports finale impose qu'un nœud partagé par
des zones de supports différents s'accorde sur toute composante commune.
Le même `consolidate` accepte un `ElementField` (opération `element_field.consolidate`) :
les sous-champs d'une même `FiniteElementSpace` fusionnent en une zone portant
l'union de leurs composantes — utile pour réunir des zones matériau bâties par
physique sur une fespace partagée (`k` thermique + `E`/`nu`/`alpha` mécanique) en
un champ matériau unique lu par chaque physique.

## Bande de valeurs (`ge` / `gt` / `le` / `lt`)

`select` et `mask` partagent la même **bande de valeurs**, fixée par quatre
bornes de comparaison qui reprennent une pour une les opérateurs Python :

| Argument | Test | Opérateur |
|---|---|---|
| `ge` | `v ≥ ge` | `>=` |
| `gt` | `v > gt`  | `>`  |
| `le` | `v ≤ le` | `<=` |
| `lt` | `v < lt`  | `<`  |

On donne **au plus une** borne basse (`ge` *ou* `gt`) et **au plus une** borne
haute (`le` *ou* `lt`), et **au moins une** borne en tout ; une borne absente
laisse ce côté ouvert. Erreur si aucune borne, ou si la borne basse dépasse la
haute.

## Sélection par valeur

`select` extrait, **zone par zone**, la partie du support d'un champ dont les
valeurs tombent dans la bande. C'est un filtre par valeur qui renvoie un
`Mesh` — un sous-maillage par zone traitée (les zones restent séparées, rien
n'est moyenné ni fusionné).

| Python | Effet |
|---|---|
| `select(field, ge=None, gt=None, le=None, lt=None, components=None)` | sous-ensemble du support du champ respectant la bande, une zone à la fois. |

- **Type de champ.** Sur un `NodeField` / `SubNodeField`, on sélectionne les
  **nœuds** : chaque zone donne un sous-maillage **POI1** des nœuds retenus.
  Sur un `ElementField` / `SubElementField`, on sélectionne les **cellules** :
  chaque zone donne un sous-maillage de **son propre type d'élément**, et une
  cellule n'est retenue que si **tous** ses points de Gauss passent (la bande
  doit tenir tout le long de la cellule).
- **Composantes.** `components=None` teste **toutes** les composantes de chaque
  zone. Une liste `components` ne teste **que** ces composantes, et seulement
  sur les zones qui les portent **toutes** — une zone à laquelle il manque une
  composante demandée est **ignorée** (aucun sous-maillage produit).
- **Combinaison (ET).** Quand plusieurs composantes sont testées, elles sont
  combinées en **ET** : un nœud / une cellule n'est retenu que si **chaque**
  composante testée est dans la bande.

```python
# Nœuds dont la température est entre 20 et 80 °C (bornes inclusives).
chauds = pyrucast.mesh.select(temperature, ge=20.0, le=80.0)

# Cellules dont la contrainte de von Mises dépasse un seuil (borne basse seule).
critiques = pyrucast.mesh.select(sigma, ge=250e6, components=["vm"])
```

## Masque par valeur

`mask` garde la **structure exacte** du champ (mêmes zones, même support, mêmes
composantes) et se contente de réécrire les valeurs : `1.0` là où la bande
tient, `0.0` sinon — **composante par composante** (le `MASQUE` de Cast3M). Le
résultat est donc du **même type** que l'entrée et se multiplie terme à terme
avec elle. Un `NodeField` est masqué par nœud, un `ElementField` par point de
Gauss.

| Python | Effet |
|---|---|
| `mask(field, ge=None, gt=None, le=None, lt=None, components=None)` | champ `0/1` de même structure que l'entrée. |

- **Pas de ET entre composantes** (contrairement à `select`) : chaque valeur est
  testée pour elle-même.
- **Composantes.** `components=None` teste toutes les composantes. Une liste
  `components` ne teste **que** celles-ci ; les autres restent à `1.0` (neutre
  pour le produit), et une zone à laquelle il manque une composante demandée
  reste tout à `1.0`.

```python
# Remet à zéro les valeurs négatives d'un champ, composante par composante.
positif = champ * champ.mask(ge=0.0)

# Sucre : les comparaisons construisent directement un masque.
positif = champ * (champ >= 0.0)  # même chose
chauds = temperature > 80.0  # NodeField 0/1
```

Les opérateurs `>=`, `>`, `<=`, `<` sur un champ (`NodeField`, `SubNodeField`,
`ElementField`, `SubElementField`) renvoient le masque correspondant contre le
scalaire de droite. `==` / `!=` gardent leur sens Python habituel (identité).

> Exemple complet et exécutable : `examples/field_mask.py` (lancer avec
> `python examples/field_mask.py` après `maturin develop`).

## Extraction et renommage de composantes (`EXCO`)

Deux opérateurs travaillent sur le **jeu de composantes** d'un champ, sans
toucher au support ni aux valeurs — l'équivalent de `EXCO` de Cast3M. Tous deux
acceptent les quatre saveurs (`NodeField`, `SubNodeField`, `ElementField`,
`SubElementField`) et renvoient la même saveur.

| Python | Effet |
|---|---|
| `filter_components(field, components)` | ne garde que les composantes nommées, zone par zone. `components` est un **nom** (`str`) ou une **liste** de noms — typiquement le résultat de `model.primal_vars()`. |
| `rename_component(field, old, new)` | renomme la composante `old` en `new` (métadonnée seule, aucune valeur déplacée). |

`filter_components` traite chaque zone indépendamment :

- une zone ne portant **aucune** des composantes demandées est **abandonnée** ;
- une zone ne portant **que** des composantes demandées (rien à retirer) voit
  son sous-champ **partagé tel quel** (handle copié, pas de duplication) ;
- une zone mixte est reconstruite sur le même support avec les seules
  composantes demandées, dans **son propre** ordre.

`components` peut être un **sur-ensemble** des composantes du champ (les noms
absents sont ignorés) : passer `model.primal_vars()` à un résultat de `solve`
pour en retirer les inconnues duales (multiplicateurs de Lagrange) est l'usage
visé. Erreur si aucune zone ne porte l'une des composantes demandées.

`rename_component` laisse **inchangée** (handle partagé) toute zone ne portant
pas `old`. Erreur si aucune zone ne porte `old`, ou si une zone concernée a déjà
une composante nommée `new`.

```python
# Retire les multiplicateurs de Lagrange d'un résultat de solve.
u = solution.filter_components(model.primal_vars())

# Renomme une composante avant export.
export = u.rename_component("u_x", "DX")
```

**Sucre d'indexation** (façon pandas/numpy) : sur **les quatre saveurs**
(`NodeField`, `SubNodeField`, `ElementField`, `SubElementField`), une clé
**chaîne** ou **liste de chaînes** appelle `filter_components` et renvoie la
même saveur. Les autres clés gardent leur sens : `int`/`slice` → accès aux
zones sur un agrégat ; le tuple d'accès à une valeur sur un sous-champ
(`sub[node, "UX"]`, `sub[cell, gauss, "E"]`) est inchangé.

```python
ux = champ["u_x"]  # == filter_components(champ, "u_x")
depl = champ[["u_x", "u_y"]]  # == filter_components(champ, ["u_x", "u_y"])
zone = champ[0]  # inchangé : la zone (SubNodeField)
val = champ[0][node, "u_x"]  # inchangé : la valeur au nœud
```

L'accesseur `champ.components()` (présent sur les quatre saveurs) donne la
liste des composantes, d'où l'idiome **« reprojeter `u1` sur les composantes
de `u2` »** :

```python
u = u1[u2.components()]  # u1 réduit au jeu de composantes de u2
```

Côté Rust, `filter_components` / `select_components` acceptent indifféremment
un `&str`, un tableau `["u_x", "u_y"]` ou un `Vec<String>` (trait
`IntoComponentNames`) — donc `field.filter_components(model.primal_vars())`
passe directement.

## Dérivation géométrique (vers les points de Gauss)

Ces opérateurs ne dépendent **que** de l'espace EF et du champ — aucune
physique. Ils produisent l'`ElementField` que le
[comportement](comportement.md) (`integrate_behavior`) consomme ensuite. Ils
partagent tous le même moteur parallèle : le driver `nodal_pointwise`
(déterministe bit-à-bit, cf. [Parallélisme](../developper/parallelisme.md)),
pendant nodal de `integrate_pointwise`.

### `interp_to_gauss(field, fespace)` → `ElementField`

Interpole un champ **nodal** vers les points de Gauss (valeurs, pas dérivées) :

\\[
f(\xi_g) = \sum_i f_i\, N_i(\xi_g).
\\]

Le résultat porte les **mêmes composantes** que l'entrée, une valeur par
`(cellule, point de Gauss)`. C'est le pendant « valeurs » de `gradient`
(direction nœuds → Gauss du `CHAN` de Cast3M) : typiquement pour porter une
température nodale aux points de Gauss avant `thermal_strain`.

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

Sur un sous-espace [axisymétrique](../coords.md#repère-de-révolution), une
quatrième composante `eps_zz` est ajoutée : la déformation **orthoradiale**
`ε_θθ = u_r / r`, que le gradient méridien ne peut pas exprimer (cf.
[Axisymétrie](../mecanique/elasticite.md#axisymétrie)).

### `thermal_strain(temperature, materials, fespace, t_ref)` → `ElementField`

Déformation thermique de libre dilatation (Cast3M `EPTH`), pour la
thermomécanique **non couplée** :

\\[
\varepsilon_{th} = \alpha\,(T - T_{ref})\,\big[\,1,1,(1),0,0,0\,\big].
\\]

`temperature` est un champ **par éléments** portant `"T"` (p. ex. produit par
`interp_to_gauss`) ; `alpha` est lu dans le champ matériau, où il voyage comme
composante **facultative** de l'élasticité (à côté de `E`/`nu`, cf.
[Élasticité](../mecanique/elasticite.md)). La sortie a **exactement la même
disposition** que `deformation` (composantes normales à `α·ΔT`, cisaillements
nuls — y compris l'orthoradiale `eps_zz` en axisymétrique, un solide de
révolution se dilatant aussi circonférentiellement), si bien que
`deformation(u, fespace) - thermal_strain(...)` donne la
déformation mécanique `ε(u) − ε_th`. Aucun couplage n'est fait ici :
l'utilisateur compose la charge thermique et la contrainte réelle
`σ = D:(ε − ε_th)` à partir des briques (`integrate_behavior`,
`internal_forces`) — cf. l'[exemple thermomécanique](../mecanique/elasticite.md#thermomécanique-non-couplée).

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
attenue = pyrucast.field.exp(temperature * -0.1)

# Magnitude d'un champ (combiné à l'arithmétique scalaire de champ).
amplitude = pyrucast.field.abs(signal)
```

## Réduction

Deux produits scalaires, à ne pas confondre — ils diffèrent par **ce qui est
réduit** et donc par le **type du résultat** :

| Python | Cast3M | Réduit | Résultat |
|---|---|---|---|
| `xty(x, y)` | `XTY` | tout (nœuds/points × composantes) | un `float` |
| `psca(x, y)` | `PSCA` | les composantes seules, nœud par nœud | un **champ** à une composante `"psca"` |

Les deux exigent la **même saveur** d'opérandes (`NodeField`, `SubNodeField`,
`ElementField`, `SubElementField`), alignent les composantes **par nom** (l'ordre
peut différer) et suivent la **même règle d'union** que l'arithmétique de
[Champ](../field.md) : la somme ne porte que sur les `(support, composante)`
**partagés** par les deux champs ; un support ou une composante d'un seul côté n'a
pas de vis-à-vis et **ne contribue pas**.

- `xty` (produit scalaire **global**, `dot` / `dot_field`) → un `float` ;
- `psca` (`pscal` / `pscal_field`) → un **champ** (composante `"psca"`), une zone
  par support partagé.

### `xty(x, y)` → `float`

Produit scalaire **global** des deux champs entiers :

\\[
x \cdot y = \sum_i \sum_c x_{i,c}\, y_{i,c},
\\]

la somme parcourant **toutes** les valeurs. Le résultat est un unique `float` :
le produit scalaire qui sert au calcul d'énergie (`F·u`), aux normes de résidu,
etc. L'addition flottante n'étant pas associative, le total dépend du nombre de
threads jusqu'au dernier ULP — comme le solveur, ce n'est pas reproductible
bit à bit.

```python
import pyrucast

# Énergie de déformation externe : travail des efforts nodaux dans le champ
# de déplacement (mêmes composantes, même maillage).
energie = pyrucast.measure.xty(forces, deplacements)
```

### `psca(x, y)` → champ (même saveur que les entrées)

Produit scalaire **nœud par nœud** (ou point par point) — réduction sur les
**composantes seules**, le support est conservé :

\\[
p_i = \sum_c x_{i,c}\, y_{i,c}.
\\]

Le résultat est un nouveau champ de la même saveur que les entrées, portant une
seule composante `"psca"` : la valeur du produit scalaire à chaque nœud. Chaque
sortie est écrite une fois (par nœud) ⇒ indépendant du nombre de threads.

```python
import pyrucast

# Norme au carré d'un champ vectoriel, nœud par nœud.
norme2 = pyrucast.field.psca(vitesse, vitesse)  # champ à une composante "psca"
```

### `integral(field, component, fespace=None)` → `float`

Intègre un champ sur son support par la **quadrature éléments finis**,
`∫_Ω f dΩ` — le total d'une composante (p.ex. la **résultante** d'une *densité*
de force distribuée) :

\\[
\int_\Omega f \, d\Omega \;=\; \sum_{\text{cell}} \sum_g f(\text{cell}, g)\, |J|_g\, w_g .
\\]

- sur un **`NodeField`** : les valeurs nodales sont relevées aux points de Gauss
  par les fonctions de forme, `∫ Σ_i f_i N_i dΩ` — `fespace` est **requis** ;
- sur un **`ElementField`** : les valeurs (déjà aux points de Gauss) sont
  intégrées directement — `fespace` est ignoré.

Comme `xty`, la somme flottante dépend du nombre de threads jusqu'au dernier ULP.
En interne, la réduction parallèle sur les cellules passe par le driver
`kernel::reduce_cells`.

```python
import pyrucast

# Résultante d'une densité de force surfacique f_y sur une plaque (via N_i).
r_y = pyrucast.measure.integral(densite, "f_y", fespace=fes)
# Mesure du domaine : ∫ 1 dΩ.
aire = pyrucast.measure.integral(champ_unite, "u", fespace=fes)
```

### Somme et `xtx`

Pour une **résultante de forces déjà nodales** (sortie de `internal_forces`,
réactions…), la résultante est une simple **somme par nœud** — exposée comme
méthode, à côté de `min` / `max` :

| Réduction | Réduit | Résultat |
|---|---|---|
| `field.min(comp)` / `field.max(comp)` | une composante | un `float` (exact) |
| `field.sum(comp)` | une composante (`Σ` nœuds/points) | un `float` |
| `xtx(field)` | toutes les valeurs au carré (`Σ v²`, `XTX`) | un `float` |
| `xtx(field, components=[…])` | seules ces composantes au carré | un `float` |

`sum` et `xtx` regroupent la somme en parallèle : dépendantes du nombre de
threads au dernier ULP (contrairement à `min` / `max`, exactes quel que soit
l'ordre).

Par défaut `xtx` somme **toutes** les composantes. En passant `components`, on
restreint la somme à celles-là (les autres sont ignorées) — utile pour mesurer
la norme d'un résidu sur un sous-jeu de degrés de liberté. Une composante
absente d'une zone y est simplement ignorée ; l'appel n'échoue que si **aucune**
zone ne porte l'une des composantes demandées.

```python
import pyrucast

# Résultante d'un champ de forces nodales, composante par composante.
rx = forces.sum("f_x")
ry = forces.sum("f_y")
# Norme du résidu au carré, pour un test de convergence.
r2 = pyrucast.measure.xtx(residu)
# Même norme, restreinte aux seules composantes de translation.
r2_uy = pyrucast.measure.xtx(residu, components=["f_y"])
```

## À venir

Le module est conçu pour accueillir d'autres dérivations sur le même patron
`(champ, espace EF) → champ` : projection L² vers les nœuds (`project_to_nodes`),
mesures non linéaires de déformation (Green-Lagrange). Elles arriveront avec les
premiers besoins.
