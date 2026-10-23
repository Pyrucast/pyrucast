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
| `coords.displace_from(field, reference, target, components=None)` | **pose** la configuration `target` en `X + u`, en absolu à partir de la configuration `reference` : rappelée, elle repart de `X`, rien ne s'accumule. La configuration active ne change pas (Cast3M `FORM`). |

`positions` est le pont géométrie → champ : on en tire un `NodeField` qu'on
peut tracer, dériver, ou réinjecter après calcul (`displace` pour passer à la
configuration déformée). En grandes transformations, la configuration courante
vit dans un jeu à part du même `Coords`, posé par `displace_from` : voir
[Grandes transformations](../mecanique/grandes-transformations.md).

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
{{#include ../../../tests/python/test_doc_ops_champs.py:select}}
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
{{#include ../../../tests/python/test_doc_ops_champs.py:mask}}
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
{{#include ../../../tests/python/test_doc_ops_champs.py:filter_rename}}
```

**Sucre d'indexation** (façon pandas/numpy) : sur **les quatre saveurs**
(`NodeField`, `SubNodeField`, `ElementField`, `SubElementField`), une clé
**chaîne** ou **liste de chaînes** appelle `filter_components` et renvoie la
même saveur. Les autres clés gardent leur sens : `int`/`slice` → accès aux
zones sur un agrégat ; le tuple d'accès à une valeur sur un sous-champ
(`sub[node, "UX"]`, `sub[cell, gauss, "E"]`) est inchangé.

```python
{{#include ../../../tests/python/test_doc_ops_champs.py:indexation}}
```

L'accesseur `champ.components()` (présent sur les quatre saveurs) donne la
liste des composantes, d'où l'idiome **« reprojeter `u1` sur les composantes
de `u2` »** :

```python
{{#include ../../../tests/python/test_doc_ops_champs.py:alignement}}
```

Côté Rust, `filter_components` / `select_components` acceptent indifféremment
un `&str`, un tableau `["u_x", "u_y"]` ou un `Vec<String>` (trait
`IntoComponentNames`) — donc `field.filter_components(model.primal_vars())`
passe directement.

## Dérivation géométrique (vers les points de Gauss)

Ces opérateurs ne dépendent **que** de l'espace EF et du champ — aucune
physique. Ils produisent l'`ElementField` que le
[comportement](comportement.md) (`integrate_behavior`) consomme ensuite ; au
niveau d'un modèle, [`strain`](#le-b-du-modèle--strainmodel-solution-materials--elementfield)
choisit pour chaque loi celui qui la nourrit. Ils
partagent tous le même moteur parallèle : le driver `nodal_pointwise`
(déterministe bit-à-bit, cf. [Parallélisme](../developper/parallelisme.md)),
pendant nodal de `element_pointwise`.

### `interp_to_gauss(field, fespace)` → `ElementField`

Interpole un champ **nodal** vers les points de Gauss (valeurs, pas dérivées) :

\\[
f(\xi_g) = \sum_i f_i\\, N_i(\xi_g).
\\]

Le résultat porte les **mêmes composantes** que l'entrée, une valeur par
`(cellule, point de Gauss)`. C'est le pendant « valeurs » de `gradient`
(direction nœuds → Gauss du `CHAN` de Cast3M) : typiquement pour porter une
température nodale aux points de Gauss avant `thermal_strain`.

### `gradient(field, fespace)` → `ElementField`

Gradient d'un champ nodal aux points de Gauss, cellule par cellule :

\\[
\nabla f = \sum_i f_i\\, \nabla N_i \quad \text{évalué en chaque } \xi_g.
\\]

Une composante de sortie `<comp>_<axe>` par couple (composante d'entrée, axe).

### `deformation(u, fespace, measure="linear")` → `ElementField`

Déformation d'un champ de déplacement aux points de Gauss, dans la **mesure**
choisie. `u` doit porter exactement `space_dim` composantes (déplacement selon
x, y, z). On note \\(H = \nabla u\\) et \\(F = I + H\\) :

| `measure` | formule | composantes |
|---|---|---|
| `"linear"` (défaut) | \\(\varepsilon = \tfrac12(H + H^\top)\\) | `eps_xx`, `eps_xy`, … |
| `"gradient"` (`"F"`) | \\(F = I + H\\) | toutes les `F_ij`, ligne par ligne |
| `"green_lagrange"` | \\(E = \tfrac12(F^\top F - I)\\) | `E_xx`, `E_xy`, … |
| `"right_cauchy_green"` (`"C"`) | \\(C = F^\top F\\) | `C_xx`, … |
| `"left_cauchy_green"` (`"b"`) | \\(b = F F^\top\\) | `b_xx`, … |
| `"hencky"` (`"log"`) | \\(\tfrac12 \ln C\\) | `H_xx`, … |
| `"euler_almansi"` (`"almansi"`) | \\(e = \tfrac12(I - b^{-1})\\) | `e_xx`, … |

Chaque mesure a son **préfixe** : on ne peut pas en lire une à la place d'une
autre. Une mesure symétrique produit une composante `<préfixe>_<ai><aj>` par
entrée indépendante `i ≤ j`. Le gradient `F`, qui n'est pas symétrique, les
produit toutes (9 en 3D, 4 en 2D). En 2D, la direction hors plan n'est pas
déformée (\\(F_{zz} = 1\\)) et ses entrées ne sont pas produites.

La mesure linéarisée est en convention **tenseur**
(`eps_xy = ½(∂u_x/∂y + ∂u_y/∂x)`, **pas** le cisaillement ingénieur `γ`).
C'est l'entrée du comportement de l'[élasticité](../mecanique/elasticite.md).
Les mesures finies valent zéro (ou \\(I\\) pour `C` et `b`) sous une
rotation rigide finie, contrairement à `eps`.

Sur un sous-espace [axisymétrique](../coords.md#repère-de-révolution), une
composante `<préfixe>_zz` est ajoutée. C'est l'entrée **orthoradiale**,
construite sur \\(H_{\theta\theta} = u_r / r\\), que le gradient méridien ne
peut pas exprimer : `eps_zz = u_r / r`, `F_zz = 1 + u_r / r`… (cf.
[Axisymétrie](../mecanique/elasticite.md#axisymétrie)).

Une transformation dégénérée (\\(\det F = 0\\)) donne des valeurs non
finies pour Hencky et Almansi : rien n'est vérifié au point de Gauss.

```python
{{#include ../../../tests/python/test_doc_ops_champs.py:mesures}}
```

### `thermal_strain(temperature, materials, fespace, t_ref, measure="linear")` → `ElementField`

Déformation thermique de libre dilatation (Cast3M `EPTH`), pour la
thermomécanique **non couplée** :

\\[
\varepsilon_{th} = \alpha\\,(T - T_{ref})\\,\big[\\,1,1,(1),0,0,0\\,\big].
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

`measure` est la mesure que lit la loi : la même dilatation y est donnée
exactement, `½((1 + αΔT)² − 1)` en Green–Lagrange (`E_*`), `ln(1 + αΔT)` en
Hencky (`H_*`) — cf. [Grandes transformations](../mecanique/grandes-transformations.md).

### `elastic_stress(model, strain, materials)` / `elastic_strain(model, stress, materials)` → `ElementField`

L'**opérateur élastique** d'un modèle aux points de Gauss, dans les deux sens
(Cast3M `ELAS`) :

\[
\sigma = \mathbb D : \varepsilon
\qquad\text{et}\qquad
\varepsilon = \mathbb D^{-1} : \sigma .
\]

`D` est l'opérateur avec lequel chaque loi **prédit** — pas la réponse de la
loi elle-même : la matrice de Hooke d'une élasticité dans sa symétrie
(isotrope, orthotrope, anisotrope), le `D` isotrope du prédicteur élastique
d'une plasticité, le `D` **non endommagé** que dégrade une loi
d'endommagement. Le matériau n'a besoin que des constantes élastiques (`E`,
`nu` en isotrope) : le `sigma_y` d'une plasticité n'y joue aucun rôle. C'est ce
dont un solveur incrémental a besoin pour transformer un incrément de
déformation imposé (thermique, donné) en la contrainte qu'il causerait si rien
ne plastifiait — la pseudo-force `F* = ∫ Bᵀ D Δε*` — ou pour ramener une
contrainte à la déformation élastique qui la porte sous d'autres modules
(`(D⁻¹ₙ₊₁ − D⁻¹ₙ) σₙ` quand le matériau varie avec la température).

L'entrée et la sortie en déformation sont en convention **tensorielle**,
disposition de `deformation` et `thermal_strain` ; en contraintes planes,
`elastic_strain` n'écrit pas `eps_zz`, que la loi plane ne lit pas. Une zone par
sous-modèle continu ; un sous-modèle sans opérateur élastique (conduction,
barre, poutre, contrainte) n'en produit pas. `elastic_strain` échoue là où `D`
est singulier (un module nul).

Deux fonctions plutôt qu'une qui lirait le sens sur les noms des composantes :
les deux sens ne font pas la même opération, et l'état que rend
`integrate_behavior` porte **à la fois** des `sigma_*` et des `eps_*`.

```python
{{#include ../../../tests/python/test_doc_ops_champs.py:elastique}}
```

### `stress_measure(stress, gradient, target)` → `ElementField`

Une contrainte convertie, point par point, d'une mesure à l'autre, avec le
gradient de la transformation `F` (`deformation(u, fespace, "gradient")`) :
Cauchy `sigma_*`, Kirchhoff `tau_*`, Piola–Kirchhoff 1 `P_*` (non symétrique)
et 2 `S_*`. La mesure d'entrée se reconnaît à ses noms, une par zone ; `target`
nomme celle qu'on veut (`"cauchy"`, `"kirchhoff"`, `"pk1"`, `"pk2"`). C'est
ainsi que la contrainte lagrangienne d'une loi en grandes transformations
devient la `P` que lisent les forces internes, ou la Cauchy des résultats —
la moitié « contraintes » de Cast3M `CONF`. Formules et exemple :
[Grandes transformations](../mecanique/grandes-transformations.md).

### `divergence(field)` → `NodeField`

Divergence **faible** (consistante) d'un champ vectoriel par éléments — l'adjoint
de `gradient` :

\\[
d_i = \int_\Omega \nabla N_i \cdot F\\, d\Omega
\approx \sum_{\text{cell}} \sum_g (\nabla N_i \cdot F)\big|_g\\, |J|_g\\, w_g,
\\]

accumulé par nœud. C'est l'opérateur `Bᵀ`, transposé du gradient : il vérifie
`⟨∇f, F⟩ = ⟨f, div F⟩`. Un tenseur se lit à ses noms : symétrique
(`A_xx, A_xy, A_yy`), ou **plein** s'il porte aussi `A_yx` — comme la
contrainte nominale `P` des grandes transformations. Le champ d'entrée doit porter exactement `space_dim`
composantes (`F_x, F_y, F_z`) ; la sortie porte une composante `"div"`, et un
nœud partagé par plusieurs sous-espaces reçoit la **somme** de leurs
contributions. Ce sont des **quantités intégrées**, pas les valeurs
ponctuelles de `∇·F` (pas de projection L²).

### `cell_integral(field, target)` → `ElementField` et `sum_to_nodes(field)` → `NodeField`

Deux briques par maille (Cast3M `INTG 'ELEM'` et `CHAN 'CHPO' 'SOMM'`) :

- `element_field.cell_integral` intègre chaque composante **maille par maille**,
  avec la quadrature du champ d'entrée :
  `v_e = Σ_g f(g)·|J|_g·w_g` (le `2πr` inclus en axisymétrique). Le résultat vit
  sur `target`, un espace **à un point par maille** du même maillage —
  `FiniteElementSpace(mesh, quadrature="REDUCED")`, un seul point au barycentre,
  le support `'GRAVITE'` de Cast3M ;
- `node_field.sum_to_nodes` rassemble un tel champ aux nœuds par **somme** :
  chaque nœud reçoit `Σ_{e ∋ i} v_e`, la somme des valeurs des mailles qui le
  touchent, zones confondues. Ce n'est pas un lissage : ni moyenne ni poids.

Intégrer `1` donne la mesure de chaque maille ; la sommer aux nœuds donne le
volume que représente chaque nœud — le poids d'un solveur incrémental pour
mettre ses résidus à une échelle comparable (l'accélération d'Anderson pondère
par `(Σ_e |Ω_e|)^{-1/2}`).

```python
{{#include ../../../tests/python/test_doc_ops_champs.py:par_maille}}
```

### `beam_deformation(field, fespace, material)` → `ElementField`

Déformations **généralisées** d'une poutre — [Bernoulli](../mecanique/bernoulli.md)
comme [Timoshenko](../mecanique/timoshenko.md) — à partir d'un champ de
déplacements et de rotations. Un opérateur pour les trois configurations, lues
sur la dimension du maillage :

| `Coords` | DDL lus | composantes produites |
|---|---|---|
| 1-D | `w, theta` | `kappa, gamma` |
| 2-D | `u_x, u_y, r_z` | `eps, kappa, gamma` |
| 3-D | six | `eps, kappa_y, kappa_z, torsion, gamma_y, gamma_z` |

C'est le produit `B · u`, avec le `B` de l'élément
(`models::beam::b_into`) — celui-là même dont la rigidité intègre `Bᵀ D B` et
dont les [forces internes](champs.md) intègrent le transposé. Les déformations
sont évaluées **à chaque point de Gauss** : la courbure varie le long d'une
travée non chargée (`M' = V`), seul le cisaillement est constant (`V' = 0`).

Le **matériau** est exigé, et c'est la signature honnête : `Φ = 12EI/(G·A_s·L²)`
décide de la distribution de courbure. Un matériau sans constantes de
cisaillement — celui d'une poutre de Bernoulli, qui ne demande ni `G` ni `A_s` —
signifie `Φ = 0`, si bien que le même opérateur sert les deux théories sans
qu'on ait à lui dire laquelle. Le résultat se donne au
[comportement](comportement.md) pour obtenir les efforts de section.

### `shell_deformation(field, fespace, model)` → `ElementField`

Déformations généralisées d'une [coque](../mecanique/coques.md), depuis les six
DDL `u_x…u_z, r_x…r_z`. `model` est la formulation, `"thick"` ou
`"kirchhoff"` : les lignes de membrane, de vrillage et de cisaillement sont
partagées, mais celles de **flexion** sont toute la différence entre les deux.

| formulation | composantes produites |
|---|---|
| `thick` | `eps_xx, eps_yy, eps_xy`, `kappa_xx, kappa_yy, kappa_xy`, `drill`, `gamma_xz, gamma_yz` |
| `kirchhoff` | les mêmes, sans les deux déformations de cisaillement |

Toutes dans le repère **local** de la facette. `drill` est le résidu de
vrillage `θ_z − ω_z`, dont l'effort conjugué est un moment que la rigidité
intègre comme les autres.

Le cisaillement transverse est **constant par élément**, échantillonné au point
réduit — le point même où la rigidité l'intègre. Ce n'est pas une
simplification : c'est l'intégration réduite qui empêche une coque mince de
bloquer, lue de l'autre côté. L'échantillonner ailleurs rapporterait une
déformation que l'élément ne porte pas.

## Le `B` du modèle : `strain(model, solution, materials)` → `ElementField`

Les producteurs ci-dessus connaissent un espace EF, pas une loi. `strain` est
leur pendant au niveau du **modèle** : il demande à chaque loi quel producteur
la nourrit, l'applique à **ses** inconnues sur **son** espace, et rend l'entrée
de toutes les lois du modèle — le `EPSI` de Cast3M, étendu à toutes les
physiques. Aussi en méthode : `model.strain(solution, materials)`.

| sous-modèle | producteur | zone rendue |
|---|---|---|
| élasticité, plasticité, endommagement, barre | `deformation` | `eps_*`, ou la mesure de la loi (`E_*`, `H_*`) |
| conduction, Fick | `gradient` | `grad_T_*`, `grad_c_<espèce>_*` |
| rayonnement | `interp_to_gauss` | `T` aux points |
| poutres Bernoulli, Timoshenko | `beam_deformation` | `eps`, `kappa*`, `gamma*`, `torsion` |
| coque | `shell_deformation` | `eps_*`, `kappa_*`, `drill`, `gamma_*` |

Chaque loi lit **ses** inconnues de `solution` par leur nom : une même solution
— déplacement, température, multiplicateurs — sert tout le modèle, et une
inconnue absente vaut zéro. Les sous-modèles sans loi (contraintes, charges,
échanges) ne produisent rien. `materials` n'est lu que par les poutres, dont la
répartition flexion / cisaillement dépend de `Φ = 12EI/(G·A_s·L²)`. Ce que la
loi déclare, c'est son producteur (`Behavior::strain_operator`, méthode
**requise** : une physique nouvelle ne peut pas l'oublier) ; l'opérateur tient
la vue de la solution une fois pour toutes les lois.

**`strain` et `internal_forces` sont transposés.** Les éléments finis portent
deux couples de variables duales : déplacement / force aux nœuds,
déformation / contrainte aux points de Gauss. Le travail se mesure des deux
côtés — `f·u` aux nœuds, `Σ_g w_g |J_g| σ:ε` aux points — et `Bᵀ W` est
l'adjoint de `B` pour ces deux produits :

\[
\sigma \cdot W (B u) = (B^\mathsf{T} W \sigma) \cdot u
\qquad \forall\, u,\ \sigma ,
\]

le principe des travaux virtuels écrit au niveau discret :

```text
   nœuds                         points de Gauss
   u      ───── strain (B) ─────▶  ε
                                   │  integrate_behavior (la loi)
   f_int  ◀── internal_forces ───  σ
                (Bᵀ W)
```

D'où `K = Bᵀ W D B`, symétrique si et seulement si `D` l'est, et, pour une loi
linéaire, `internal_forces(integrate_behavior(strain(u))) = K·u` à l'arrondi
près — ce que vérifie l'exemple. [`divergence`](#divergencefield--nodefield) est
le cas scalaire du même `Bᵀ`, et l'adjoint de `gradient`. En grandes
transformations, la transposition lie la *variation* `δE = sym(Fᵀ∇₀δu)` aux
efforts : `internal_forces` lit alors la contrainte nominale `P = F S`
(cf. [Grandes transformations](../mecanique/grandes-transformations.md)), et la
tangente gagne la rigidité géométrique. Elle ne tient plus pour un terme non
variationnel (advection, pression suiveuse, relation non associée), ni dès que
les deux côtés n'utilisent pas les mêmes points : c'est pourquoi
`shell_deformation` lit le cisaillement au point réduit où la rigidité
l'intègre.

**Un modèle couplé se passe de `consolidate`.** Deux lois sur un même maillage
(thermique et mécanique) donnent deux zones sur un même espace. Chaque
consommateur — `integrate_behavior` (entrée et état précédent),
`internal_forces`, `tangent`, `geometric` — prend la zone qui porte ce qu'il lit,
comme le matériau est déjà résolu par ses composantes.

**Pourquoi `strain` et pas `deformation`.** En anglais, *deformation* désigne la
transformation elle-même (`F`, et les tenseurs `C`, `b`, qui valent `I` en
mouvement rigide) ; *strain* en est une **mesure**, qui s'annule en mouvement
rigide (`ε`, `E`, `ln U`), et le mot consacré pour les déformations
généralisées des poutres et coques. L'opérateur géométrique, qui produit les
deux, garde donc `deformation` ; ce que lit une loi est un *strain*. Le
gradient thermique en est un abus assumé, comme les `internal_forces` d'un
problème thermique.

```python
{{#include ../../../tests/python/test_doc_ops_champs.py:strain_modele}}
```

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
{{#include ../../../tests/python/test_doc_ops_champs.py:maths}}
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
x \cdot y = \sum_i \sum_c x_{i,c}\\, y_{i,c},
\\]

la somme parcourant **toutes** les valeurs. Le résultat est un unique `float` :
le produit scalaire qui sert au calcul d'énergie (`F·u`), aux normes de résidu,
etc. L'addition flottante n'étant pas associative, le total dépend du nombre de
threads jusqu'au dernier ULP — comme le solveur, ce n'est pas reproductible
bit à bit.

```python
{{#include ../../../tests/python/test_doc_ops_champs.py:xty}}
```

### `psca(x, y)` → champ (même saveur que les entrées)

Produit scalaire **nœud par nœud** (ou point par point) — réduction sur les
**composantes seules**, le support est conservé :

\\[
p_i = \sum_c x_{i,c}\\, y_{i,c}.
\\]

Le résultat est un nouveau champ de la même saveur que les entrées, portant une
seule composante `"psca"` : la valeur du produit scalaire à chaque nœud. Chaque
sortie est écrite une fois (par nœud) ⇒ indépendant du nombre de threads.

```python
{{#include ../../../tests/python/test_doc_ops_champs.py:psca}}
```

### `integral(field, component, fespace=None)` → `float`

Intègre un champ sur son support par la **quadrature éléments finis**,
`∫_Ω f dΩ` — le total d'une composante (p.ex. la **résultante** d'une *densité*
de force distribuée) :

\\[
\int_\Omega f \\, d\Omega \\;=\\; \sum_{\text{cell}} \sum_g f(\text{cell}, g)\\, |J|_g\\, w_g .
\\]

- sur un **`NodeField`** : les valeurs nodales sont relevées aux points de Gauss
  par les fonctions de forme, `∫ Σ_i f_i N_i dΩ` — `fespace` est **requis** ;
- sur un **`ElementField`** : les valeurs (déjà aux points de Gauss) sont
  intégrées directement — `fespace` est ignoré.

Comme `xty`, la somme flottante dépend du nombre de threads jusqu'au dernier ULP.
En interne, la réduction parallèle sur les cellules passe par le driver
`kernel::reduce_cells`.

```python
{{#include ../../../tests/python/test_doc_ops_champs.py:integral}}
```

### Somme et `xtx`

Pour une **résultante de forces déjà nodales** (sortie de `internal_forces`,
réactions…), la résultante est une simple **somme par nœud** — exposée comme
méthode, à côté de `min` / `max` :

| Réduction | Réduit | Résultat |
|---|---|---|
| `field.min(comp)` / `field.max(comp)` | une composante | un `float` (exact) |
| `field.min()` / `field.max()` | toutes les composantes | un `float` (exact) |
| `field.sum(comp)` | une composante (`Σ` nœuds/points) | un `float` |
| `xtx(field)` | toutes les valeurs au carré (`Σ v²`, `XTX`) | un `float` |
| `xtx(field, components=[…])` | seules ces composantes au carré | un `float` |

`sum` et `xtx` regroupent la somme en parallèle : dépendantes du nombre de
threads au dernier ULP (contrairement à `min` / `max`, exactes quel que soit
l'ordre).

Appelées **sans argument**, `min` et `max` lisent le champ comme la liste plate
de ses valeurs — toutes composantes confondues, et, au niveau agrégat, toutes
zones confondues. C'est l'esprit de `xtx`, et la même mise en garde : sur un
champ dont les composantes ne portent pas la même unité (`sigma_xx` à côté de
`sigma_xy`), la réponse est « la plus petite valeur là-dedans », pas une
grandeur physique — il faut alors nommer la composante. Pour la liste des
extremums composante par composante, une compréhension suffit :
`{c: f.min(c) for c in f.components()}`.

Par défaut `xtx` somme **toutes** les composantes. En passant `components`, on
restreint la somme à celles-là (les autres sont ignorées) — utile pour mesurer
la norme d'un résidu sur un sous-jeu de degrés de liberté. Une composante
absente d'une zone y est simplement ignorée ; l'appel n'échoue que si **aucune**
zone ne porte l'une des composantes demandées.

```python
{{#include ../../../tests/python/test_doc_ops_champs.py:sommes}}
```

## À venir

Le module est conçu pour accueillir d'autres dérivations sur le même patron
`(champ, espace EF) → champ` : projection L² vers les nœuds (`project_to_nodes`),
mesures non linéaires de déformation (Green-Lagrange). Elles arriveront avec les
premiers besoins.
