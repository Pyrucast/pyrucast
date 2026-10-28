# Grandes transformations

Une loi de petites déformations — Hooke, von Mises, Mazars — lit `ε` et rend
`σ`. Quand les rotations ou les déformations deviennent grandes, elle doit lire
une **mesure** de déformation qui ne voie que ce qui déforme vraiment la
matière. pyrucast suit la voie de MFront : le **lagrangien total**. La loi lit
une mesure lagrangienne totale, rend sa contrainte duale, et tout son état reste
sur la configuration de référence. Rien n'est transporté d'une configuration à
l'autre.

## Choisir la mesure

La mesure est un **paramètre de construction** de la loi :

```python
{{#include ../../../tests/python/test_doc_grandes_transformations.py:modele}}
```

| `strain_measure` | la loi lit | elle écrit | domaine |
|---|---|---|---|
| `"linear"` (défaut) | `eps_*` : `ε = sym ∇u` | `sigma_*` : Cauchy | petites déformations, petites rotations |
| `"green_lagrange"` | `E_*` : `E = ½(FᵀF − I)` | `S_*` : Piola–Kirchhoff 2 | petites déformations, **grandes rotations** |
| `"hencky"` | `H_*` : `½ ln C` | `T_*` : sa duale | grandes déformations plastiques |

Chaque composante porte le nom de ce qu'elle contient : une loi en
Green–Lagrange ne lit pas des `eps_*` qui contiendraient `E`, elle lit `E_*`.
Une loi de petites déformations relie une déformation à sa duale sans dire
laquelle : la lire sur une autre mesure donne un **autre matériau**. Hooke sur
`E` est Saint-Venant–Kirchhoff ; Hooke sur `H` est l'élasticité de Hencky ; J2
sur `H` est le cadre de Miehe, Apel et Lambrecht. MFront fait ce choix dans la
loi (`@StrainMeasure`), code_aster à la mise en données (`DEFORMATION`).

Le paramètre existe pour `elasticity`, pour les dix lois de plasticité et pour
les trois lois d'endommagement. Deux combinaisons sont refusées : une mesure qui
n'est pas une entrée de loi (`gradient`, Cauchy–Green, Euler–Almansi, que
[`deformation`](../operateurs/champs.md)
sait pourtant produire) et Hencky en contraintes planes, où `ln C` demande
`C_zz`, que seule la loi connaît.

La dilatation thermique suit la mesure :
[`thermal_strain`](../operateurs/champs.md)
reçoit `measure` et donne l'étirement isotrope `1 + αΔT` exactement dans
chacune — `αΔT`, `½((1 + αΔT)² − 1)`, `ln(1 + αΔT)` — sous les noms que lit
la loi.

## Une rotation rigide ne déforme rien

```python
{{#include ../../../tests/python/test_doc_grandes_transformations.py:rotation}}
```

Pour une rotation `R`, `F = R` et `E = ½(RᵀR − I) = 0` : pas de déformation,
pas de contrainte, pas d'effort. La déformation linéarisée du même mouvement
vaut `ε = (cos θ − 1) I`, soit `−½` sur la diagonale pour 60° : une contrainte
fictive considérable alors que rien ne se déforme. C'est pourquoi les
structures élancées ont besoin d'une mesure finie même quand elles se déforment
peu.

## Les forces internes : `P = F S`

La contrainte que rend la loi n'est pas celle que lit l'équilibre. Les efforts
intérieurs sont le travail virtuel de `S` dans la variation de `E` :

\[
\delta E = \operatorname{sym}\!\left(F^{\mathsf T} \nabla_0 \delta u\right),
\qquad
f_{ia} = \int_{\Omega_0} P_{ab}\, \frac{\partial N_i}{\partial X_b}\, dV_0,
\qquad
P = F\,S .
\]

`δE` est la ligne du `B` en grandes transformations : `B` dépend de `F`, ce
qui fait de `internal_forces` l'adjoint de `strain` au point `u`. Avec `F = I`,
on retrouve le `B` habituel. En pratique, la loi rend `S`, on la convertit
en `P` (la contrainte nominale, **non symétrique**), et
[`internal_forces`](../operateurs/assemblage.md) applique son `Bᵀ` sur la
configuration de référence : c'est la même boucle qu'en petites déformations,
qui lit simplement un tenseur plein (`P_xx, P_xy, P_yx, P_yy`).

## Les mesures de contrainte : `stress_measure`

`stress_measure(stress, gradient, target)` convertit, point par point, une
contrainte d'une mesure à l'autre avec le gradient de la transformation `F`
(`deformation(u, fes, "gradient")`). La mesure d'entrée se reconnaît à ses
noms ; la cible est nommée par la contrainte qu'on veut :

| `target` | contrainte | à partir de `S` | duale de |
|---|---|---|---|
| `"cauchy"` (`sigma`) | `sigma_*`, vraie contrainte | `J⁻¹ F S Fᵀ` | — (eulérienne) |
| `"kirchhoff"` (`tau`) | `tau_*` | `F S Fᵀ` | — (eulérienne) |
| `"pk1"` (`P`) | `P_*`, non symétrique | `F S` | `F` |
| `"pk2"` (`S`) | `S_*` | `S` | `E` (Green–Lagrange) |
| `"hencky"` (`T`) | `T_*` | `S̃_ab / θ_ab` | `H = ½ ln C` (Hencky) |

```python
{{#include ../../../tests/python/test_doc_grandes_transformations.py:cauchy}}
```

**Hencky.** La duale `T` de `H = ½ ln C` se ramène à `S` par `S = T : ∂H/∂E`.
Dans la base propre de `C = FᵀF = Σ c_a N_a⊗N_a` (le tilde), cette dérivée agit
composante par composante :

\[
\tilde S_{ab} = \theta_{ab}\,\tilde T_{ab},
\qquad
\theta_{aa} = \frac{1}{c_a},
\qquad
\theta_{ab} = \frac{\ln c_a - \ln c_b}{c_a - c_b} \quad (a \neq b),
\]

la différence divisée du logarithme, qui tend vers `1/c` quand deux valeurs
propres se confondent (calculée alors par son développement, sans perte de
chiffres). En traction uniaxiale (`ν = 0`), la loi rend `T = E ln λ`, d'où
`S = T/λ²` et une contrainte nominale `P = E ln λ / λ` ; Saint-Venant–Kirchhoff
donne `P = λ E (λ² − 1)/2`. Les deux formes closes sont vérifiées par
`tests/large_transformations.rs`, avec la dualité énergie–efforts dans chaque
mesure.

C'est la moitié « contraintes » de `CONF` de Cast3M, entre la configuration de
référence et la courante. Le reste de `CONF` (transporter des déformations,
tourner par Jaumann) n'a pas lieu d'être : l'état ne change jamais de
configuration. La conversion est inversible : de la Cauchy, `"pk2"` rend `S`.

## Deux configurations

Le **corps** se calcule sur la configuration de référence : sa mesure, sa loi,
son état, ses forces internes. Ce qui agit sur la **forme
actuelle** se calcule sur une configuration courante `x = X + u` : l'opérateur
de Newton, la pression qui suit la paroi, la thermique de la pièce
déformée. Le contact, lui, se réapparie sur `X + u` sans changer de
configuration : [`update_contact`](../contraintes/contact.md#grands-glissements--mettre-le-contact-à-jour)
lit `u` et range un jeu rapporté à la référence. Le `Coords` les porte
toutes les deux ; on passe de l'une à l'autre en changeant d'indice, sans
recopier le maillage. [`in_config`](../coords.md#plusieurs-configurations) rend
la courante active le temps d'un bloc, puis rétablit celle d'avant, même si le
bloc échoue :

```python
{{#include ../../../tests/python/test_doc_grandes_transformations.py:configuration_courante}}
```

[`displace_from`](../operateurs/champs.md) pose la
configuration courante **en absolu**, `X + u`, à partir de la référence :
rappelée avec un autre `u`, elle repart de `X`, et rien ne s'accumule — ce
qu'exigent un sous-pas repris ou une itération accélérée. La configuration
active ne change pas.

Les deux membres de l'équilibre sont des **forces nodales**, des vecteurs en
axes globaux qui ne dépendent pas de la configuration où on les calcule :
`∫Ω₀ P:∇₀δv dV₀ = ∫Ω σ:∇ˢδv dv`. On peut donc calculer les forces intérieures
sur la référence et une charge suiveuse sur la configuration courante, puis les
soustraire nœud par nœud.

## La pression suiveuse

Une pression pousse sur la paroi **où elle est**. Le mode normal de
[`flux`](../operateurs/assemblage.md) — la ligne `"n"`, une densité `phi_n`
portée par la normale de la maille — lit la normale de la configuration
active : évaluée sur la configuration courante, la charge suit la paroi.

```python
{{#include ../../../tests/python/test_doc_grandes_transformations.py:pression}}
```

Pour un contour parcouru dans le sens trigonométrique, la normale `(t_y, −t_x)`
d'une arête sort de la matière : une pression `p` qui pousse sur la paroi
s'écrit `phi_n = −p`. Sur la plaque étirée, la paroi du bas mesure 2,4 et reçoit
`p · 2,4` vers le haut. Comme dans Cast3M, la pression n'a pas de raideur de
chargement : elle n'entre que dans le résidu, recalculée à chaque itéré.

## La rigidité géométrique

L'opérateur de Newton se construit sur la configuration courante avec les
fonctions existantes : la rigidité élastique `stiffness`, et la rigidité
géométrique `geometric` avec la contrainte de Cauchy. C'est l'opérateur de
Cast3M en grands déplacements, `K_e(GEOM2) + K_σ`. Deux propriétés le
justifient, que vérifient les tests de `tests/large_transformations.rs` :

- la rigidité géométrique **spatiale**, `∫Ω ∇N·σ·∇N dv` sur la configuration
  courante, est égale à la **lagrangienne**, `∫Ω₀ ∇₀N·S·∇₀N dV₀` :
  `∇N = F⁻ᵀ∇₀N`, `dv = J dV₀` et `σ = J⁻¹FSFᵀ` se compensent au point ;
- après une rotation rigide, la rigidité élastique évaluée sur la configuration
  courante est la tangente exacte des forces intérieures, pour un matériau
  isotrope. Hors rotation pure, elle en diffère d'un terme de l'ordre de la
  déformation, ce qu'un Newton modifié tolère : l'opérateur ne règle que la
  vitesse de convergence, la solution est fixée par le résidu.

K_σ ne change donc que la convergence, sauf là où elle donne la seule raideur
(un câble ou une membrane tendus, transversalement) et à l'approche du
flambement, où `K + K_σ` devient singulière.

## Les barres et les câbles

Une barre porte un effort le long de son axe, pas un tenseur. Construite avec
`strain_measure="green_lagrange"`, sa loi lit `E_*` et en prend la composante
le long de la direction de référence `t₀` de la maille : sur un segment, le
gradient est tangentiel, `F t₀ = (x_B − x_A)/L` est la corde courante rapportée
à la longueur au repos, et

\[
t_0 \cdot E\, t_0 = \tfrac12 (\lambda^2 - 1) = \frac{l^2 - L^2}{2L^2},
\]

exactement, quelle que soit la rotation. La loi écrit l'effort de
Piola–Kirchhoff 2 axial `S_n = E A (t₀·E t₀)`. `stress_measure` reconnaît une
barre à ce nom et en fait :

| `target` | composantes | valeur |
|---|---|---|
| `"pk1"` | `P_x, P_y(, P_z)` | `S_n F t₀`, le long de la corde courante |
| `"cauchy"` | `n` | `λ S_n`, l'effort vrai |
| `"pk2"` | `S_n` | inchangé |

Les forces intérieures d'une barre en Green–Lagrange lisent `P` sur la
référence : `−P` au premier nœud, `+P` au second. La rigidité géométrique lit
l'effort vrai `n` sur la configuration courante et vaut `n/l · I` : c'est le
terme de contrainte de la tangente exacte, `S_n/L₀ = n/l`, dans toutes les
directions, l'axiale comprise. La rigidité élastique sur la configuration
courante, `E A/l c⊗c`, diffère de la tangente exacte `E A λ²/L₀` d'un facteur
`λ³`, comme celle du milieu continu en traction uniaxiale : seule la vitesse de
convergence en dépend.

```python
{{#include ../../../tests/python/test_doc_grandes_transformations.py:barre}}
```

Hencky est refusé pour une barre : `t₀·½ ln C·t₀` ne vaut `ln λ` que tant que
`t₀` est une direction propre de `C`, ce qu'une barre qui tourne n'est plus.
Une barre tendue, un câble, n'a de raideur transversale que par `K_σ` : sans
elle, l'opérateur du pas à pas est singulier.

## La conduction

La conductivité donnée au matériau est celle de la matière **telle qu'elle
est** : déformer la pièce ne change pas ses valeurs. La thermique se calcule
donc sur la pièce déformée, avec la conductivité donnée — conduction, échanges
(convection, rayonnement, flux imposés) — et la loi écrit le flux réel,
`q = −k ∇T`. Une barre **étirée** à `λ` est plus longue : à écart de
température donné, la chaleur qui la traverse est `k ΔT A / (λ L₀)`.

Une conduction **orientée** porte ses axes dans le matériau (`V1*`, `V2*`),
donnés sur la référence. Sur la pièce déformée, ils tournent avec elle, de la
rotation `R` de la décomposition polaire `F = R U` : `k = R k₀ Rᵀ`, et la
répartition de température d'une pièce tournée est celle de la pièce au repos.
L'étirement `U` ne touche ni les axes ni les valeurs. `rotate_frame` tourne les
axes d'un champ de matériau, avec `F` calculé **sur la référence** :

```python
{{#include ../../../tests/python/test_doc_grandes_transformations.py:axes}}
```

La **capacité** seule reste sur la référence : la masse se conserve,
`ρ c dv = ρ₀ c dV₀`, si bien que la capacité de référence est exacte.
L'opérateur réunit la conduction et les échanges, assemblés sur la
configuration courante, et la capacité, assemblée sur la référence : une
matrice garde les valeurs de la configuration où elle a été assemblée, et
[la somme](../operateurs/assemblage.md#la-somme) les additionne telles quelles.

Cast3M résout aussi toute la thermique sur la géométrie courante, avec la
conductivité donnée. Il range les axes d'une conduction orientée dans le repère
local de chaque élément (la tangente à la première coordonnée de référence, au
centre), qu'il recalcule sur la géométrie courante à chaque assemblage : sous
une rotation rigide, ses axes tournent avec la pièce comme ici ; sous un
cisaillement, ils suivent cette ligne de l'élément, là où pyrucast suit la
rotation `R`.

## Dans le pas à pas

`stepping.step_by_step` lit la mesure sur les lois et suit cette voie de
lui-même : forces intérieures sur `P` et capacité thermique en référence,
pression suiveuse, thermique (axes matériau tournés) et opérateur sur la
configuration courante.
Ses résultats gardent la contrainte lagrangienne de la loi ; la Cauchy s'en
déduit comme plus haut, avec le déplacement enregistré. Voir
[Calcul pas à pas](../pas-a-pas.md#grands-déplacements). Les tests
`tests/python/test_stepping_large.py` y vérifient :

- une bande tournée de 90° par un bord tourne sans contrainte ;
- un quart de carré sous pression suiveuse garde une contrainte de Cauchy
  `−p I` exacte ;
- un cube parfaitement plastique en Hencky, étiré à `λ = 1,5`, tire
  `A₀ σ_y / λ` ;
- une bande étirée à `λ` conduit comme la bande plus longue qu'elle est, flux
  `k ΔT / (λ L₀)` ;
- une bande à conduction orientée, tournée, garde en chaque nœud la
  température du repos ;
- une bande refroidie par convection sur une face, étirée à `λ` en long et `μ`
  en travers : la face est à `T₀ G_c / (G_c + G_h)`, avec
  `G_c = k μ H₀ / (λ L₀)` et `G_h = h μ H₀`, par les deux schémas thermiques
  non linéaire et linéaire ;
- la même bande, isolée sauf sur sa face refroidie et de conductivité assez
  grande pour rester uniforme : chaque pas implicite donne
  `T ← T C / (C + G_h Δt)`, avec la capacité de référence `C = ρ c V₀`.
- un treillis de von Mises, poussé au sommet en pilotage automatique, suit
  `P(w) = E A ((h² − (h − w)²) (h − w)) / L³` à travers ses deux points
  limites, la taille des pas étant la flèche du sommet (crochet
  `automatic_measure`) ;
- une corde tendue, chargée en son milieu, prend la flèche de l'équilibre
  exact `2 S_n w / L = P` ; sans `K_σ`, l'opérateur est singulier ;
- un nœud tiré par une barre le long d'un sol rigide brisé, un plat puis une
  rampe, quitte le plat pour la rampe : il s'arrête où l'équilibre le long de
  la rampe le place, la réaction normale à la rampe — et non où un contact
  figé sur le plat l'aurait laissé.

## Correspondance Cast3M

| Cast3M | pyrucast |
|---|---|
| `MODE … 'EPSILON' 'LINEAIRE'` / `'QUADRATIQUE'` | `strain_measure="linear"` / `"green_lagrange"` |
| `LAGRANGIEN TOTAL` (loi en `(S, E)` sur `FOR0`) | la voie de pyrucast, pour toute loi |
| `CONF` (contraintes) | `stress_measure` |
| `FORM` avec un déplacement | `coords.displace_from` + `Coords.select` |
| `KSIG` | `matrix.geometric` |
| `BARR` en `GRANDS_DEPLACEMENTS` | `model.truss(fes, strain_measure="green_lagrange")` |
| `PRES` sur la géométrie courante | `model.flux(…, "n")` évalué sur la configuration courante |
| `RFCO` à chaque reconstruction de K | `model.update_contact`, appelé par le pas à pas |
