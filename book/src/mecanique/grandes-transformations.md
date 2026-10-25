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
son état, ses forces internes, sa conduction. Ce qui agit sur la **forme
actuelle** se calcule sur une configuration courante `x = X + u` : l'opérateur
de Newton, la pression qui suit la paroi, les échanges thermiques de la pièce
déformée. Le `Coords` les porte
toutes les deux ; on passe de l'une à l'autre en changeant d'indice, sans
recopier le maillage :

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

## La conduction

La conductivité donnée au matériau est la conductivité **lagrangienne** `K₀`,
celle de la matière dans ses axes de référence. Sur la pièce déformée, elle se
pousse en `k = J⁻¹ F K₀ Fᵀ`, comme `S` se pousse en `σ`. Écrite sur la
référence, la conduction n'a donc rien à pousser :
`∇T = F⁻ᵀ∇₀T` et `dv = J dV₀` donnent, au point,

```text
∫Ω ∇T · k ∇δT dv = ∫Ω₀ ∇₀T · K₀ ∇₀δT dV₀
```

et la capacité devient exacte, `ρ₀ c dV₀`. Deux conséquences :

- des **axes matériau tournent avec la pièce** : sous une rotation `R`,
  `k = R K₀ Rᵀ`, et la répartition de température d'une pièce tournée est celle
  de la pièce au repos ;
- une barre **étirée** à `λ` voit sa conductivité dans l'axe devenir `k λ` : la
  chaleur qui la traverse, à écart de température donné, est celle de la barre
  au repos.

La loi écrit le **flux nominal** `K₀ ∇₀T`, par unité de surface de référence ; le
flux de la pièce déformée s'en déduit par `q = J⁻¹ F Q₀`.

Les **échanges** (convection, rayonnement, flux imposés) agissent sur la surface
déformée : ils restent sur la configuration courante. L'opérateur réunit les
deux : la conduction assemblée sur la référence, les échanges assemblés sur la
configuration courante. Une matrice garde les valeurs de la configuration où
elle a été assemblée, et [la somme](../operateurs/assemblage.md#la-somme) les
additionne telles quelles.

Cast3M résout toute la thermique sur la géométrie courante, avec la
conductivité donnée, sans la pousser. Il range les axes d'une conduction
orientée dans le repère local de chaque élément (la tangente à la première
coordonnée de référence, au centre), qu'il recalcule sur la géométrie courante
à chaque assemblage : sous une rotation rigide, ses axes tournent avec la pièce
comme ici ; sous une déformation, ils suivent cette ligne de l'élément, et les
conductivités gardent leurs valeurs là où pyrucast les pousse.

## Dans le pas à pas

`stepping.step_by_step` lit la mesure sur les lois et suit cette voie de
lui-même : forces intérieures sur `P` et conduction en référence, pression
suiveuse, échanges thermiques et opérateur sur la configuration courante.
Ses résultats gardent la contrainte lagrangienne de la loi ; la Cauchy s'en
déduit comme plus haut, avec le déplacement enregistré. Voir
[Calcul pas à pas](../pas-a-pas.md#grands-déplacements). Les tests
`tests/python/test_stepping_large.py` y vérifient :

- une bande tournée de 90° par un bord tourne sans contrainte ;
- un quart de carré sous pression suiveuse garde une contrainte de Cauchy
  `−p I` exacte ;
- un cube parfaitement plastique en Hencky, étiré à `λ = 1,5`, tire
  `A₀ σ_y / λ` ;
- une bande étirée conduit la même chaleur qu'au repos, `k ΔT / L₀` ;
- une bande à conduction orientée, tournée, garde en chaque nœud la
  température du repos ;
- une bande refroidie par convection sur une face étirée à `λ` : la face est à
  `T₀ G_c / (G_c + G_h)`, avec `G_c = k H₀ / L₀` et `G_h = h λ H₀`, par les deux
  schémas thermiques non linéaire et linéaire.

## Correspondance Cast3M

| Cast3M | pyrucast |
|---|---|
| `MODE … 'EPSILON' 'LINEAIRE'` / `'QUADRATIQUE'` | `strain_measure="linear"` / `"green_lagrange"` |
| `LAGRANGIEN TOTAL` (loi en `(S, E)` sur `FOR0`) | la voie de pyrucast, pour toute loi |
| `CONF` (contraintes) | `stress_measure` |
| `FORM` avec un déplacement | `coords.displace_from` + `Coords.select` |
| `KSIG` | `matrix.geometric` |
| `PRES` sur la géométrie courante | `model.flux(…, "n")` évalué sur la configuration courante |
