# Élasticité orthotrope et anisotrope

## Introduction

L'[élasticité linéaire](elasticite.md) suppose le matériau **isotrope** : deux
constantes, aucune direction privilégiée. Beaucoup de matériaux n'obéissent pas à
cette hypothèse — un composite tissé, un bois, une tôle laminée, un monocristal
ont des raideurs différentes selon la direction.

pyrucast traite cela comme un **axe à part entière**, la *symétrie matériau*,
porté par le même modèle `elasticity` et **orthogonal** à l'hypothèse cinématique
(contraintes planes, déformations planes, axisymétrie, massif) :

| symétrie | constantes indépendantes | ce qu'elle décrit |
|---|---|---|
| `isotropic` | 2 | pas de direction privilégiée |
| `orthotropic` | 9 | trois plans de symétrie orthogonaux |
| `anisotropic` | 21 | le cas général |

C'est la convention de Cast3M, où `ISOTROPE` / `ORTHOTROPE` / `ANISOTROPE`
qualifie le **matériau** d'une formulation plutôt que de nommer un modèle
différent. Les degrés de liberté ne changent donc pas : déplacement
`u_x, u_y(, u_z)` en primal, force nodale `f_x, …` en dual, exactement comme en
isotrope. Seule la matrice de Hooke `D` change.

## Équations continues résolues

Les mêmes qu'en élasticité linéaire — équilibre `∇·σ + f = 0` en petites
déformations, avec `ε = ½(∇u + ∇uᵀ)`. C'est la **loi de comportement** qui se
généralise :

\\[
\sigma_{ij} = C_{ijkl}\\,\varepsilon_{kl}
\\]

où \\( C \\) est le tenseur d'élasticité d'ordre 4. Ses symétries — mineures
\\( C_{ijkl} = C_{jikl} = C_{ijlk} \\), qui viennent de celles de
\\( \sigma \\) et \\( \varepsilon \\), et **majeure**
\\( C_{ijkl} = C_{klij} \\), qui vient de l'existence d'un potentiel élastique
\\( W = \tfrac12\\,\varepsilon : C : \varepsilon \\) — le réduisent d'un tenseur à
81 composantes à une matrice \\( 6\times6 \\) symétrique en notation de Voigt,
soit **21 constantes** dans le cas général.

L'**orthotropie** est le cas où le matériau possède trois plans de symétrie
orthogonaux. Dans ses axes propres, la souplesse `S = C⁻¹` se découple : les
termes normaux ne sont couplés qu'entre eux, et chaque cisaillement est isolé.

\\[
S =
\begin{bmatrix}
1/E_1 & -\nu_{21}/E_2 & -\nu_{31}/E_3 & & & \\\\
-\nu_{12}/E_1 & 1/E_2 & -\nu_{32}/E_3 & & & \\\\
-\nu_{13}/E_1 & -\nu_{23}/E_2 & 1/E_3 & & & \\\\
 & & & 1/G_{23} & & \\\\
 & & & & 1/G_{13} & \\\\
 & & & & & 1/G_{12}
\end{bmatrix}
\\]

Les relations de réciprocité \\( \nu_{ji}/E_j = \nu_{ij}/E_i \\) rendent la
matrice symétrique, d'où **neuf** constantes seulement : trois modules d'Young,
trois coefficients de Poisson, trois modules de cisaillement. Le bloc normal et
les trois cisaillements sont **découplés**, ce qui est la définition même de
l'orthotropie : une traction selon un axe propre ne produit aucun cisaillement.

> **Attention** — un jeu de constantes n'est pas physique par construction : `S`
> doit rester définie positive, ce qui impose `ν_ij² < E_i/E_j`. pyrucast le
> vérifie en inversant `S` et **erronne** si elle est singulière, plutôt que
> d'assembler une raideur non définie positive en silence.

## Le repère d'orthotropie

Les constantes sont données dans les **axes matériau**, qui ne coïncident pas
avec les axes globaux. Il faut donc dire où ils pointent.

pyrucast suit Cast3M et les décrit par des **vecteurs**, pas par des angles
d'Euler (`MATE 'DIRECTION' V1 V2`). Ils voyagent dans le champ matériau comme
n'importe quel autre coefficient :

| espace | composantes | signification |
|---|---|---|
| 2-D | `V1X`, `V1Y` | le premier axe matériau ; le deuxième est sa normale dans le plan |
| 3-D | `V1X…V1Z`, `V2X…V2Z` | les deux premiers axes ; le troisième est `V1 × V2` |

Ils sont **orthonormalisés** en interne (Gram-Schmidt) : `V2` n'a besoin d'être
que *grossièrement* perpendiculaire à `V1`, c'est le plan qu'ils engendrent qui
compte. Des vecteurs plutôt que des angles, parce qu'il n'y a aucune convention à
retenir, aucun cas de blocage de cardan — et surtout parce que le repère varie
alors **naturellement d'une maille à l'autre** (un composite bobiné, une pièce
courbe), en passant par le canal matériau existant.

Un `V1` nul, ou un `V2` parallèle à `V1`, est un repère dégénéré : il est
**signalé**, jamais complété arbitrairement.

## Forme discrétisée

La chaîne est celle de l'élasticité — `K_e = Σ_g Bᵀ D B |J| w`, avec le **même**
opérateur `B`. Ce qui change tient en trois étapes, faites une fois par maille :

1. construire `D` **dans les axes matériau**, où l'orthotropie est diagonale ;
2. le **tourner** vers les axes globaux ;
3. le **réduire** au modèle cinématique (bloc `[xx, yy, xy]` en déformations
   planes, sa condensation statique sur `ε_zz` en contraintes planes, le bloc
   `[rr, zz, θθ, rz]` en axisymétrie, le `6×6` complet en massif).

La rotation passe par le **tenseur d'ordre 4**, pas par une matrice de Bond
`6×6` :

\\[
C'_{pqrs} = R_{pi}\\,R_{qj}\\,R_{rk}\\,R_{sl}\\;C_{ijkl},
\qquad
R = \big[\\,V_1\ \ V_2\ \ V_1 \times V_2\\,\big],
\\]

`R` étant la rotation qui porte les axes matériau sur les axes globaux.

C'est un choix délibéré. En cisaillement **ingénieur** (`γ = 2ε`), le passage
Voigt ↔ tenseur ne porte **aucun facteur** — `C_ijkl = D[voigt(i,j)][voigt(k,l)]`
— et la rotation d'ordre 4 s'écrit sans la moindre convention à mémoriser. Le
coût, quelques centaines de multiplications par maille, est négligeable devant
l'assemblage, et il achète l'élimination de toute une famille d'erreurs
d'indices et de facteurs 2. L'isotropie, elle, **court-circuite** ce chemin et
garde ses formes fermées : les calculs isotropes existants conservent leurs
nombres exacts.

## Variables et matériau

Primales `u_x, u_y(, u_z)`, duales `f_x, f_y(, f_z)` — inchangées.

| symétrie | composantes matériau requises |
|---|---|
| `isotropic` | `E`, `nu` |
| `orthotropic` | `E_1, E_2, E_3`, `nu_12, nu_13, nu_23`, `G_12, G_13, G_23` + le repère |
| `anisotropic` | `C_11 … C_66` (21, triangle supérieur) + le repère |

Les trois contrats sont **disjoints**. Ce n'est pas un détail : l'assembleur
résout la zone matériau d'une physique par l'ensemble des composantes qu'elle
exige, si bien qu'une zone isotrope et une zone orthotrope peuvent partager un
maillage sans consolidation explicite.

Les constantes anisotropes sont nommées d'après le triangle supérieur de la
matrice de Voigt, dans l'ordre de ce dépôt `[xx, yy, zz, yz, xz, xy]` : `C_11`,
`C_12`, …, `C_16`, `C_22`, …, `C_66`. Ainsi `C_44` est la raideur en `yz`, `C_66`
celle en `xy`.

Même en 2-D, les neuf constantes orthotropes sont exigées : la raideur
hors-plan intervient en déformations planes et en axisymétrie, et le tenseur
complet est de toute façon construit avant d'être réduit.

Le comportement (`COMP`) est linéaire, `σ = D·ε`, et rend les mêmes composantes
`sigma_*` que l'élasticité isotrope.

## Mise en donnée (Rust, testé)

```rust,ignore
{{#include ../../../tests/orthotropic.rs:example}}
```

## Exemple Python

Le balayage du repère matériau, où l'on voit l'effet propre à l'orthotropie :

```python
{{#include ../../../examples/plaque_orthotrope.py}}
```

Il produit :

```text
  angle    u_x(1,0)    écart u_x sur le bord droit
  ----------------------------------------------
    0.0°    0.010000       -0.000000
   22.5°    0.017888       -0.004075
   45.0°    0.031195       -0.008035
   67.5°    0.039603       -0.005663
   90.0°    0.040000       -0.000000
```

Les deux bornes sont analytiques — `S/E₁` quand l'axe rigide est aligné sur la
traction, `S/E₂` quand c'est l'axe souple. Entre les deux, le bord droit **se
gauchit** : hors de ses axes, un matériau orthotrope couple traction et
cisaillement (le terme `D₁₆` du tenseur tourné n'est plus nul). C'est exactement
ce qu'un calcul isotrope ne peut pas produire, et le signe le plus visible que la
rotation fait son travail.

## Compléments

**Ce qui vaut comme vérification.** Deux dégénérescences encadrent
l'implémentation, et sont testées de bout en bout :

- une loi **orthotrope** nourrie de constantes isotropes doit se comporter comme
  l'isotrope, **quel que soit son repère** — c'est le contrôle le plus sévère de
  la rotation, puisque toute erreur d'indice brise l'invariance ;
- une loi **anisotrope** nourrie du tenseur isotrope doit faire de même, ce qui
  fixe l'ordre de lecture des 21 constantes : une permutation placerait les
  modules de cisaillement dans les mauvaises cases de Voigt.

**Axisymétrie.** L'orthotropie s'y combine sans rien de particulier. En 2-D le
troisième axe matériau est la direction hors-plan, c'est-à-dire l'orthoradiale
`θ` — ce qui est le comportement voulu pour un tube bobiné, dont la direction de
fibre est justement circonférentielle.

**Ce que cela ne couvre pas.** La symétrie matériau porte sur l'**élasticité**.
Les lois non linéaires (plasticité, endommagement) restent bâties sur une
élasticité isotrope ; l'endommagement orthotrope est le sujet d'un modèle propre,
pas d'un axe de symétrie.

La même mécanique sert la [conduction thermique orientée](../thermique.md) et la
[diffusion](../diffusion.md), avec un tenseur d'ordre 2 au lieu de 4.
