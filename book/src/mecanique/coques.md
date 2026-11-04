# Coques

## En bref

Une coque est une **surface** qui porte à la fois des efforts de membrane
(dans son plan) et des moments de flexion : une tôle, une paroi de réservoir,
une dalle mince.

| | |
|---|---|
| opérateur | `pyrucast.model.shell(fespace, formulation)` |
| tag | `mechanical` |
| type | domaine avec comportement (`label()` : `"Shell"`) |
| éléments | `TRI3`, `QUA4`, dans un espace 3-D |

`formulation` est `"thick"` (Reissner-Mindlin, cisaillement transverse intégré
de façon réduite) ou `"kirchhoff"` (Kirchhoff discret, DKT et DKQ, sans
cisaillement). Les deux partagent les inconnues, le repère local et la
membrane ; seuls la flexion et le cisaillement changent. Chaque maille est une
facette plane : une forte courbure demande un maillage plus fin.

## Théorie

**Six inconnues par nœud.** Trois translations et trois rotations,
`u_x`…`u_z`, `r_x`…`r_z`, comme le cadre spatial des poutres : une coque et un
portique partagent leurs nœuds sans adaptateur. La sixième, la rotation autour
de la normale (le **vrillage**), n'a pas de raideur physique sur une facette
plane ; elle est liée à la rotation propre de la membrane, ce qui ne s'oppose
pas à une rotation rigide de la facette. Cette liaison travaille : elle a un
effort conjugué, `M_drill`.

**Le repère local** de chaque élément est construit depuis ses nœuds : la
première arête et la normale de la facette. Les dérivées dans le plan de la
facette sont le gradient tangent de la géométrie, projeté sur ce repère.

**Reissner-Mindlin** (`thick`). La fibre normale reste droite mais pas normale
à la surface déformée ; dans l'épaisseur \\(z \in [-h/2, h/2]\\),

\\[
u_x = u + z\\,\theta_y, \qquad u_y = v - z\\,\theta_x, \qquad u_z = w .
\\]

Les déformations se séparent en membrane \\(\varepsilon\\), courbures
\\(\kappa\\) et cisaillement transverse \\(\gamma\\). Pour un matériau homogène
en contraintes planes, intégré dans l'épaisseur, les lois de section sont

\\[
D_m = \frac{E h}{1-\nu^2}\begin{bmatrix} 1 & \nu & 0 \\\\ \nu & 1 & 0 \\\\ 0 & 0 & \tfrac{1-\nu}2 \end{bmatrix},
\qquad
D_b = \frac{h^2}{12}\\,D_m,
\qquad
D_s = k_s\\,G\\,h ,
\\]

avec \\(k_s = 5/6\\) par défaut (composante `k_s`), qui corrige un cisaillement
supposé uniforme dans l'épaisseur. La raideur est
\\(\int_A (B_m^{\mathsf T} D_m B_m + B_b^{\mathsf T} D_b B_b + B_s^{\mathsf T} D_s B_s)\\),
la membrane et la flexion intégrées complètement, le cisaillement en **un
point**. Intégré complètement, ce terme, en \\(h\\), écraserait la flexion, en
\\(h^3\\), quand la coque s'amincit : l'élément refuserait de fléchir (blocage
en cisaillement).

**Kirchhoff discret** (`kirchhoff`). La fibre reste normale, \\(\gamma = 0\\).
Plutôt que des dérivées secondes de la flèche, qui demanderaient une base
\\(C^1\\), la rotation est interpolée quadratiquement et
\\(\gamma = 0\\) est imposé aux sommets et le long de chaque arête ; les
rotations de milieu d'arête sont ensuite éliminées. Rien n'est dérivé deux fois
et la base reste de Lagrange. La limite mince est exacte par construction : il
n'y a pas de cisaillement à bloquer. L'effort tranchant d'une plaque mince est
alors une réaction, pas une sortie de la loi.

## Variables

| | `thick` | `kirchhoff` |
|---|---|---|
| primales | `u_x`, `u_y`, `u_z`, `r_x`, `r_y`, `r_z` | idem |
| duales | `f_x`, `f_y`, `f_z`, `m_x`, `m_y`, `m_z` | idem |
| matériau requis | `E`, `nu`, `h` | `E`, `nu`, `h` |
| matériau facultatif | `rho`, `k_s` | `rho` |
| entrée du comportement | `eps_xx`, `eps_yy`, `eps_xy`, `kappa_xx`, `kappa_yy`, `kappa_xy`, `drill`, `gamma_xz`, `gamma_yz` | les sept premières |
| sortie du comportement | `N_xx`, `N_yy`, `N_xy`, `M_xx`, `M_yy`, `M_xy`, `M_drill`, `Q_xz`, `Q_yz` | les sept premières |
| état | aucun | aucun |

L'entrée vient de `pyrucast.element_field.shell_deformation`, ou de `strain` :
membrane, flexion et vrillage à chaque point de Gauss, cisaillement au seul
point réduit.

## Ce qu'elle produit

| | |
|---|---|
| raideur | membrane, flexion, cisaillement (`thick`) et vrillage |
| masse | — |
| géométrique | — |
| tangente | = raideur |
| forces internes | le transposé du même \\(B\\), ramené aux axes globaux : \\(\int B^{\mathsf T}\sigma = K\\,u\\) exactement |

## Exemple

Une facette carrée, ses inconnues et son matériau :

```python
{{#include ../../../tests/python/test_doc_mecanique.py:coque}}
```

## Côté Rust

`Shell` (`src/models/shell.rs`, `src/models/shell/`) porte sa formulation en
attribut (`ShellFormulation`). Les deux formulations construisent leur
\\(B\\) au même endroit (`shell::b_into`) et ne diffèrent que par ses lignes de
flexion. `thick` intègre sur deux sous-espaces du même maillage, à quadrature
complète et réduite ; la physique les construit elle-même.

Les tests retrouvent la flèche d'une plaque carrée encastrée sous charge
uniforme et vérifient que le maillage converge ; ils mesurent la flèche
normalisée d'une plaque mince sur deux décades d'épaisseur, qu'un élément
bloqué perdrait ; ils vérifient la membrane, et qu'une rotation rigide ne
coûte aucune énergie.

```rust,ignore
{{#include ../../../tests/shell.rs:example}}
```

La même plaque en Kirchhoff discret, triangles et quadrangles :

```rust,ignore
{{#include ../../../tests/shell.rs:kirchhoff}}
```
