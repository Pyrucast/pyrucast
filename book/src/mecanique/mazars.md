# Endommagement de Mazars

## En bref

Le béton qui se fissure : il perd de sa raideur, sans déformation permanente.
Un seul scalaire d'endommagement \\(D \in [0, 1)\\) affaiblit la contrainte.

| | |
|---|---|
| opérateur | `pyrucast.model.mazars(fespace, kinematics, strain_measure="linear")` |
| tag | `mechanical` |
| type | domaine avec comportement (`label()` : `"Damage"`) |
| éléments | 2-D, en plan ou en axisymétrie : `TRI3`, `QUA4`, `TRI6`, `QUA8`, `QUA9` ; 3-D : `TET4`, `PYRA5`, `PENTA6`, `HEX8`, `TET10`, `PENTA15`, `HEX20`, `HEX27` |
| cinématiques | `plane_stress`, `plane_strain`, `axisymmetric`, `full_3d` |

## Théorie

La loi est **sécante** : la contrainte est la contrainte élastique affaiblie,

\\[
\sigma = (1 - D)\\,D_\text{el} : \varepsilon .
\\]

L'endommagement est piloté par la **déformation équivalente**, bâtie sur les
parts positives des déformations principales,

\\[
\tilde\varepsilon = \sqrt{\textstyle\sum_I \langle\varepsilon_I\rangle_+^2},
\qquad
\kappa = \max\big(\varepsilon_{d0},\ \max_t \tilde\varepsilon\big),
\\]

où \\(\varepsilon_{d0}\\) est le seuil. \\(\kappa\\) ne décroît jamais : le
matériau ne guérit pas. L'endommagement mêle une branche de traction et une
branche de compression,

\\[
D_t = 1 - \frac{\varepsilon_{d0}(1 - A_t)}{\kappa} - \frac{A_t}{e^{B_t(\kappa - \varepsilon_{d0})}},
\qquad
D = \alpha_t\\,D_t + \alpha_c\\,D_c ,
\\]

\\(D_c\\) s'écrivant de même avec \\(A_c\\) et \\(B_c\\). Les poids
\\(\alpha_t\\), \\(\alpha_c\\) viennent du partage de la contrainte effective
\\(D_\text{el} : \varepsilon\\) en parts positive et négative, sur ses valeurs
principales.

Le calcul est mené sur le tenseur 3-D : en déformations planes,
\\(\varepsilon_{zz} = 0\\) ; en contraintes planes,
\\(\varepsilon_{zz} = -\nu\\,(\varepsilon_{xx} + \varepsilon_{yy})/(1 - \nu)\\) ;
en axisymétrie, l'orthoradiale est mesurée.

La raideur est la raideur **élastique non endommagée** : l'opérateur
d'itération d'un Newton modifié. La loi n'a pas de tangente.

## Variables

Selon la cinématique :

| | `plane_stress`, `plane_strain` | `axisymmetric` | `full_3d` |
|---|---|---|---|
| primales | `u_x`, `u_y` | `u_x`, `u_y` | `u_x`, `u_y`, `u_z` |
| duales | `f_x`, `f_y` | `f_x`, `f_y` | `f_x`, `f_y`, `f_z` |
| entrée du comportement | `eps_xx`, `eps_yy`, `eps_xy` | `eps_xx`, `eps_yy`, `eps_zz`, `eps_xy` | `eps_xx`, `eps_yy`, `eps_zz`, `eps_yz`, `eps_xz`, `eps_xy` |
| contrainte | `sigma_xx`, `sigma_yy`, `sigma_xy` | `sigma_xx`, `sigma_yy`, `sigma_zz`, `sigma_xy` | `sigma_xx`, `sigma_yy`, `sigma_zz`, `sigma_yz`, `sigma_xz`, `sigma_xy` |

En axisymétrie, `eps_zz` et `sigma_zz` sont les composantes orthoradiales.

La sortie du comportement est la contrainte, suivie de l'état :

| | |
|---|---|
| matériau requis | `E`, `nu`, `eps_d0`, `A_t`, `B_t`, `A_c`, `B_c` |
| matériau facultatif | `rho`, `alpha` |
| état, écrit après la contrainte | `damage`, `kappa` |
| état relu (`prev`) | `kappa` |

La contrainte ne dépend que de la déformation totale et de \\(\kappa\\) : il
n'y a pas d'incrément. Au premier pas, `prev` est l'état de repos,
`element_field.initial_state(modele, materiaux)`, où `kappa` vaut 0 : la loi
prenant le maximum avec le seuil, cela revient à
\\(\kappa = \varepsilon_{d0}\\).

## Ce qu'elle produit

| | |
|---|---|
| raideur | élastique, non endommagée |
| masse | exige `rho` |
| géométrique | sous la contrainte `sigma_*` |
| tangente | aucune : `matrix.tangent` lève une erreur |
| forces internes | \\(\int B^{\mathsf T}\sigma\\), la contrainte endommagée |

## Exemple

Une maille étirée à deux fois le seuil :

```python
{{#include ../../../tests/python/test_doc_mecanique.py:mazars}}
```

## Côté Rust

`Damage` (`src/models/damage.rs`) porte sa loi en attribut (`DamageLaw`), comme
`Plasticity` : Mazars, [Damage TC et SiC/SiC](endommagement.md). Mazars vit dans
`src/models/damage/mazars.rs` ; la décomposition spectrale passe par
`nalgebra`. `tests/damage_laws.rs` vérifie que Mazars ne rend qu'un
endommagement scalaire.
