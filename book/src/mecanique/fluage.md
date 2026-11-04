# Fluage et viscoplasticité

## En bref

Des lois où l'écoulement prend du temps : la contrainte peut sortir du domaine
élastique, et c'est l'écart qui fixe la vitesse à laquelle elle y revient. Le
résultat d'un pas dépend de sa durée.

| | |
|---|---|
| opérateurs | `pyrucast.model.creep_norton(fespace, kinematics, strain_measure="linear")`, `pyrucast.model.creep_lemaitre(fespace, kinematics, strain_measure="linear")`, `pyrucast.model.creep_blackburn(fespace, kinematics, strain_measure="linear")`, `pyrucast.model.viscoplasticity_chaboche(fespace, kinematics, strain_measure="linear")`, `pyrucast.model.viscoplasticity_lemaitre_chaboche(fespace, kinematics, strain_measure="linear")` |
| tag | `mechanical` |
| type | domaine avec comportement (`label()` : `"Plasticity"`) |
| éléments | 2-D, en plan ou en axisymétrie : `TRI3`, `QUA4`, `TRI6`, `QUA8`, `QUA9` ; 3-D : `TET4`, `PYRA5`, `PENTA6`, `HEX8`, `TET10`, `PENTA15`, `HEX20`, `HEX27` |
| cinématiques | `plane_stress`, `plane_strain`, `axisymmetric`, `full_3d` |

`integrate_behavior` et `matrix.tangent` exigent le pas de temps `dt` pour ces
lois, et lèvent une erreur sans lui : intégrer une loi visqueuse comme si
elle était instantanée donnerait un nombre plausible et faux.

| loi | vitesse | ce qu'elle décrit |
|---|---|---|
| `creep_norton` | \\(\dot p = (q/K)^n\\) | fluage secondaire, stationnaire |
| `creep_lemaitre` | \\(\dot p = (q/K)^N\\,p^{-M}\\) | fluage primaire, par écrouissage en déformation |
| `creep_blackburn` | primaire saturant et secondaire, en `sinh` | les deux stades |
| `viscoplasticity_chaboche` | \\(\dot p = \langle (J(\sigma - X) - R - k)/K\rangle^n\\) | viscoplasticité cyclique |
| `viscoplasticity_lemaitre_chaboche` | la même, sur \\(\sigma/(1-D)\\) | et endommagement ductile |

## Théorie

Le cadre est celui de la [plasticité](plasticite.md#théorie), sans les
conditions de Kuhn-Tucker : la vitesse d'écoulement n'est plus l'inconnue d'une
condition de consistance, elle est **donnée** par une loi,

\\[
\dot\varepsilon^{vp} = \dot p\\,\frac32\\,\frac{s}{q}, \qquad \dot p = g(q, p, \dots).
\\]

L'écoulement est déviatorique, sans variation de volume, et radial.

**Un solveur pour toutes.** Le pas est intégré par Euler implicite : la
vitesse est évaluée en fin de pas. La direction étant celle du prédicteur, le
déviateur ne fait que se contracter, \\(q(\Delta p) = q^{tr} - 3\mu\\,\Delta p\\),
et le pas se ramène à une équation scalaire,

\\[
R(\Delta p) = \Delta p - \Delta t\\,g\big(q(\Delta p), p_A + \Delta p, \dots\big) = 0 ,
\\]

dont la racine est unique dans \\([0, q^{tr}/3\mu]\\). Les lois de vitesse étant
raides, elle est cherchée par un Newton encadré, doublé d'une dichotomie.

**Norton.** \\(\dot p = (q/K)^n\\), sans seuil : toute contrainte flue. \\(K\\)
est la contrainte de référence, \\(n\\) la sensibilité.

**Lemaitre.** \\(\dot p = (q/K)^N\\,p^{-M}\\), \\(M > 0\\) : la déformation
accumulée ralentit l'écoulement ; à contrainte constante,
\\(p(t) \propto t^{1/(1+M)}\\). \\(p\\) a un plancher minuscule, pour que
\\(p^{-M}\\) reste fini au premier pas.

**Blackburn.** La déformation primaire est une variable à part, `p_prim` :

\\[
\dot p_\text{prim} = r\\,\big(\varepsilon_\infty(q) - p_\text{prim}\big),
\qquad \varepsilon_\infty(q) = A\\,\sinh(\alpha q),
\qquad \dot p = \dot p_\text{prim} + B\\,\sinh(\beta q) .
\\]

Le primaire tend vers son asymptote à la vitesse \\(r\\) puis s'éteint ; le
secondaire persiste. En Euler implicite, le primaire s'inverse en un pas,
\\(p^B_\text{prim} = (p^A_\text{prim} + \Delta t\\,r\\,A\\,\sinh(\alpha q))/(1 + \Delta t\\,r)\\).

**Chaboche.** Une surface translatée par une contrainte de rappel \\(X\\) et
agrandie par un écrouissage isotrope \\(R\\) :

\\[
f = J(\sigma - X) - R - k, \qquad \dot p = \Big\langle \frac fK \Big\rangle^n, \qquad
\dot X = \tfrac23 C\\,\dot\varepsilon^{vp} - \gamma\\,X\\,\dot p, \qquad
\dot R = b\\,(Q - R)\\,\dot p ,
\\]

où \\(\langle\cdot\rangle\\) est la partie positive : sous la surface, le
comportement est élastique. \\(X\\) produit l'effet Bauschinger et sature à
\\(C/\gamma\\) ; \\(R\\) sature à \\(Q\\). La direction est gelée au
prédicteur, et les deux variables sont implicites en \\(\Delta p\\) :
\\(X_B = (X_A + \tfrac23 C\\,\Delta p\\,\hat n)/(1 + \gamma\\,\Delta p)\\),
\\(R_B = (R_A + b\\,Q\\,\Delta p)/(1 + b\\,\Delta p)\\).

**Lemaitre-Chaboche** remplace la contrainte par la contrainte effective
\\(\tilde\sigma = \sigma/(1 - D)\\), et fait croître l'endommagement avec
l'écoulement :

\\[
\dot D = \Big(\frac YS\Big)^s \dot p, \qquad Y = \frac{\tilde\sigma_\text{eq}^2}{2E} .
\\]

Un matériau endommagé flue plus vite, ce qui l'endommage davantage : c'est le
fluage tertiaire. \\(D\\) ne décroît jamais et s'arrête à `D_c`.

**La tangente** est obtenue par perturbation et symétrisée, comme celle de
[Drucker-Prager](lois-plastiques.md#la-tangente-par-perturbation).

## Variables

Selon la cinématique :

| | `plane_stress`, `plane_strain` | `axisymmetric` | `full_3d` |
|---|---|---|---|
| primales | `u_x`, `u_y` | `u_x`, `u_y` | `u_x`, `u_y`, `u_z` |
| duales | `f_x`, `f_y` | `f_x`, `f_y` | `f_x`, `f_y`, `f_z` |
| entrée du comportement | `eps_xx`, `eps_yy`, `eps_xy` | `eps_xx`, `eps_yy`, `eps_zz`, `eps_xy` | `eps_xx`, `eps_yy`, `eps_zz`, `eps_yz`, `eps_xz`, `eps_xy` |
| contrainte | `sigma_xx`, `sigma_yy`, `sigma_xy` | `sigma_xx`, `sigma_yy`, `sigma_zz`, `sigma_xy` | `sigma_xx`, `sigma_yy`, `sigma_zz`, `sigma_yz`, `sigma_xz`, `sigma_xy` |

En axisymétrie, `eps_zz` et `sigma_zz` sont les composantes orthoradiales.

La sortie du comportement est la contrainte, suivie d'un état commun aux
cinq lois, puis de l'état propre à chacune :

| | |
|---|---|
| matériau facultatif | `rho`, `alpha` |
| état commun | la déformation plastique `eps_p_xx`, `eps_p_yy`, `eps_p_zz`, `eps_p_yz`, `eps_p_xz`, `eps_p_xy` ; la déformation plastique cumulée `p` ; la déformation totale `eps_xx`, `eps_yy`, `eps_zz`, `eps_yz`, `eps_xz`, `eps_xy` ; et, en `plane_stress` et `plane_strain`, `sigma_zz` |
| état relu (`prev`) | toute la sortie du pas précédent |

| loi | matériau requis | état propre, après l'état commun |
|---|---|---|
| `creep_norton` | `E`, `nu`, `K`, `n` | aucun |
| `creep_lemaitre` | `E`, `nu`, `K`, `N`, `M` | aucun |
| `creep_blackburn` | `E`, `nu`, `A_1`, `alpha_1`, `r_1`, `B_s`, `beta_s` | `p_prim` |
| `viscoplasticity_chaboche` | `E`, `nu`, `k`, `K`, `n`, `C_1`, `gamma_1`, `b`, `Q` | `X_xx`, `X_yy`, `X_zz`, `X_yz`, `X_xz`, `X_xy`, `R` |
| `viscoplasticity_lemaitre_chaboche` | `E`, `nu`, `k`, `K`, `n`, `C_1`, `gamma_1`, `b`, `Q`, `S`, `s`, `D_c` | `X_xx`, `X_yy`, `X_zz`, `X_yz`, `X_xz`, `X_xy`, `R`, `damage` |

## Ce qu'elles produisent

| | |
|---|---|
| raideur | élastique |
| masse | exige `rho` |
| géométrique | sous la contrainte `sigma_*` |
| tangente | par perturbation, symétrisée ; exige `dt` |
| forces internes | \\(\int B^{\mathsf T}\sigma\\) |

## Exemple

Une déformation maintenue sous Norton : sans `dt`, l'intégration est refusée ;
avec, la contrainte se relaxe d'un pas à l'autre.

```python
{{#include ../../../tests/python/test_doc_mecanique.py:fluage}}
```

Les paramètres sont des composantes du matériau. Une dépendance à la
température se calcule par une [évolution](../evolution.md#appliquer-une-courbe-à-un-champ)
appliquée au champ de température.

## Côté Rust

Les cinq lois vivent dans `src/models/plasticity/viscous.rs`, et déclarent
`requires_dt`. Le test vérifie que Norton suit sa loi de vitesse sur un pas
court, qu'une déformation maintenue se relaxe d'autant plus que le pas est
long, que le primaire de Blackburn sature, que la contrainte de rappel de
Chaboche plafonne sous \\(C/\gamma\\), que l'endommagement ne dépasse pas
\\(D_c\\), et que toutes refusent d'intégrer sans `dt`.

```rust,ignore
{{#include ../../../tests/viscous_laws.rs:example}}
```
