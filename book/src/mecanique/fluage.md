# Fluage et viscoplasticité

## En bref

Des lois où l'écoulement prend du temps : la contrainte peut sortir du domaine
élastique, et c'est l'écart qui fixe la vitesse à laquelle elle y revient. Le
résultat d'un pas dépend de sa durée.

| | |
|---|---|
| opérateurs | `pyrucast.model.creep_norton`, `creep_lemaitre`, `creep_blackburn`, `viscoplasticity_chaboche`, `viscoplasticity_lemaitre_chaboche`, tous `(fespace, kinematics, strain_measure="linear")` |
| tag | `mechanical` |
| type | domaine avec comportement (`label()` : `"Plasticity"`) |
| éléments | ceux de l'[élasticité](elasticite.md) ; toutes les cinématiques |

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

| | |
|---|---|
| primales, duales | celles de l'[élasticité](elasticite.md#variables) |
| matériau requis | Norton : `E`, `nu`, `K`, `n` ; Lemaitre : `E`, `nu`, `K`, `N`, `M` ; Blackburn : `E`, `nu`, `A_1`, `alpha_1`, `r_1`, `B_s`, `beta_s` ; Chaboche : `E`, `nu`, `k`, `K`, `n`, `C_1`, `gamma_1`, `b`, `Q` ; Lemaitre-Chaboche : ceux de Chaboche et `S`, `s`, `D_c` |
| matériau facultatif | `rho`, `alpha` |
| entrée du comportement | celle de la [plasticité](plasticite.md#variables) |
| sortie du comportement | celle de la plasticité, plus : `p_prim` (Blackburn) ; `X_xx`…`X_xy` et `R` (Chaboche) ; et `damage` (Lemaitre-Chaboche) |
| état relu (`prev`) | toute la sortie du pas précédent |

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
