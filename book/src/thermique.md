# Conduction thermique

## En bref

La conduction de la chaleur dans un solide : la température `T` est
l'inconnue, et la chaleur circule des zones chaudes vers les zones froides.

| | |
|---|---|
| opérateur | `pyrucast.model.heat_conduction(fespace, symmetry="isotropic")` |
| tag | `thermal` |
| type | domaine avec comportement (`label()` : `"HeatConduction"`) |
| éléments | toute maille de Lagrange, de degré 1 ou 2, en 1-D, 2-D ou 3-D |

`symmetry` choisit la forme de la conductivité : `"isotropic"` (un scalaire
`k`), `"orthotropic"` ou `"anisotropic"` (un tenseur orienté par des axes
matériau).

## Théorie

En régime transitoire, l'équation de la chaleur s'écrit

\\[
\rho\\,c_p\\,\frac{\partial T}{\partial t} - \nabla\cdot(\mathbf K\\,\nabla T) = Q,
\\]

avec \\(\mathbf K\\) le tenseur de conductivité, \\(\rho\\,c_p\\) la capacité
volumique et \\(Q\\) une source. En régime permanent, le premier terme
disparaît. Multipliée par une température virtuelle \\(v\\) et intégrée par
parties, elle devient

\\[
\int_\Omega \nabla v\cdot\mathbf K\\,\nabla T\\,d\Omega +
\int_\Omega v\\,\rho\\,c_p\\,\dot T\\,d\Omega =
\int_\Omega v\\,Q\\,d\Omega + \int_\Gamma v\\,\bar q\\,d\Gamma ,
\\]

où \\(\bar q\\) est le flux qui entre par le bord. Un bord sans condition est
**isolé** : rien n'y entre.

Avec \\(T \approx \sum_j N_j\\,T_j\\), chaque maille donne une matrice de
**conduction** et une matrice de **capacité** :

\\[
K_{ij} = \int_e \nabla N_i^{\mathsf T}\\,\mathbf K\\,\nabla N_j\\,d\Omega ,
\qquad
C_{ij} = \int_e \rho\\,c_p\\,N_i\\,N_j\\,d\Omega ,
\\]

intégrées aux points de Gauss, où le matériau est lu : la conductivité et la
capacité peuvent varier à l'intérieur d'une maille. Le système semi-discret
est \\(C\\,\dot T + K\\,T = F\\) ; un pas d'Euler implicite résout
\\((C/\Delta t + K)\\,T_{n+1} = F + (C/\Delta t)\\,T_n\\).

**Conductivité orientée.** Un matériau fibré ou feuilleté conduit mieux dans
certaines directions. Le tenseur \\(\mathbf K\\) est alors construit dans les
**axes matériau**, diagonal (`orthotropic`, conductivités principales `k_1`,
`k_2`, `k_3`) ou complet (`anisotropic`, `k_11`…`k_33`), puis tourné dans les
axes globaux. Les axes sont des vecteurs du champ matériau : `V1X`, `V1Y` en
2-D, le second axe étant la normale dans le plan ; `V1X…V1Z` et `V2X…V2Z` en
3-D, le troisième étant `V1 × V2`. Ils sont normés en interne et peuvent
varier d'une maille à l'autre. En 2-D, `k_3` (la direction normale au plan)
est demandé aussi : le tenseur est construit en 3-D, puis réduit au plan.

## Variables

| | |
|---|---|
| primale | `T` |
| duale | `q`, le flux de chaleur nodal |
| matériau requis | `k` ; ou `k_1`, `k_2`, `k_3` et les axes (`orthotropic`) ; ou `k_11`, `k_12`, `k_13`, `k_22`, `k_23`, `k_33` et les axes (`anisotropic`) |
| matériau facultatif | `rho`, `cp`, demandés par la capacité seule |
| entrée du comportement | `grad_T_x`, `grad_T_y`, `grad_T_z` : le gradient, un par direction de l'espace |
| sortie du comportement | `flux_x`, `flux_y`, `flux_z` |
| état | aucun |

La sortie est `𝐊·∇T`, le flux de la forme faible. Le flux de Fourier, celui
qui va du chaud vers le froid, est son opposé : `−𝐊·∇T`.

## Ce qu'elle produit

| | |
|---|---|
| raideur | \\(K\\), la conduction |
| masse | \\(C\\), la capacité ; exige `rho` et `cp` |
| géométrique | — |
| tangente | = raideur : la loi est linéaire |
| forces internes | \\(q_i = \int \nabla N_i\cdot\text{flux}\\), soit \\(K\\,T\\) |
| forces externes | aucune ; une source ou un flux de bord se donne par un chargement, ou par [`flux`](echanges.md#flux-imposé) |

## Exemple

Une barre de conductivité 2, chauffée par un flux de 10 à gauche, à 20 à
droite. Le profil est linéaire, \\(T(x) = 20 + Q\\,(1 - x)/k\\) :

```python
{{#include ../../tests/python/test_doc_thermique.py:conduction}}
```

Le gradient et le flux aux points de Gauss, et les forces internes, qui
rendent la source :

```python
{{#include ../../tests/python/test_doc_thermique.py:comportement}}
```

Un pas de temps, depuis une barre à 20 partout :

```python
{{#include ../../tests/python/test_doc_thermique.py:transitoire}}
```

Une conductivité orthotrope, d'axe principal `(0,6 ; 0,8)` :

```python
{{#include ../../tests/python/test_doc_thermique.py:orthotrope}}
```

## Côté Rust

`HeatConduction` (`src/models/heat_conduction.rs`) garde son sous-espace et sa
`MaterialSymmetry`. Les matrices élémentaires sont des fonctions pures,
`element_stiffness` et `element_capacity`, qui lisent le matériau ligne par
ligne aux points de Gauss par les indices résolus une fois par zone
(`ElementLayout`). Le tenseur orienté est construit par
`src/models/symmetry.rs`, le même code que pour la diffusion et l'élasticité.

Le même problème que l'exemple, en Rust :

```rust,ignore
{{#include ../../tests/thermal_line.rs:example}}
```

Le test de la conduction orientée impose un champ linéaire au bord, qui doit
être reproduit à l'intérieur pour tout tenseur constant, puis relit le flux et
le compare à `𝐊·∇T` calculé à la main :

```rust,ignore
{{#include ../../tests/anisotropic_conduction.rs:example}}
```
