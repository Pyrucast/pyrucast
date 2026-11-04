# TRI3 — triangle linéaire

## En bref

Le triangle à trois nœuds, l'élément 2-D le plus simple : son interpolation
est affine, et un gradient y est constant. Plongé dans l'espace, il décrit une
surface : une coque, la peau d'un volume.

| | |
|---|---|
| nom | `"TRI3"` |
| nœuds | 3, aux sommets |
| interpolation | `"LAGRANGE1"` : affine |
| dimension | 2 ; dans des `Coords` 2-D ou 3-D |
| bord | 3 `SEG2` |
| quadrature | `"GAUSS"` : 3 points, degré 2 ; `"REDUCED"` : 1 point |
| version quadratique | [TRI6](tri6.md) |
| obtenu par | `triangulate_surface`, `sweep`, `transfinite` ; `convert` d'un `QUA4` ; `skin` d'un `TET4`, `PYRA5` ou `PENTA6` |

## Élément de référence

Le triangle \\( \xi, \eta \ge 0 \\), \\( \xi + \eta \le 1 \\), parcouru dans le
sens trigonométrique.

| Nœud | \\( (\xi, \eta) \\) |
|---:|:---:|
| 0 | \\( (0, 0) \\) |
| 1 | \\( (1, 0) \\) |
| 2 | \\( (0, 1) \\) |

## Fonctions de forme

Les **coordonnées barycentriques** du point :

\\[
N_0 = 1 - \xi - \eta, \qquad N_1 = \xi, \qquad N_2 = \eta.
\\]

## Dérivées de référence

Constantes :

\\[
\nabla_\xi N_0 = (-1, -1), \qquad
\nabla_\xi N_1 = (1, 0), \qquad
\nabla_\xi N_2 = (0, 1).
\\]

Le jacobien est donc constant sur la maille, comme le gradient d'un champ et
la déformation d'un solide.

## Degrés de liberté

| physique | inconnues par nœud | par maille |
|---|---|---:|
| conduction, diffusion | `T` | 3 |
| élasticité plane ou axisymétrique | `u_x`, `u_y` | 6 |
| coque, dans l'espace | `u_x`, `u_y`, `u_z`, `r_x`, `r_y`, `r_z` | 18 |
| échanges, rayonnement, sur la face d'un volume | `T` | 3 |

## Quadrature

`"GAUSS"` : règle de Hammer à 3 points, aux milieux des côtés, exacte au
degré 2.

\\[
\xi_g \in \left\\{ \left(\tfrac12, 0\right), \left(\tfrac12, \tfrac12\right), \left(0, \tfrac12\right) \right\\}, \qquad
w_g = \tfrac{1}{6}, \qquad \sum_g w_g = \tfrac12.
\\]

Un point suffirait à la raideur, de degré 0 ; la masse
\\( \int N_i N_j \\), de degré 2, en demande trois.

`"REDUCED"` : un point au centre de gravité \\( (\tfrac13, \tfrac13) \\), de
poids \\( \tfrac12 \\).

## Exemple

```python
{{#include ../../../tests/python/test_doc_elements.py:tri3}}
```

## Côté Rust

`Tri3` (`src/atoms/element_kind/tri3.rs`) implémente `ElementKind`.
