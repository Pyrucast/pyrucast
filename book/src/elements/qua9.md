# QUA9 — quadrangle biquadratique

## En bref

Le quadrangle à neuf nœuds : les sommets, les milieux des côtés et le centre.
Son interpolation est le produit de deux paraboles, polynôme de degré 2 complet
dans chaque direction, terme \\( \xi^2\eta^2 \\) compris.

| | |
|---|---|
| nom | `"QUA9"` |
| nœuds | 9 : les sommets, les milieux des côtés, le centre |
| interpolation | `"LAGRANGE2"` : complète |
| dimension | 2 ; dans des `Coords` 2-D ou 3-D |
| bord | 4 `SEG3` |
| quadrature | `"GAUSS"` : 3×3 points, degré 5 par direction ; `"REDUCED"` : 1 point |
| parent linéaire | [QUA4](qua4.md) |
| obtenu par | `sweep`, `transfinite`, `pave_surface`, `grid_surface` ; `skin` d'un `HEX27` |

## Élément de référence

Le carré \\( [-1, +1]^2 \\) : les nœuds du [QUA8](qua8.md), puis le centre.

| Nœud | \\( (\xi, \eta) \\) | rôle |
|---:|:---:|---|
| 0 | \\( (-1, -1) \\) | sommet |
| 1 | \\( (+1, -1) \\) | sommet |
| 2 | \\( (+1, +1) \\) | sommet |
| 3 | \\( (-1, +1) \\) | sommet |
| 4 | \\( (0, -1) \\) | milieu de \\( (0,1) \\) |
| 5 | \\( (+1, 0) \\) | milieu de \\( (1,2) \\) |
| 6 | \\( (0, +1) \\) | milieu de \\( (2,3) \\) |
| 7 | \\( (-1, 0) \\) | milieu de \\( (3,0) \\) |
| 8 | \\( (0, 0) \\) | centre |

## Fonctions de forme

Le produit des trois fonctions de Lagrange quadratiques d'un segment, sur les
positions \\( -1, 0, +1 \\) :

\\[
\ell_{-}(t) = \tfrac12 t(t-1), \qquad
\ell_{0}(t) = 1 - t^2, \qquad
\ell_{+}(t) = \tfrac12 t(t+1),
\\]

et \\( N_i(\xi, \eta) = \ell_a(\xi)\\,\ell_b(\eta) \\), où \\( a \\) et
\\( b \\) sont les positions du nœud dans chaque direction. Le centre a
\\( N_8 = (1-\xi^2)(1-\eta^2) \\).

## Dérivées de référence

\\[
\frac{\partial N_i}{\partial \xi} = \ell_a\'(\xi)\\,\ell_b(\eta), \qquad
\frac{\partial N_i}{\partial \eta} = \ell_a(\xi)\\,\ell_b\'(\eta),
\\]

avec \\( \ell_{-}\'(t) = t - \tfrac12 \\), \\( \ell_0\'(t) = -2t \\),
\\( \ell_{+}\'(t) = t + \tfrac12 \\).

## Degrés de liberté

| physique | inconnues par nœud | par maille |
|---|---|---:|
| conduction, diffusion | `T` | 9 |
| élasticité plane ou axisymétrique | `u_x`, `u_y` | 18 |
| échanges, rayonnement, sur la face d'un volume quadratique | `T` | 9 |

## Quadrature

`"GAUSS"` : produit 3×3 de Gauss-Legendre, 9 points, exact au degré 5 par
direction ; \\( \sum_g w_g = 4 \\). La masse, de degré 4 par direction, est
exacte sur un parallélogramme à côtés droits.

`"REDUCED"` : un point au centre, de poids 4.

## Exemple

```python
{{#include ../../../tests/python/test_doc_elements.py:qua9}}
```

## Côté Rust

`Qua9` (`src/atoms/element_kind/qua9.rs`) implémente `ElementKind`.
