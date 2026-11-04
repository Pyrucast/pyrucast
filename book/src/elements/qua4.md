# QUA4 — quadrangle bilinéaire

## En bref

Le quadrangle à quatre nœuds. Son interpolation est bilinéaire : elle
contient le terme \\( \xi\eta \\) qui manque au triangle, si bien qu'un
gradient y varie.

| | |
|---|---|
| nom | `"QUA4"` |
| nœuds | 4, aux sommets |
| interpolation | `"LAGRANGE1"` : bilinéaire |
| dimension | 2 ; dans des `Coords` 2-D ou 3-D |
| bord | 4 `SEG2` |
| quadrature | `"GAUSS"` : 2×2 points, degré 3 par direction ; `"REDUCED"` : 1 point |
| versions quadratiques | [QUA8](qua8.md) (sérendipité), [QUA9](qua9.md) (complet) |
| obtenu par | `sweep`, `transfinite`, `pave_surface`, `grid_surface`, `triangulate_surface` ; `extrude`, `revolve` d'un `SEG2` ; `skin` d'un `HEX8`, `PYRA5` ou `PENTA6` |

## Élément de référence

Le carré \\( \xi, \eta \in [-1, +1] \\), parcouru dans le sens
trigonométrique.

| Nœud | \\( (\xi_i, \eta_i) \\) |
|---:|:---:|
| 0 | \\( (-1, -1) \\) |
| 1 | \\( (+1, -1) \\) |
| 2 | \\( (+1, +1) \\) |
| 3 | \\( (-1, +1) \\) |

## Fonctions de forme

Pour le nœud \\( i \\) de coordonnées \\( (\xi_i, \eta_i) \\) :

\\[
N_i(\xi, \eta) = \tfrac{1}{4}\\,(1 + \xi_i\\,\xi)\\,(1 + \eta_i\\,\eta).
\\]

Chacune est le produit d'une fonction linéaire en \\( \xi \\) et d'une autre en
\\( \eta \\) : le quadrangle reproduit exactement \\( a + b\xi + c\eta + d\xi\eta \\).

## Dérivées de référence

\\[
\frac{\partial N_i}{\partial \xi} = \tfrac14\\,\xi_i\\,(1 + \eta_i\\,\eta), \qquad
\frac{\partial N_i}{\partial \eta} = \tfrac14\\,\eta_i\\,(1 + \xi_i\\,\xi).
\\]

\\( \partial N_i/\partial\xi \\) varie avec \\( \eta \\) : sauf sur un
parallélogramme, le jacobien n'est pas constant sur la maille.

## Degrés de liberté

| physique | inconnues par nœud | par maille |
|---|---|---:|
| conduction, diffusion | `T` | 4 |
| élasticité plane ou axisymétrique | `u_x`, `u_y` | 8 |
| coque, dans l'espace | `u_x`, `u_y`, `u_z`, `r_x`, `r_y`, `r_z` | 24 |
| échanges, rayonnement, sur la face d'un volume | `T` | 4 |

## Quadrature

`"GAUSS"` : produit 2×2 de Gauss-Legendre, exact au degré 3 par direction.

\\[
\xi_g = \left(\pm\tfrac{1}{\sqrt 3}, \pm\tfrac{1}{\sqrt 3}\right), \qquad
w_g = 1, \qquad \sum_g w_g = 4.
\\]

La masse, de degré 2 par direction, est exacte sur un parallélogramme. Sur un
quadrangle quelconque, le jacobien varie, et la raideur ne l'est plus.

`"REDUCED"` : un point au centre \\( (0, 0) \\), de poids 4.

## Exemple

```python
{{#include ../../../tests/python/test_doc_elements.py:qua4}}
```

## Côté Rust

`Qua4` (`src/atoms/element_kind/qua4.rs`) implémente `ElementKind`.
