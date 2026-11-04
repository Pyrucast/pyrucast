# HEX27 — hexaèdre triquadratique

## En bref

L'hexaèdre à vingt-sept nœuds : sommets, milieux des arêtes, centres des faces
et centre. Son interpolation est le produit de trois paraboles, polynôme de
degré 2 complet dans chaque direction. Aucun mailleur de pyrucast ne le
produit : il vient d'un fichier gmsh ou de `from_arrays`.

| | |
|---|---|
| nom | `"HEX27"` |
| nœuds | 27 : sommets, milieux des arêtes, centres des faces, centre |
| interpolation | `"LAGRANGE2"` : complète |
| dimension | 3 ; dans des `Coords` 3-D |
| bord | 6 `QUA9` |
| quadrature | `"GAUSS"` : 3×3×3 points, degré 5 par direction ; `"REDUCED"` : 1 point |
| parent linéaire | [HEX8](hex8.md) |
| obtenu par | lecture : `read_gmsh`, `from_arrays` |

## Élément de référence

Le cube \\( [-1, +1]^3 \\) : les sommets 0 à 7 et les milieux d'arête 8 à 19
dans l'ordre du [HEX20](hex20.md), les centres de face 20 à 25 (faces
\\( \xi = -1 \\), \\( \xi = +1 \\), \\( \eta = -1 \\), \\( \eta = +1 \\),
\\( \zeta = -1 \\), \\( \zeta = +1 \\)), et le centre 26 en \\( (0,0,0) \\).

## Fonctions de forme

Le produit des trois fonctions de Lagrange quadratiques d'un segment, sur les
positions \\( -1, 0, +1 \\) :

\\[
\ell_{-}(t) = \tfrac12 t(t-1), \qquad
\ell_{0}(t) = 1 - t^2, \qquad
\ell_{+}(t) = \tfrac12 t(t+1),
\\]

et \\( N_i(\xi, \eta, \zeta) = \ell_a(\xi)\\,\ell_b(\eta)\\,\ell_c(\zeta) \\),
où \\( a, b, c \\) sont les positions du nœud dans chaque direction. Le centre
a \\( N_{26} = (1-\xi^2)(1-\eta^2)(1-\zeta^2) \\).

## Dérivées de référence

\\[
\frac{\partial N_i}{\partial \xi} = \ell_a\'(\xi)\\,\ell_b(\eta)\\,\ell_c(\zeta),
\\]

et de même en \\( \eta \\) et \\( \zeta \\), avec
\\( \ell_{-}\'(t) = t - \tfrac12 \\), \\( \ell_0\'(t) = -2t \\),
\\( \ell_{+}\'(t) = t + \tfrac12 \\).

## Degrés de liberté

| physique | inconnues par nœud | par maille |
|---|---|---:|
| conduction, diffusion | `T` | 27 |
| élasticité 3-D | `u_x`, `u_y`, `u_z` | 81 |

## Quadrature

`"GAUSS"` : produit 3×3×3 de Gauss-Legendre, 27 points, exact au degré 5 par
direction ; \\( \sum_g w_g = 8 \\). La masse, de degré 4 par direction, est
exacte sur un parallélépipède à arêtes droites.

`"REDUCED"` : un point au centre, de poids 8.

## Exemple

```python
{{#include ../../../tests/python/test_doc_elements.py:hex27}}
```

## Côté Rust

`Hex27` (`src/atoms/element_kind/hex27.rs`) implémente `ElementKind`.
