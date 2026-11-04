# PENTA6 — prisme linéaire

## En bref

Le prisme à six nœuds, un triangle étiré dans une direction. C'est ce que
donne un maillage de triangles extrudé, ou tourné autour d'un axe : un
maillage par couches.

| | |
|---|---|
| nom | `"PENTA6"` |
| nœuds | 6 : le triangle du bas, puis celui du haut |
| interpolation | `"LAGRANGE1"` : affine dans le triangle, linéaire en hauteur |
| dimension | 3 ; dans des `Coords` 3-D |
| bord | 2 `TRI3`, 3 `QUA4` |
| quadrature | `"GAUSS"` : 3 × 2 = 6 points ; `"REDUCED"` : 1 point |
| version quadratique | [PENTA15](penta15.md) |
| obtenu par | `extrude`, `revolve` d'un `TRI3` ; `sweep_solid` ; `pave_volume` |

## Élément de référence

Le triangle \\( \xi, \eta \ge 0 \\), \\( \xi + \eta \le 1 \\), étiré sur
\\( \zeta \in [0, 1] \\) : le triangle du bas (\\( \zeta = 0 \\)) puis celui du
haut (\\( \zeta = 1 \\)), chacun dans le sens trigonométrique.

| Nœud | \\( (\xi, \eta, \zeta) \\) | | Nœud | \\( (\xi, \eta, \zeta) \\) |
|---:|:---:|---|---:|:---:|
| 0 | \\( (0, 0, 0) \\) | | 3 | \\( (0, 0, 1) \\) |
| 1 | \\( (1, 0, 0) \\) | | 4 | \\( (1, 0, 1) \\) |
| 2 | \\( (0, 1, 0) \\) | | 5 | \\( (0, 1, 1) \\) |

## Fonctions de forme

Le produit des coordonnées barycentriques du triangle,
\\( L_1 = 1-\xi-\eta \\), \\( L_2 = \xi \\), \\( L_3 = \eta \\), et d'une
fonction linéaire en \\( \zeta \\) :

\\[
N_{j-1} = L_j\\,(1 - \zeta), \qquad N_{j+2} = L_j\\,\zeta \qquad (j = 1, 2, 3).
\\]

## Dérivées de référence

Les dérivées des \\( L_j \\) sont constantes ; celle en \\( \zeta \\) vient du
facteur de hauteur. Pour le nœud 0 :

\\[
\nabla_\xi N_0 = \big(-(1-\zeta),\ -(1-\zeta),\ -L_1\big).
\\]

## Degrés de liberté

| physique | inconnues par nœud | par maille |
|---|---|---:|
| conduction, diffusion | `T` | 6 |
| élasticité 3-D | `u_x`, `u_y`, `u_z` | 18 |

## Quadrature

`"GAUSS"` : le produit de la règle du `TRI3` (3 points, poids \\( \tfrac16 \\))
et de Gauss-Legendre à 2 points sur \\( [0, 1] \\)
(\\( \zeta_g = \tfrac12 \mp \tfrac{1}{2\sqrt3} \\), poids \\( \tfrac12 \\)),
soit 6 points.

\\[
w_g = \tfrac16\cdot\tfrac12 = \tfrac{1}{12}, \qquad \sum_g w_g = \tfrac12.
\\]

Exacte au degré 2 dans le triangle et 3 en hauteur : la masse et la raideur
sont exactes sur un prisme dont le triangle du haut est celui du bas déplacé.

`"REDUCED"` : un point en \\( (\tfrac13, \tfrac13, \tfrac12) \\), de poids
\\( \tfrac12 \\).

## Exemple

```python
{{#include ../../../tests/python/test_doc_elements.py:penta6}}
```

## Côté Rust

`Penta6` (`src/atoms/element_kind/penta6.rs`) implémente `ElementKind`.
