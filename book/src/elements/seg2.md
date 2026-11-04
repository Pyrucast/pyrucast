# SEG2 — segment linéaire

## En bref

Un segment à deux nœuds : une barre, une poutre, un fil conducteur, le bord
d'une surface.

| | |
|---|---|
| nom | `"SEG2"` |
| nœuds | 2, aux extrémités |
| interpolation | `"LAGRANGE1"` : linéaire ; aussi `"HERMITE3"` et `"MODEL_EMBEDDED"` pour les poutres |
| dimension | 1 ; dans des `Coords` 1-D, 2-D ou 3-D |
| bord | ses deux extrémités |
| quadrature | `"GAUSS"` : 2 points, degré 3 ; `"REDUCED"` : 1 point |
| version quadratique | [SEG3](seg3.md) |
| obtenu par | `line`, `arc`, `circle` ; `border`, le bord d'une surface `TRI3` ou `QUA4` |

## Élément de référence

\\( \xi \in [-1, +1] \\).

| Nœud | \\( \xi \\) |
|---:|:---:|
| 0 | \\( -1 \\) |
| 1 | \\( +1 \\) |

## Fonctions de forme

\\[
N_0(\xi) = \tfrac{1}{2}(1 - \xi), \qquad
N_1(\xi) = \tfrac{1}{2}(1 + \xi).
\\]

Dans des `Coords` 2-D ou 3-D, le segment est **plongé** : sa longueur
s'obtient par la métrique \\( \sqrt{J^{\mathsf T}J} \\), et le gradient d'un
champ est le gradient le long du segment (voir [Jacobien et grandeurs
physiques](../fe-space.md#jacobien-et-grandeurs-physiques)).

## Dérivées de référence

Constantes :

\\[
\frac{\partial N_0}{\partial \xi} = -\tfrac{1}{2}, \qquad
\frac{\partial N_1}{\partial \xi} = +\tfrac{1}{2}.
\\]

Sur un segment de longueur \\( L \\), \\( |J| = L/2 \\) : un gradient constant
le long du segment.

## Degrés de liberté

| physique | inconnues par nœud | par maille |
|---|---|---:|
| conduction, diffusion | `T` | 2 |
| échanges, rayonnement, sur le bord d'une surface | `T` | 2 |
| barre | `u_x` en 1-D ; `u_x`, `u_y` en plan ; `u_x`, `u_y`, `u_z` dans l'espace | 2, 4, 6 |
| poutre de [Bernoulli](../mecanique/bernoulli.md) (`"HERMITE3"`), de [Timoshenko](../mecanique/timoshenko.md) (`"MODEL_EMBEDDED"`) | `w`, `theta` en 1-D ; `u_x`, `u_y`, `r_z` en plan ; six dans l'espace | 4, 6, 12 |

Sous `"HERMITE3"`, la flèche est interpolée par quatre fonctions cubiques, une
valeur et une pente par nœud. Sous `"MODEL_EMBEDDED"`, la poutre porte ses
propres fonctions, et l'espace n'en tabule aucune pour le champ.

## Quadrature

`"GAUSS"` : Gauss-Legendre à 2 points, exacte au degré 3.

\\[
\xi_g = \pm\frac{1}{\sqrt 3}, \qquad w_g = 1, \qquad \sum_g w_g = 2.
\\]

La masse d'un segment droit,
\\( \int N_i N_j\\,ds = \tfrac{L}{6}\begin{bmatrix}2&1\\\\1&2\end{bmatrix} \\),
est de degré 2 : intégrée exactement. Sous `"HERMITE3"`, la raideur de flexion
l'est aussi : ses dérivées secondes sont linéaires, leur produit de degré 2.

`"REDUCED"` : un point en \\( \xi = 0 \\), de poids 2.

## Exemple

```python
{{#include ../../../tests/python/test_doc_elements.py:seg2}}
```

## Côté Rust

`Seg2` (`src/atoms/element_kind/seg2.rs`) implémente `ElementKind`. La base
d'Hermite vit à part, dans `src/atoms/element_kind/hermite.rs` : seul un
`SEG2` l'accepte.
