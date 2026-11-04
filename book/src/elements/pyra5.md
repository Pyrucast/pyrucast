# PYRA5 — pyramide linéaire

## En bref

La pyramide à base carrée relie des hexaèdres à des tétraèdres : sa face carrée
s'appuie sur une face de `HEX8`, ses quatre faces triangulaires sur des faces
de `TET4`. Une couche d'hexaèdres se referme ainsi sur un cœur de tétraèdres
sans nœud isolé au milieu d'une face.

| | |
|---|---|
| nom | `"PYRA5"` |
| nœuds | 5 : les quatre coins de la base, le sommet |
| interpolation | `"LAGRANGE1"` : bilinéaire sur chaque section, rationnelle |
| dimension | 3 ; dans des `Coords` 3-D |
| bord | 1 `QUA4`, 4 `TRI3` |
| quadrature | `"GAUSS"` : 8 points, degré 3 ; `"REDUCED"` : 1 point |
| version quadratique | aucune : `to_quadratic` refuse une pyramide |
| obtenu par | `pave_volume` |

## Élément de référence

\\( \zeta \in [0, 1] \\), et \\( \xi, \eta \in [-(1-\zeta),\ +(1-\zeta)] \\) :
la section carrée rétrécit avec \\( \zeta \\) jusqu'au sommet. La base tourne
dans le sens trigonométrique vue depuis le sommet.

| Nœud | \\( (\xi, \eta, \zeta) \\) | | Nœud | \\( (\xi, \eta, \zeta) \\) |
|---:|:---:|---|---:|:---:|
| 0 | \\( (-1, -1, 0) \\) | | 3 | \\( (-1, 1, 0) \\) |
| 1 | \\( (1, -1, 0) \\) | | 4 | \\( (0, 0, 1) \\) |
| 2 | \\( (1, 1, 0) \\) | | | |

## Fonctions de forme

Ses fonctions ne sont **pas polynomiales** : la base carrée doit se réduire à
un point au sommet, et aucun polynôme ne le fait en restant bilinéaire sur la
base. Avec \\( m = 1 - \zeta \\), la demi-largeur de la section, ce sont les
fonctions bilinéaires du carré dans les coordonnées \\( \xi/m \\),
\\( \eta/m \\), multipliées par \\( m \\) :

\\[
N_i = \frac{m}{4}\left(1 + \xi_i\\,\frac{\xi}{m}\right)\left(1 + \eta_i\\,\frac{\eta}{m}\right)
\quad (i = 0 \dots 3), \qquad N_4 = \zeta,
\\]

où \\( (\xi_i, \eta_i) \\) sont les signes du nœud \\( i \\). Développé,

\\[
N_i = \frac{1}{4}\left(m + \xi_i\\,\xi + \eta_i\\,\eta + \xi_i\eta_i\\,\frac{\xi\eta}{m}\right).
\\]

Le terme \\( \xi\eta/m \\) est la partie rationnelle. Il reste borné dans
l'élément, \\( |\xi\eta/m| \le m \\), mais n'est pas défini **au** sommet, où
le code pose directement \\( N_4 = 1 \\) et les autres à 0.

Sur la base, \\( \zeta = 0 \\), les fonctions sont exactement celles d'un
`QUA4` ; le long d'une arête de la base au sommet, elles sont linéaires. C'est
ce qui raccorde la pyramide sans trou à un `HEX8` par sa face carrée, et à un
`TET4` par une face triangulaire.

## Dérivées de référence

Avec \\( u = \xi/m \\), \\( v = \eta/m \\) :

\\[
\frac{\partial N_i}{\partial \xi} = \frac{\xi_i}{4}\\,(1 + \eta_i v), \qquad
\frac{\partial N_i}{\partial \eta} = \frac{\eta_i}{4}\\,(1 + \xi_i u), \qquad
\frac{\partial N_i}{\partial \zeta} = \frac{1}{4}\\,(-1 + \xi_i\eta_i\\,u v),
\\]

et \\( \nabla_\xi N_4 = (0, 0, 1) \\).

## Degrés de liberté

| physique | inconnues par nœud | par maille |
|---|---|---:|
| conduction, diffusion | `T` | 5 |
| élasticité 3-D | `u_x`, `u_y`, `u_z` | 15 |

## Quadrature

La pyramide n'est le produit d'aucune paire d'éléments plus simples. Sa règle
`"GAUSS"` est **conique** : on écrit un point \\( \xi = a(1-\zeta) \\),
\\( \eta = b(1-\zeta) \\), avec \\( a, b \in [-1, 1] \\), et le changement de
variables donne

\\[
d\xi\\,d\eta = (1-\zeta)^2\\,da\\,db ,
\\]

le rétrécissement de la section. La règle croise Gauss-Legendre 2×2 sur
\\( (a, b) \\) et, en \\( \zeta \\), une règle de **Gauss-Jacobi** à 2 points,
faite pour intégrer contre ce poids \\( (1-\zeta)^2 \\). Ses deux points sont
les racines de \\( z^2 - \tfrac23 z + \tfrac1{15} \\),

\\[
\zeta_g = \frac13 \mp \frac{\sqrt{10}}{15} \simeq 0{,}12251 \ \text{et}\ 0{,}54415,
\\]

et ses poids, \\( 0{,}23255 \\) et \\( 0{,}10079 \\), se déduisent des deux
premiers moments de \\( (1-z)^2 \\) sur \\( [0,1] \\) :
\\( \sum w = 1/3 \\), \\( \sum w z = 1/12 \\). Soit 8 points, et une somme des
poids \\( 2 \times 2 \times \tfrac13 = \tfrac43 \\), le volume de la pyramide
de référence.

Exprimées en \\( (a, b, \zeta) \\), les fonctions de forme redeviennent des
polynômes : la règle intègre exactement les monômes jusqu'au degré 3, et la
masse et la raideur de toute pyramide dont la base est un parallélogramme.
Sur une base quelconque, la raideur n'est plus exacte, comme sur un `QUA4` ou
un `HEX8` déformé.

`"REDUCED"` : un point au centre de gravité \\( (0, 0, \tfrac14) \\), au quart
de la hauteur, de poids \\( \tfrac43 \\).

## Exemple

```python
{{#include ../../../tests/python/test_doc_elements.py:pyra5}}
```

## Côté Rust

`Pyra5` (`src/atoms/element_kind/pyra5.rs`) implémente `ElementKind`. Au
sommet, `shape_into` et `dshape_into` rendent des valeurs fixées au lieu de
diviser par \\( m \\). `pyra5_jacobian_volume`, dans
`src/containers/finite_element_space/`, vérifie le volume d'une pyramide à
base rectangulaire et à sommet décentré.
