# PENTA6 — prisme (pentaèdre) linéaire

Prisme à **6 nœuds**, interpolation **Lagrange-1**. C'est l'**extrusion d'un
TRI3** le long de \\( \zeta \\) : produit d'un triangle (coordonnées
barycentriques) par un segment linéaire. Produit par l'opérateur `extrude`
(TRI3 → PENTA6) et par `sweep_solid`.

## Repère de référence

Triangle \\( \xi, \eta \in [0, 1] \\), \\( \xi + \eta \le 1 \\), extrudé sur
\\( \zeta \in [0, 1] \\). Triangle inférieur (\\( \zeta = 0 \\)) puis triangle
supérieur (\\( \zeta = 1 \\)), chacun CCW.

| Nœud | \\( (\xi, \eta, \zeta) \\) | | Nœud | \\( (\xi, \eta, \zeta) \\) |
|---:|:---:|---|---:|:---:|
| 0 | \\( (0, 0, 0) \\) | | 3 | \\( (0, 0, 1) \\) |
| 1 | \\( (1, 0, 0) \\) | | 4 | \\( (1, 0, 1) \\) |
| 2 | \\( (0, 1, 0) \\) | | 5 | \\( (0, 1, 1) \\) |

## Fonctions de forme

Avec les barycentriques du triangle \\( L_1 = 1-\xi-\eta \\), \\( L_2 = \xi \\),
\\( L_3 = \eta \\) et le facteur linéaire en \\( \zeta \\) :

\\[
N_j = L_j\,(1 - \zeta) \quad (j = 0, 1, 2), \qquad
N_{j+3} = L_{j+1}\,\zeta \quad (j = 0, 1, 2).
\\]

(les nœuds 0..2 portent \\( L_1, L_2, L_3 \\) à \\( \zeta=0 \\) ; les nœuds
3..5, les mêmes à \\( \zeta=1 \\)).

## Dérivées de référence

Par exemple, pour le nœud 0 (\\( L_1(1-\zeta) \\)) :

\\[
\nabla_\xi N_0 = \big(-(1-\zeta),\ -(1-\zeta),\ -L_1\big),
\\]

les autres suivant le même schéma (dérivées de \\( L_j \\) constantes,
\\( \partial_\zeta \\) porté par le facteur \\( \zeta \\)).

## Quadrature (défaut)

**Produit tensoriel** de la règle TRI3 (3 points, \\( w = 1/6 \\)) par la règle
de Gauss à 2 points sur \\( \zeta \in [0, 1] \\) (\\( \zeta_g = \tfrac12 \pm \tfrac{1}{2\sqrt3} \\),
\\( w = \tfrac12 \\)), soit **6 points** :

\\[
w_g = \tfrac16\cdot\tfrac12 = \tfrac{1}{12}, \qquad \sum_g w_g = \tfrac12.
\\]

## Notes

- Dimensions valides : \\( d_r = d_s = 3 \\).
- Exact au degré \\( \le 2 \\) dans le plan du triangle et \\( \le 3 \\) selon
  \\( \zeta \\).
- Utile pour mailler par couches (extrusion d'un maillage surfacique TRI3).
- Version quadratique sérendipité : [PENTA15](penta15.md).
