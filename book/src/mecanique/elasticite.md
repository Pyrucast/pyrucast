# Élasticité linéaire

## En bref

Un solide qui se déforme réversiblement, proportionnellement à la charge (loi
de Hooke), en petites déformations.

| | |
|---|---|
| opérateur | `pyrucast.model.elasticity(fespace, kinematics, symmetry="isotropic", strain_measure="linear")` |
| tag | `mechanical` |
| type | domaine avec comportement (`label()` : `"Elasticity"`) |
| éléments | les mailles de Lagrange 2-D (`TRI3`, `QUA4`, `TRI6`, `QUA8`, `QUA9`) et 3-D (`TET4`, `PYRA5`, `PENTA6`, `HEX8` et leurs quadratiques) |

`kinematics` est l'hypothèse cinématique (voir [Mécanique](../mecanique.md)),
`symmetry` la symétrie du matériau (voir
[Orthotropie](orthotropie.md)), `strain_measure` la mesure de déformation (voir
[Grandes transformations](grandes-transformations.md)). Cette page décrit le cas
isotrope en petites déformations.

Un maillage de bord (des segments en 2-D) est refusé comme domaine : \\(B\\) y
serait bâti sur le seul gradient tangent, et la raideur n'aurait aucune
rigidité dans la direction normale.

## Théorie

Sur le domaine \\(\Omega\\), avec \\(b\\) les forces de volume :

\\[
\nabla\cdot\sigma + b = 0, \qquad
\sigma = \mathbb D : \varepsilon, \qquad
\varepsilon = \tfrac12\big(\nabla u + \nabla u^{\mathsf T}\big).
\\]

La forme faible : pour tout déplacement virtuel \\(v\\),

\\[
\int_\Omega \varepsilon(v) : \mathbb D : \varepsilon(u)\\,d\Omega =
\int_\Omega v\cdot b\\,d\Omega + \int_{\Gamma_N} v\cdot t\\,d\Gamma ,
\\]

où \\(t\\) est la traction imposée sur le bord \\(\Gamma_N\\).

Avec \\(u \approx \sum_i N_i\\,u_i\\), la déformation s'écrit
\\(\varepsilon = B\\,u_e\\). En 2-D,
\\(\varepsilon = [\varepsilon_{xx}, \varepsilon_{yy}, \gamma_{xy}]\\) et le bloc
du nœud \\(i\\) est

\\[
B_i = \begin{bmatrix}
\partial_x N_i & 0 \\\\
0 & \partial_y N_i \\\\
\partial_y N_i & \partial_x N_i
\end{bmatrix};
\\]

en 3-D, \\(\varepsilon = [\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{zz}, \gamma_{yz}, \gamma_{xz}, \gamma_{xy}]\\)
avec un bloc de six lignes de même forme. La raideur et la masse de l'élément
sont

\\[
K_e = \int_{\Omega_e} B^{\mathsf T} D\\,B\\,d\Omega ,
\qquad
M_e = \int_{\Omega_e} \rho\\,N^{\mathsf T} N\\,d\Omega ,
\\]

intégrées aux points de Gauss, où le matériau est lu.

**La matrice de Hooke \\(D\\)**, avec \\(E\\) le module d'Young et \\(\nu\\) le
coefficient de Poisson :

- `plane_stress` (\\(\sigma_{zz} = 0\\)), avec \\(c = E/(1-\nu^2)\\) :
  \\(D = c\\,[[1, \nu, 0], [\nu, 1, 0], [0, 0, (1-\nu)/2]]\\) ;
- `plane_strain` (\\(\varepsilon_{zz} = 0\\)), avec
  \\(c = E/((1+\nu)(1-2\nu))\\) :
  \\(D = c\\,[[1-\nu, \nu, 0], [\nu, 1-\nu, 0], [0, 0, (1-2\nu)/2]]\\) ;
- `full_3d`, même \\(c\\) : le bloc normal \\(3\times3\\) de
  \\(c(1-\nu)\\) sur la diagonale et \\(c\nu\\) ailleurs, et
  \\(G = c\\,(1-2\nu)/2\\) sur les trois cisaillements ;
- `axisymmetric`, même \\(c\\) : le bloc normal isotrope sur
  \\([rr, zz, \theta\theta]\\) et \\(G\\) sur \\(rz\\).

**Axisymétrie.** Un solide de révolution se maille dans son plan méridien
\\((r, z)\\). Deux choses changent. La mesure d'intégration devient
\\(2\pi r\\,|J|\\), pour toutes les intégrales (raideur, masse, forces) ; c'est
la géométrie qui l'apporte, pas la physique. Et la déformation orthoradiale
\\(\varepsilon_{\theta\theta} = u_r / r\\) ajoute une ligne à \\(B\\),
\\([N_i/r,\\; 0]\\) ; elle est rangée dans `eps_zz`. Les points de Gauss étant
intérieurs aux mailles, \\(r > 0\\) même pour une maille qui touche l'axe.

**Dilatation thermique.** Une température impose une déformation libre
\\(\varepsilon_{th} = \alpha\\,(T - T_\text{ref})\\), sans effet de la mécanique
sur la thermique. La raideur ne change pas ; le terme thermique agit sur la
contrainte et le second membre :

\\[
\sigma = D : (\varepsilon(u) - \varepsilon_{th}), \qquad
f_{th} = \int_\Omega B^{\mathsf T} D\\,\varepsilon_{th}\\,d\Omega .
\\]

`alpha` est une composante facultative du matériau ;
`pyrucast.element_field.thermal_strain(temperature, materiaux, fespace, t_ref)`
calcule \\(\varepsilon_{th}\\) aux points de Gauss depuis une température
portée aux points de Gauss par `interp_to_gauss` ; `integrate_behavior` puis
`internal_forces` en tirent la charge thermique.

## Variables

En 2-D plan, isotrope, en petites déformations :

| | |
|---|---|
| primales | `u_x`, `u_y` (et `u_z` en 3-D) |
| duales | `f_x`, `f_y` (et `f_z`) |
| matériau requis | `E`, `nu` |
| matériau facultatif | `rho`, `alpha` |
| entrée du comportement | `eps_xx`, `eps_yy`, `eps_xy` (six composantes en 3-D, quatre en axisymétrie) |
| sortie du comportement | `sigma_xx`, `sigma_yy`, `sigma_xy` (de même) |
| état | aucun |

## Ce qu'elle produit

| | |
|---|---|
| raideur | \\(K_e\\) |
| masse | \\(M_e\\), exige `rho` |
| géométrique | \\(\int G^{\mathsf T}\hat\sigma\\,G\\) sous la contrainte courante, lue dans `sigma_*` |
| tangente | analytique : le module élastique \\(D\\), d'où la raideur |
| forces internes | \\(\int B^{\mathsf T}\sigma\\) |

La raideur géométrique est décrite avec les
[opérateurs d'assemblage](../operateurs/assemblage.md).

## Exemple

Une plaque carrée en contraintes planes, tenue en `u_x` sur son bord gauche et
en `u_y` à son coin inférieur gauche, tirée par une traction \\(S = 100\\) sur
son bord droit. La solution est exacte : \\(u_x = S\\,x/E\\),
\\(u_y = -\nu\\,S\\,y/E\\) :

```python
{{#include ../../../tests/python/test_doc_mecanique.py:elasticite}}
```

## Côté Rust

`Elasticity` (`src/models/elasticity.rs`) délègue sa cinématique à un
`Continuum` (`src/models/continuum/`), qui porte \\(B\\) pour chaque
hypothèse et le noyau de forces internes ; la matrice \\(D\\) vient de
`src/models/symmetry.rs`, l'isotropie gardant ses formes fermées.

```rust,ignore
{{#include ../../../tests/elasticity.rs:example}}
```

Sur un solide de révolution, `tests/axisymmetric.rs` vérifie la solution de
Lamé d'un cylindre épais : avec `QUA8`, l'erreur sur les déplacements nodaux
décroît à l'ordre 4 environ et celle des contraintes à l'ordre 2, sans jamais
s'annuler, \\(1/r\\) n'étant pas un polynôme.

La barre chauffée vérifie la dilatation thermique : encastrée aux deux bouts,
\\(\sigma_{xx} = -E\\,\alpha\\,\Delta T\\) ; libre, une dilatation sans
contrainte :

```rust,ignore
{{#include ../../../tests/thermoelastic_bar.rs:example}}
```
