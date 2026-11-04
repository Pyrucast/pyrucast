# Damage TC, SiC/SiC et Gurson

## En bref

Trois lois où un matériau perd sa raideur en se dégradant :

| | |
|---|---|
| opérateurs | `pyrucast.model.damage_tc(fespace, kinematics, strain_measure="linear")`, `pyrucast.model.damage_sic_sic(fespace, kinematics, strain_measure="linear")`, `pyrucast.model.gurson(fespace, kinematics, strain_measure="linear")` |
| tag | `mechanical` |
| type | domaine avec comportement (`label()` : `"Damage"` ; `"Plasticity"` pour Gurson) |
| éléments | 2-D, en plan ou en axisymétrie : `TRI3`, `QUA4`, `TRI6`, `QUA8`, `QUA9` ; 3-D : `TET4`, `PYRA5`, `PENTA6`, `HEX8`, `TET10`, `PENTA15`, `HEX20`, `HEX27` |
| cinématiques | `plane_stress`, `plane_strain`, `axisymmetric`, `full_3d` |

| loi | variables d'endommagement | ce qu'elle décrit |
|---|---|---|
| `damage_tc` | \\(d^+\\), \\(d^-\\) | béton, traction et compression séparées |
| `damage_sic_sic` | \\(d_1\\), \\(d_2\\), \\(d_3\\) | composite tissé, un endommagement par direction |
| `gurson` | la porosité \\(f\\) | métal ductile, rupture par croissance de cavités |

Damage TC et SiC/SiC sont des lois d'endommagement, comme
[Mazars](mazars.md) ; Gurson est une loi plastique.

## Théorie

**Le cadre de l'endommagement.** La contrainte effective
\\(\tilde\sigma = D_\text{el} : \varepsilon\\) est celle du matériau sain sous
la même déformation ; la contrainte est sa version dégradée. Un moteur
scalaire \\(\tau\\), construit sur \\(\varepsilon\\) ou \\(\tilde\sigma\\),
alimente une variable d'histoire \\(r = \max(r_0, \max_t \tau)\\), qui ne
décroît jamais, et l'endommagement est une fonction croissante de \\(r\\). Il
n'y a pas de déformation permanente, et la mise à jour se fait en un pas, sans
itérer.

### Damage TC

Les parts positive et négative de la contrainte effective, découpées sur ses
valeurs principales, sont dégradées chacune par sa variable :

\\[
\sigma = (1 - d^+)\\,\tilde\sigma^+ + (1 - d^-)\\,\tilde\sigma^- .
\\]

Une fissure ouverte en traction qui se referme reprend donc toute la raideur en
compression : c'est l'effet **unilatéral**, que le scalaire de Mazars ne peut
pas rendre. Chaque variable a son moteur et son histoire :

\\[
\tau^+ = \sqrt{\tilde\sigma^+ : \varepsilon}, \qquad
\tau^- = \sqrt{\sqrt3\\,\big|K\\,\tilde\sigma^-\_\text{oct} + \tilde\tau^-\_\text{oct}\big|}, \qquad
r_0^+ = \frac{f_t}{\sqrt E}, \quad r_0^- = \frac{f_c}{\sqrt E},
\\]

l'énergie élastique de la part positive en traction, une mesure octaédrique
(contrainte normale \\(\tfrac13\operatorname{tr}\tilde\sigma^-\\) et
cisaillement) sensible au confinement en compression, avec \\(K = 0{,}171\\).
L'adoucissement est exponentiel en traction, durcissant puis adoucissant en
compression :

\\[
d^+ = 1 - \frac{r_0^+}{r^+}\\,e^{A_t(1 - r^+/r_0^+)}, \qquad
d^- = 1 - \frac{r_0^-}{r^-}(1 - A_c) - A_c\\,e^{2(1 - r^-/r_0^-)} .
\\]

### SiC/SiC

Dans un composite tissé, la matrice fissure dans des plans normaux aux torons,
et les fibres continuent de porter la charge : la raideur chute par direction.
Chaque direction matériau \\(i\\) a son endommagement, piloté par la
déformation **dans les axes matériau** \\(\varepsilon^\text{mat} = R^{\mathsf T}\varepsilon R\\) :

\\[
\kappa_i = \max_t \langle\varepsilon^\text{mat}\_{ii}\rangle_+, \qquad
d_i = d_{\max,i}\Big(1 - e^{-(\kappa_i - \varepsilon_{0,i})/\varepsilon_{c,i}}\Big)
\ \text{si } \kappa_i > \varepsilon_{0,i} .
\\]

Une direction comprimée n'est pas dégradée. La dégradation s'applique terme à
terme dans les axes matériau,
\\(\sigma^\text{mat}\_{ij} = \sqrt{1-d_i}\sqrt{1-d_j}\\;\tilde\sigma^\text{mat}\_{ij}\\),
ce qui garde l'opérateur symétrique. Chaque \\(d_i\\) sature à \\(d_{\max,i}\\)
au lieu d'atteindre 1 : les fibres restent. Les axes sont ceux de
l'[orthotropie](orthotropie.md), les vecteurs `V1` (et `V2` en 3-D) du
matériau ; l'élasticité, elle, est isotrope.

### Gurson

La porosité \\(f\\) rétrécit la surface de charge :

\\[
\Phi = \Big(\frac{q}{\sigma_y}\Big)^2 + 2\\,q_1 f^* \cosh\\!\Big(\frac{3\\,q_2\\,\sigma_m}{2\\,\sigma_y}\Big) - \big(1 + q_3\\,f^{*2}\big),
\\]

qui redonne von Mises à \\(f = 0\\). Le `cosh` rend la surface sensible à la
pression : les cavités croissent en traction triaxiale. Au-delà d'une porosité
critique \\(f_c\\), une porosité effective \\(f^*\\), bilinéaire en \\(f\\),
accélère la coalescence et atteint \\(1/q_1\\) à la porosité de rupture
\\(f_f\\). La porosité croît par conservation de la masse,
\\(\dot f = (1 - f)\\,\operatorname{tr}\dot\varepsilon^p\\), à partir de sa valeur
initiale \\(f_0\\) ; la germination n'est pas modélisée. L'écoulement n'est
donc pas isochore. Le retour se fait par plan sécant à normale numérique,
comme celui d'[Ottosen](lois-plastiques.md#ottosen), et la tangente par
perturbation.

## Variables

Selon la cinématique :

| | `plane_stress`, `plane_strain` | `axisymmetric` | `full_3d` |
|---|---|---|---|
| primales | `u_x`, `u_y` | `u_x`, `u_y` | `u_x`, `u_y`, `u_z` |
| duales | `f_x`, `f_y` | `f_x`, `f_y` | `f_x`, `f_y`, `f_z` |
| entrée du comportement | `eps_xx`, `eps_yy`, `eps_xy` | `eps_xx`, `eps_yy`, `eps_zz`, `eps_xy` | `eps_xx`, `eps_yy`, `eps_zz`, `eps_yz`, `eps_xz`, `eps_xy` |
| contrainte | `sigma_xx`, `sigma_yy`, `sigma_xy` | `sigma_xx`, `sigma_yy`, `sigma_zz`, `sigma_xy` | `sigma_xx`, `sigma_yy`, `sigma_zz`, `sigma_yz`, `sigma_xz`, `sigma_xy` |

En axisymétrie, `eps_zz` et `sigma_zz` sont les composantes orthoradiales.

La sortie du comportement est la contrainte, suivie de l'état. Matériau
facultatif pour les trois lois : `rho`, `alpha`.

| loi | matériau requis | état, écrit après la contrainte | état relu (`prev`) |
|---|---|---|---|
| `damage_tc` | `E`, `nu`, `f_t`, `f_c`, `A_t`, `A_c` | `damage`, `r_plus`, `r_minus`, `d_plus`, `d_minus` | `r_plus`, `r_minus`, `d_plus`, `d_minus` |
| `damage_sic_sic` | `E`, `nu`, `eps_0_1`, `eps_c_1`, `d_max_1`, `eps_0_2`, `eps_c_2`, `d_max_2`, `eps_0_3`, `eps_c_3`, `d_max_3`, et les axes : `V1X`, `V1Y` en plan et en axisymétrie ; `V1X`, `V1Y`, `V1Z`, `V2X`, `V2Y`, `V2Z` en 3-D | `damage`, `kappa_1`, `kappa_2`, `kappa_3`, `d_1`, `d_2`, `d_3` | `kappa_1`, `kappa_2`, `kappa_3`, `d_1`, `d_2`, `d_3` |
| `gurson` | `E`, `nu`, `sigma_y`, `q_1`, `q_2`, `q_3`, `f_0`, `f_c`, `f_f` | la déformation plastique `eps_p_xx`, `eps_p_yy`, `eps_p_zz`, `eps_p_yz`, `eps_p_xz`, `eps_p_xy` ; la déformation plastique cumulée `p` ; la déformation totale `eps_xx`, `eps_yy`, `eps_zz`, `eps_yz`, `eps_xz`, `eps_xy` ; et, en `plane_stress` et `plane_strain`, `sigma_zz` ; puis `porosity` | toute la sortie du pas précédent |

## Ce qu'elles produisent

| | |
|---|---|
| raideur | élastique, non endommagée |
| masse | exige `rho` |
| géométrique | sous la contrainte `sigma_*` |
| tangente | aucune pour Damage TC et SiC/SiC (`matrix.tangent` lève une erreur) ; par perturbation pour Gurson |
| forces internes | \\(\int B^{\mathsf T}\sigma\\) |

## Exemple

Une maille en traction sous Damage TC : seule la variable de traction croît.

```python
{{#include ../../../tests/python/test_doc_mecanique.py:damage_tc}}
```

## Côté Rust

Damage TC et SiC/SiC vivent dans `src/models/damage/`, Gurson dans
`src/models/plasticity/gurson.rs`. Une loi déclare l'état qu'elle relit.
L'état de repos que rend `initial_state` est nul, sauf la porosité de Gurson :
la loi déclare qu'elle part de `f_0` (`initial_internal_sources`), et l'état
de repos la lit dans le matériau.

`tests/damage_laws.rs` vérifie qu'une fissure refermée reprend la charge en
compression, que l'écrasement laisse l'endommagement de traction intact,
qu'étirer un axe de tissage n'endommage que lui et sature à son plafond, que
la porosité de Gurson croît en traction triaxiale et qu'un métal poreux
plastifie avant un métal dense.

```rust,ignore
{{#include ../../../tests/damage_laws.rs:example}}
```
