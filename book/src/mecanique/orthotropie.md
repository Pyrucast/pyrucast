# Élasticité orthotrope et anisotrope

## En bref

Un composite tissé, un bois, une tôle laminée ou un monocristal n'ont pas la
même raideur dans toutes les directions. C'est l'option `symmetry` de
l'[élasticité](elasticite.md), indépendante de la cinématique :

| `symmetry` | constantes | ce qu'elle décrit |
|---|---|---|
| `"isotropic"` | 2 | aucune direction privilégiée |
| `"orthotropic"` | 9 | trois plans de symétrie orthogonaux |
| `"anisotropic"` | 21 | le cas général |

Les variables ne changent pas : seule la matrice de Hooke \\(D\\) change. Les
lois non linéaires (plasticité, fluage, endommagement de Mazars et Damage TC)
gardent une élasticité isotrope ; la loi [SiC/SiC](endommagement.md) a ses
propres axes.

## Théorie

La loi de comportement est \\(\sigma_{ij} = C_{ijkl}\\,\varepsilon_{kl}\\). Les
symétries de \\(\sigma\\) et \\(\varepsilon\\) et l'existence d'un potentiel
élastique réduisent \\(C\\) à une matrice \\(6\times6\\) symétrique en notation
de Voigt : **21 constantes** dans le cas général.

**Orthotropie.** Dans ses axes propres, la souplesse \\(S = C^{-1}\\) découple
les termes normaux des cisaillements :

\\[
S =
\begin{bmatrix}
1/E_1 & -\nu_{21}/E_2 & -\nu_{31}/E_3 & & & \\\\
-\nu_{12}/E_1 & 1/E_2 & -\nu_{32}/E_3 & & & \\\\
-\nu_{13}/E_1 & -\nu_{23}/E_2 & 1/E_3 & & & \\\\
 & & & 1/G_{23} & & \\\\
 & & & & 1/G_{13} & \\\\
 & & & & & 1/G_{12}
\end{bmatrix},
\qquad \frac{\nu_{ji}}{E_j} = \frac{\nu_{ij}}{E_i} .
\\]

Les relations de réciprocité ne laissent que **neuf** constantes. \\(S\\) doit
être définie positive : un jeu de constantes qui la rend singulière est
refusé, plutôt que d'assembler une raideur fausse.

**Les axes matériau** sont donnés par des vecteurs du champ matériau : `V1X`,
`V1Y` en 2-D (le second axe est la normale dans le plan, le troisième la
direction hors du plan, l'orthoradiale en axisymétrie) ; `V1X…V1Z` et
`V2X…V2Z` en 3-D (le troisième est \\(V_1 \times V_2\\)). Ils sont
orthonormalisés en interne : \\(V_2\\) n'a besoin d'être qu'à peu près
perpendiculaire à \\(V_1\\). Un \\(V_1\\) nul, ou un \\(V_2\\) parallèle à
\\(V_1\\), est refusé. Comme toute composante matériau, les axes peuvent varier
d'une maille et d'un point de Gauss à l'autre : un composite bobiné, une pièce
courbe.

En chaque point de Gauss, \\(D\\) est construit dans les axes matériau, tourné
dans les axes globaux par le tenseur d'ordre 4,

\\[
C\'\_{pqrs} = R_{pi}\\,R_{qj}\\,R_{rk}\\,R_{sl}\\,C_{ijkl},
\qquad R = \big[\\,V_1\ \ V_2\ \ V_1\times V_2\\,\big],
\\]

puis réduit à la cinématique : le bloc `[xx, yy, xy]` en déformations planes,
sa condensation sur \\(\varepsilon_{zz}\\) en contraintes planes, le bloc
`[rr, zz, θθ, rz]` en axisymétrie. En cisaillement de l'ingénieur, le passage
Voigt ↔ tenseur ne porte aucun facteur, et la rotation d'ordre 4 n'a aucune
convention à retenir.

## Variables

| `symmetry` | matériau requis |
|---|---|
| `"isotropic"` | `E`, `nu` |
| `"orthotropic"` | `E_1`, `E_2`, `E_3`, `nu_12`, `nu_13`, `nu_23`, `G_12`, `G_13`, `G_23` et les axes |
| `"anisotropic"` | `C_11`…`C_66`, le triangle supérieur dans l'ordre `[xx, yy, zz, yz, xz, xy]`, et les axes |

Les neuf constantes orthotropes sont demandées même en 2-D : le tenseur est
construit en 3-D avant d'être réduit. Les trois jeux de composantes sont
disjoints, si bien qu'une zone isotrope et une zone orthotrope partagent un
maillage sans rien consolider. L'entrée et la sortie du comportement sont
celles de l'élasticité isotrope.

## Exemple

Les constantes d'un matériau quatre fois plus raide selon son axe 1 :

```python
{{#include ../../../tests/python/test_doc_mecanique.py:orthotrope}}
```

Sur la plaque en traction de la page [Élasticité](elasticite.md#exemple), le
fichier de tests vérifie les deux bornes : \\(u_x = S/E_1\\) quand l'axe raide
suit la traction, \\(S/E_2\\) quand il lui est perpendiculaire.

## Côté Rust

`src/models/symmetry.rs` construit les tenseurs orientés pour l'élasticité,
la conduction et la diffusion ; `MaterialSymmetry` est l'énumération des trois
symétries. Le test ci-dessous étire un carré orthotrope selon son premier axe
matériau ; le même fichier nourrit une loi orthotrope de constantes isotropes,
puis une loi anisotrope du tenseur isotrope, qui doivent se comporter comme
l'isotrope quel que soit le repère, et refuse un repère dégénéré.

```rust,ignore
{{#include ../../../tests/orthotropic.rs:example}}
```
