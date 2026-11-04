# Poutre d'Euler-Bernoulli

## En bref

La théorie classique des poutres : les sections restent planes **et
normales** à l'axe déformé, si bien que la rotation de la section est la
pente, et qu'il n'y a aucun cisaillement transverse.

| | |
|---|---|
| opérateur | `pyrucast.model.bernoulli(fespace)` |
| tag | `mechanical` |
| type | domaine avec comportement (`label()` : `"Bernoulli"`) |
| éléments | `SEG2`, sur un espace `FiniteElementSpace(maillage, interpolation="HERMITE3")` |

La configuration se lit sur la dimension des `Coords` :

| `Coords` | inconnues par nœud | ce qui s'ajoute à la flexion |
|---|---|---|
| 1-D | `w`, `theta` | rien : flexion pure |
| 2-D | `u_x`, `u_y`, `r_z` | l'effort axial, et la rotation vers les axes globaux : un portique plan |
| 3-D | `u_x`, `u_y`, `u_z`, `r_x`, `r_y`, `r_z` | l'axial, la torsion et la flexion selon deux axes principaux : un cadre spatial |

## Théorie

La section rigide tourne avec la pente :

\\[
u_x(x, y) = u(x) - y\\,w\'(x), \qquad u_y = w(x), \qquad \theta = w\' ,
\\]

d'où une déformation axiale affine dans la section,
\\(\varepsilon_{xx} = \varepsilon_0 - y\\,\chi\\), de courbure
\\(\chi = w\'\'\\), et aucune distorsion. Intégrée sur la section, la loi de
Hooke donne les efforts de section,

\\[
N = E A\\,\varepsilon_0, \qquad M = E I\\,\chi ,
\\]

et l'équilibre deux équations découplées :

\\[
(E A\\,u\')\' + n = 0, \qquad (E I\\,w\'\')\'\' = q .
\\]

La seconde est du **quatrième ordre** : sa forme faible demande une
interpolation **\\(C^1\\)**, le déplacement et sa pente continus d'un élément
au suivant. C'est ce que donne l'interpolation
[`HERMITE3`](../fe-space.md#fonctions-de-forme-dhermite-hermite3), qui prend la
flèche et la pente à chaque extrémité. Un espace de Lagrange est refusé : sa
flèche linéaire aurait une courbure nulle.

La courbure est linéaire sur l'élément : il est **exact aux nœuds** pour toute
travée chargée à ses seules extrémités, et un élément par barre suffit. La
raideur de flexion \\(\int_0^L E I\\,N\'\'^{\mathsf T} N\'\'\\) est intégrée depuis
la base et vaut la forme fermée classique,

\\[
K_b = \frac{E I}{L^3}
\begin{bmatrix}
12 & 6L & -12 & 6L \\\\
6L & 4L^2 & -6L & 2L^2 \\\\
-12 & -6L & 12 & -6L \\\\
6L & 2L^2 & -6L & 4L^2
\end{bmatrix},
\qquad [\\,w_A,\ \theta_A,\ w_B,\ \theta_B\\,].
\\]

L'effort axial s'y ajoute par le terme de barre \\(E A/L\\), sans couplage. En
3-D, la flexion selon le second axe principal (\\(M_z = E I_z\\,\chi_z\\)) et la
torsion de Saint-Venant (\\(T = G J\\,\varphi\'\\)) s'y ajoutent ; le repère local
est déduit d'une référence globale `Z` (ou `Y` pour une barre presque
verticale), sans donnée d'orientation, ce qui convient aux sections
symétriques.

Bernoulli est la limite sans cisaillement de [Timoshenko](timoshenko.md). Une
physique à part évite d'y arriver par \\(G A_s \to \infty\\), où le terme de
cisaillement noierait la flexion en virgule flottante, et évite de demander
`G` et `A_s`, que cette théorie n'utilise pas.

## Variables

| | 1-D | 2-D | 3-D |
|---|---|---|---|
| primales | `w`, `theta` | `u_x`, `u_y`, `r_z` | `u_x`, `u_y`, `u_z`, `r_x`, `r_y`, `r_z` |
| duales | `f_w`, `m_theta` | `f_x`, `f_y`, `m_z` | `f_x`, `f_y`, `f_z`, `m_x`, `m_y`, `m_z` |
| matériau requis | `E`, `I` | `E`, `A`, `I` | `E`, `A`, `I_y`, `I_z`, `J`, `G` |
| matériau facultatif | `A`, `rho` | `rho` | `rho` |
| entrée du comportement | `kappa` | `eps`, `kappa` | `eps`, `kappa_y`, `kappa_z`, `torsion` |
| sortie du comportement | `M` | `N`, `M` | `N`, `M_y`, `M_z`, `T` |
| état | aucun | aucun | aucun |

L'entrée vient de `pyrucast.element_field.beam_deformation`, ou de `strain`.

## Ce qu'elle produit

| | |
|---|---|
| raideur | flexion, et axial et torsion selon la configuration |
| masse | exige `rho` (et `A` en 1-D) ; le bloc de Timoshenko pris à \\(\Phi = 0\\) |
| géométrique | sous l'effort `N` ; aucune en 1-D, sans effort axial |
| tangente | = raideur |
| forces internes | le transposé du même \\(B\\) que la raideur : \\(\int B^{\mathsf T}\sigma = K\\,u\\) exactement |

## Exemple

Une console de longueur 2 dans le plan, encastrée, chargée en bout. La flèche
est exacte, \\(F\\,L^3/(3 E I)\\) :

```python
{{#include ../../../tests/python/test_doc_mecanique.py:bernoulli}}
```

## Côté Rust

`Bernoulli` (`src/models/bernoulli.rs`) refuse à la construction un espace
qui n'est pas `HERMITE3`. Son \\(B\\) est celui de `src/models/beam.rs`, partagé
avec Timoshenko pour les forces internes. `tests/hermite.rs` compare la
raideur intégrée à la forme fermée, à la précision machine ; le test
ci-dessous compare la console sous charge et sous moment en bout, la traction
et la torsion aux formules du cours, à `1e-12` :

```rust,ignore
{{#include ../../../tests/bernoulli.rs:example}}
```
