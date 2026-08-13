# Poutre d'Euler-Bernoulli

## Introduction

La théorie classique des poutres : les sections planes restent planes **et
normales** à l'axe déformé, si bien que la rotation de section *est* la pente,
`θ = w'`, et qu'il n'y a aucun cisaillement transverse. C'est là toute la
différence avec [Timoshenko](timoshenko.md), le [portique 2D](portique.md) et le
[cadre 3D](cadre3d.md), qui conservent une souplesse de cisaillement.

Trois configurations partagent la même théorie :

| `model` | DDL par nœud | matériau | ce qui s'ajoute à la flexion |
|---|---|---|---|
| `planar_1d` | `w`, `theta` | `E, I` | rien — flexion pure |
| `frame_2d` | `u_x, u_y, r_z` | `+ A` | l'effort axial, et une rotation vers les axes globaux |
| `frame_3d` | 6 DDL | `+ I_y, I_z, J, G` | l'axial, la torsion, et la flexion selon **deux** axes principaux |

## Équations continues résolues

La cinématique est celle d'une section rigide qui **tourne avec la pente** :

\\[
u_x(x, y, z) = u(x) - y\\,w'(x), \qquad u_y = w(x),
\qquad \theta = w' .
\\]

D'où une déformation axiale affine dans la section, et **aucune** distorsion :

\\[
\varepsilon_{xx} = u' - y\\,w'' = \varepsilon_0 - y\\,\chi,
\qquad \chi = w'',
\qquad \gamma_{xy} \equiv 0 .
\\]

\\( \chi \\) est la **courbure**. En intégrant \\( \sigma_{xx} = E\varepsilon_{xx} \\)
sur la section, les efforts généralisés se découplent — c'est le choix de l'axe
neutre, \\( \int_A y\\,dA = 0 \\) — et la loi de section s'écrit

\\[
N = EA\\,\varepsilon_0, \qquad M = EI\\,\chi, \qquad I = \int_A y^2\\,dA .
\\]

L'équilibre local intégré sur la section donne alors les deux équations
classiques, découplées elles aussi :

\\[
(EA\\,u')' + n = 0, \qquad (EI\\,w'')'' = q .
\\]

La seconde est du **quatrième ordre** — et c'est toute la différence avec
[Timoshenko](timoshenko.md), qui en fait deux du second ordre en gardant
\\( \theta \\) indépendant de \\( w' \\). En 3-D s'y ajoutent la flexion selon le
second axe principal (\\( M_z = EI_z\\,w_y'' \\)) et la torsion de Saint-Venant
\\( M_t = GJ\\,\varphi' \\), qui ne se couple à rien pour une section symétrique.

## Pourquoi une physique à part, et non une aire de cisaillement infinie

On pourrait atteindre Bernoulli en faisant tendre `G·A_s → ∞` dans un élément de
Timoshenko, et le résultat serait juste en arithmétique exacte. En virgule
flottante il ne l'est pas : le terme de cisaillement domine alors la raideur de
plusieurs ordres de grandeur et la réponse en flexion s'y noie — le **blocage en
cisaillement** classique, atteint par l'autre bout.

Écrire la théorie directement supprime la question, et supprime au passage deux
constantes matériau (`G`, `A_s`) qu'un modèle de Bernoulli n'a aucune raison de
demander. Réclamer une constante qu'une théorie n'utilise pas, c'est inviter la
mauvaise.

## L'élément

L'équation étant du quatrième ordre, sa forme faible demande une interpolation
**\\( C^1 \\)** : le déplacement *et* sa pente doivent être continus d'un élément
au suivant. C'est exactement ce que fournit la famille
**[`Hermite3`](../fe-space.md#fonctions-de-forme-cubiques-dhermite-hermite-3)**,
qui prend pour degrés de liberté la flèche et la pente à chaque extrémité — deux
fonctions de forme par nœud au lieu d'une.

Ce n'est donc pas une interpolation de Lagrange, et le modèle **exige** un
espace `HERMITE3` :

```python
fes = pyrucast.FiniteElementSpace(maillage, interpolation="HERMITE3")
poutre = pyrucast.Model.bernoulli(fes, "planar_1d")
```

Un espace de Lagrange porterait une flèche linéaire, de courbure identiquement
nulle : le modèle le refuse plutôt que d'assembler une raideur qui ne
correspondrait pas à la base déclarée.

La courbure est alors **linéaire** sur l'élément, donc l'espace d'approximation
contient exactement la solution d'une travée chargée seulement à ses
extrémités : l'élément est *exact aux nœuds* pour toute charge laissant la travée
libre d'efforts répartis — c'est pourquoi un élément par barre suffit pour un
portique. La raideur \\( K_b = \int_0^L EI\\,N''^\top N''\\,dx \\) est **intégrée**
depuis cette base, et vaut exactement la forme fermée classique :

\\[
K_b = \frac{EI}{L^3}
\begin{bmatrix}
 12   &  6L   & -12  &  6L \\\\
 6L   &  4L^2 & -6L  &  2L^2 \\\\
-12   & -6L   &  12  & -6L \\\\
 6L   &  2L^2 & -6L  &  4L^2
\end{bmatrix},
\qquad \text{DDL } [\\,w_A,\ \theta_A,\ w_B,\ \theta_B\\,].
\\]

L'intégration plutôt que la forme fermée est un choix : elle laisse **une seule
source de vérité**, et rend l'interpolation déclarée *porteuse*. Une base fausse
produirait désormais une raideur fausse, que les tests de poutre attraperaient ;
avec une matrice écrite en dur, elle aurait pu être n'importe quoi. La forme
fermée reste, comme **oracle de test** — c'est elle que `tests/hermite.rs`
compare à l'intégrale, à la précision machine.

L'effort axial y est ajouté par le terme de barre \\( EA/L \\), qui ne s'y couple
pas.

Le repère local 3-D est déduit automatiquement d'une référence globale Z (globale
Y pour une barre quasi verticale), comme pour le [cadre 3D](cadre3d.md) : aucune
donnée d'orientation à fournir, ce qui convient aux sections symétriques.

## Variables et matériau

Le comportement (`COMP`) rend les efforts de section — `M` en 1-D, `N, M` en
plan, `N, M_y, M_z, T` dans l'espace — par une loi linéaire, comme tout élément
structural.

## Mise en donnée (Rust, testé)

```rust,ignore
{{#include ../../../tests/bernoulli.rs:example}}
```

## Exemple Python

```python
model = pyrucast.Model.bernoulli(fes, "frame_2d")
materials = pyrucast.element_field.material_field(
    model, [("E", 210_000.0), ("A", 1e-2), ("I", 1e-4)]
)
k = pyrucast.matrix.stiffness(model, materials)
```

## Compléments

**Ce que valent les tests.** Un élément de poutre gagne sa place en étant *exact
aux nœuds* : les tests comparent donc aux formules du cours à la **précision
machine** (1e-12), et non à une tolérance de discrétisation. Console sous charge
en bout (`PL³/3EI`), sous moment en bout (`ML²/2EI` — un cas qu'un signe faux
dans la matrice d'Hermite raterait tout en passant le premier), traction axiale
découplée de la flexion, et torsion `TL/GJ`.

**Quand préférer Timoshenko.** Un dernier test mesure ce qui sépare les deux
théories : une poutre **élancée** donne le même résultat aux deux (à 1 % près),
une poutre **trapue** fléchit nettement plus avec le cisaillement. Bernoulli est
exactement la théorie qui dit qu'elle ne le fait pas — à utiliser tant que
l'élancement le permet, et à quitter sinon.

> Ce test-là maille les deux poutres. Bernoulli est exact avec un seul élément,
> mais l'interpolation linéaire de Timoshenko ne l'est pas : comparer les deux
> théories sur un élément unique mesurerait le maillage, pas la physique.
