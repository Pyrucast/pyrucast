# Poutre de Timoshenko

## En bref

Une poutre **déformable en cisaillement** : la section reste plane mais pas
normale à l'axe déformé. Pour une poutre trapue, la flèche due au
cisaillement s'ajoute à celle de la flexion.

| | |
|---|---|
| opérateur | `pyrucast.model.timoshenko(fespace)` |
| tag | `mechanical` |
| type | domaine avec comportement (`label()` : `"Timoshenko"`) |
| éléments | `SEG2`, sur un espace `FiniteElementSpace(maillage, interpolation="MODEL_EMBEDDED")` |

Comme pour [Euler-Bernoulli](bernoulli.md), la configuration se lit sur la
dimension des `Coords` : flexion pure en 1-D, portique plan en 2-D, cadre
spatial en 3-D.

## Théorie

La rotation \\(\theta\\) est un champ indépendant de la pente, et la distorsion
\\(\gamma = w\' - \theta\\) une déformation à part entière :

\\[
\kappa = \theta\', \qquad \gamma = w\' - \theta, \qquad
M = E I\\,\kappa, \qquad V = G A_s\\,\gamma, \qquad
V\' + q = 0, \qquad M\' - V = 0 ,
\\]

\\(A_s\\) étant l'aire de cisaillement. Deux équations du second ordre : une
interpolation \\(C^0\\) suffit.

L'élément est la **solution exacte** de ces équations sur une travée sans
charge répartie : il est exact aux nœuds, et un élément par barre suffit. Ses
fonctions de forme, cubiques en \\(w\\) et quadratiques en \\(\theta\\),
dépendent du matériau par

\\[
\Phi = \frac{12\\,E I}{G A_s L^2},
\\]

le rapport des souplesses de flexion et de cisaillement. La raideur de flexion
est

\\[
K_b = \frac{E I}{L^3(1+\Phi)}
\begin{bmatrix}
12 & 6L & -12 & 6L \\\\
6L & (4+\Phi)L^2 & -6L & (2-\Phi)L^2 \\\\
-12 & -6L & 12 & -6L \\\\
6L & (2-\Phi)L^2 & -6L & (4+\Phi)L^2
\end{bmatrix},
\\]

où la raideur en flèche est la mise en série des deux souplesses,
\\(1/(L^3/12 E I + L/G A_s)\\). \\(\Phi = 0\\) redonne Euler-Bernoulli terme à
terme.

Parce que la base dépend du matériau, aucun espace ne peut la tabuler :
l'espace déclare
[`MODEL_EMBEDDED`](../fe-space.md#pas-de-base-du-tout--model_embedded), une
interpolation que la physique porte elle-même.

La masse est intégrée des mêmes fonctions de forme :
\\(\int \rho A\\,N_w^{\mathsf T} N_w + \int \rho I\\,N_\theta^{\mathsf T} N_\theta\\),
le second terme étant l'inertie de rotation de la section ; l'axial et la
torsion gardent la masse linéaire \\((\rho L/6)\\,[[2, 1], [1, 2]]\\). À
\\(\Phi = 0\\), elle redonne la table classique
\\(\rho A L/420\cdot[156, 22L, 54, -13L, \dots]\\).

Aux points de Gauss, la courbure varie linéairement et le cisaillement est
constant sur une travée sans charge, comme l'imposent \\(M\' = V\\) et
\\(V\' = 0\\). Les reconstruire demande le matériau, puisque \\(\Phi\\) en
dépend.

## Variables

| | 1-D | 2-D | 3-D |
|---|---|---|---|
| primales, duales | celles de [Bernoulli](bernoulli.md#variables) | | |
| matériau requis | `E`, `I`, `G`, `A_s` | `E`, `A`, `I`, `G`, `A_s` | `E`, `A`, `I_y`, `I_z`, `J`, `G`, `A_sy`, `A_sz` |
| matériau facultatif | `A`, `rho` | `rho` | `rho` |
| entrée du comportement | `kappa`, `gamma` | `eps`, `kappa`, `gamma` | `eps`, `kappa_y`, `kappa_z`, `torsion`, `gamma_y`, `gamma_z` |
| sortie du comportement | `M`, `V`, `phi` | `N`, `M`, `V`, `phi` | `N`, `M_y`, `M_z`, `T`, `V_y`, `V_z`, `phi_y`, `phi_z` |
| état | aucun | aucun | aucun |

La sortie porte \\(\Phi\\) (`phi`) à côté des efforts : le noyau des forces
internes reçoit l'état, pas le matériau, et son \\(B\\) dépend de \\(\Phi\\).

## Ce qu'elle produit

| | |
|---|---|
| raideur | l'élément exact |
| masse | exige `rho` (et `A` en 1-D) |
| géométrique | sous l'effort `N` ; aucune en 1-D |
| tangente | = raideur |
| forces internes | le transposé du même \\(B\\) : \\(\int B^{\mathsf T}\sigma = K\\,u\\) exactement |

## Exemple

La console de la page [Euler-Bernoulli](bernoulli.md#exemple), avec une aire
de cisaillement : la flèche gagne \\(F\\,L/(G A_s)\\).

```python
{{#include ../../../tests/python/test_doc_mecanique.py:timoshenko}}
```

## Côté Rust

`Timoshenko` (`src/models/timoshenko.rs`) partage avec Bernoulli le \\(B\\) de
`src/models/beam.rs`. Le test compare une console à
\\(P L^3/(3 E I) + P L/(G A_s)\\) :

```rust,ignore
{{#include ../../../tests/timoshenko.rs:example}}
```

Le portique plan, où l'axial et la flexion se découplent :

```rust,ignore
{{#include ../../../tests/frame.rs:example}}
```

Le cadre spatial, avec sa torsion :

```rust,ignore
{{#include ../../../tests/frame3d.rs:example}}
```
