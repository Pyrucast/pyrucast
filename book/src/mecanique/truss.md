# Barre

## En bref

Une barre ne transmet que l'**effort axial** : un treillis, un câble tendu,
un tirant. Elle se pose sur des segments, en 1-D, 2-D ou 3-D.

| | |
|---|---|
| opérateur | `pyrucast.model.truss(fespace, strain_measure="linear")` |
| tag | `mechanical` |
| type | domaine avec comportement (`label()` : `"Truss"`) |
| éléments | `SEG2` |

Une barre n'a aucune raideur transversale : chaque nœud doit être tenu dans
les directions que la barre ne porte pas (par d'autres barres, un treillis
triangulé, ou des appuis), sinon la matrice est singulière. Une géométrie de
révolution est refusée.

## Théorie

Le long de l'axe \\(s\\) de la barre, de section \\(A\\) :

\\[
N = E\\,A\\,\varepsilon, \qquad \varepsilon = \frac{du_s}{ds}, \qquad \frac{dN}{ds} + f = 0 .
\\]

La direction de la barre est le vecteur unitaire
\\(c = (x_B - x_A)/L\\), tiré des coordonnées des nœuds. Avec l'interpolation
linéaire, la déformation est constante dans l'élément, et la raideur
s'écrit directement dans les axes globaux, sans matrice de repère :

\\[
K_e = \frac{E\\,A}{L}
\begin{bmatrix} c\\,c^{\mathsf T} & -c\\,c^{\mathsf T} \\\\ -c\\,c^{\mathsf T} & c\\,c^{\mathsf T} \end{bmatrix}.
\\]

En 1-D, \\(c = 1\\) et l'on retrouve \\((EA/L)\\,[[1, -1], [-1, 1]]\\). La
masse est \\((\rho A L/6)\\,[[2, 1], [1, 2]]\\) sur chaque direction, et la
raideur géométrique sous l'effort \\(N\\) vaut
\\((N/L)\\,[[I, -I], [-I, I]]\\) : elle raidit toutes les directions, l'axiale
comprise.

Avec `strain_measure="green_lagrange"`, la barre suit les grands
déplacements : voir [Grandes transformations](grandes-transformations.md#les-barres-et-les-câbles).

## Variables

En 2-D, en petites déformations :

| | |
|---|---|
| primales | `u_x`, `u_y` (et `u_z` en 3-D) |
| duales | `f_x`, `f_y` (et `f_z`) |
| matériau requis | `E`, `A` |
| matériau facultatif | `rho` |
| entrée du comportement | la déformation du point : `eps_xx`, `eps_xy`, `eps_yy` |
| sortie du comportement | `n`, l'effort axial \\(E\\,A\\,c^{\mathsf T}\varepsilon\\,c\\) |
| état | aucun |

## Ce qu'elle produit

| | |
|---|---|
| raideur | \\(K_e\\) |
| masse | exige `rho` |
| géométrique | sous l'effort `n` |
| tangente | = raideur |
| forces internes | \\(n\\,c\\) aux deux nœuds, de signes opposés : le même \\(B\\) que la raideur |

## Exemple

Une barre de longueur 2, de module 200 000 et de section 0,01, tirée par une
force de 1000 : \\(u = F\\,L/(E\\,A) = 1\\).

```python
{{#include ../../../tests/python/test_doc_mecanique.py:barre}}
```

## Côté Rust

`Truss` (`src/models/truss.rs`) calcule ses matrices par des fonctions pures,
`element_stiffness`, `element_mass` et `element_geometric`, et ses forces
internes par `internal_force_element`, qui applique le même \\(B\\) que la
raideur : pour une loi linéaire, \\(\int B^{\mathsf T}\sigma\\) vaut exactement
\\(K\\,u\\).

```rust,ignore
{{#include ../../../tests/truss.rs:example}}
```
