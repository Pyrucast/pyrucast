# Échanges et flux imposé

Trois physiques sans comportement, qui s'appliquent à une autre physique, leur
**cible**, et en prennent les variables et le tag :

- l'**échange de bord**, entre un bord et un milieu extérieur connu ;
- le **transfert d'interface**, entre deux maillages qui se font face ;
- le **flux imposé**, une densité donnée sur un bord ou un volume.

Les deux échanges suivent la même loi : un flux proportionnel à un écart,

\\[
q\cdot n = h\\,(a - b).
\\]

Pour l'échange de bord, \\(b\\) est une **donnée**, la valeur ambiante ; pour
le transfert d'interface, c'est une **inconnue**, le champ de l'autre
maillage. Les deux partagent leur noyau, `transfer::exchange_matrix`.

**Ce qui est échangé est donné par l'appelant.** On passe la liste des
variables échangées (`["T"]`, `["c_H2"]`, `["u_x", "u_y"]`). Chacune doit être
une inconnue de la cible, qui lui donne sa duale, la ligne où le flux est
écrit. Le coefficient \\(h\\) est un par variable, nommé d'après elle : `h_T`,
`h_c_H2`, `h_u_x`. L'échange prend le tag de la cible : `thermal` sur une
conduction, `diffusion` sur une diffusion, `mechanical` sur une élasticité.
Un nom inconnu de la cible, ou des variables de deux tags dans un même
échange, sont refusés à la construction.

## Échange de bord

### En bref

Un échange entre un bord et un milieu extérieur : la convection avec un fluide
(loi de Newton), le transfert de masse en surface, ou, sur les déplacements,
une fondation élastique.

| | |
|---|---|
| opérateur | `pyrucast.model.boundary_transfer(fespace, target, variables)` |
| tag | celui de la cible |
| type | domaine sans comportement (`label()` : `"BoundaryTransfer"`) |
| éléments | le bord de la cible : segments en 2-D, faces en 3-D |

### Théorie

Sur le bord \\(\Gamma\\), le flux sortant vaut \\(h\\,(a - a_\text{ext})\\).
Dans la forme faible, il donne une matrice de film et un second membre :

\\[
K_{ij} = \int_\Gamma h\\,N_i\\,N_j\\,d\Gamma ,
\qquad
f_i = \int_\Gamma h\\,a_\text{ext}\\,N_i\\,d\Gamma .
\\]

Aucune normale n'entre dans le calcul : elle est déjà dans \\(q\cdot n\\), et
il ne reste sous l'intégrale qu'un scalaire et la mesure du bord, qui ne
dépend pas de l'orientation des mailles. Le film rend la matrice définie : un
problème avec un flux imposé et un échange, sans aucune température imposée,
a une solution unique.

### Variables

Pour la variable `T` d'une conduction :

| | |
|---|---|
| primale | `T`, celle de la cible |
| duale | `q`, celle que la cible associe à `T` |
| matériau requis | `h_T`, `a_ext_T` (la valeur ambiante) |
| comportement | aucun ; les forces internes lisent `T` interpolée aux points de Gauss |

### Ce qu'il produit

| | |
|---|---|
| raideur | le film, \\(\int h\\,N_i\\,N_j\\) |
| masse, géométrique | — |
| tangente | = raideur |
| forces internes | \\(\int h\\,a\\,N_i\\) |
| forces externes | \\(\int h\\,a_\text{ext}\\,N_i\\), rendues par `pyrucast.node_field.external_forces` |

### Exemple

Un carré de conductivité 2, avec une source de 10 sur le bord gauche (un
[flux imposé](#flux-imposé)) et un film `h = 5` vers 20 sur le bord droit. Tout
ce qui entre ressort par le film, d'où
\\(T = T_\text{ext} + Q/h\\) à droite et \\(T_\text{ext} + Q/h + Q/k\\) à
gauche :

```python
{{#include ../../tests/python/test_doc_echanges.py:echange_bord}}
```

Le même échange sur les déplacements est une fondation élastique : un ressort
réparti par direction.

```python
{{#include ../../tests/python/test_doc_echanges.py:fondation}}
```

## Transfert d'interface

### En bref

Un échange entre deux maillages qui se font face sans partager leurs nœuds :
une résistance de contact, un revêtement, un joint, une membrane. Le champ
**saute** à la traversée, et le flux qui passe est proportionnel au saut.

| | |
|---|---|
| opérateur | `pyrucast.model.interface_transfer(side_a, side_b, target, variables, tol=1e-9)` |
| tag | celui de la cible |
| type | domaine sans comportement (`label()` : `"InterfaceTransfer"`) |
| éléments | deux bords **conformes** |

`side_a` et `side_b` sont les espaces des deux faces. Elles doivent être
conformes : même type de maille, même nombre de mailles, la maille `i` face à
la maille `i`, et leur nœud local `k` à moins de `tol` l'un de l'autre. C'est
vérifié à la construction, et une interface non conforme est refusée. Un
sous-modèle est créé par paire de zones des deux côtés, prises dans l'ordre.

### Théorie

La forme faible du terme d'interface est

\\[
\int_\Gamma h\\,(a_1 - a_2)\\,(\delta a_1 - \delta a_2)\\,d\Gamma ,
\\]

qui donne quatre blocs sur les inconnues des deux côtés :

\\[
\begin{bmatrix} +K & -K \\\\ -K & +K \end{bmatrix},
\qquad
K_{ij} = \int_\Gamma h\\,N_i\\,N_j\\,d\Gamma .
\\]

Les deux blocs diagonaux sont deux films, un par côté. Les deux autres ont
leurs lignes sur un maillage et leurs colonnes sur l'autre ; chacun seul n'est
pas symétrique, leur paire l'est. Les variables échangées ne se couplent pas
entre elles : la chaleur qui traverse un joint ne pousse pas l'hydrogène.

C'est la version par pénalité de ce qu'un [multi-points](contraintes/mpc.md)
impose exactement : quand \\(h\\) grandit, le saut tend vers zéro. Si \\(h\\)
est une donnée physique (une résistance de contact mesurée), c'est un
transfert ; s'il n'est là que pour lier deux surfaces, c'est une contrainte.

| | `mpc` | `interface_transfer` |
|---|---|---|
| inconnues | ajoute `lambda` | aucune |
| matrice | point-selle | définie positive |
| quand \\(h\\) grandit | — | la matrice se conditionne mal |

### Variables

Pour la variable `T` :

| | |
|---|---|
| primale | `T`, des deux côtés |
| duale | `q`, des deux côtés |
| matériau requis | `h_T` |
| comportement | aucun ; les forces internes lisent le saut `jump_T` aux points de Gauss |

### Ce qu'il produit

| | |
|---|---|
| raideur | les quatre blocs |
| masse, géométrique | — |
| tangente | = raideur |
| forces internes | \\(\int h\\,(a_1 - a_2)\\,N_i\\) sur le côté A, son opposé sur le côté B |
| forces externes | aucune |

### Exemple

Deux carrés côte à côte, un joint `h = 4` entre eux, un flux de 1 entrant à
gauche et `T = 0` à droite. Le saut à la traversée vaut \\(Q/h\\) :

```python
{{#include ../../tests/python/test_doc_echanges.py:interface}}
```

## Flux imposé

### En bref

Une densité donnée, intégrée en forces nodales : une source de chaleur sur un
bord ou dans un volume, une traction, une pression. Elle ne dépend pas de la
solution, et n'entre dans aucune matrice.

| | |
|---|---|
| opérateur | `pyrucast.model.flux(fespace, target, dual)` |
| tag | celui de la cible |
| type | domaine sans comportement (`label()` : `"Flux"`) |
| éléments | toute maille, en volume ou sur un bord |

`dual` est la ligne de la cible où la densité est écrite : `"q"` pour une
source de chaleur, `"f_x"` pour une traction en x. `"n"` désigne une
**pression** : la densité agit selon la normale de la maille et s'écrit dans
toutes les lignes de déplacement (`f_x`, `f_y`, `f_z`). La normale est celle de
la configuration active : sur une configuration déformée, la pression suit la
paroi.

### Théorie

\\[
f_i = \int \varphi\\,N_i\\,d\Gamma ,
\qquad
f_{i,a} = \int \varphi\\,n_a\\,N_i\\,d\Gamma \\;\text{pour une pression.}
\\]

### Variables

| | |
|---|---|
| primale | aucune |
| duale | `dual`, prise dans la cible |
| matériau requis | `phi_<dual>` : `phi_q`, `phi_f_x`, `phi_n` |
| comportement | aucun |

La densité est une composante du champ matériau : uniforme, ou variable d'un
point de Gauss à l'autre.

### Ce qu'il produit

| | |
|---|---|
| raideur, masse, géométrique, tangente | — |
| forces internes | aucune |
| forces externes | \\(\int \varphi\\,N_i\\), rendues par `pyrucast.node_field.external_forces` |

### Exemple

La source de l'[échange de bord](#exemple) ci-dessus est un flux imposé :
`model.flux(gauche, conduction, "q")` et la composante `phi_q` du matériau.

## Côté Rust

`BoundaryTransfer` (`src/models/boundary_transfer.rs`) et `InterfaceTransfer`
(`src/models/interface_transfer.rs`) s'appuient sur `src/models/transfer.rs`,
qui porte leur noyau commun, `exchange_matrix`, la vérification des variables
contre la cible et le choix du tag (`owner_physics`). Ils implémentent
`Domain` sans `Behavior`. Ce qu'ils lisent pour leurs forces internes est la
primale interpolée aux points de Gauss (`residual_input`), et non un état.

Le transfert d'interface rend ses quatre blocs par `contributions` : deux
blocs ordinaires et deux blocs de couplage (`Contribution::Coupling`), dont les
lignes et les colonnes vivent sur deux sous-espaces différents.

`Flux` (`src/models/flux.rs`) ne contribue à aucune matrice ; il déclare
seulement son terme de forces externes (`external_force_contribution`).

Le test de la fondation élastique compare le déplacement d'une barre poussée
sur un ressort réparti à \\(u = q / (E/L + h)\\), sur quatre décades de
raideur :

```rust,ignore
{{#include ../../tests/transfer.rs:foundation}}
```

Le test du transfert d'interface vérifie le profil de part et d'autre et le
saut \\(q/h\\) :

```rust,ignore
{{#include ../../tests/interface_transfer.rs:example}}
```
