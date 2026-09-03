# Coques

## Introduction

Une coque est une **surface** qui porte à la fois des efforts de membrane et des
moments de flexion. Ses éléments sont des variétés (`ref_dim = 2` dans un espace
3-D) — précisément le cas que refuse le garde-fou des milieux continus : un noyau
massif construirait `B` à partir du gradient tangentiel et serait déficient en
rang à travers l'épaisseur. Une coque a son noyau et sa cinématique propres.

Comme partout ailleurs ici, la **formulation est un attribut** (`ShellModel`) :
les DDL, le repère local, la loi de membrane et la rotation vers les axes globaux
sont partagés, seul le traitement flexion/cisaillement change.

| formulation | cisaillement transverse | éléments |
|---|---|---|
| `thick` (Reissner-Mindlin) | oui, intégré réduit | TRI3, QUA4 |
| `kirchhoff` (DKT/DKQ) | imposé nul en des points discrets | TRI3, QUA4 |

Membrane et vrillage sont **une seule routine** partagée : ils ne doivent rien à
la théorie de flexion, et un test vérifie que les deux formulations donnent bien
le même allongement de membrane.

## Six DDL par nœud, et celui de vrillage

La cinématique naturelle d'une coque compte cinq degrés de liberté — trois
translations et deux rotations de la fibre normale. Mais le cinquième et le
sixième ne se distinguent que dans le repère **local**, alors qu'un assembleur
global numérote les DDL par leur nom. L'élément en porte donc six,
`u_x…u_z, r_x…r_z`, exactement comme le [cadre 3D](timoshenko.md) — ce qui permet
aussi à une coque et à un portique spatial de partager des nœuds sans adaptateur.

Le sixième, la rotation autour de la normale, est le DDL de **vrillage**, et une
facette plane n'oppose aucune raideur physique à son sujet : laissé seul, il rend
la matrice élémentaire singulière. Il est donc lié à la rotation propre de la
membrane,

\\[
\omega_z = \tfrac12\left(\frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}\right),
\qquad K_\text{vrillage} = \alpha\\,G h \int (\theta_z - \omega_z)^2\\, dA
\\]

ce qui est un énoncé physique (la rotation de vrillage doit suivre celle de la
matière) et non une béquille numérique. Une pénalité diagonale serait le raccourci
tentant, et serait **fausse** : elle s'oppose à une rotation rigide de la facette
autour de sa normale, qui ne coûte aucune énergie. C'est ce que vérifie un test.

Et puisque cette contrainte **travaille**, elle a un effort conjugué : le
comportement rend `M_drill = α·G·h·(θ_z − ω_z)` au même titre que `N_xx` ou
`M_xx`. Il a longtemps manqué à la liste — non par choix, mais parce que
personne n'avait encore demandé à une coque son résidu. Un `∫ Bᵀσ` qui
l'omettrait serait faux du terme même qui désingularise l'élément.

## Le repère local est par élément

Les DDL nodaux étant globaux, la rotation local → global doit être une matrice
par **élément**, pas par point de Gauss. Elle est construite depuis les
coordonnées des nœuds — la première arête, et la normale de la facette — ce qui
la rend exacte pour une facette plane et raisonnable pour un quadrilatère
légèrement gauche.

Les dérivées locales, elles, ne coûtent rien : le gradient tangentiel
`CellGeom::dn_dx` est déjà dans le plan tangent, donc le projeter sur `e₁`, `e₂`
*est* la dérivée locale — aucune inversion, aucun second jacobien.

## Reissner-Mindlin (`thick`)

### Équations continues résolues

La fibre normale reste droite mais **non normale**. Sa rotation est un champ
indépendant, ce qui donne une cinématique affine dans l'épaisseur
(\\( z \in [-h/2,\ h/2] \\)) :

\\[
u_x(x,y,z) = u(x,y) + z\\,\theta_y, \quad
u_y(x,y,z) = v(x,y) - z\\,\theta_x, \quad
u_z = w(x,y).
\\]

Les déformations s'y séparent en trois familles — membrane, flexion et
cisaillement transverse :

\\[
\varepsilon =
\begin{bmatrix}
\partial u/\partial x \\\\
\partial v/\partial y \\\\
\partial u/\partial y + \partial v/\partial x
\end{bmatrix},
\quad
\kappa =
\begin{bmatrix}
\partial \theta_y/\partial x \\\\
-\\,\partial \theta_x/\partial y \\\\
\partial \theta_y/\partial y - \partial \theta_x/\partial x
\end{bmatrix},
\quad
\gamma =
\begin{bmatrix}
\partial w/\partial x + \theta_y \\\\
\partial w/\partial y - \theta_x
\end{bmatrix},
\\]

la déformation dans le plan à la cote `z` valant
\\( \varepsilon(z) = \varepsilon + z\\,\kappa \\). C'est \\( \gamma \\) qui fait
toute la différence avec Kirchhoff-Love, où l'on impose \\( \gamma = 0 \\), donc
\\( \theta = -\nabla w \\), et où la courbure redevient un jeu de dérivées
**secondes** de la seule flèche.

### Les lois de section

L'intégration dans l'épaisseur d'un matériau homogène en contraintes planes
découple les trois familles et donne, avec
\\( N = \int \sigma\\,dz \\), \\( M = \int z\\,\sigma\\,dz \\) et
\\( T = \int \tau\\,dz \\) :

\\[
D_m = \frac{Eh}{1-\nu^2}
\begin{bmatrix}
1 & \nu & 0 \\\\
\nu & 1 & 0 \\\\
0 & 0 & \tfrac{1-\nu}{2}
\end{bmatrix},
\qquad
D_b = \frac{h^2}{12}\\,D_m,
\qquad
D_s = k_s\\,G\\,h .
\\]

La loi de flexion est celle de membrane multipliée par \\( h^2/12 \\) : c'est
tout le contenu de « les sections restent planes » — le même matériau en
contraintes planes, intégré dans l'épaisseur avec un poids \\( z^2 \\)
(\\( \int_{-h/2}^{h/2} z^2 dz = h^3/12 \\)).

Le facteur \\( k_s = 5/6 \\) corrige le fait que la cinématique impose un
cisaillement **uniforme** dans l'épaisseur là où la solution exacte est
parabolique et nulle aux peaux ; il est choisi pour restituer la bonne énergie de
cisaillement d'une section rectangulaire, et se règle par la composante matériau
`k_s`.

La raideur élémentaire est alors la somme des trois formes,

\\[
K_e = \int_A \Big(
B_m^\top D_m B_m + B_b^\top D_b B_b + B_s^\top D_s B_s
\Big)\\,dA,
\\]

**chacune avec sa quadrature** — et c'est là le point suivant.

### Pourquoi le cisaillement est intégré réduit

À mesure que la coque s'amincit, `D_s` (linéaire en `h`) écrase `D_b` (cubique en
`h`) d'un facteur `1/h²`. Intégré à la quadrature complète, le terme de
cisaillement impose alors `γ = 0` **point par point**, ce qu'un élément linéaire
ne peut satisfaire qu'en refusant de fléchir : le déplacement s'effondre vers
zéro et aucun raffinement ne le récupère. C'est le **blocage en cisaillement**.

L'intégrer en un seul point relâche la contrainte en moyenne, l'élément fléchit,
et le calcul converge. La [poutre de Timoshenko](timoshenko.md) a connu le même
blocage et y a répondu de la même manière ; elle a depuis été remplacée par un
élément **exact**, qui possède son interpolation au lieu d'en intégrer une. La
coque épaisse est donc aujourd'hui le seul élément multi-quadrature du code : le
patron reste général, son utilisateur ne l'est plus.

Et le second sous-espace n'est **pas** un argument de `model.shell` : rien en lui
n'appartient à l'appelant (même sous-maillage, même interpolation, seule la
quadrature change), et `element_matrix` lit les deux `CellGeom` comme *une seule*
maille — un invariant qu'il vaut mieux établir par construction que valider après
coup. Le choix qui est réel, lui, est bien un argument : la formulation.

Le remède alternatif est de ne pas avoir de contrainte du tout — c'est le
Kirchhoff discret, ci-dessous, qui n'a rien à bloquer.

## Kirchhoff discret (`kirchhoff`)

### Ce que dit la théorie, et pourquoi on ne l'écrit pas telle quelle

Kirchhoff-Love impose que la fibre normale reste **normale** :
\\( \gamma = \nabla w + \beta = 0 \\), donc \\( \beta = -\nabla w \\) et la
courbure redevient un jeu de dérivées **secondes** de la seule flèche :

\\[
\kappa = \begin{bmatrix}
-\\,\partial^2 w/\partial x^2 \\\\
-\\,\partial^2 w/\partial y^2 \\\\
-\\,2\\,\partial^2 w/\partial x \partial y
\end{bmatrix}.
\\]

C'est une équation d'ordre quatre, et un élément conforme pour elle réclame une
base C¹ — la [poutre de Bernoulli](bernoulli.md) en dimension un, où le cubique
d'Hermite la fournit. En dimension deux, la même construction (le bicubique
d'Hermite, quatre DDL par nœud dont le vrillage \\( \partial^2 w/\partial x
\partial y \\)) n'est conforme que sur des **rectangles alignés aux axes** :
ailleurs le jacobien varie, la conversion des DDL nodaux diffère d'un élément à
l'autre au nœud partagé, et la continuité C¹ est perdue. Aucun mailleur d'ici ne
produit une telle grille.

### Ce qu'on écrit à la place

La réponse du **Kirchhoff discret** est de garder la rotation comme champ
interpolé et de n'imposer \\( \gamma = 0 \\) qu'en des points choisis. Rien n'est
jamais dérivé deux fois, la base reste Lagrange, et la limite mince est exacte
par construction plutôt qu'approchée depuis un cisaillement qu'il faudrait
empêcher de bloquer.

La rotation \\( \beta \\) est interpolée **quadratiquement** — les six fonctions
d'un `TRI6`, les huit d'un `QUA8` — sur un élément dont la géométrie reste
linéaire. Ses valeurs de milieu d'arête sont ensuite éliminées, arête par arête,
par trois énoncés :

| énoncé | où | ce qu'il donne |
|---|---|---|
| \\( \gamma = 0 \\) | à chaque sommet | \\( \beta_i = -\nabla w_i \\) |
| \\( \gamma_s = 0 \\) | au milieu de chaque arête | \\( \beta_{sk} = -\tfrac{3}{2l}(w_j - w_i) - \tfrac14(\beta_{si} + \beta_{sj}) \\) |
| \\( \beta_n \\) linéaire | le long de chaque arête | \\( \beta_{nk} = \tfrac12(\beta_{ni} + \beta_{nj}) \\) |

Le deuxième lit la pente à mi-portée de la **cubique** que suit la flèche le long
d'une arête : c'est par là que l'exactitude d'une poutre d'Euler-Bernoulli entre
dans une plaque, sans qu'aucune base d'Hermite soit jamais assemblée.

Après élimination, chaque fonction de milieu d'arête porte une combinaison fixe
des DDL de sommet, et tout l'élément tient en cinq nombres par arête :

\\[
a = \frac{-x_{ij}}{l^2}, \quad
b = \frac{3}{4}\frac{x_{ij} y_{ij}}{l^2}, \quad
c = \frac{\tfrac14 x_{ij}^2 - \tfrac12 y_{ij}^2}{l^2}, \quad
d = \frac{-y_{ij}}{l^2}, \quad
e = \frac{\tfrac14 y_{ij}^2 - \tfrac12 x_{ij}^2}{l^2}.
\\]

\\( a \\) et \\( d \\) portent la flèche dans la rotation — ils sont **impairs**
dans le sens de l'arête, d'où le changement de signe entre ses deux extrémités ;
\\( b \\), \\( c \\) et \\( e \\) projettent une rotation de sommet sur la
tangente et la normale de l'arête, et sont pairs.

DKT et DKQ ne diffèrent **que** par le nombre de sommets, la base quadratique et
la quadrature : c'est donc une seule routine, pas deux. Les tables \\( H_x \\),
\\( H_y \\) publiées par Batoz pour l'un et pour l'autre en ressortent, et c'est
ce que vérifie le test unitaire — l'élimination est gardée comme une **matrice
sur les fonctions de forme** plutôt que comme le \\( H_x(\xi, \eta) \\) assemblé
de la littérature, si bien que \\( \partial H/\partial \xi = C \cdot \partial
N/\partial \xi \\) réutilise la même matrice et qu'aucune seconde table n'est à
tenir cohérente avec la première.

### Ce qu'il n'a pas

Pas de déformation de cisaillement, donc pas de \\( Q \\) issu d'une loi de
comportement : l'effort tranchant d'une plaque mince est une **réaction**,
retrouvée par le gradient des moments. Le comportement s'arrête aux sept
résultantes de membrane, de flexion et de vrillage, là où `thick` en rend neuf.

## Un seul `B`, lu dans les deux sens

Les deux formulations bâtissent leurs lignes de `B` au même endroit
(`models::shell::b_into`), et ne diffèrent que par les trois de flexion — ce qui
*est* la différence entre elles. La rigidité en intègre `Bᵀ D B`, les
[forces internes](../operateurs/champs.md) le transposé `Bᵀ σ`, puis ramènent le
résultat aux axes globaux par la transposée du trièdre de la facette.

Les déformations, elles, s'obtiennent par
[`shell_deformation`](../operateurs/champs.md) — le produit `B · u`. Sa
particularité tient à la double quadrature : membrane, flexion et vrillage à
chaque point de Gauss, cisaillement transverse **au seul point réduit**, écrit
ensuite à tous. C'est le pendant de l'intégration réduite de la rigidité, et
c'est ce qui fait tomber `∫ Bᵀσ` exactement sur `K·u` — ce que mesure
`tests/internal_forces.rs`, pour les deux formulations sur TRI3 et QUA4.

## Mise en donnée (Rust, testé)

```rust,ignore
{{#include ../../../tests/shell.rs:example}}
```

La même plaque en Kirchhoff discret, triangles et quadrangles :

```rust,ignore
{{#include ../../../tests/shell.rs:kirchhoff}}
```

## Exemple Python

```python
{{#include ../../../tests/python/test_doc_mecanique.py:coques}}
```

## Compléments

**Ce que valent les tests.** Une plaque carrée encastrée sous charge uniforme a
une flèche de cours, `w = 0,00126·qa⁴/D` : le test la retrouve à 6 % près et
vérifie que le maillage **converge**, par le dessous, avec des incréments qui
décroissent.

Mais le test qui compte est celui du **blocage**. Normalisée par la raideur de
plaque, la flèche d'une plaque mince est une constante de la théorie,
indépendante de l'épaisseur. Le test la mesure sur deux décades d'épaisseur : un
élément qui bloque la perd de plusieurs ordres de grandeur, un élément
correctement sous-intégré la conserve.

S'y ajoutent le comportement de membrane, exact à `1e-9`, et le vrillage rigide
qui ne coûte pas d'énergie tout en laissant un vrillage parasite en coûter.

Pour le **Kirchhoff discret**, l'énoncé de non-blocage est plus tranchant
encore : la flèche normalisée n'est pas seulement stable quand la plaque
s'amincit, elle est *invariante* à la précision machine — l'épaisseur n'entre
dans la raideur de flexion que par un facteur \\( h^3 \\), et le problème de
flexion pure est donc exactement sans échelle. Le test l'exige à `1e-6` près sur
deux décades et demie, pour DKT comme pour DKQ.

La convergence, elle, ne se dit **pas** de la même manière : la facette de
Mindlin est un modèle en déplacements compatible, donc sa flèche ne peut que
monter vers la réponse. Un élément à Kirchhoff discret ne l'est pas — éliminer
les rotations de milieu d'arête sous des contraintes qui ne tiennent qu'en des
points laisse une interpolation discontinue au travers d'une arête, et la borne
variationnelle s'en va avec. Le test vérifie donc la convergence elle-même
(chaque raffinement tombe plus près, les incréments décroissent), pas le côté
d'où elle arrive : sur cette plaque encastrée, la DKQ converge par le **haut**.

Enfin, une bascule rigide de la plaque — `w = x` et la rotation qui l'accompagne
— ne doit coûter aucune énergie. Assemblé, ce test attrape ce que les tables
d'élément ne peuvent pas : la convention de signe liant la flèche à la rotation
(\\( \beta = -\nabla w \\), donc `r_y = −1` pour `u_z = x`), et la rotation
local → global qui la transporte.

**Ce qui n'est pas couvert.** Pas de matrice de masse ni de raideur géométrique
pour l'instant ; pas de coque courbe au sens propre — une facette plane par
élément, ce qui est la formulation usuelle des coques facettisées et demande un
maillage plus fin sur une forte courbure. Aucun opérateur de déformation de
coque non plus : les lois de section sont écrites et testées par la raideur, mais
la reconstruction des résultantes depuis un champ solution reste à faire.
