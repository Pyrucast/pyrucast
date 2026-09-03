# Échanges (frontière et interface)

## Introduction

Deux physiques, une seule loi. Un échange fait traverser un flux
proportionnellement à un **écart** :

\\[
q\cdot n = h\\,\big(a - b\big)
\\]

et toute la différence entre les deux tient à **de quel côté du signe égal vit
le milieu d'en face** :

| modèle | l'autre côté est… | où il va |
|---|---|---|
| [`boundary_transfer`](#échange-avec-une-ambiante) | une **donnée** (une valeur ambiante, `a_ext_<primale>`) | au second membre, \\( h\\,a_\text{ext}\int N\\,d\Gamma \\), rendu par `external_forces` |
| [`interface_transfer`](#échange-entre-deux-maillages) | une **inconnue** (le champ de l'autre maillage) | dans la matrice, en bloc de couplage |

Le bloc hors-diagonale *est* le second membre rendu implicite. C'est pourquoi les
deux partagent leur noyau (`src/models/transfer.rs`) au lieu d'en porter chacun
une copie : **le terme de frontière est le terme d'interface avec les deux côtés
sur la même maille.**

## Ce qui est transféré appartient à l'appelant

Aucun des deux ne sait ce qu'il transporte. On lui donne une liste de couples
`(primale, duale)` — la forme que prennent déjà
[`embedded`](contraintes/embedded.md) et [`contact`](contraintes/contact.md), les
deux autres lois qui lient des maillages — et tout le reste en découle :

| transféré | matériau | entrée du comportement | sortie |
|---|---|---|---|
| `("T", "q")` | `h_T` | `T` / `jump_T` | `flux_T` |
| `("c_H2", "j_H2")` | `h_c_H2` | `c_H2` / `jump_c_H2` | `flux_c_H2` |
| `("u_x", "f_x")` | `h_u_x` | `u_x` / `jump_u_x` | `flux_u_x` |

**Un coefficient par grandeur transférée, nommé d'après elle.** C'est ce qui
permet une raideur par direction, et ce qu'un `h` unique ne pouvait pas exprimer.

Nommer les DDL de la physique de volume est aussi ce qui fait que le terme s'y
**couple directement** : un `boundary_transfer` sur `("T", "q")` entre dans la
raideur d'un [`heat_conduction`](thermique.md) sans adaptateur, parce que la
matrice est indexée par le couple *(nœud, nom de champ)*.

La **nature physique** (`"thermal"`, `"diffusion"`, `"mechanical"`) est le seul
argument qui ne se déduit pas : des noms de variables libres ne peuvent pas
l'impliquer, et c'est elle que `model.filter(...)` sélectionne. On la déclare.

## Forme discrétisée

### Échange avec une ambiante

La forme faible du terme de bord se scinde en deux ingrédients, dont le
sous-modèle ne porte que le premier :

\\[
\underbrace{K_{ij} = h \int_\Gamma N_i\\,N_j\\,d\Gamma}_{\text{matrice de film (raideur)}}
\qquad
\underbrace{f_i = h\\,a_\text{ext} \int_\Gamma N_i\\,d\Gamma}_{\text{charge (second membre)}}
\\]

La part ambiante est un **chargement**, et le sous-modèle le porte : `a_ext` est
une composante matériau **exigée**, à côté de son coefficient, et le terme se
récupère par [`external_forces`](operateurs/comportement.md). Elle a longtemps
été bâtie à la main avec l'opérateur `flux`, ce qui laissait l'oublier — et un
ambiant oublié ne se voit pas : il se lit comme un ambiant nul, donc comme une
paroi qui échange avec le zéro absolu. Le terme de film rend par ailleurs la
matrice **définie** : un problème purement Neumann + échange est bien posé
**sans Dirichlet**.

**Aucune normale à choisir.** Elle est déjà consommée en passant de
\\( q\cdot n \\) à \\( h(a - a_\text{ext}) \\) ; ce qui reste sous l'intégrale
est un scalaire fois la mesure \\( d\Gamma = \sqrt{\det(J^\top J)} \\), une
magnitude **indépendante de l'orientation** du maillage de bord — contrairement à
une pression ou à un flux signé.

### Échange entre deux maillages

Quand l'autre côté est une inconnue, la forme faible devient

\\[
\int_\Gamma h\\,(a_1 - a_2)\\,(\delta a_1 - \delta a_2)\\; d\Gamma,
\\]

qui se développe en une structure \\( 2\times2 \\) sur les DDL des deux côtés :

\\[
\begin{bmatrix} +K & -K \\\\ -K & +K \end{bmatrix},
\qquad
K_{ij} = h \int_\Gamma N_i\\,N_j\\; d\Gamma .
\\]

Les deux blocs **diagonaux** sont exactement deux termes de frontière, un par
côté. Les deux autres ont leurs **lignes sur un maillage et leurs colonnes sur
l'autre** : c'est le genre de contribution `Coupling`, dont ce modèle est le seul
utilisateur (voir
[Ajouter une physique](ajouter-une-physique.md#un-bloc-inter-maillages--coupling)).

Les quatre sortent d'une **seule** fonction, `transfer::exchange_matrix` : les
diagonaux avec la même maille des deux côtés et le signe `+`, les croisés avec la
maille en vis-à-vis et le signe `−`. Le signe est donc porté par le noyau et non
par un facteur qu'il faudrait faire circuler dans l'assembleur. Chaque bloc de
couplage pris seul est **non symétrique** ; leur réunion l'est — exactement comme
la paire C / Cᵀ de [Dirichlet](contraintes/dirichlet.md).

Les grandeurs transférées **ne se couplent pas entre elles** : la chaleur qui
traverse un joint n'y pousse pas l'hydrogène. Seule la diagonale en indice de
variable est écrite, donc le coût reste linéaire en nombre de composantes.

### Conformité

Les deux côtés doivent être **conformes** : même type d'élément, même nombre de
mailles, maille `i` face à maille `i`, et nœud local `k` face au nœud local `k`.
C'est vérifié géométriquement à la construction — les nœuds appariés doivent être
colocalisés — et **signalé** plutôt qu'approché. Une interface non conforme est un
problème de maillage ; la rattraper par une projection silencieuse fabriquerait
des flux faux sans le dire.

Les deux maillages sont donc géométriquement confondus mais **numérotés
séparément** : c'est cette duplication des nœuds qui laisse le champ sauter, là
où un nœud partagé l'interdirait.

## Mise en donnée (Rust, testé)

```rust,ignore
{{#include ../../tests/transfer.rs:foundation}}
```

## Exemple Python

```python
{{#include ../../tests/python/test_doc_ops_physiques.py:echanges}}
```

## Compléments

### Ce que la généralisation apporte

Rien dans la loi n'est thermique ni diffusif, donc deux modèles sortent sans une
ligne de mécanique :

- une **fondation élastique de Winkler** — un échange de frontière sur les
  déplacements. Une barre poussée sur sa face libre, cette face reposant sur un
  ressort réparti, c'est la barre et le ressort en parallèle sous la traction
  appliquée : \\( u = q / (E/L + h) \\), ce que le test vérifie à `1e-9` sur
  quatre décades de raideur ;
- un **joint collé de raideur finie** — le même sur une interface.

### Échange ou contrainte ?

Un `interface_transfer` est la **régularisation par pénalité** de ce qu'un
[MPC](contraintes/mpc.md) impose exactement :

```text
mpc([(T, maillage₁, +1), (T, maillage₂, −1)]) = 0   ⟺  T₁ = T₂, exactement
interface_transfer(maillage₁, maillage₂), h → ∞      ⟶  T₁ = T₂, à 1/h près
```

| | `mpc` | `interface_transfer` |
|---|---|---|
| `h` | n'existe pas — un multiplicateur n'est pas un matériau | une **constante physique** |
| inconnues | ajoute `lambda_mpc` | aucune |
| système | point-selle, indéfini | reste défini positif |
| conditionnement | insensible | se dégrade quand `h` monte |

Le critère de choix tient en une phrase : **si `h` vient d'une mesure, c'est de la
physique ; s'il a été choisi « assez grand », il fallait une contrainte.** Un
joint imparfait, une résistance thermique de contact, une couche adhésive ont un
`h` mesurable ; « lier deux surfaces » n'en a pas.

### Ce que ça vaut comme vérification

Le film thermique est confronté à une solution analytique — une dalle chauffée
d'un côté, refroidie de l'autre, dont tout le flux ressort par convection (voir
[Conduction thermique](thermique.md#convection-de-surface-robin--film)).

L'interface l'est au **saut** : deux carrés côte à côte, un flux `q` injecté d'un
côté, la concentration imposée de l'autre, et un saut `q/h` à la traversée (voir
[Diffusion](diffusion.md#transfert-à-travers-une-interface)). C'est ce saut qui
distingue une interface d'un nœud partagé, et il est porté entièrement par les
blocs hors-diagonale ; quand `h → ∞` il s'efface et l'on retrouve le corps
continu.

S'y ajoute la structure : la raideur assemblée reste **symétrique** alors
qu'aucun bloc de couplage ne l'est, ce qui est la vérification que les quatre
blocs atterrissent où ils doivent.

### Ce qui n'est pas couvert

Pas d'échange entre maillages **non coïncidents** : la liaison exacte a sa
version non conforme avec [`embedded`](contraintes/embedded.md) et ses poids
d'interpolation, l'échange fini n'en a pas. Le mécanisme existe, il n'a
simplement jamais été branché là.
