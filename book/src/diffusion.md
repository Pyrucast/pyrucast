# Diffusion (loi de Fick)

## Introduction

La diffusion d'une espèce dans un milieu — humidité dans un béton, hydrogène
dans un acier, chlorures dans un enrobage — obéit à la **loi de Fick**, dont
l'opérateur est celui de la [conduction thermique](thermique.md).

C'est pourtant une **physique distincte**, et pyrucast la traite comme telle. La
variable primale est la concentration `c`, la duale le flux de matière `j`, et sa
nature déclarée est `Physics::Diffusion`. Partager un opérateur n'est pas
partager une physique : dans un problème couplé thermo-diffusif, on doit pouvoir
écrire `model.filter("diffusion")` et n'obtenir que la partie diffusive, sans
traîner la thermique avec.

Le modèle vit sur n'importe quel espace EF volumique (2-D ou 3-D, linéaire ou
quadratique), avec un degré de liberté scalaire par nœud.

## Équations continues résolues

La première loi de Fick relie le flux au gradient de concentration :

\\[
\mathbf j = -\\,\mathsf D\\,\nabla c
\\]

et la conservation de l'espèce, en régime transitoire avec un coefficient de
stockage \\( \varphi \\) (la porosité, pour une espèce diffusant dans un solide
poreux) :

\\[
\varphi\\,\frac{\partial c}{\partial t} + \nabla\\!\cdot\mathbf j = 0
\qquad\Longleftrightarrow\qquad
\varphi\\,\frac{\partial c}{\partial t} - \nabla\\!\cdot(\mathsf D\\,\nabla c) = 0 .
\\]

En stationnaire c'est l'équation de Laplace pondérée par \\( \mathsf D \\). C'est
**la même équation** que la conduction thermique, à un changement de noms près
(\\( c \leftrightarrow T \\), \\( \mathsf D \leftrightarrow k \\),
\\( \varphi \leftrightarrow \rho c_p \\)) — d'où un modèle qui partage la
totalité du noyau de [conduction thermique](thermique.md), et n'en diffère que
par ses variables et sa nature physique.

La forme faible, après intégration par parties, s'écrit : trouver `c` tel que
pour tout `δc` admissible,

\\[
\int_\Omega \varphi\\,\delta c\\,\dot c\\; d\Omega
+ \int_\Omega \nabla \delta c \cdot \mathsf D\\,\nabla c\\; d\Omega
= -\int_{\partial\Omega} \delta c\\;\mathbf j\\!\cdot\\!\mathbf n\\; d\Gamma .
\\]

## Forme discrétisée

\\[
K_{ij} = \int_\Omega \nabla N_i^\top\\,\mathsf D\\;\nabla N_j\\; d\Omega
\quad \text{(rigidité de diffusion — Cast3M \texttt{COND})},
\\]
\\[
C_{ij} = \int_\Omega \varphi\\,N_i\\,N_j\\; d\Omega
\quad \text{(stockage — Cast3M \texttt{CAPA})}.
\\]

`D` est un **tenseur**, dont le cas isotrope `D = D·I` redonne le produit
scalaire habituel `∇N_i · ∇N_j`. Les trois symétries matériau décrites au
chapitre [Élasticité orthotrope](mecanique/orthotropie.md) s'appliquent
identiquement, avec un tenseur d'ordre 2 au lieu de 4 :

| symétrie | composantes matériau |
|---|---|
| `isotropic` | `D` |
| `orthotropic` | `D_1`, `D_2`, `D_3` + le repère matériau |
| `anisotropic` | `D_11`, `D_12`, `D_13`, `D_22`, `D_23`, `D_33` (symétrique) + le repère |

Le repère est donné par les vecteurs `V1X, V1Y` (2-D) ou `V1X…V1Z, V2X…V2Z`
(3-D), exactement comme en mécanique.

## Variables et matériau

| | |
|---|---|
| primale | `c` (concentration, colonnes) |
| duale | `j` (flux de matière, lignes) |
| matériau requis | la diffusivité, selon la symétrie |
| matériau optionnel | `poro` — le coefficient de stockage, exigé par la seule matrice de masse |
| nature | `Physics::Diffusion` |

Le comportement (`COMP`) rend le flux **sous forme faible** `D·∇c`, en
composantes `j_x, j_y(, j_z)`. Comme en thermique, c'est l'**opposé** du flux
physique de Fick : ce choix garantit `∫ Bᵀ·j = K·c`, donc l'accord entre le
comportement et la rigidité dans le cas linéaire. Les composantes sont nommées
d'après la variable duale (`j_*`) et non `flux_*`, afin qu'un modèle portant à la
fois conduction et diffusion garde deux champs de flux non ambigus.

L'entrée du comportement est le gradient `grad_c_x, …`, tel que le produit
l'opérateur [`gradient`](operateurs/geometrie.md) sur un champ dont la composante
est `c`.

## Mise en donnée (Rust, testé)

```rust,ignore
{{#include ../../tests/fick.rs:example}}
```

## Exemple Python

```python
{{#include ../../examples/diffusion_1d.py}}
```

## Compléments

### Coexister avec la thermique

Les deux physiques peuvent vivre sur le **même maillage** sans se gêner :

```python
model = pyrucast.Model.fick(fes) | pyrucast.Model.heat_conduction(fes)
materials = pyrucast.element_field.material_field(model, [("D", 2.0), ("k", 5.0)])
k = pyrucast.matrix.stiffness(model, materials)

len(model.filter("diffusion"))  # 1
len(model.filter("thermal"))  # 1
```

Un seul champ matériau porte les deux jeux de coefficients. L'assembleur résout
la zone de chaque physique par les **composantes qu'elle exige** (`D` ici, `k`
là) — il n'y a rien à consolider à la main. Et parce que les deux natures sont
distinctes, `filter` les sépare de nouveau après coup, aussi bien sur le modèle
que sur la matrice assemblée.

Les degrés de liberté restent séparés (`c` d'un côté, `T` de l'autre) : le
système est bloc-diagonal. Un vrai **couplage** — une diffusivité fonction de la
température, ou une thermodiffusion — se pilote depuis Python, en réassemblant la
partie diffusive à chaque pas avec un champ matériau recalculé.

### Transfert à travers une interface

Deux corps qui se touchent ne partagent pas forcément leurs nœuds. Un contact
imparfait, un revêtement, un joint, une membrane laissent le champ **sauter** à
la traversée, tandis qu'un flux la franchit proportionnellement à ce saut :

\\[
j\cdot n = h\\,\big(c_1 - c_2\big)
\\]

`h_c` est le coefficient de transfert (son inverse est la résistance de
contact) : un par grandeur transférée, nommé d'après elle.
La même loi décrit une résistance de contact **thermique**, avec `T` et `q` — d'où
le paramètre `kind` du constructeur.

```python
model = (
    pyrucast.Model.fick(gauche)
    | pyrucast.Model.fick(droite)
    | pyrucast.Model.interface_transfer(
        face_gauche, face_droite, [("c", "j")], "diffusion"
    )
)
materials = pyrucast.element_field.material_field(model, [("D", 2.0), ("h_c", 5.0)])
```

#### Quatre blocs, dont deux hors-diagonale

La forme faible du terme d'échange sur l'interface \\( \Gamma \\) est

\\[
\int_\Gamma h\\,(c_1 - c_2)\\,(\delta c_1 - \delta c_2)\\; d\Gamma,
\\]

qui se développe en une structure \\( 2\times2 \\) sur les degrés de liberté des
deux côtés :

\\[
\begin{bmatrix} +K & -K \\\\ -K & +K \end{bmatrix},
\qquad
K_{ij} = h \int_\Gamma N_i\\,N_j\\; d\Gamma .
\\]

Les deux blocs diagonaux sont des blocs *calculés* ordinaires. Les deux autres
ont leurs **lignes sur un maillage et leurs colonnes sur l'autre** : c'est le
genre de contribution `Coupling`, dont ce modèle est le premier utilisateur (voir
[Ajouter une physique](ajouter-une-physique.md#un-bloc-inter-maillages--coupling)).
Le **signe** est porté par le noyau et non par un facteur : le bloc diagonal rend
`+h∫NᵢNⱼ`, le bloc de couplage `−h∫NᵢNⱼ`, et comme chacun choisit son noyau
depuis sa propre variante de contribution, l'assembleur n'a rien à savoir des
interfaces. Chaque bloc de couplage pris seul est **non symétrique** ; leur
réunion l'est — exactement comme la paire C / Cᵀ de Dirichlet.

#### Conformité

Les deux côtés doivent être **conformes** : même type d'élément, même nombre de
mailles, maille `i` face à maille `i`, et nœud local `k` face au nœud local `k`.
C'est vérifié géométriquement à la construction — les nœuds appariés doivent être
colocalisés — et **signalé** plutôt qu'approché. Une interface non conforme est un
problème de maillage ; la rattraper par une projection silencieuse fabriquerait
des flux faux sans le dire.

#### Ce que ça vaut comme vérification

Deux carrés côte à côte, un flux `q` injecté d'un côté, la concentration imposée
de l'autre : le profil est linéaire par morceaux avec une chute `q/D` dans chaque
carré et un **saut `q/h`** à l'interface. C'est ce saut qui distingue une
interface d'un nœud partagé, et il est porté entièrement par les blocs
hors-diagonale. Quand `h → ∞`, le saut s'efface et l'on retrouve le corps
continu.

```rust,ignore
{{#include ../../tests/interface_transfer.rs:example}}
```

### Régime transitoire

La matrice de stockage s'assemble avec `matrix.mass(...)`, qui exige alors la
composante `poro`. L'intégration en temps est orchestrée en Python, comme pour la
thermique transitoire — le noyau Rust fournit `K` et `C`, pas la boucle.

### Bilan de matière

Comme en thermique, le multiplicateur de Lagrange d'une concentration imposée
**est** le flux d'espèce qui traverse la frontière. C'est la vérification la plus
directe d'un calcul de diffusion, et c'est ce que contrôle le test d'intégration
ci-dessus : la réaction au bord imposé égale exactement le flux injecté à l'autre
bout.
