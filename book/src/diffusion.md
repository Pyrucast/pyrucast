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

```text
j = −D · ∇c
```

et la conservation de l'espèce, en régime transitoire avec un coefficient de
stockage `φ` (la porosité, pour une espèce diffusant dans un solide poreux) :

```text
φ ∂c/∂t + ∇·j = 0     ⇔     φ ∂c/∂t − ∇·(D ∇c) = 0
```

En stationnaire c'est l'équation de Laplace pondérée par `D`. La forme faible,
après intégration par parties, donne la rigidité `∫ ∇N_i · D · ∇N_j` et, pour le
terme instationnaire, la matrice de « masse » `∫ φ N_i N_j`.

## Forme discrétisée

```text
K_ij = ∫_Ω ∇N_iᵀ · D · ∇N_j dΩ        (rigidité de diffusion — Cast3M COND)
C_ij = ∫_Ω φ · N_i N_j dΩ             (stockage — Cast3M CAPA)
```

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
