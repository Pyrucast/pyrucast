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

## Six DDL par nœud, et celui de vrillage

La cinématique naturelle d'une coque compte cinq degrés de liberté — trois
translations et deux rotations de la fibre normale. Mais le cinquième et le
sixième ne se distinguent que dans le repère **local**, alors qu'un assembleur
global numérote les DDL par leur nom. L'élément en porte donc six,
`u_x…u_z, r_x…r_z`, exactement comme le [cadre 3D](cadre3d.md) — ce qui permet
aussi à une coque et à un portique spatial de partager des nœuds sans adaptateur.

Le sixième, la rotation autour de la normale, est le DDL de **vrillage**, et une
facette plane n'oppose aucune raideur physique à son sujet : laissé seul, il rend
la matrice élémentaire singulière. Il est donc lié à la rotation propre de la
membrane,

\\[
\omega_z = \tfrac12\left(\frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}\right),
\qquad K_\text{vrillage} = \alpha\,G h \int (\theta_z - \omega_z)^2\, dA
\\]

ce qui est un énoncé physique (la rotation de vrillage doit suivre celle de la
matière) et non une béquille numérique. Une pénalité diagonale serait le raccourci
tentant, et serait **fausse** : elle s'oppose à une rotation rigide de la facette
autour de sa normale, qui ne coûte aucune énergie. C'est ce que vérifie un test.

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

La fibre normale reste droite mais **non normale** : sa rotation est un champ
indépendant, si bien que le cisaillement transverse `γ = ∇w + θ` est une
déformation à part entière.

```text
membrane   ε = [∂u/∂x, ∂v/∂y, ∂u/∂y + ∂v/∂x]
flexion    κ = [∂θ_y/∂x, −∂θ_x/∂y, ∂θ_y/∂y − ∂θ_x/∂x]
cisaillement γ = [∂w/∂x + θ_y, ∂w/∂y − θ_x]
```

```text
D_m = Eh/(1−ν²)·[[1, ν, 0], [ν, 1, 0], [0, 0, (1−ν)/2]]
D_b = D_m · h²/12
D_s = k_s·G·h                        k_s = 5/6 par défaut
```

La loi de flexion est celle de membrane multipliée par `h²/12` : c'est tout le
contenu de « les sections restent planes » — le même matériau en contraintes
planes, intégré dans l'épaisseur avec un poids `z²`.

### Pourquoi le cisaillement est intégré réduit

À mesure que la coque s'amincit, `D_s` (linéaire en `h`) écrase `D_b` (cubique en
`h`) d'un facteur `1/h²`. Intégré à la quadrature complète, le terme de
cisaillement impose alors `γ = 0` **point par point**, ce qu'un élément linéaire
ne peut satisfaire qu'en refusant de fléchir : le déplacement s'effondre vers
zéro et aucun raffinement ne le récupère. C'est le **blocage en cisaillement**.

L'intégrer en un seul point relâche la contrainte en moyenne, l'élément fléchit,
et le calcul converge. C'est le même remède et le même mécanisme que pour la
[poutre de Timoshenko](timoshenko.md) — d'où le partage de la structure
multi-quadrature plutôt que deux inventions parallèles.

## Mise en donnée (Rust, testé)

```rust,ignore
{{#include ../../../tests/shell.rs:example}}
```

## Exemple Python

```python
model = pyrucast.Model.shell(fes, "thick")
materials = pyrucast.element_field.material_field(
    model, [("E", 210_000.0), ("nu", 0.3), ("h", 0.01)]
)
k = pyrucast.matrix.stiffness(model, materials)
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

**Ce qui n'est pas couvert.** Pas de matrice de masse ni de raideur géométrique
pour l'instant ; pas de coque courbe au sens propre — une facette plane par
élément, ce qui est la formulation usuelle des coques facettisées et demande un
maillage plus fin sur une forte courbure.
