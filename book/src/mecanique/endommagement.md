# Lois d'endommagement

## Introduction

L'endommagement décrit la **perte de raideur** d'un matériau qui se fissure, sans
déformation permanente : la contrainte est la contrainte effective, dégradée.
Comme pour la plasticité, la loi est un **attribut** du même modèle
([`DamageLaw`]) — mêmes degrés de liberté, même montage incrémental, seule change
la loi qui transforme une déformation en contrainte dégradée.

| loi | variable(s) | ce qu'elle capture |
|---|---|---|
| `mazars` | un scalaire `D` | béton, deux branches mélangées |
| `damage_tc` | `d⁺`, `d⁻` | traction et compression séparées — l'effet unilatéral |
| `damage_sic_sic` | `d_1, d_2, d_3` | composite tissé, endommagement **orthotrope** |

S'y ajoute, du côté plastique, la **plasticité poreuse** de Gurson
(`Model.gurson`), où l'endommagement est une porosité qui rétrécit la surface de
charge — voir plus bas.

## Mazars — un scalaire

L'endommagement est piloté par une déformation équivalente construite sur les
déformations principales positives, `ε̃ = √(Σ⟨ε_I⟩₊²)` : il répond à l'extension
et est aveugle à la compression hydrostatique. Deux branches, traction et
compression, sont mélangées par des poids `α_t`, `α_c` issus du découpage de la
contrainte effective — c'est ce qui permet à une seule loi de décrire un matériau
un ordre de grandeur plus résistant en compression.

La variable d'histoire unique est `κ = max_t ε̃` : l'endommagement ne guérit pas.

## Damage TC — deux variables

Mélanger les deux branches en un scalaire a un coût : un matériau endommagé en
compression l'est autant en traction, et le modèle **ne peut pas** représenter une
fissure qui se **referme** et reprend de la charge.

Damage TC les garde séparées :

\\[
\sigma = (1 - d^+)\,\tilde\sigma^+ + (1 - d^-)\,\tilde\sigma^-
\\]

où `σ̃⁺` et `σ̃⁻` sont les parties positive et négative de la contrainte
**effective**, découpées sur ses valeurs principales. Chacune est dégradée par sa
propre variable, si bien qu'un déchargement de la traction vers la compression
retrouve la raideur en compression — l'**effet unilatéral**, ce qui rend la loi
utilisable en cyclique.

Chaque endommagement a son moteur et son histoire : une énergie élastique en
traction, une mesure octaédrique en compression (qui répond au confinement et non
à la seule extension), et deux lois d'adoucissement distinctes — exponentielle en
traction, une fissure est fragile ; durcissante puis adoucissante en compression,
le béton s'écrase, il ne casse pas net.

## SiC/SiC — endommagement orthotrope

Un composite SiC/SiC est une matrice de carbure de silicium renforcée par des
torons de fibres, généralement tissés. Il ne rompt ni comme un métal ni comme un
béton : la matrice fissure d'abord, dans des plans normaux aux directions de
torons, tandis que les fibres continuent de porter la charge **à travers** ces
fissures. La raideur chute donc **par direction**, et très inégalement.

Aucun endommagement scalaire ne peut exprimer cela. Cette loi porte **un
endommagement par direction matériau** :

```text
d_i = d_max,i · (1 − exp(−(⟨ε_i⟩₊ − ε_0,i)/ε_c,i))      pour ⟨ε_i⟩₊ > ε_0,i
```

La partie positive est essentielle : une fissure de matrice s'ouvre en
**extension** et se referme en compression, si bien qu'une direction comprimée
n'est pas dégradée du tout.

### Le repère est le tissage

Les directions d'endommagement sont les **axes matériau**, fournis exactement
comme pour l'[élasticité orthotrope](orthotropie.md) — par les vecteurs `V1`,
`V2` du champ matériau. Ce n'est pas une coïncidence : pour un composite tissé
ce **sont** les directions de torons, et réutiliser le même repère donne
gratuitement les bonnes directions maille par maille sur une pièce courbe.

### Saturation, pas rupture

Chaque `d_i` sature à `d_max,i` plutôt que d'atteindre 1. C'est l'énoncé physique
que la fissuration matricielle ne prend **pas** toute la raideur : les fibres
restent, et un composite saturé porte encore le long de ses torons. Une loi
laissant l'endommagement atteindre 1 prédirait un effondrement qui n'a pas lieu.

## Gurson — la porosité comme endommagement

Un métal ductile ne rompt pas en atteignant une contrainte : il rompt parce que
des cavités germent, croissent et coalescent jusqu'à ce que les ligaments entre
elles ne portent plus. La surface de Gurson fait de la porosité `f` une variable
interne explicite, qui **rétrécit** la surface de charge :

\\[
\Phi = \left(\frac{q}{\sigma_y}\right)^2 + 2q_1 f^* \cosh\!\left(\frac{3q_2\sigma_m}{2\sigma_y}\right) - (1 + q_3 f^{*2})
\\]

À `f = 0` cela redonne exactement von Mises. Le `cosh` rend la contraction
dépendante de la contrainte **hydrostatique** : les cavités croissent en traction
triaxiale et se referment en compression. C'est cette sensibilité à la pression
qui fait qu'une loi J2 ne peut jamais prédire une rupture ductile.

**Coalescence** — au-delà d'une porosité critique les cavités coalescent et
l'effondrement s'accélère. Tvergaard et Needleman le modélisent en donnant à la
surface une porosité *effective* `f*` qui atteint `1/q₁` — surface réduite à
rien — à la porosité de rupture `f_f`.

**Croissance** — `ḟ = (1 − f)·tr(ε̇_p)`, conservation de la masse et rien de
plus. L'écoulement plastique sur cette surface n'est **pas** isochore, ce qui le
distingue précisément de von Mises. La germination n'est **pas** modélisée : seule
la croissance depuis une porosité initiale `f_0`.

## Mise en donnée (Rust, testé)

```rust,ignore
{{#include ../../../tests/damage_laws.rs:example}}
```

## Exemple Python

```python
model = pyrucast.Model.damage_tc(fes, "solid")
materials = pyrucast.element_field.material_field(
    model,
    [
        ("E", 30_000.0),
        ("nu", 0.2),
        ("f_t", 3.0),
        ("f_c", 30.0),
        ("A_t", 0.9),
        ("A_c", 0.5),
    ],
)
strain = pyrucast.element_field.deformation(u, fes)
state = pyrucast.element_field.integrate_behavior(model, strain, materials)
# `state` porte d_plus, d_minus, r_plus, r_minus — et redevient le `prev` du pas suivant.
```

## Compléments

**Ce que valent les tests.** Chaque loi est épinglée sur ce qu'elle fait et que
les autres ne font pas : une fissure qui se **referme** reprend toute la charge
en compression (Damage TC), l'écrasement laisse l'endommagement de traction
intact, l'étirement selon un axe de tissage n'endommage **que** cette direction
et sature à son plafond (SiC/SiC), et la porosité de Gurson croît sous traction
triaxiale sans jamais décroître.

**Une subtilité relevée en écrivant les tests.** Sous une déformation purement
compressive, les poids `α` de Mazars s'annulent et la loi ne rapporte *aucun*
endommagement — propriété connue du modèle, et la raison même pour laquelle une
variable compressive séparée vaut son coût.

**État absent ≠ état nul.** Une loi dont l'état démarre d'une constante matériau
— la porosité initiale de Gurson — doit pouvoir distinguer « pas encore d'état »
de « état à zéro ». Le premier pas passe donc un vecteur de variables **vide**,
et non un vecteur de zéros. Sans cela un métal poreux démarrerait dense et ne
s'endommagerait jamais.

**Pas de tangente cohérente** pour les lois d'endommagement : comme Mazars, elles
ne déclarent pas de `tangent_layout`, et l'opérateur d'itération reste la rigidité
élastique non endommagée. Gurson, qui est une plasticité, en a une (numérique).
