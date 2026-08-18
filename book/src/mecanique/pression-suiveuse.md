# Pression suiveuse

## Introduction

Une pression est toujours **normale à la surface** sur laquelle elle s'exerce.
Quand le corps se déforme, cette surface bouge et bascule : la direction de la
charge bouge avec elle. C'est une charge **suiveuse**.

L'ignorer n'est exact qu'en petits déplacements. Sur une membrane qui se gonfle,
une coque qui flambe, une aube qui tourne, la différence n'est pas un détail :
c'est elle qui décide de la charge critique.

\\[
\mathbf t = -p\\,\mathbf n(u),
\\]

\\( \mathbf n(u) \\) étant la normale de la surface **déformée**.

Les degrés de liberté sont ceux de la mécanique — déplacement `u_x, u_y(, u_z)`,
force nodale `f_x, …` — et le modèle vit sur un maillage de **bord** : SEG2 en
2-D, TRI3/QUA4 en 3-D.

## Pourquoi c'est un modèle et pas un chargement

Une charge morte se construit une fois avec
[`flux`](../operateurs/champs.md) et ne se regarde plus. Une pression suiveuse ne
le peut pas : sa direction dépend du déplacement courant, donc elle doit être
**recalculée à chaque évaluation du résidu**. C'est exactement ce que fait une
physique — elle intègre un comportement et contribue aux forces internes — donc
c'en est une :

```text
u  ──gradient──▶  ∇_s u  ──integrate_behavior──▶  t(u)  ──internal_forces──▶  f(u)
```

C'est dans l'intégration du comportement que la direction se rafraîchit. Rien
d'autre dans la chaîne ne change, et la boucle de Newton reste pilotée depuis
Python comme les autres non-linéarités.

## Équations continues résolues

Le travail virtuel de la pression sur la configuration **déformée** :

\\[
\delta W = -\int_{\gamma} p\\, \mathbf{n}\cdot\delta\mathbf{u}\\; da
\\]

où \\(\gamma\\) et \\(\mathbf{n}\\) sont la surface et la normale *actuelles*.
Tout le travail consiste à ramener cette intégrale sur la configuration de
référence, ce qui demande à la fois la **rotation** de la normale et le
**changement d'aire**.

## Forme discrétisée — par les tangentes déformées

Les deux viennent des tangentes de la surface. Si \\(a_k = \partial x/\partial
\xi_k\\) sont les tangentes de référence, les tangentes déformées sont

\\[
\bar{a}_k = a_k + \frac{\partial u}{\partial \xi_k} = a_k + (\nabla_s u)\cdot a_k
\\]

et la normale multipliée par le rapport d'aires est leur produit vectoriel (leur
rotation de −90° en 2-D), divisé par celui de référence :

\\[
\mathbf t = -p\\;\frac{\bar a_1 \times \bar a_2}{\lVert a_1 \times a_2 \rVert}
\quad \text{(3-D)},
\qquad
\mathbf t = -p\\;\frac{(\bar a_y,\\; -\bar a_x)}{\lVert a \rVert}
\quad \text{(2-D)},
\\]

les forces s'en déduisant par la mesure de **référence**, comme n'importe quelle
force interne :

\\[
f_i = \int_{\Gamma_0} N_i\\,\mathbf t\\; d\Gamma_0 .
\\]

Garder la traction **référentielle** est ce qui permet à l'intégrale des forces
internes d'utiliser la mesure de référence habituelle : la formulation reste
totalement lagrangienne, et sans déplacement elle redonne exactement `t = −p·N`.

### Pourquoi pas Nanson

\\(n\\,da = \det(F)\\,F^{-T}N\\,dA\\) est la route classique, et c'est la
**mauvaise** ici. Sur une variété, le gradient tangentiel n'a aucune composante
selon la normale : \\(I + \nabla_s u\\) n'est donc pas un gradient de
transformation. Un quart de tour de la surface envoie son déterminant à zéro et
la formule explose sur une rotation parfaitement ordinaire. Les tangentes, elles,
ne dégénèrent jamais ainsi — elles tournent avec la surface. C'est sur elles
qu'une charge surfacique doit être bâtie.

C'est le genre d'écueil qu'un test attrape et qu'une relecture laisse passer :
la rotation à 90° est dans la suite de tests pour cette raison.

## Variables et matériau

| | |
|---|---|
| primales | `u_x, u_y(, u_z)` |
| duales | `f_x, f_y(, f_z)` |
| matériau | `p` (la pression) |
| entrée du comportement | `grad_u_x_x`, … (le gradient surfacique de `u`) |
| sortie du comportement | `t_x, t_y(, t_z)` (la traction référentielle) |
| nature | `Mechanical` |

### L'orientation est l'affaire du maillage

La normale suit le **sens de parcours** du maillage de bord. Un `p` positif
pousse *contre* elle — compression — donc un bord orienté vers l'extérieur donne
le signe habituel.

C'est le seul endroit où l'orientation d'un maillage de bord compte. Par
contraste, la [convection](../thermique.md#convection-de-surface-robin--film) et
le [rayonnement](../thermique.md#rayonnement-à-linfini-stefan-boltzmann) y sont
aveugles : leur direction est déjà consommée en écrivant `q·n`, et la mesure
`det_j_w` est une magnitude invariante.

## Ce qu'elle contribue

Des forces internes, et rien d'autre. Elle déclare un `stiffness_layout` — c'est
de là que la dispersion des forces internes est pilotée — mais ses
`contributions()` sont **vides** pour tous les genres de matrice. La raideur de
suivi \\(\partial f/\partial u\\) (non symétrique) n'est pas implémentée : une
boucle de Newton converge sans elle, plus lentement.

## Mise en donnée (Rust, testé)

```rust,ignore
{{#include ../../../tests/follower_pressure.rs:example}}
```

## Exemple Python

```python
{{#include ../../../tests/python/test_doc_mecanique.py:pression_suiveuse}}
```

## Compléments

**Ce que ça vaut comme vérification.** Une charge suiveuse ne se contrôle pas en
comparant une valeur à une formule une fois : elle se contrôle en vérifiant
qu'elle **bouge** comme la surface. Trois régimes l'épinglent :

| déplacement | charge attendue |
|---|---|
| aucun | `(−p, 0)` — exactement la charge morte |
| rotation rigide de `θ` | `(−p cosθ, −p sinθ)` — même module, tournée de `θ` |
| étirement `λ` le long du bord | `(−pλ, 0)` — l'aire déformée a grandi |

Le deuxième est ce qu'une charge non suiveuse rate ; le troisième est ce que
rate une charge suiveuse qui se contenterait de tourner sa direction en oubliant
le changement d'aire.
