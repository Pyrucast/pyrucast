# Lois d'écoulement plastique

## Introduction

La [plasticité parfaite](plasticite.md) est un cas particulier. Toutes les lois
élastoplastiques de pyrucast partagent **la même physique** — mêmes degrés de
liberté, même rigidité élastique comme opérateur d'itération, même montage
incrémental A → B, même état interne — et ne diffèrent que par leur **surface de
charge** et leur règle d'écoulement.

La loi est donc un **attribut** du modèle de plasticité, pas un modèle à part.
C'est la convention de Cast3M, où `PLASTIQUE PARFAIT`, `PLASTIQUE ISOTROPE`,
`PLASTIQUE DRUCKER_PRAGER` et `PLASTIQUE OTTOSEN` sont des variantes d'une seule
formulation.

| loi | surface | ce qu'elle capture | matériau |
|---|---|---|---|
| `plasticity_perfect` | `q = σ_y` | métal sans écrouissage | `E, nu, sigma_y` |
| `plasticity_isotropic` | `q = σ_y + H·p` | métal écrouissable | `+ H` |
| `drucker_prager` | `q + α·I₁ = k` | sols, roches, poudres — sensibilité à la pression | `E, nu, alpha, k, psi` |
| `ottosen` | 4 paramètres, dépendante de l'angle de Lode | béton — traction ≠ compression | `E, nu, a, b, k_1, k_2, sigma_c` |

Ce qui est mutualisé dans `src/models/plastic/` : le prédicteur élastique, la
**boucle sécante des contraintes planes** (qui résout `σ_zz(B) = 0` *autour* de
n'importe quelle loi, si bien qu'aucune loi ne connaît les contraintes planes),
le retour par **plan sécant** pour les surfaces sans forme fermée, et la tangente
cohérente.

L'état est toujours porté en **3-D complet** (six `eps_p_*` et un `p` cumulé)
quel que soit le modèle 2-D : chaque retour est ainsi identique en contraintes
planes, déformations planes, axisymétrie et massif — seules les projections
d'entrée et de sortie changent.

## Écrouissage isotrope

La surface se dilate avec la déformation plastique cumulée :

\\[
f(\sigma, p) = q - \big(\sigma_y + H\,p\big)
\\]

Le retour reste **radial** et fermé, la consistance donnant le multiplicateur en
un pas :

```text
Δp = (q_trial − σ_y(p_A)) / (3μ + H)
```

`H = 0` redonne exactement la loi parfaite — un seul chemin de code sert les
deux, ce que vérifie un test.

## Drucker-Prager

Sols, roches, bétons et poudres sont **plus résistants en compression qu'en
traction** : leur seuil dépend de la pression hydrostatique, que von Mises
ignore. Drucker-Prager est le cône le plus simple qui le capture :

\\[
f(\sigma) = q + \alpha\,I_1 - k
\\]

### Un écoulement non associé

Un écoulement associé sur ce cône ferait dilater le matériau sous cisaillement
d'exactement ce que son frottement implique — bien trop pour un milieu granulaire
réel. Le potentiel plastique porte donc **sa propre** pente, la dilatance `ψ` :

\\[
g(\sigma) = q + \psi\,I_1, \qquad \psi \le \alpha
\\]

`ψ = α` redonne l'écoulement associé ; `ψ = 0` donne un écoulement plastique
isochore à résistance frottante.

### Le sommet

Un cône a une pointe, en `I₁ = k/α`. Une contrainte d'essai au-delà y retourne,
et non sur le flanc : le retour lisse pousserait sinon la contrainte équivalente
dans le négatif, ce qui n'a pas de sens. C'est **le** cas qu'une implémentation
naïve rate silencieusement sous forte traction.

Au sommet la contrainte est figée, donc **la tangente y est nulle** — et
délibérément. Renvoyer le module élastique, la solution de facilité, rendrait la
tangente incohérente avec le retour. Un corps entièrement à son sommet assemble
donc une tangente singulière : ce n'est pas un artefact, c'est le constat honnête
qu'un tel matériau ne porte plus rien.

## Ottosen

Le béton casse très différemment en traction et en compression, et sa résistance
sous une pression donnée dépend de **la direction déviatorique** de la
contrainte. Ni von Mises (aveugle à la pression) ni Drucker-Prager (aveugle à
cette direction) ne le capturent. La surface d'Ottosen le fait par une dépendance
à l'**angle de Lode** :

\\[
f(\sigma) = a\frac{J_2}{\sigma_c^2} + \lambda(\theta)\frac{\sqrt{J_2}}{\sigma_c}
          + b\frac{I_1}{\sigma_c} - 1
\\]

```text
λ(θ) = k₁·cos[⅓ arccos(k₂ cos3θ)]              si cos3θ ≥ 0
λ(θ) = k₁·cos[π/3 − ⅓ arccos(−k₂ cos3θ)]       si cos3θ < 0
```

Les méridiens sont **courbes** et la section déviatorique est un triangle arrondi
qui s'ouvre vers la compression — ce qui est tout l'objet.

### Intégrée par plan sécant, avec une normale numérique

Il n'existe pas de retour fermé exploitable sur cette surface. Pire, la normale
`∂f/∂σ` demande de dériver `λ(θ)` à travers `arccos` et `J₃` — une expression
assez longue pour qu'une erreur de signe y soit invisible en relecture et ne se
manifeste que par une direction d'écoulement légèrement fausse.

Le retour passe donc par l'algorithme du **plan sécant**, qui n'a besoin que du
scalaire `f(σ)`, la normale étant obtenue par **différences centrées**. Le critère
est alors exact et le gradient précis à `O(h²)`. Échanger un gradient analytique
invérifiable contre un gradient numérique qui ne peut pas être mal dérivé est le
bon compromis ici.

## La tangente cohérente, et deux limites assumées

Seule von Mises garde une tangente **analytique**, parce que seule sa forme
fermée a été confrontée à une différence finie.

> La dérivation analytique de Drucker-Prager, écrite d'abord, était **fausse de
> 24 %** — plausible, et fausse. Seul l'oracle par différences finies de
> `tests/plastic_laws.rs` l'a dit. La tangente numérique qui l'a remplacée ne
> peut pas être mal dérivée, coûte douze évaluations d'une mise à jour fermée, et
> laisse la convergence de Newton quadratique.

**La tangente stockée est symétrique.** `D_alg` voyage dans le champ d'état sous
forme de triangle supérieur (`ktan_i_j`, i ≤ j) et est relue en miroir : le format
ne peut pas porter la tangente réellement **non symétrique** d'une loi non
associée. Celle de Drucker-Prager est donc symétrisée — le compromis d'ingénierie
usuel, qui coûte à Newton son taux quadratique sur cette loi et rien d'autre, et
garde symétriques tous les consommateurs en aval.

**Une tangente doublement numérique n'est précise que jusqu'à un point.** Ottosen
dérive `f` pour obtenir sa normale, puis la tangente dérive toute cette carte
itérative ; les deux échelles d'erreur se composent, pour environ 10 % d'écart à
la dérivée exacte. Newton converge quand même — il lui faut une tangente assez
bonne pour converger, pas une tangente exacte à la précision machine — et le test
annonce le chiffre plutôt que de le cacher derrière une tolérance lâche partout.

## Mise en donnée (Rust, testé)

```rust,ignore
{{#include ../../../tests/plastic_laws.rs:example}}
```

## Exemple Python

```python
model = pyrucast.Model.drucker_prager(fes, "solid")
materials = pyrucast.element_field.material_field(
    model,
    [("E", 20_000.0), ("nu", 0.2), ("alpha", 0.3), ("k", 30.0), ("psi", 0.1)],
)
strain = pyrucast.element_field.deformation(u, fes)
state = pyrucast.element_field.integrate_behavior(model, strain, materials)
k_t = pyrucast.matrix.tangent(model, materials, state)
```

La boucle de Newton reste orchestrée en Python, comme pour toute non-linéarité :
le noyau Rust fournit la mise à jour ponctuelle et `D_alg`, pas la boucle.

## Compléments

**Ce que vaut chaque test.** Un test de plasticité qui n'affirme que « la
contrainte a baissé » ne prouve rien. Chaque loi est épinglée par sa propriété
définissante : la consistance `q = σ_y + H·p` pour l'écrouissage, l'atterrissage
sur le cône et l'effondrement au sommet pour Drucker-Prager, l'atterrissage sur
la surface à quatre paramètres et l'écart traction/compression pour Ottosen. Et
**toutes** ont leur tangente confrontée à une différence centrée des forces
internes.

**Renommage.** `Model.plasticity` s'appelle désormais `Model.plasticity_perfect`,
pour que les quatre lois se nomment de la même façon. Seul le nom a changé.
