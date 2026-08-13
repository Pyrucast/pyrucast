# Endommagement de Mazars

Modèle d'**endommagement isotrope** du béton (Mazars, 1986), formulation
**classique à deux variables**. Loi *sécante* : la contrainte est la contrainte
élastique (effective) affaiblie par un scalaire d'endommagement `D ∈ [0, 1)`.
Mêmes éléments et degrés de liberté que l'[élasticité linéaire](elasticite.md)
(2-D `TRI3` / `QUA4`, 3-D `TET4` / `HEX8`).

## Équations continues résolues

- **contrainte** : `σ = (1 − D) · D_el : ε` ;
- **déformation équivalente** : `ε̃ = √(Σ ⟨ε_I⟩₊²)`, somme sur les parts
  positives des **déformations principales** ;
- **historique** : `κ = maxₜ ε̃`, initialisé au seuil `eps_d0` ; l'endommagement
  ne croît que lorsque `κ` augmente (irréversibilité, pas de guérison).

L'endommagement combine une branche **traction** et une branche
**compression** :

\\[
D_t = 1 - \frac{\varepsilon_{d0}(1-A_t)}{\kappa} - \frac{A_t}{e^{B_t(\kappa-\varepsilon_{d0})}},
\qquad
D = \alpha_t D_t + \alpha_c D_c,
\\]

(et `D_c` de même avec `A_c, B_c`). Les poids `α_t, α_c` proviennent de la
**décomposition traction/compression** de la contrainte effective
`σ̃ = D_el : ε` : on sépare ses contraintes principales en parts positive /
négative, on en déduit les déformations associées `εᵗ, εᶜ`, puis
`α_t = Σ ⟨εᵗ_I⟩₊⟨ε_I⟩₊ / ε̃²` (idem `α_c`). Le coefficient de cisaillement
`β` est fixé à 1.

La décomposition spectrale (déformations principales) utilise `nalgebra`.

## Forme discrétisée

Comme la [plasticité](plasticite.md), Mazars expose deux briques, la boucle de
Newton restant **pilotée en Python** (voir
[Comportement](../operateurs/comportement.md)) :

- **rigidité** : la **rigidité élastique** (non endommagée)
  `K = ∫ Bᵀ D_el B dΩ`, opérateur d'itération ;
- **comportement** (`COMP`) : la mise à jour `σ`, `D`, `κ` point par point.

La boucle de Newton résout \\( K\\,\delta u = F_{\text{ext}} - F_{\text{int}} \\)
avec les **forces internes** endommagées

\\[
F_{\text{int}} = \int_\Omega B^\top \sigma\\, d\Omega, \qquad
\sigma = (1 - D)\\,D_{\text{el}} : \varepsilon,
\\]

où \\( B \\) est la matrice déformation-déplacement de
l'[élasticité](elasticite.md#forme-discrétisée). La rigidité **sécante**
(élastique non endommagée) sert d'opérateur d'itération ; la loi étant *sécante*
et non incrémentale, seule la variable d'historique `κ` porte l'irréversibilité.

Le calcul interne est mené en **3-D**. La **déformation plane** impose
`ε_zz = 0` ; la **contrainte plane** pose `ε_zz = −ν/(1−ν)(ε_xx+ε_yy)` (le
facteur `(1−D)` se simplifie dans `σ_zz = 0`).

## Variables et matériau

- **primal** : `u_x, u_y(, u_z)` — **dual** : `f_x, f_y(, f_z)`.
- **matériau** : `E`, `nu`, `eps_d0` (seuil), `A_t, B_t` (traction),
  `A_c, B_c` (compression).
- **état de début de pas A** (entrée `prev`) : la variable scalaire d'historique
  `kappa`. `None` au premier pas — elle est plafonnée à `eps_d0` dans la mise à
  jour, donc l'absence est correcte. La contrainte effective ne dépend que de la
  déformation totale `ε(B)` : la mécanique de l'endommagement n'a pas
  d'incrément, seul `kappa` est historique.
- **sortie du `COMP`** (= `prev` du pas suivant) : contrainte (`sigma_*`),
  `damage` (le scalaire `D`), et `kappa` mis à jour.
- **modèles** : `plane_stress`, `plane_strain`, `axisymmetric` (2-D) et `solid`
  (3-D).

## Axisymétrie

Le modèle `"axisymmetric"` s'applique sur une géométrie de révolution
([`Coords.axisymmetric()`](../coords.md#repère-de-révolution)) : Voigt à quatre
composantes `[εrr, εzz, εθθ, γrz]`, nommées `eps_xx, eps_yy, eps_zz, eps_xy`
avec **`zz` = orthoradial** (convention Cast3M). Le modèle et le repère doivent
s'accorder dans les deux sens, comme en [élasticité](elasticite.md#axisymétrie).

La déformation équivalente étant bâtie sur les **déformations principales du
tenseur 3-D complet**, les modèles 2-D ne diffèrent que par la reconstruction de
ce tenseur : la déformation plane force `ε_zz = 0`, la contrainte plane la
déduit, et l'axisymétrie lit l'orthoradiale `ε_θθ = u_r/r` mesurée.

## Exemple Python

```python
import pyrucast

model = pyrucast.Model.mazars(fes, "plane_stress")
materials = pyrucast.element_field.material_field(
    model,
    [
        ("E", 30_000.0),
        ("nu", 0.2),
        ("eps_d0", 1e-4),
        ("A_t", 0.8),
        ("B_t", 20_000.0),
        ("A_c", 1.4),
        ("B_c", 1_900.0),
    ],
)

strain = pyrucast.element_field.deformation(u, fes)
state = pyrucast.element_field.integrate_behavior(
    model, strain, materials, prev=prev_state
)
d = state[0].value(0, 0, "damage")  # endommagement scalaire D
kappa = state[0].value(0, 0, "kappa")  # variable d'historique
```

L'historique `kappa` se réinjecte au pas suivant en **passant `state` comme
`prev`** — la sortie le porte déjà —, ce qui garantit l'irréversibilité.
