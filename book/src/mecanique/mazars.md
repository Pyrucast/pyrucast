# Endommagement de Mazars

Modèle d'**endommagement isotrope** du béton (Mazars, 1986), formulation
**classique à deux variables**. Loi *sécante* : la contrainte est la contrainte
élastique (effective) affaiblie par un scalaire d'endommagement `D ∈ [0, 1)`.
Mêmes éléments et degrés de liberté que l'[élasticité linéaire](elasticite.md)
(2-D `TRI3` / `QUA4`, 3-D `TET4` / `HEX8`).

## Équations résolues

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

## Linéarisation et comportement

Comme la [plasticité](plasticite.md), Mazars expose deux briques, la boucle de
Newton restant **pilotée en Python** (voir
[Comportement](../operateurs/comportement.md)) :

- **rigidité** : la **rigidité élastique** (non endommagée)
  `K = ∫ Bᵀ D_el B dΩ`, opérateur d'itération ;
- **comportement** (`COMP`) : la mise à jour `σ`, `D`, `κ` point par point.

Le calcul interne est mené en **3-D**. La **déformation plane** impose
`ε_zz = 0` ; la **contrainte plane** pose `ε_zz = −ν/(1−ν)(ε_xx+ε_yy)` (le
facteur `(1−D)` se simplifie dans `σ_zz = 0`).

## Variables, matériau, état

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

## Exemple Python (la brique `COMP`)

```python
import pyrucast

model = pyrucast.Model.mazars(fes, "plane_stress")
materials = pyrucast.material_field(
    model,
    [
        ("E", 30_000.0), ("nu", 0.2), ("eps_d0", 1e-4),
        ("A_t", 0.8), ("B_t", 20_000.0),
        ("A_c", 1.4), ("B_c", 1_900.0),
    ],
)

strain = pyrucast.deformation(u, fes)
state = pyrucast.integrate_behavior(model, strain, materials, prev=prev_state)
d = state[0].value(0, 0, "damage")  # endommagement scalaire D
kappa = state[0].value(0, 0, "kappa")  # variable d'historique
```

L'historique `kappa` se réinjecte au pas suivant en **passant `state` comme
`prev`** — la sortie le porte déjà —, ce qui garantit l'irréversibilité.
