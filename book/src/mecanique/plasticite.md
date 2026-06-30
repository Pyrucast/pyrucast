# Plasticité parfaite (von Mises)

Élastoplasticité **parfaite** (sans écrouissage) en petites déformations,
critère de **von Mises** (J2), écoulement associé. Mêmes éléments et mêmes
degrés de liberté que l'[élasticité linéaire](elasticite.md) : 2-D (`TRI3` /
`QUA4`) ou 3-D (`TET4` / `HEX8`).

## Équations résolues

- **équilibre** : `∇·σ + b = 0` ;
- **partition** : `ε = εᵉ + εᵖ` (déformation élastique + plastique) ;
- **élasticité** : `σ = D : εᵉ` ;
- **critère** : `f(σ) = q − σ_y ≤ 0`, avec `q = √(3/2 · s:s)` la contrainte
  équivalente de von Mises (`s` = déviateur de `σ`) ;
- **écoulement associé** : `ε̇ᵖ = λ̇ · ∂f/∂σ`, conditions de Kuhn–Tucker
  `λ̇ ≥ 0, f ≤ 0, λ̇ f = 0`.

Sans écrouissage, `σ_y` est constant : la contrainte équivalente est
**plafonnée** à `σ_y` (plateau parfaitement plastique).

## Linéarisation et comportement

Conformément au découpage du cœur (voir
[Comportement](../operateurs/comportement.md) — la boucle de Newton est
**pilotée en Python**, le cœur Rust ne fournit que les briques), cette physique
expose deux briques :

- **rigidité** (`build_stiffness_blocks`) : la **rigidité élastique**
  `K = ∫ Bᵀ D B dΩ`, utilisée comme opérateur d'itération (la matrice
  tangente cohérente `KTAN` viendra plus tard) ;
- **comportement** (`COMP`, `integrate_behavior`) : le **retour radial**
  exact, point de Gauss par point de Gauss.

### Retour radial (algorithme)

À partir de la déformation totale `ε` et de l'état plastique précédent
(`εᵖ`, `p`) :

1. **prédiction élastique** : `σ_trial = D : (ε − εᵖ)`, `q = √(3/2 s_trial:s_trial)` ;
2. si `f = q − σ_y ≤ 0` → pas élastique, état inchangé ;
3. sinon (plasticité parfaite) : `Δp = f / (3μ)`, le déviateur est ramené
   `s = s_trial · σ_y / q`, `Δεᵖ = Δp · (3/2) s_trial / q`, puis
   `εᵖ ← εᵖ + Δεᵖ`, `p ← p + Δp`.

Le calcul interne est mené en **3-D** quel que soit le modèle. La
**déformation plane** impose `ε_zz = ε_yz = ε_xz = 0` ; la **contrainte plane**
résout la condition `σ_zz = 0` par une méthode de la sécante locale autour du
retour radial.

## Variables, matériau, état

- **primal** : `u_x, u_y(, u_z)` — **dual** : `f_x, f_y(, f_z)`.
- **matériau** : `E` (Young), `nu` (Poisson), `sigma_y` (limite d'élasticité).
- **état interne** (`VAR0` → `VAR1`, transitant par le champ d'entrée/sortie du
  `COMP`) : tenseur de déformation plastique `eps_p_xx … eps_p_xy` (toujours
  **6 composantes 3-D**) et déformation plastique cumulée `p`. Absent au
  premier pas, il est pris **nul** par défaut.
- **sortie du `COMP`** : contrainte (`sigma_*` dans l'ordre de Voigt du modèle)
  suivie de l'état mis à jour.

## Exemple Python (la brique `COMP`)

La boucle de Newton (assemblage du résidu, résolution, mise à jour de l'état)
s'écrit en Python ; voici l'usage **d'un pas** de la brique d'intégration :

```python
import pyrucast

model = pyrucast.Model.plasticity(fes, "plane_stress")
materials = pyrucast.material_field(
    model, [("E", 210_000.0), ("nu", 0.3), ("sigma_y", 250.0)]
)

# Déformation issue du champ de déplacement courant (op géométrique).
strain = pyrucast.deformation(u, fes)
# Intégration du comportement : contrainte + état plastique mis à jour.
state = pyrucast.integrate_behavior(model, strain, materials)
sigma_xx = state[0].value(0, 0, "sigma_xx")
p = state[0].value(0, 0, "p")  # déformation plastique cumulée
```

Pour réinjecter l'état au pas suivant, on fusionne `eps_p_*` / `p` de `state`
avec la nouvelle déformation (op [`merge`](../operateurs/champs.md)) avant
d'appeler de nouveau `integrate_behavior`.
