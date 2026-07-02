# Opérateurs de comportement

Le module `ops::behavior` **intègre la loi de comportement** d'un
[`Model`](../model.md) — le `COMP` de cast3m (« intégrer le comportement ») ; le
module `ops::internal_forces` en calcule les **forces internes** — le `BSIG` de
cast3m (`∫ Bᵀ σ`).

## `integrate_behavior(model, deformation, materials)` → `ElementField`

Là où [`stiffness`](assemblage.md) produit la **linéarisation** du modèle (une
matrice), `integrate_behavior` produit la **réponse ponctuelle exacte** comme
[champ aux points de Gauss](../element-field.md).

Le découpage est volontaire :

1. l'**entrée de déformation** est construite **séparément et géométriquement**
   par [`gradient`](champs.md) (`∇T`…), [`deformation`](champs.md) (`ε`) ou
   [`beam_deformation`](champs.md) (`κ, γ`) — ces opérateurs ne dépendent que de
   l'espace EF, pas du modèle ; le choix de *quelle* déformation nourrir reste
   donc à l'appelant ;
2. `integrate_behavior` prend cette déformation (plus, optionnellement, l'état
   interne d'entrée `VAR0`) et le **matériau par zone**, et applique la loi de
   chaque physique **point par point** ;
3. il renvoie l'**état matériau** : le flux / la contrainte dual(e) plus les
   variables internes mises à jour `VAR1`.

Les sous-modèles de **contrainte** (Dirichlet…) sont ignorés : un sous-modèle
participe ssi il déclare un espace EF de comportement. Les zones de déformation
et de matériau sont appariées par sous-espace EF.

Pour une loi **linéaire**, le résultat est cohérent avec `stiffness`
(`∫ Bᵀ·flux = K·u`) ; une loi **non linéaire** s'écarte de cette tangente — c'est
tout l'intérêt d'intégrer le comportement exactement.

## Exemple : efforts de section d'une poutre

```python
# Solution (w, theta) déjà obtenue par le solveur.
eps = pyrucast.beam_deformation(solution, fes)  # (κ, γ) par élément
forces = pyrucast.integrate_behavior(model, eps, materials)
# forces porte le moment M = E·I·κ et l'effort tranchant V = G·A_s·γ.
```

Les pages [Barre](../mecanique/truss.md), [Élasticité](../mecanique/elasticite.md)
et [Timoshenko](../mecanique/timoshenko.md) détaillent l'intégrande de
comportement (`COMP`) de chaque physique. Pour les lois **non linéaires** avec
variables internes (`VAR0` → `VAR1`), voir
[Plasticité parfaite](../mecanique/plasticite.md) (retour radial von Mises) et
[Endommagement de Mazars](../mecanique/mazars.md).

## `internal_forces(model, stresses)` → `NodeField`

Les **forces internes** `f = ∫ Bᵀ σ dΩ` (le `BSIG` de cast3m) sont la
**transposée** de l'opérateur de déformation `B` : là où
[`deformation`](champs.md) applique `B` au déplacement (`ε = B·u`),
`internal_forces` applique `Bᵀ` à la contrainte et **rassemble** le résultat aux
nœuds. C'est la **généralisation mécanique** de [`divergence`](champs.md) (qui
est exactement `Bᵀ q` pour un transport scalaire) : une composante de sortie par
DDL dual.

`stresses` est le champ d'état matériau renvoyé par `integrate_behavior`. Chaque
sous-modèle porteur d'un comportement applique **son propre** `Bᵀ` — c'est
pourquoi l'opérateur prend un **modèle** et pas un simple espace EF : un même
`SEG2` peut être une barre (`Bᵀ` axial, DDL déplacement) ou une poutre (`Bᵀ` à
deux quadratures flexion + cisaillement, DDL `w, θ`), et seul le modèle tranche.
Solides continus, barres et poutres sont donc tous couverts.

Pour une loi **linéaire**, le résultat égale la rigidité appliquée à la solution
(`K·u`) ; pour une loi **non linéaire**, il donne les forces internes exactes, de
sorte que `r = f_ext − f_int` est le **résidu** d'équilibre.

```python
# Solution déjà obtenue par le solveur.
eps    = pyrucast.deformation(solution, fes)          # ε = B·u
sig    = pyrucast.integrate_behavior(model, eps, materials)  # COMP : σ
f_int  = pyrucast.internal_forces(model, sig)         # BSIG : ∫ Bᵀ σ
residu = f_ext - f_int                                # équilibre
```

### `internal_forces_continuum(stresses, fespace)` → `NodeField`

Variante **sans modèle** pour le cas **continu** (élasticité, Mazars,
plasticité), où `B` est le gradient symétrique universel et les DDL sont toujours
un déplacement : elle ne demande que la géométrie (`fespace`) et la contrainte en
notation de Voigt (`sigma_xx`, `sigma_xy`…), et renvoie `space_dim` composantes
`f_x, f_y, f_z` par nœud. **Barres et poutres ne sont pas couvertes** — leur `B`
n'est pas le gradient symétrique — : utiliser `internal_forces(model, stresses)`
pour celles-ci.
