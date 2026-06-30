# Opérateur de comportement

Le module `ops::behavior` **intègre la loi de comportement** d'un
[`Model`](../model.md) — le `COMP` de cast3m (« intégrer le comportement »).

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
