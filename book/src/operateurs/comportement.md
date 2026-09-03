# Opérateurs de comportement

Le module `ops::element_field::behavior` **intègre la loi de comportement** d'un
[`Model`](../model.md) — le `COMP` de cast3m (« intégrer le comportement ») ;
l'opérateur `ops::node_field::internal_forces` en calcule les **forces internes** —
le `BSIG` de cast3m (`∫ Bᵀ σ`).

## `integrate_behavior(model, deformation, materials, prev=None, dt=None)` → `ElementField`

Là où [`stiffness`](assemblage.md) produit la **linéarisation** du modèle (une
matrice), `integrate_behavior` produit la **réponse ponctuelle exacte** comme
[champ aux points de Gauss](../element-field.md).

C'est un **montage incrémental A → B** : la loi intègre le comportement entre
l'état convergé du **début de pas A** et la déformation de **fin de pas B**.

1. l'**entrée de déformation** `ε(B)` est construite **séparément et
   géométriquement** par [`gradient`](champs.md) (`∇T`…), [`deformation`](champs.md)
   (`ε`), [`beam_deformation`](champs.md) (`κ, γ`) ou
   [`shell_deformation`](champs.md) (`ε, κ, γ`) — ces opérateurs ne dépendent
   que de l'espace EF, pas du modèle ; le choix de *quelle* déformation nourrir
   reste donc à l'appelant ;
2. `prev` est l'**état convergé de A** — la **sortie du pas précédent** : la
   contrainte `σ(A)`, les variables internes `VAR(A)` et la déformation `ε(A)`.
   Il vaut `None` au **premier pas**, où A est la configuration de référence
   (`σ(A)=0`, `ε(A)=0`) ;
3. `dt` est l'**incrément de temps**, `None` pour une loi indépendante du temps
   (une loi visqueuse future erreurera s'il vaut `None`) ;
4. `integrate_behavior` prend ces entrées plus le **matériau par zone** et
   applique la loi de chaque physique **point par point** ;
5. il renvoie l'**état matériau de B** : le flux / la contrainte dual(e) plus les
   variables internes mises à jour `VAR1` — le champ à **réinjecter comme `prev`
   au pas suivant**.

Les sous-modèles de **contrainte** (Dirichlet…) sont ignorés : un sous-modèle
participe ssi il déclare un espace EF de comportement. Les zones de déformation,
de matériau et d'état précédent sont appariées par sous-espace EF.

> **Pourquoi le montage incrémental ?** Fournir l'état de A séparément de la
> déformation de B (plutôt que fusionnés dans un même champ) rend le fil d'état
> robuste — un `ElementField` = une zone par support, sans ambiguïté — et ouvre
> les **grandes déformations** et les **lois visqueuses** : elles exigent l'accès
> à `σ(A)` et à un incrément daté, que cette interface porte déjà. En petites
> déformations le prédicteur incrémental `σ_trial = σ(A) + C:Δε` est
> rigoureusement identique à la forme en déformation totale.

Pour une loi **linéaire**, le résultat est cohérent avec `stiffness`
(`∫ Bᵀ·flux = K·u`) ; une loi **non linéaire** s'écarte de cette tangente — c'est
tout l'intérêt d'intégrer le comportement exactement.

### Boucle multi-pas (fil d'état)

```python
{{#include ../../../tests/python/test_doc_ops_physiques.py:pas_a_pas}}
```

## Exemple : efforts de section d'une poutre

```python
{{#include ../../../tests/python/test_doc_ops_physiques.py:beam_deformation}}
```

## Exemple : résultantes d'une coque

Le même montage, avec la formulation en argument : ce sont ses lignes de
flexion qui distinguent `thick` de `kirchhoff`.

```python
{{#include ../../../tests/python/test_doc_ops_physiques.py:shell_deformation}}
```

Les pages [Barre](../mecanique/truss.md), [Élasticité](../mecanique/elasticite.md)
et [Timoshenko](../mecanique/timoshenko.md) détaillent l'intégrande de
comportement (`COMP`) de chaque physique. Pour les lois **non linéaires** avec
variables internes (`VAR0` → `VAR1`), voir
[Plasticité parfaite](../mecanique/plasticite.md) (retour radial von Mises) et
[Endommagement de Mazars](../mecanique/mazars.md).

## `internal_forces(model, state)` → `NodeField`

Les **forces internes** `f = ∫ Bᵀ σ dΩ` (le `BSIG` de cast3m) sont la
**transposée** de l'opérateur de déformation `B` : là où
[`deformation`](champs.md) applique `B` au déplacement (`ε = B·u`),
`internal_forces` applique `Bᵀ` à la contrainte et **rassemble** le résultat aux
nœuds. C'est la **généralisation mécanique** de [`divergence`](champs.md) (qui
est exactement `Bᵀ q` pour un transport scalaire) : une composante de sortie par
DDL dual.

`state` est le champ d'état matériau renvoyé par `integrate_behavior`. Chaque
sous-modèle porteur d'un comportement applique **son propre** `Bᵀ` — c'est
pourquoi l'opérateur prend un **modèle** et pas un simple espace EF : un même
`SEG2` peut être une barre (`Bᵀ` axial, DDL déplacement) ou une poutre (`Bᵀ` à
deux quadratures flexion + cisaillement, DDL `w, θ`), et seul le modèle tranche.
Solides continus, barres et poutres sont donc tous couverts.

Pour une loi **linéaire**, le résultat égale la rigidité appliquée à la solution
(`K·u`) ; pour une loi **non linéaire**, il donne les forces internes exactes.

C'est le côté **intérieur** du bilan `Σ f_int = Σ f_ext`, dont l'écart est le
résidu d'équilibre. Le miroir nodal de [`stiffness`](assemblage.md) : là où
l'assemblage demande à chaque sous-modèle ses blocs de `∂r/∂u`, celui-ci lui
demande son terme de `r`. Un sous-modèle qui n'a pas de terme de ce côté n'en
déclare aucun et n'apparaît pas dans le résultat.

```python
{{#include ../../../tests/python/test_doc_ops_physiques.py:forces_internes}}
```

## `external_forces(model)` → `NodeField`

L'autre côté du bilan : la **donnée** de chaque terme, à droite du signe égal.
Une physique dont le terme n'est qu'une réponse à `u` — élasticité, conduction,
barre — n'en a aucun, si bien qu'un modèle qui n'en contient que de celles-là
rend un champ **vide**, ce qui est la réponse juste et non un échec.

Séparer les deux côtés est ce qui garde les signes hors des fichiers de physique
: l'auteur écrit ses deux moitiés positivement, comme la forme faible se lit, et
l'unique soustraction vit chez l'appelant. De quel côté un terme se range est
une question de physique — le côté du signe égal où il se trouve — et non de
comptabilité.

### `internal_forces_continuum(stresses, fespace)` → `NodeField`

Variante **sans modèle** pour le cas **continu** (élasticité, Mazars,
plasticité), où `B` est le gradient symétrique universel et les DDL sont toujours
un déplacement : elle ne demande que la géométrie (`fespace`) et la contrainte en
notation de Voigt (`sigma_xx`, `sigma_xy`…), et renvoie `space_dim` composantes
`f_x, f_y, f_z` par nœud. **Barres et poutres ne sont pas couvertes** — leur `B`
n'est pas le gradient symétrique — : utiliser `internal_forces(model, state)`
pour celles-ci.
