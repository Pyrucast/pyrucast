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

1. l'**entrée de déformation** `ε(B)` est produite par
   [`strain(model, u, materials)`](champs.md#le-b-du-modèle--strainmodel-solution-materials--elementfield),
   qui applique à chaque loi le producteur qu'elle déclare —
   [`deformation`](champs.md) (`ε`), [`gradient`](champs.md) (`∇T`…),
   [`beam_deformation`](champs.md) (`κ, γ`), [`shell_deformation`](champs.md)
   (`ε, κ, γ`). Ces producteurs géométriques ne dépendent que de l'espace EF et
   restent disponibles pour construire une entrée à la main ;
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
> les **lois visqueuses**, qui exigent un incrément daté. C'est le montage de
> MFront (`s0` / `s1` de MGIS).

> **La convention de toutes les lois.** Chaque loi calcule la contrainte en B
> comme **le matériau de B appliqué à la déformation élastique de B** :
> `σ(B) = C(B) : ε_e(B)`. L'élasticité et l'endommagement le font directement,
> la plasticité par `ε_e = ε − ε_p` (prédicteur `σ_trial = C:(ε(B) − ε_p(A))`).
> Une contrainte écrite ainsi reste **sécante** quand le matériau dépend de
> l'état : un module qui baisse avec la température baisse la contrainte à
> déformation élastique égale, sans aucune correction. Une forme incrémentale
> `σ(A) + C(B):Δε` garderait la contrainte de l'ancien module — Cast3M, dont
> toutes les lois sont incrémentales, ajoute pour cela la correction
> `(D⁻¹(B) − D⁻¹(A)) : σ(A)` ; MFront, comme pyrucast, n'en a pas besoin. Seule
> limite : une précontrainte donnée seule (sans déformation élastique qui la
> porte) n'est pas représentée par une loi plastique ; elle s'introduit comme une
> déformation plastique initiale `ε_p = −C⁻¹σ₀`.

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
[`strain`](champs.md#le-b-du-modèle--strainmodel-solution-materials--elementfield)
applique `B` au déplacement (`ε = B·u`),
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

### Sans modèle, ce sont des divergences

Il n'y a **pas** de variante sans modèle des forces internes, et c'est voulu :
privée de son modèle, l'opération ne connaît plus aucune mécanique. Il lui reste
la géométrie et des noms, et sous ce jour `∫ Bᵀ σ` est exactement la
**divergence du tenseur des contraintes** — une divergence faible par ligne de
`σ`. C'est donc [`divergence(field, "sigma")`](champs.md) qui la rend, avec des
composantes `div_sigma_x, div_sigma_y…` que l'appelant renomme en lignes duales
s'il veut en faire des forces.

Ce renommage n'est pas une formalité administrative : c'est l'endroit, et le
seul, où ces nombres deviennent de la mécanique.

**Barres et poutres ne sont pas couvertes** par cette voie — leur `B` n'est pas
le gradient symétrique et leurs DDL ne sont pas un déplacement : pour elles,
`internal_forces(model, …)`, qui répartit par physique.
