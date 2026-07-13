# Opérateurs de construction

Le module `ops::build` **construit des conteneurs** à partir de données ou
d'autres conteneurs. Aujourd'hui il héberge la construction des **champs
matériau** prêts pour l'assemblage ; à terme, c'est le foyer prévu pour la
migration des générateurs de maillage (actuellement sous
[`ops::mesher`](maillage.md)) et des futurs helpers de remaillage / raffinement.

## Champs matériau

Un assemblage a besoin d'un [`ElementField`](../element-field.md) portant les
propriétés matériau (conductivité `k`, module `E`, Poisson `nu`…) **aux points
de Gauss**. Plutôt que de construire ce champ à la main, ces opérateurs le
fabriquent en appariant les zones aux sous-modèles qui consomment du matériau,
et en **ignorant** ceux qui n'en ont pas (les [contraintes](../contraintes.md)
comme Dirichlet).

| Python | Effet |
|---|---|
| `material_field(model, [(nom, valeur), …])` | un `ElementField` **uniforme** : une zone par sous-modèle consommateur de matériau, chaque composante demandée mise à la valeur donnée |
| `material_field_per_sub_model(model, [[(nom, valeur), …], …])` | idem mais avec une liste de paires **par** sous-modèle (matériaux différents par zone) |
| `sub_material_field(sub_model, [(nom, valeur), …])` | une **seule** zone (`SubElementField`) pour un sous-modèle donné |

```python
import pyrucast

# Thermique : conductivité uniforme.
materials = pyrucast.build.material_field(model, [("k", 1.0)])

# Élasticité : deux propriétés.
materials = pyrucast.build.material_field(model, [("E", 210e9), ("nu", 0.3)])
```

Le champ produit est ensuite passé tel quel à
[`stiffness(model, materials)`](assemblage.md) et à
[`integrate_behavior`](comportement.md). Comme l'assembleur sélectionne, pour
chaque sous-modèle, la zone dont le `SubFiniteElementSpace` correspond au sien,
un champ couvrant seulement certains sous-espaces reste valide tant que chaque
sous-modèle gourmand en matériau trouve sa zone.

> Une fois construit, le champ matériau est un `ElementField` ordinaire : on
> peut le faire varier dans l'espace (écriture par zone, par cellule) ou le
> mettre à l'échelle (arithmétique de champ) — voir
> [Champ aux points de Gauss](../element-field.md).
