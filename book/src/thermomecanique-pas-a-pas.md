# Thermo-mécanique pas-à-pas (couche Python haut niveau)

Au-dessus des opérateurs de bas niveau (assemblage, comportement, solveur…),
pyrucast livre une **couche Python pure** de plus haut niveau. Elle n'ajoute
aucun code Rust : elle orchestre les opérateurs déjà exposés. La première brique
est une résolution **thermo-mécanique pas-à-pas**.

## Empaquetage : du Python pur dans le même package

pyrucast est un package *mixed Rust/Python* (maturin) : l'extension compilée est
le sous-module privé `_pyrucast` (tous les `#[pyfunction]`/`#[pyclass]`), et le
package public `pyrucast` la ré-exporte tout en ajoutant des modules Python purs.

```text
python/pyrucast/
├── __init__.py          # from ._pyrucast import *  + ré-export du haut niveau
├── py.typed             # PEP 561
├── thermomechanics.py   # step_by_step / thermal_step / mechanical_step
└── _pyrucast/…          # stub .pyi de l'extension (généré)
```

`import pyrucast` donne donc accès **à la fois** à toute l'API Rust et aux
fonctions Python de plus haut niveau, sans distinction à l'usage :

```python
import pyrucast as pc

pc.matrix.stiffness(...)  # opérateur Rust (extension)
pc.thermomechanics.step_by_step(...)  # fonction Python pure (thermomechanics.py)
```

Côté configuration, cela tient à trois lignes de `pyproject.toml`
(`[tool.maturin] python-source = "python"`, `module-name = "pyrucast._pyrucast"`)
et au `__init__.py` qui fait `from ._pyrucast import *`.

## Modèle de calcul

- **Thermique stationnaire par pas.** La librairie n'a pas (encore) de terme
  transitoire de capacité ; chaque pas résout un problème thermique
  **stationnaire** `K_th · T = charges`. La dépendance au temps vient des charges
  et matériaux, interpolés à l'instant courant (voir [Évolution](evolution.md)).
- **Couplage faible, sens unique** thermo → méca. La température du pas fournit la
  déformation thermique `ε_th` (opérateur [`thermal_strain`](operateurs/champs.md),
  Cast3M `EPTH`), retirée de la déformation totale avant l'intégration de la loi.
  Il n'y a pas de rétroaction méca → thermique.
- **Mécanique non linéaire.** Newton **modifié** : l'opérateur d'itération est la
  rigidité **élastique** (assemblée une fois par pas, factorisation mise en cache
  par `solve`), **accéléré par l'accélération d'Anderson** (historique `m = 3`,
  garde-fou de descente). L'état interne (plasticité, endommagement…) est propagé
  d'un pas au suivant via l'interface incrémentale
  [`integrate_behavior(..., prev=…, dt=…)`](operateurs/comportement.md).

## Les trois fonctions

| Fonction | Rôle |
|---|---|
| `step_by_step(data) -> dict` | Mise en donnée + boucle sur les instants. Découpe le modèle par physique, appelle `thermal_step` puis `mechanical_step` à chaque pas, complète `data["results"]`. |
| `thermal_step(thermal_model, materials, loads) -> NodeField` | Une résolution thermique stationnaire. |
| `mechanical_step(mechanical_model, fespace, mesh, materials, loads, temperature, u, state_prev, dt, …) -> (u, out, info)` | Une résolution mécanique non linéaire (Newton modifié + Anderson) du pas. |

La découpe par physique s'appuie sur [`Model.filter`](model.md)
(`"thermal"` / `"mechanical"` / `"constraint"`) ; les contraintes de Dirichlet
sont ré-attachées à la physique dont elles contraignent une variable (`T` →
thermique, `u_*` → mécanique).

## Dictionnaire d'entrée / sortie

`step_by_step` prend **un seul dictionnaire** et le renvoie complété :

| Clé | Type | Rôle |
|---|---|---|
| `times` | `list[float]` | instants de calcul |
| `model` | `Model` | modèle complet (thermique + mécanique + Dirichlet). L'espace EF et le maillage en sont **déduits** — voir ci-dessous |
| `loads` | `NodeField` \| `Evolution` | **un seul champ unioné** : `q`/`imposed_T` (thermique) + `f_*`/`imposed_u` (mécanique) |
| `materials` | `ElementField` \| `Evolution` | **un seul champ unioné** : `k`/`h` + `E`/`nu`/`alpha` |
| `t_ref` | `float` (opt.) | température de référence pour `ε_th` |
| `free_mesh` | `Mesh` (opt.) | DDL libres pour la norme de résidu (recommandé avec Dirichlet) |
| `anderson_depth` / `max_newton` / `tol_rel` | (opt.) | réglages du solveur mécanique |

Seul le `model` porte la donnée EF : `step_by_step` en déduit l'espace et le
maillage mécaniques par [`Model.fespace()`](model.md) (les sous-espaces des
sous-modèles de domaine, contraintes exclues) puis `FiniteElementSpace.mesh()`.

Chaque étape ne lit du champ unioné que ce dont elle a besoin : `solve`
n'échantillonne le second membre qu'aux DDL de sa matrice et ignore les
composantes surnuméraires ; `material_field` remplit par nom les composantes de
chaque physique. Comme thermique et mécanique partagent la fespace, le champ
matériau porte **deux zones** (aux composantes disjointes) sur le même support ;
nul besoin de les fusionner : les opérateurs (`stiffness`, `integrate_behavior`,
`thermal_strain`) résolvent leur zone matière **par les composantes qu'ils
requièrent** (`k` pour la conduction, `E`/`nu` pour l'élasticité, `alpha` pour la
dilatation). Pour fusionner explicitement des zones qui partagent légitimement un
support, [`element_field.consolidate`](operateurs/champs.md) reste disponible.

En sortie, `data["results"]` est une liste (un élément par instant) :

```python
{
    "time",
    "temperature",
    "displacement",
    "state",
    "mech_iters",
    "mech_anderson",
    "converged",
}
```

## Exemple

```python
import pyrucast as pc

# … maillage `mesh`, `fes`, modèle thermo-mécanique `model`, `materials`, `loads` …

data = {
    "times": [0.0, 0.25, 0.5, 0.75, 1.0],
    "model": model,  # fespace + maillage déduits du modèle
    "loads": loads,  # NodeField unioné ou Evolution de champ
    "materials": materials,  # ElementField unioné ou Evolution de champ
    "t_ref": 20.0,
}

pc.thermomechanics.step_by_step(data)

for r in data["results"]:
    print(r["time"], r["mech_iters"], r["converged"])
```

Démonstration complète (plaque chauffée, dilatation libre, contrôle analytique
`u = α·ΔT·x`) : `examples/thermomecanique_pas_a_pas.py`. Passer de
`Model.elasticity` à `Model.plasticity` suffit pour une mécanique
élasto-plastique — le même appel gère la boucle non linéaire.
