# Baignage (embedded)

## En bref

Lie chaque nœud d'un maillage **immergé** à l'interpolation d'un maillage
**hôte** au même point : une barre d'armature baignée dans un béton, qui suit
ses déplacements sans partager ses nœuds.

| | |
|---|---|
| opérateur | `pyrucast.model.embedded(immersed, host, target, variables, tol=1e-6)` |
| tag | `constraint` et celui de la cible |
| type | contrainte (`label()` : `"Embedded"`) |
| support | les nœuds du maillage immergé, et les mailles de l'hôte |

`variables` sont les inconnues liées (`["u_x", "u_y", "u_z"]`), toutes
inconnues de la cible. Les deux maillages partagent un même `Coords`. L'hôte
peut être de tout type de maille à domaine de référence (segments, triangles,
quadrangles, tétraèdres, prismes, hexaèdres, linéaires et quadratiques) ; le
maillage immergé n'est lu que par ses nœuds.

## Théorie

Pour un nœud immergé \\(p\\) logé dans la maille hôte de fonctions de forme
\\(N_i\\), et pour chaque variable \\(c\\) :

\\[
u_c(p) - \sum_i N_i(\xi_p)\\,u_c(\text{hôte}_i) = g_c ,
\\]

\\(g_c = 0\\) étant la liaison rigide. C'est une [MPC](mpc.md) dont les
coefficients, \\(+1\\) sur le nœud immergé et \\(-N_i(\xi_p)\\) sur chaque nœud
hôte, varient d'un nœud à l'autre.

Les poids sont calculés **une fois, à la construction**, en localisant chaque
nœud immergé dans l'hôte : un Newton sur \\(x - \sum_i N_i(\xi)\\,X_i\\), avec
un test d'appartenance au domaine de référence de la maille, à `tol` près. Les
mailles hôtes sont parcourues une à une, filtrées par leur boîte englobante.
Un nœud immergé qui ne tombe dans aucune maille hôte est refusé.

Chaque relation (un nœud immergé et une variable) a son nœud multiplicateur,
placé au nœud immergé ; `multiplier_mesh()` porte une zone par variable. Le
second membre \\(g_c\\) s'écrit au chargement par `constraint_rhs` ; absent, il
vaut 0.

## Variables

| | |
|---|---|
| primale | `lambda`, la force de liaison |
| duale | `imposed` |
| ce qu'elle contraint | les `variables`, au nœud immergé et aux nœuds hôtes |
| matériau, comportement | aucun |

## Ce qu'elle produit

| | |
|---|---|
| raideur | une paire \\(C\\), \\(C^{\mathsf T}\\) par variable |
| masse, géométrique | — |
| tangente | = raideur |
| forces internes | \\(C^{\mathsf T}\lambda\\) sur les duales, \\(C\\,u\\) sur les lignes `imposed` |
| forces externes | aucune |

## Exemple

Un cube `HEX8` en conduction, ses coins tenus au champ linéaire
\\(T = 1 + 2x + 3y + 4z\\), que l'interpolation trilinéaire reproduit
exactement ; un nœud immergé au cœur prend la valeur de l'hôte en ce point.

```python
{{#include ../../../tests/python/test_doc_contraintes.py:embedded_complet}}
```

`examples/barre_baignee_elastique.py` baigne une barre dans un volume
élastique, sur les trois déplacements.

## Côté Rust

`Embedded` (`src/models/embedded.rs`) implémente `Constraint`. La localisation
passe par `ops::geom::locate_points`, l'inversion de la transformation de
référence.
