# Multi-points (MPC)

## En bref

Impose une relation linéaire entre plusieurs inconnues : deux nœuds à la même
température, une périodicité, une liaison affine. Avec `sense`, une borne :
une liaison à jeu.

| | |
|---|---|
| opérateur | `pyrucast.model.mpc(terms, target, sense="=")` |
| tag | `constraint` et celui de la cible ; `constraint` seul si la relation lie deux physiques |
| type | contrainte (`label()` : `"Mpc"`) |
| support | un maillage POI1 par terme |

`terms` est une liste de termes `(maillage, variable, coefficient)`. Les
maillages des termes sont appariés **maille par maille** : la relation \\(r\\)
relie la maille \\(r\\) de chacun. Une périodicité entre deux surfaces de
\\(N\\) nœuds est donc \\(N\\) relations d'un coup. L'ordre des mailles est à
la charge de l'appelant ; la contrainte vérifie que tout est POI1, sur un
même `Coords`, avec le même nombre de mailles par zone. `target` donne à
chaque variable sa duale (`Model.dual_of`) ; une relation entre deux physiques
se pose sur leur union.

## Théorie

Chaque relation s'écrit

\\[
\sum_k a_k\\,u(n_k, v_k) = g \qquad (\text{ou } \ge,\ \le\ g).
\\]

C'est [Dirichlet](dirichlet.md) à plusieurs termes et coefficients quelconques.
Chaque terme apporte une paire de blocs de coefficient \\(a_k\\) :
\\(C\\), ligne `(multiplicateur, imposed)`, colonne `(n_k, v_k)` ; et
\\(C^{\mathsf T}\\), ligne `(n_k, duale de v_k)`, colonne
`(multiplicateur, lambda)`. Tous les termes d'une relation partagent son nœud
multiplicateur et sa ligne `imposed`, ce qui les additionne en une équation.
La réaction, la force que la relation exerce sur la pièce, est
\\(-a_k\\,\lambda\\) au nœud de chaque terme.

Le nœud multiplicateur de chaque relation est créé à la position du nœud de
son premier terme. Le second membre \\(g\\) s'écrit au chargement, à la
composante `imposed` du nœud multiplicateur, par
`constraint_rhs([(un_noeud_de_la_relation, g)])`, ou par
`constraint_rhs_by_index([(r, g)])` quand un nœud appartient à plusieurs
relations. Une relation absente du chargement impose 0.

Les coefficients sont un par terme, les mêmes pour toutes ses mailles. Des
coefficients qui varient d'un nœud à l'autre sont le cas du
[baignage](embedded.md).

## Variables

| | |
|---|---|
| primale | `lambda` |
| duale | `imposed` |
| ce qu'elle contraint | les variables des termes, inconnues de la cible |
| matériau, comportement | aucun |

## Ce qu'elle produit

| | |
|---|---|
| raideur | une paire \\(C\\), \\(C^{\mathsf T}\\) par terme et par zone |
| masse, géométrique | — |
| tangente | = raideur |
| forces internes | \\(C^{\mathsf T}\lambda\\) sur les duales, \\(C\\,u\\) sur les lignes `imposed` |
| forces externes | aucune |

## Exemple

Sur une barre en conduction, \\(T(0) = 0\\) par Dirichlet et
\\(T(1) - T(0) = 1\\) par une MPC : d'où \\(T = x\\).

```python
{{#include ../../../tests/python/test_doc_contraintes.py:mpc_complet}}
```

`examples/mpc_periodicite.py` pose une périodicité.

## Côté Rust

`Mpc` (`src/models/mpc.rs`) implémente `Constraint`. Ses blocs sortent du même
`constraint_block_pair` que Dirichlet, avec le coefficient \\(a_k\\) au lieu de
1.
