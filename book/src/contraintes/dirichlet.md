# Dirichlet

La condition de **Dirichlet** impose la valeur d'une variable primale,
`u(n) = u_d`, sur un ensemble de nœuds. C'est une [contrainte](../contraintes.md)
imposée par **multiplicateurs de Lagrange** : aucun matériau, aucune loi de
comportement. Les seuls nœuds qu'elle crée sont ses multiplicateurs, un par
relation.

Implémentation : `src/models/dirichlet.rs` ; opérateur `ops::model::dirichlet(…)`.

## Un maillage fourni, les multiplicateurs créés

L'utilisateur fournit **un maillage POI1**, **`imposed_mesh`** : les nœuds
contraints (partagés avec la physique cible), une relation par cellule.

Chaque relation reçoit un **nœud-multiplicateur à elle**, créé à la
construction par le mailleur [`barycenter`](../operateurs/maillage.md) : un
nœud neuf colocalisé avec le nœud contraint, dans la même structure de
sous-maillage. On les retrouve avec `Model.multiplier_mesh()` ; le plus souvent
on n'y touche pas, `constraint_rhs` écrivant `u_d` à partir du nœud contraint.

## Les noms des variables

| rôle | nom | fourniture |
|---|---|---|
| le modèle **contraint** | `target` | requis |
| variable imposée (une primale de `target`) | `variable` (ex `"T"`) | requis |
| duale de la cible (ligne où atterrit la réaction `Cᵀ`) | lue dans `target` | déduit |
| primale propre = multiplicateur (inconnue du système) | `lambda` | commun à toute relation |
| duale propre = ligne de contrainte + **slot** où l'utilisateur écrit `u_d` | `imposed` | commun à toute relation |

Les deux derniers sont les noms de **toute** relation de contrainte, quelle que
soit la contrainte qui l'a posée : une fois dans la matrice, une relation est
une relation. Deux relations ne se rencontrent jamais sur un même DDL, puisque
chacune a son nœud.

Signature complète :

```text
model.dirichlet(imposed_mesh, target, variable, sense="=")
```

Quatre arguments, et chacun porte une décision. Le **support** vient en tête,
comme pour tous les opérateurs de `pyrucast.model` : les nœuds contraints. La
**cible** suit : c'est le modèle qu'on
contraint, `variable` doit être une de ses primales, et sa ligne duale s'y lit
— le passage `T → q` est l'affaire de la physique visée, pas quelque chose à
retaper. Le donner permet aussi de refuser à la construction une variable que
la cible ne déclare pas, au lieu de découvrir une matrice singulière.

**Une variable, et une seule.** Pour encastrer, on compose :
`dirichlet(nœuds, m, "u_x") | dirichlet(nœuds, m, "u_y")`, chacun avec ses propres
nœuds-multiplicateurs. Une contrainte est une famille de relations scalaires
avec un multiplicateur par relation ; on n'empaquette plusieurs composantes que
lorsqu'elles partagent quelque chose de coûteux, ce qu'un Dirichlet ne fait
pas. La valeur imposée, elle, se donne variable par variable.

`sense` (`"="`, `">="`, `"<="`) rend la contrainte **unilatérale** (`u ≥ u_d` :
une butée) — voir la section « Relations unilatérales » de la page
[Contraintes](../contraintes.md) et le [solveur](../operateurs/solveur.md), qui lit le sens dans la matrice.

## Les deux blocs unité

À l'assemblage, Dirichlet contribue **une paire de blocs unité par
sous-maillage**, chacun marqué **non-symétrique** (seule l'union `C ∪ Cᵀ` l'est
— propriété **globale** du système point-selle ; cf. le drapeau `symmetric` de
la [Matrice](../matrix.md)) :

- **bloc C** : `(multiplier_node, imposed) × (imposed_node, imposed_variable) = 1`
- **bloc Cᵀ** : `(imposed_node, target_dual) × (multiplier_node, lambda) = 1`

Le bloc `C` exprime la relation `u(n) = u_d` ; le bloc `Cᵀ` réinjecte la
réaction dans l'équation de la physique cible (ligne `target_dual`).

## Valeur imposée et réaction

- La **valeur imposée** `u_d` n'est **pas** stockée dans le SubModel :
  l'utilisateur la fournit dans le `NodeField` de **chargement**, à la position
  `(multiplier_node, imposed)`. Le plus simple est de la donner au nœud
  contraint et de laisser `constraint_rhs` la porter sur le nœud-multiplicateur
  de la relation ; on peut aussi l'écrire directement sur
  `multiplier_mesh()`.
- Le **multiplicateur** se retrouve dans la solution sous le nom `lambda` au
  nœud-multiplicateur ; sa valeur **est** la force de réaction de la
  contrainte.

Les nœuds-multiplicateurs appartiennent au SubModel qui les a créés : ils
vivent tant que lui **ou** un champ défini sur `multiplier_mesh()` les
référence, puis deviennent collectables.

## Exemple : Poisson 1-D `-u'' = 0`, `u(0)=0`, `u(1)=1`

Solution analytique `u(x) = x`, multiplicateurs aux bords = flux `±1`. On
compose la conduction thermique avec deux contraintes Dirichlet par l'union `|`
(cf. [Modèle physique](../model.md)) :

```python
{{#include ../../../tests/python/test_doc_contraintes.py:dirichlet_complet}}
```

La forme Rust équivalente (opérateurs au niveau parent, composés par
`union`) est dans le chapitre [Modèle physique](../model.md).

## Limitations actuelles

- `imposed_mesh` est un maillage **POI1** (contrainte par nœud). Les contraintes réparties (sur une arête entière) passeront par un
  bloc `C` issu d'une intégration, comme [`flux`](../operateurs/assemblage.md) pour
  les seconds membres.
- Seule la valeur imposée **constante par nœud** est gérée ; une valeur
  spatialement variable se fournit nœud par nœud dans le chargement.
