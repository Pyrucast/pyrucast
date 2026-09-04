# Dirichlet

La condition de **Dirichlet** impose la valeur d'une variable primale,
`u(n) = u_d`, sur un ensemble de nœuds. C'est une [contrainte](../contraintes.md)
imposée par **multiplicateurs de Lagrange** : aucun matériau, aucune loi de
comportement. Elle ne crée **aucun nœud** et ne mute jamais le `Coords`.

Implémentation : `src/models/dirichlet.rs` ; opérateur `ops::model::dirichlet(…)`.

## Deux maillages fournis par l'utilisateur

L'utilisateur fournit **deux maillages POI1** :

- **`imposed_mesh`** — les nœuds contraints (partagés avec la physique cible) ;
- **`multiplier_mesh`** — le support des **multiplicateurs**, apparié
  élément-par-élément avec `imposed_mesh` (même structure de sous-maillage,
  même nombre de cellules par paire).

On fabrique typiquement `multiplier_mesh` depuis `imposed_mesh` avec le mesher
générique [`barycenter`](../operateurs/maillage.md) (des nœuds neufs colocalisés
au centre de gravité de chaque cellule). Mais l'utilisateur reste libre :
nœuds colocalisés, décalés, ou même réutiliser les nœuds contraints eux-mêmes.

## Quatre noms de variables

Deux sont requis, deux sont déduits et **surchargeables** :

| rôle | nom | fourniture |
|---|---|---|
| le modèle **contraint** | `target` | requis |
| variable imposée (une primale de `target`) | `variable` (ex `"T"`) | requis |
| duale de la cible (ligne où atterrit la réaction `Cᵀ`) | lue dans `target` | déduit |
| primale propre = multiplicateur (inconnue du système) | `lambda_<variable>` | déduit |
| duale propre = ligne de contrainte + **slot** où l'utilisateur écrit `u_d` | `imposed_<variable>` | déduit |

Signature complète :

```text
model.dirichlet(target, variable, imposed_mesh, multiplier_mesh, sense="=")
```

Cinq arguments, et chacun porte une décision. La **cible** est le modèle qu'on
contraint : `variable` doit être une de ses primales, et sa ligne duale s'y lit
— le passage `T → q` est l'affaire de la physique visée, pas quelque chose à
retaper. Le donner permet aussi de refuser à la construction une variable que
la cible ne déclare pas, au lieu de découvrir une matrice singulière.

**Une variable, et une seule.** Pour encastrer, on compose :
`dirichlet(m, "u_x", …) | dirichlet(m, "u_y", …)`, les deux partageant leur
maillage multiplicateur — `lambda_u_x` et `lambda_u_y` sur un même nœud sont des
DDL distincts. Une contrainte est une famille de relations scalaires avec un
multiplicateur par relation ; on n'empaquette plusieurs composantes que
lorsqu'elles partagent quelque chose de coûteux, ce qu'un Dirichlet ne fait pas.

`sense` (`"="`, `">="`, `"<="`) rend la contrainte **unilatérale** (`u ≥ u_d` :
une butée) — voir la section « Relations unilatérales » de la page
[Contraintes](../contraintes.md) et le solveur `solve_unilateral`.

## Les deux blocs unité

À l'assemblage, Dirichlet contribue **une paire de blocs unité par
sous-maillage**, chacun marqué **non-symétrique** (seule l'union `C ∪ Cᵀ` l'est
— propriété **globale** du système point-selle ; cf. le drapeau `symmetric` de
la [Matrice](../matrix.md)) :

- **bloc C** : `(multiplier_node, imposed_value) × (imposed_node, imposed_variable) = 1`
- **bloc Cᵀ** : `(imposed_node, target_dual) × (multiplier_node, multiplier) = 1`

Le bloc `C` exprime la relation `u(n) = u_d` ; le bloc `Cᵀ` réinjecte la
réaction dans l'équation de la physique cible (ligne `target_dual`).

## Valeur imposée et réaction

- La **valeur imposée** `u_d` n'est **pas** stockée dans le SubModel :
  l'utilisateur la fournit dans le `NodeField` de **chargement**, à la position
  `(multiplier_node, imposed_value)` — c'est-à-dire au slot `imposed_<v>` du
  nœud-multiplicateur.
- Le **multiplicateur** se retrouve dans la solution sous le nom `multiplier`
  (`lambda_<v>`) au nœud-multiplicateur ; sa valeur **est** la force de
  réaction de la contrainte.

Les nœuds-multiplicateurs vivent tant que leur maillage **ou** le SubModel les
référence (refcounts) ; quand les deux disparaissent, ils deviennent
collectables. Le SubModel ne décrémente que ce qu'il partage — il n'a rien créé.

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

- `imposed_mesh` et `multiplier_mesh` sont des maillages **POI1** (contrainte
  par nœud). Les contraintes réparties (sur une arête entière) passeront par un
  bloc `C` issu d'une intégration, comme [`flux`](../operateurs/assemblage.md) pour
  les seconds membres.
- Seule la valeur imposée **constante par nœud** est gérée ; une valeur
  spatialement variable se fournit nœud par nœud dans le chargement.
