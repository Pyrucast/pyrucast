# Multi-points (MPC)

Une **contrainte multi-points** (MPC) impose une **relation linéaire** entre
degrés de liberté :

\\[
\sum_k a_k \\, u(n_k, v_k) = g .
\\]

C'est une [contrainte](../contraintes.md) imposée par **multiplicateurs de
Lagrange**, exactement comme [Dirichlet](dirichlet.md) — dont elle est la
**généralisation** : Dirichlet est la relation à un seul terme `1·u = u_d`
(coefficient 1), une MPC en a autant qu'on veut, avec des coefficients
quelconques. Aucun matériau, aucune loi de comportement ; les seuls nœuds
qu'elle crée sont ses multiplicateurs, un par relation.

Implémentation : `src/models/mpc.rs` ; opérateur `ops::model::mpc(…)`.

## Mise en donnée : un maillage par terme

Chaque **terme** est un tuple `(maillage POI1, variable, coefficient)`. Tous
les terme-maillages sont **appariés cellule par cellule** : la relation `r`
relie la `r`-ème cellule de *chaque* terme-maillage. Une périodicité entre deux
surfaces de `N` nœuds est donc `N` relations d'un coup, **vectorisées** sur les
cellules — sans boucle Python.

Chaque relation reçoit un **nœud-multiplicateur à elle**, créé à la
construction par [`barycenter`](../operateurs/maillage.md) sur le premier
terme-maillage : un nœud neuf colocalisé avec le nœud du premier terme. On les
retrouve avec `Model.multiplier_mesh()`.

> **Contrat d'appariement.** L'ordre cohérent des maillages appariés est à la
> charge de l'utilisateur : la cellule `r` de chaque terme-maillage doit
> désigner des nœuds **partenaires** (p. ex. le nœud d'entrée et son image
> périodique). Le modèle vérifie seulement que tout est POI1, partage un même
> `Coords`, et a le même nombre de cellules par sous-maillage.

La **cible** est **une** : le modèle contraint, qui donne à chaque variable sa
duale (la ligne où atterrit la réaction `aₖ·λ`) — appariement positionnel
`primal_vars[i] ↔ dual_vars[i]` de la physique qui déclare la variable
(`"T" → "q"`, `"u_x" → "f_x"`, …), lisible avec **`Model.dual_of(variable)`**.
Une relation entre deux physiques se pose sur leur union.

## Les noms des variables

Ceux de **toute** relation de contrainte (voir [Dirichlet](dirichlet.md)) :

| rôle | nom |
|---|---|
| primale propre = multiplicateur `λ` (inconnue du système) | `lambda` |
| duale propre = ligne de contrainte + **slot** où l'utilisateur écrit `g` | `imposed` |

Signature complète :

```text
model.mpc(target, terms, sense="=")
# terms : [(maillage, variable, coefficient), …]
```

`sense` (`"="`, `">="`, `"<="`) rend les relations **unilatérales**
(`Σₖ aₖ·uₖ ≥ g` : une liaison à jeu) — voir la section « Relations
unilatérales » de la page [Contraintes](../contraintes.md) et le solveur
`solve_unilateral`.

## Les blocs `C` / `Cᵀ`

À l'assemblage, l'MPC contribue **une paire de blocs par (sous-maillage, terme)**,
via le même code partagé que Dirichlet mais avec le coefficient `aₖ` au
lieu de `1` (chaque bloc est marqué non-symétrique ; seule l'union `C ∪ Cᵀ`
l'est — propriété **globale** du système point-selle) :

- **bloc C** : `(multiplier_node, imposed) × (nœud_k, variable_k) = aₖ`
- **bloc Cᵀ** : `(nœud_k, dual_k) × (multiplier_node, lambda) = aₖ`

Tous les termes d'une relation partagent le **même nœud multiplicateur** et la
**même ligne `imposed`** : c'est ce qui les additionne dans une seule
équation `Σₖ aₖ uₖ = g`.

## Second membre et réaction

- Le second membre `g` n'est **pas** stocké dans le SubModel : l'utilisateur
  l'écrit dans le `NodeField` de **chargement**, au slot `imposed` du
  nœud-multiplicateur (défaut `g = 0` — le cas homogène des égalités et
  périodicités). `constraint_rhs([(nœud d'un terme, g), …])` l'y range.
- Le **multiplicateur** se retrouve dans la solution sous `lambda` au
  nœud-multiplicateur ; sa valeur **est** la force de réaction de la contrainte.

## Exemple : relation `T(1) − T(0) = 1`

Sur la barre 1-D `-u'' = 0`, un Dirichlet `T(0) = 0` et une MPC à deux termes
`1·T(1) − 1·T(0) = 1` imposent `T(1) = 1`, d'où la solution linéaire `u(x) = x` :

```python
{{#include ../../../tests/python/test_doc_contraintes.py:mpc_complet}}
```

L'exemple complet est dans `examples/mpc_periodicite.py`.

## Limitations actuelles

- Les terme-maillages sont **POI1** (un nœud par relation) ; l'appariement est
  positionnel (aucun outil géométrique fourni).
- Les coefficients sont des **scalaires** par terme (constants sur les cellules
  du terme-maillage). Les coefficients variant par nœud (poids d'interpolation,
  bras de levier) sont couverts par une contrainte dédiée,
  [Baignage (embedded)](embedded.md).
- Une **seule** relation reliant un grand nombre de termes (`Σ` sur 100 nœuds)
  demanderait autant de terme-maillages ; ce cas passera par une extension
  « cellule multi-nœuds ». Beaucoup de relations **parallèles** (le cas courant)
  sont déjà vectorisées.
