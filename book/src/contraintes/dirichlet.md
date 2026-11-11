# Dirichlet

## En bref

Impose la valeur d'une inconnue sur des nœuds : `T = 20` sur un bord,
`u_x = 0` sur un appui. Avec `sense`, impose une borne : `u_x ≥ 0` est une
butée.

| | |
|---|---|
| opérateur | `pyrucast.model.dirichlet(imposed_mesh, target, variable, sense="=")` |
| tag | `constraint` et celui de la cible |
| type | contrainte (`label()` : `"Dirichlet"`) |
| support | un maillage POI1 : les nœuds imposés, une relation par maille |

`target` est le modèle contraint, `variable` l'une de ses inconnues ; une
variable que la cible ne déclare pas est refusée à la construction. Une seule
variable par contrainte : pour encastrer, on compose
`dirichlet(noeuds, m, "u_x") | dirichlet(noeuds, m, "u_y")`.

## Théorie

Chaque nœud \\(n\\) du support porte une relation à un seul terme,

\\[
u(n) = u_d \qquad (\text{ou } \ge,\ \le\ u_d),
\\]

imposée par un multiplicateur \\(\lambda\\) (voir [Contraintes](../contraintes.md)).
Elle apporte deux blocs à la raideur, chacun de coefficient 1 :

- le bloc \\(C\\) : ligne `(multiplicateur, imposed)`, colonne `(n, variable)` ;
- le bloc \\(C^{\mathsf T}\\) : ligne `(n, duale de la variable)`, colonne
  `(multiplicateur, lambda)`.

À la solution, \\(\lambda\\) est l'effort de la pièce sur le blocage : le flux
ou la force qui tient la valeur imposée, vu de la pièce. La **réaction**, ce
que le blocage exerce sur la pièce, est \\(-\lambda\\) : une poutre encastrée
qui porte un poids \\(P\\) vers le bas a des multiplicateurs `u_z` de somme
\\(-P\\), et une réaction \\(P\\) vers le haut (voir
[Contraintes](../contraintes.md)).

Chaque relation a son **nœud multiplicateur**, créé par la contrainte à la
position du nœud imposé, dans le même `Coords` ; `multiplier_mesh()` les
donne. La valeur imposée \\(u_d\\) n'est pas stockée dans la contrainte : elle
s'écrit dans le chargement, à la composante `imposed` du nœud multiplicateur,
ce que fait `constraint_rhs([(noeud_impose, u_d)])`. Une relation absente du
chargement impose 0.

## Variables

| | |
|---|---|
| primale | `lambda`, le multiplicateur |
| duale | `imposed`, la ligne de la relation |
| ce qu'elle contraint | `variable`, une inconnue de la cible, dont la duale reçoit la réaction |
| matériau, comportement | aucun |

## Ce qu'elle produit

| | |
|---|---|
| raideur | les blocs \\(C\\) et \\(C^{\mathsf T}\\), une paire par zone du support |
| masse, géométrique | — |
| tangente | = raideur |
| forces internes | \\(C^{\mathsf T}\lambda\\) sur la duale de la cible, \\(C\\,u\\) sur les lignes `imposed` |
| forces externes | aucune ; la valeur imposée est un chargement |

## Exemple

Une barre \\([0, 1]\\) en conduction, \\(T(0) = 0\\) et \\(T(1) = 1\\) : le
profil est \\(T = x\\), et les multiplicateurs sont les flux \\(\pm1\\) aux
deux bouts.

```python
{{#include ../../../tests/python/test_doc_contraintes.py:dirichlet_complet}}
```

## Côté Rust

`Dirichlet` (`src/models/dirichlet.rs`) implémente `Constraint` : ses
relations (`relations()`) et ses nœuds multiplicateurs. Ses blocs sortent de
`constraint_block_pair`, partagé avec les autres contraintes, qui donne aux
deux blocs une même identité de paire (`Symmetry::Half`) : la matrice n'est
symétrique que si elle les tient tous les deux. Ils entrent dans la raideur
par `contributions`, comme blocs écrits.
