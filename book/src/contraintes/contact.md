# Contact (nœud-surface)

## En bref

Empêche les nœuds d'un maillage **esclave** de pénétrer une surface
**maître**, sans frottement : deux blocs posés l'un sur l'autre, une pièce
contre un appui.

| | |
|---|---|
| opérateur | `pyrucast.model.contact(slave, master, target, variables)` |
| tag | `constraint` et celui de la cible |
| type | contrainte (`label()` : `"Contact"`) |
| support | les nœuds esclaves, et les facettes maîtres : segments en 2-D, triangles et quadrangles en 3-D |

`variables` sont les déplacements de la cible, un par dimension de l'espace,
dans l'ordre (`["u_x", "u_y"]`). Les deux maillages partagent un même
`Coords`. La surface maître doit être **orientée** vers le corps esclave : en
2-D, la normale d'un segment est sa tangente tournée de −90°,
\\(n = (t_y, -t_x)\\) ; en 3-D, elle suit la règle de la main droite sur
l'ordre des nœuds.

## Théorie

Chaque nœud esclave \\(s\\) est apparié à son point le plus proche sur la
surface maître : une facette, des poids \\(N_i(\xi)\\), une normale \\(n\\) et
un jeu signé \\(g_0\\), positif quand c'est décollé. La non-pénétration
linéarisée est une relation **unilatérale** par nœud esclave :

\\[
g_0 + n\cdot u(s) - \sum_i N_i(\xi)\\,n\cdot u(\text{maître}_i) \ \ge\ 0 .
\\]

Une seule relation scalaire par nœud esclave : la normale couple les
composantes, avec les coefficients \\(+n_c\\) sur l'esclave et
\\(-N_i\\,n_c\\) sur chaque nœud maître. `solve` y trouve des inégalités et lance
l'ensemble actif (voir [Contraintes](../contraintes.md#relations-unilatérales-inégalités)) :
le multiplicateur \\(\lambda \le 0\\) porte la réaction de contact, et vaut
exactement 0 quand la paire est décollée.

L'appariement est calculé **à la construction**, par projection au point le
plus proche (`ops::geom::project_points`), \\(\xi\\) ramené au domaine de
référence : un nœud face à un bord se projette sur le bord. Il est ensuite
figé : petits déplacements, petits glissements.

**Le jeu est un chargement que porte le contact.** Le second membre de chaque
relation est \\(-g_0\\), une donnée géométrique : `external_forces` le rend, à
la composante `imposed` de chaque nœud multiplicateur, avec les autres
chargements du modèle.

**Grands glissements.** `update_contact(modele, deplacement)` (ou
`modele.update_contact(deplacement)`) rend le modèle avec ses contacts
**appariés de nouveau** sur les positions \\(X + u\\), \\(X\\) étant la
configuration active ; les autres sous-modèles sont partagés tels quels. Les
nœuds multiplicateurs ne changent pas ; les coefficients et le second membre,
si. Le jeu rangé est celui du nouvel appariement mesuré sur \\(X\\),

\\[
\tilde g = n\cdot\Big(X_s - \textstyle\sum_i N_i\\,X_{m_i}\Big),
\\]

si bien que \\(\tilde g + C\\,u\\) est le jeu réel en \\(X + u_k\\), linéarisé
là. Pour une projection au point le plus proche, la variation du jeu se réduit
à \\(n\cdot(\delta x_s - \sum N_i\\,\delta x_{m_i})\\) ; seul le terme du second
ordre, la raideur géométrique du contact, est négligé. `update_contact(modele, 0)`
redonne l'appariement de construction. Le [pas à pas](../pas-a-pas.md#grands-déplacements)
met ses contacts à jour chaque fois qu'il reconstruit son opérateur.

## Variables

| | |
|---|---|
| primale | `lambda`, la réaction de contact |
| duale | `imposed` |
| ce qu'il contraint | les déplacements `variables`, aux nœuds esclaves et maîtres |
| matériau, comportement | aucun |

## Ce qu'il produit

| | |
|---|---|
| raideur | une paire \\(C\\), \\(C^{\mathsf T}\\) |
| masse, géométrique | — |
| tangente | = raideur |
| forces internes | \\(C^{\mathsf T}\lambda\\) sur les duales, \\(C\\,u\\) sur les lignes `imposed` |
| forces externes | \\(-g_0\\) sur les lignes `imposed` |

## Exemple

Deux blocs élastiques empilés, `u_x` bloqué partout, une pression `S` sur le
bloc du haut : le contact se ferme et transmet \\(\sigma_{yy} = -S\\), les
réactions \\(-\lambda_i\\) étant les forces nodales de la pression.

```python
{{#include ../../../tests/python/test_doc_contraintes.py:contact}}
```

Le jeu que porte le contact, ici nul :

```python
{{#include ../../../tests/python/test_doc_contraintes.py:contact_rhs}}
```

Le bloc du haut glissé d'un quart, et les contacts appariés de nouveau :

```python
{{#include ../../../tests/python/test_doc_contraintes.py:update_contact}}
```

## Côté Rust

`Contact` (`src/models/contact.rs`) implémente `Constraint` et déclare son
jeu comme force externe. `tests/contact.rs` et `tests/python/test_contact.py`
déroulent les deux blocs en 2-D et en 3-D, et vérifient que soulever le bloc
du haut relâche toutes les paires.
