# Contraintes

Une **contrainte** impose une relation aux inconnues du problème (une valeur
imposée, une liaison) sans être une physique au sens d'une loi de
comportement : elle n'a **ni matériau, ni intégrande volumique**. pyrucast les
traite comme des [`SubModel`](model.md) ordinaires, contribuant leurs blocs à
la **même** matrice globale que les physiques — pas de système point-selle
séparé à orchestrer côté utilisateur.

## Multiplicateurs de Lagrange

Les contraintes sont imposées par **multiplicateurs de Lagrange**. Une
relation `C·u = u_d` introduit une inconnue supplémentaire `λ` (le
multiplicateur) et deux blocs **rectangulaires** dans la matrice :

\\[
\begin{bmatrix} K & C^\top \\\\ C & 0 \end{bmatrix}
\begin{bmatrix} u \\\\ \lambda \end{bmatrix}
=
\begin{bmatrix} f \\\\ u_d \end{bmatrix}
\\]

- le bloc **`C`** porte la relation de contrainte (lignes = duale propre,
  colonnes = primale contrainte de la cible) ;
- le bloc **`Cᵀ`** réinjecte la **réaction** dans l'équation de la physique
  cible (la force qui maintient la contrainte) ;
- à la solution, le multiplicateur `λ` **est** cette force de réaction.

Chacun des blocs `C` et `Cᵀ` est, pris isolément, **non symétrique** ; seule
leur union `C ∪ Cᵀ` l'est — c'est une propriété **globale** du système
point-selle, pas de chaque bloc (voir le drapeau `symmetric` de la
[Matrice](matrix.md)).

Les **nœuds-multiplicateurs** sont des nœuds comme les autres, fournis par
l'utilisateur via un maillage : la contrainte ne crée jamais de nœud et ne
mute jamais le `Coords`.

## Second membre : le helper `constraint_rhs`

Le second membre `u_d` / `g` n'est **pas** stocké dans la contrainte :
l'utilisateur l'écrit dans le `NodeField` de chargement, à la composante duale
propre de la contrainte (`imposed_<v>` pour Dirichlet, `mpc_rhs` pour la MPC),
**au nœud-multiplicateur** de la relation. Retrouver ce nœud et cette composante
à la main est fastidieux ; le helper le fait :

```python
rhs = dirichlet.constraint_rhs([(noeud_contraint, u_d), …])
rhs = mpc.constraint_rhs([(noeud_terme, g), …])
```

- on **désigne chaque relation par un nœud** : le nœud contraint pour Dirichlet
  (un seul par relation), n'importe quel nœud-terme pour une MPC ;
- le helper résout ce nœud vers le **nœud-multiplicateur** de sa relation (via
  `relations()`) et y écrit la valeur, à la composante duale de la contrainte ;
- il renvoie un `NodeField` neuf sur **tous** les nœuds-multiplicateurs (les
  relations non citées valent `0`), à **fusionner** dans le chargement global
  avec `|` : `load | dirichlet.constraint_rhs(…) | mpc.constraint_rhs(…)`.

Le modèle passé doit porter **exactement une** contrainte (l'objet `dirichlet`
ou `mpc`). Une erreur est levée si un nœud n'appartient à aucune relation, ou
s'il en désigne **plusieurs** (ambigu).

### Désigner par index de relation

Quand un même nœud participe à plusieurs relations (le keying par nœud est alors
ambigu), on désigne la relation par son **index** (0-based, dans l'ordre de
`relations()`) :

```python
rhs = mpc.constraint_rhs_by_index([(index_relation, g), …])
```

Le champ renvoyé et la fusion par `|` sont identiques ; une erreur est levée si
un index dépasse le nombre de relations.

## Contraintes disponibles

- [Dirichlet](contraintes/dirichlet.md) — impose la valeur d'une variable
  primale (`T = u_d`, `u_x = 0`, …) sur un ensemble de nœuds. C'est la relation
  **à un seul terme** `1·u = u_d`.
- [Multi-points (MPC)](contraintes/mpc.md) — impose une relation linéaire
  **à N termes** `Σₖ aₖ·u(nœudₖ, varₖ) = g` entre plusieurs DOFs (égalité,
  périodicité, liaison affine…). Généralise Dirichlet.

D'autres contraintes (contact linéarisé…) suivront le même patron : une struct
implémentant `SubModelKind`, des blocs `C`/`Cᵀ` littéraux. Voir [Ajouter une
physique](ajouter-une-physique.md).
