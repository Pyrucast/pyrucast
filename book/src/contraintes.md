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

## Imposition par élimination (condensation)

Voie **alternative** aux multiplicateurs, sélectionnée au moment de la résolution
avec [`solve_eliminate`](operateurs/solveur.md) au lieu de `solve`. Plutôt que
d'agrandir le système, on **élimine** chaque relation : un terme *esclave* `s`
est exprimé par les *maîtres*,

\\[
u_s = \frac{1}{a_s}\Big(g - \sum_{k \neq s} a_k\\, u_k\Big),
\\]

d'où une transformation globale `u = T·û + u₀` (`û` = DOFs retenus). Le système
se réduit à `K̂ û = f̂` avec `K̂ = Tᵀ K T`, `f̂ = Tᵀ(f − K·u₀)`, résolu par le
**même** LU creux — mais sur une matrice **plus petite et définie**, sans DOF
multiplicateur. La solution est prolongée `u = T·û + u₀` ; la **réaction**
(équivalent du multiplicateur) est récupérée en post-traitement, `−(K·u − f)` à
la ligne duale de chaque esclave (`= aₛ·λ`).

Les deux voies lisent la **même** description méthode-neutre des contraintes
(`Constraint::relations()`) et acceptent le **même** chargement (le second membre
`g` au nœud-multiplicateur, ci-dessous) : passer de l'une à l'autre ne change que
l'appel de résolution.

**Périmètre v1** : non chaîné, esclaves disjoints — chaque relation élimine un
esclave distinct, jamais réutilisé comme maître ni esclave dans une autre
relation (couvre la périodicité). Un système chaîné est refusé avec une erreur
explicite ; il reste résoluble par la voie Lagrange (`solve`).

## Relations unilatérales (inégalités)

Toute relation d'égalité peut devenir **unilatérale** : `Σₖ aₖ·uₖ ≥ g` (ou
`≤ g`) au lieu de `= g`. C'est le paramètre optionnel `sense` des contraintes
(`"="` par défaut, `">="`, `"<="`) — une butée `u ≥ a` est un Dirichlet
unilatéral, une liaison à jeu est une MPC unilatérale :

```python
{{#include ../../tests/python/test_doc_contraintes.py:butee}}
```

Une relation unilatérale obéit aux conditions de **complémentarité** (KKT) : ou
bien elle est **active** (l'égalité tient et le multiplicateur porte la
réaction), ou bien elle est **inactive** (le jeu `C·u − g` est du côté
admissible et `λ = 0`). Avec la convention des blocs assemblés
(`K·u + Cᵀ·λ = f`), le signe admissible du multiplicateur est `λ ≤ 0` pour une
relation `≥` active, `λ ≥ 0` pour une `≤` active.

Comme l'ensemble actif n'est pas connu d'avance, la résolution est itérative —
la **méthode du statut** (*active-set*), portée par l'opérateur
[`solve_unilateral`](operateurs/solveur.md) :

1. statut initial : toutes les inégalités actives (ou le statut convergé
   précédent quand le cache est chaud — *warm start*) ;
2. résolution du système point-selle avec, pour chaque relation **inactive**,
   sa ligne de contrainte remplacée par `λ = 0` (la matrice garde sa taille,
   seules les valeurs changent) ;
3. mise à jour du statut : une relation active dont le `λ` tire (signe
   inadmissible) est **relâchée**, une inactive dont le jeu pénètre est
   **activée** ;
4. statut stable ⇒ convergé ; sinon on refactorise et on répète (boucle finie
   de la méthode du statut classique, bornée par `max_iter`).

```python
{{#include ../../tests/python/test_doc_contraintes.py:solve_unilateral}}
```

Les relations inactives sortent avec `λ = 0` exact ; la solution a la même
forme que la voie Lagrange (primal + multiplicateurs). L'assemblage est
**inchangé** — le `sense` n'est lu que par les solveurs : `solve` sur un modèle
unilatéral résout la version « tout collé » (toutes les relations en égalité),
et `solve_eliminate` le **refuse** avec une erreur explicite. Un modèle sans
inégalité retombe sur le `solve` ordinaire.

Le second membre `g` (la borne) suit le mécanisme habituel : au slot
`imposed_value` du nœud-multiplicateur, donc `constraint_rhs` fonctionne tel
quel.

## Second membre : le helper `constraint_rhs`

Le second membre `u_d` / `g` n'est **pas** stocké dans la contrainte :
l'utilisateur l'écrit dans le `NodeField` de chargement, à la composante duale
propre de la contrainte (`imposed_<v>` pour Dirichlet, `mpc_rhs` pour la MPC),
**au nœud-multiplicateur** de la relation. Retrouver ce nœud et cette composante
à la main est fastidieux ; le helper le fait :

```python
{{#include ../../tests/python/test_doc_contraintes.py:constraint_rhs}}
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
{{#include ../../tests/python/test_doc_contraintes.py:constraint_rhs_by_index}}
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
- [Baignage (embedded)](contraintes/embedded.md) — lie chaque nœud d'un maillage
  **immergé** à l'**interpolation** d'un maillage hôte au même point
  (`u_c(p) = Σᵢ Nᵢ(ξ_p)·u_c(hôteᵢ)`) : une barre baignée dans un volume. Les
  poids `Nᵢ` sont calculés par **localisation de point** à la construction ; c'est
  une MPC dont les coefficients **varient par nœud**.
- [Contact (nœud-surface)](contraintes/contact.md) — empêche les nœuds d'un
  maillage **esclave** de pénétrer une surface **maître** : une relation
  **unilatérale** (`≥`) par nœud esclave, à coefficients `n·Nᵢ` calculés par
  **projection** à la construction (petits glissements, sans frottement).
  Résolu par `solve_unilateral`.

D'autres contraintes suivront le même patron : une struct
implémentant `SubModelKind`, des blocs `C`/`Cᵀ` littéraux. Voir [Ajouter une
physique](ajouter-une-physique.md).
