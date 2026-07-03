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

## Contraintes disponibles

- [Dirichlet](contraintes/dirichlet.md) — impose la valeur d'une variable
  primale (`T = u_d`, `u_x = 0`, …) sur un ensemble de nœuds.

D'autres contraintes (égalité de DOFs, périodicité, contact linéarisé…)
suivront le même patron : une struct implémentant `SubModelKind`, deux blocs
unité par sous-maillage. Voir [Ajouter une physique](ajouter-une-physique.md).
