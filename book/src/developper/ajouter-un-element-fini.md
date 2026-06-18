# Ajouter un élément fini

Ce chapitre liste les points de code à toucher pour ajouter un **type
d'élément** (un `ElementType`), puis le rendre utilisable comme **élément
fini** (interpolation + quadrature). L'ajout est **purement additif** : on
étend des `match` exhaustifs, le compilateur signale tout ce qui reste à
compléter.

Deux niveaux, indépendants :

1. **Géométrie seule** — une nouvelle variante d'`ElementType` suffit à
   construire des [`Mesh`](../mesh.md) / `SubMesh` de ce type (connectivité,
   refcount, viz).
2. **Élément fini** — pour poser un [`FiniteElementSpace`](../fe-space.md)
   dessus, il faut en plus une **interpolation** (fonctions de forme) et une
   **quadrature** (points de Gauss).

## 1. Le type d'élément — `containers/mesh/element_type.rs`

Ajouter une variante à l'enum `ElementType` et ses métadonnées :

- **nombre de nœuds** par cellule ;
- **dimension topologique** (0 pour POI1, 1 SEG, 2 face, 3 volume) ;
- **nom court** cast3m (`"PRI6"`, `"PYR5"`…), utilisé pour le parse / display
  côté Python (le type est passé en **chaîne** : `pyrucast.Mesh(c, "PRI6")`).

À ce stade, le type est utilisable géométriquement. Aucun wrapper PyO3
supplémentaire n'est nécessaire — `ElementType` voyage comme une chaîne.

## 2. L'élément de référence

Avant d'écrire les fonctions de forme, **fixer la convention de l'élément de
référence** (comme dans le tableau de [Espace éléments finis](../fe-space.md)) :

- le **repère** \\( \xi \\) (domaine de référence) ;
- la **numérotation locale** des nœuds (l'ordre dans la connectivité) ;
- l'**orientation** (CCW pour les faces, ordre des faces pour les volumes).

Cette convention doit être **cohérente** avec le reste du code (orientation des
triangles de `fill_surface`, ordre des nœuds produits par `extrude`…). Elle se
documente dans le rustdoc d'`ElementType` **et** dans le tableau du chapitre
[Espace éléments finis](../fe-space.md).

## 3. L'interpolation — `containers/finite_element_space/interpolation.rs`

Étendre l'enum `Interpolation` (aujourd'hui `Lagrange1` seul) pour le nouveau
type :

- `Interpolation::supports(element_type)` — déclarer le couple `(ElementType,
  Interpolation)` supporté ;
- les **fonctions de forme** \\( N_i(\xi) \\) (propriété de Kronecker
  \\( N_i(\xi_j) = \delta_{ij} \\)) ;
- les **dérivées de référence** `dshape_dxi(et, &xi)` — buffer plat row-major
  \\( \mathtt{dN}[i \times d_r + k] = \partial N_i / \partial \xi_k \\).

> Ajouter un type **quadratique** (TRI6, QUA8…) suppose en général une nouvelle
> variante d'interpolation `Lagrange2` en parallèle de la variante
> d'`ElementType`.

Le Jacobien, son déterminant (y compris le cas **manifold** \\( d_s > d_r \\)),
et les dérivées physiques \\( \partial N_i / \partial x \\) sont **génériques** :
ils ne dépendent que des dérivées de référence et des coordonnées — rien à
écrire de spécifique (cf. [Espace éléments finis](../fe-space.md)).

## 4. La quadrature — `containers/finite_element_space/quadrature.rs`

Étendre l'enum `QuadratureRule` (`Gauss`, `Reduced`) pour fournir, par type :

- les **points** \\( \xi_g \\) et **poids** \\( w_g \\) de la règle par défaut
  (`Gauss`), calibrée pour intégrer exactement la matrice de masse Lagrange-1 ;
- si pertinent, la règle **réduite** (`Reduced`, un point au centroïde — utile
  par exemple pour le cisaillement d'une poutre, anti-verrouillage).

La somme des poids doit valoir la **mesure de référence** de l'élément (2 pour
SEG2, 1/2 pour TRI3, 8 pour HEX8…) — c'est un test à ajouter.

## 5. Validation à la construction

`SubFiniteElementSpace::new` rejette déjà les couples non supportés : un
nouveau type sans interpolation/quadrature déclarée échouera proprement. Une
fois les §3 et §4 faits, la construction passe automatiquement.

## 6. Visualisation (optionnel) — `viz/mesh_draw.rs`

Pour tracer le nouveau type, ajouter un bras au `match` de
`submesh_primitives` : convertir une cellule en primitive(s) géométrique(s)
(point, arête, face(s)). Sans cela, le maillage reste calculable mais pas
traçable.

## 7. Tests

Les invariants vérifiés par les tests des autres types s'appliquent au nouveau,
et constituent une bonne checklist :

- **partition de l'unité** : \\( \sum_i N_i(\xi) = 1 \\) en tout \\( \xi \\) ;
- **somme des dérivées nulle** : \\( \sum_i \partial N_i / \partial \xi_k = 0 \\) ;
- **somme des poids** = mesure de référence ;
- **déterminant du Jacobien** sur un élément droit de géométrie connue (par
  exemple \\( |J| = \\) volume × constante) ;
- round-trip de **sérialisation** (`Persist`) si des données nouvelles sont
  introduites.

## Récapitulatif

| Étape | Fichier | Obligatoire pour… |
|---|---|---|
| variante + métadonnées | `containers/mesh/element_type.rs` | la géométrie |
| convention de référence | (doc) `element_type.rs` + [fe-space](../fe-space.md) | l'élément fini |
| fonctions de forme + dérivées | `containers/finite_element_space/interpolation.rs` | l'élément fini |
| points / poids de Gauss | `containers/finite_element_space/quadrature.rs` | l'élément fini |
| primitive de rendu | `viz/mesh_draw.rs` | la visualisation |
| tests d'invariants | `tests/` / doctests | le merge |

Comme pour [ajouter une physique](../ajouter-une-physique.md), tout le code
**générique** (assemblage, Jacobien, viz hors `match`) reste inchangé.
