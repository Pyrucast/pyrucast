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
- **nom court** cast3m (`"PENTA6"`, `"PYR5"`…), utilisé pour le parse / display
  côté Python (le type est passé en **chaîne** : `pyrucast.Mesh(c, "PENTA6")`).

À ce stade, le type est utilisable géométriquement. Aucun wrapper PyO3
supplémentaire n'est nécessaire — `ElementType` voyage comme une chaîne.

> **Exemple concret** : le prisme `PENTA6` (extrusion d'un TRI3) suit
> exactement ce chapitre, des métadonnées jusqu'à l'export — un bon patron de
> référence.

## 2. L'élément de référence

Avant d'écrire les fonctions de forme, **fixer la convention de l'élément de
référence** (comme dans le tableau de [Espace éléments finis](../fe-space.md)) :

- le **repère** \\( \xi \\) (domaine de référence) ;
- la **numérotation locale** des nœuds (l'ordre dans la connectivité) ;
- l'**orientation** (CCW pour les faces, ordre des faces pour les volumes).

Cette convention doit être **cohérente** avec le reste du code (orientation des
triangles de `triangulate_surface`, ordre des nœuds produits par `extrude`…). Elle se
documente dans le rustdoc d'`ElementType` **et** dans le tableau du chapitre
[Espace éléments finis](../fe-space.md).

## 3. L'interpolation — `containers/finite_element_space/interpolation.rs`

Étendre l'enum `Interpolation` (`Lagrange1` pour les types linéaires,
`Lagrange2` pour les quadratiques) pour le nouveau type :

- `Interpolation::is_compatible_with(element_type)` — déclarer le couple
  `(ElementType, Interpolation)` supporté ; le **degré** doit correspondre au
  type (un type linéaire va avec `Lagrange1`, un type quadratique avec
  `Lagrange2`) ;
- les **fonctions de forme** \\( N_i(\xi) \\) (propriété de Kronecker
  \\( N_i(\xi_j) = \delta_{ij} \\)) ;
- les **dérivées de référence** `dshape_dxi(et, &xi)` — buffer plat row-major
  \\( \mathtt{dN}[i \times d_r + k] = \partial N_i / \partial \xi_k \\).

> Un type **quadratique** (TRI6, QUA8, TET10…) se branche sur l'interpolation
> `Lagrange2` déjà en place. Un ordre encore supérieur (cubique…) demanderait
> une nouvelle variante `Lagrange3`. Astuce de validation : recouper les
> dérivées analytiques par différences finies (comme le fait
> `check_dshape_matches_fd`).

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

## 6. Visualisation (optionnel) — `viz/`

Pour tracer le nouveau type :

- **`viz/mesh_draw.rs`** — ajouter un bras au `match` de `submesh_primitives`
  (convertir une cellule en point / arête / face(s)) et compléter
  `element_edges` (arêtes du fil de fer). Pour un **volume**, définir la table
  des faces (à l'image de `TET4_FACES` / `HEX8_FACES` / `PENTA6_FACES`) et
  l'ajouter à `boundary_faces`, pour que seules les faces de peau soient
  émises (les faces internes, cachées dans le solide opaque, sont retirées).
- **`viz/subdivide.rs`** — pour le **rendu interpolé** (couleur variant dans
  l'élément), ajouter les coordonnées de référence des nœuds (`ref_nodes`) et
  un bras à `subdivide` qui découpe chaque face en sous-triangles. Un prisme
  réutilise `tri_face` pour ses deux triangles et `quad_face` pour ses trois
  faces latérales.

Sans cela, le maillage reste calculable mais pas traçable.

## 7. Interopérabilité (optionnel)

Pour que le type traverse les entrées/sorties et les mailleurs :

- **export VTK** — `ops/export/vtk.rs` : associer le **code de cellule VTK**
  (`vtk_cell_type`) ; l'ordre local coïncidant avec VTK, la connectivité est
  copiée telle quelle (prisme = *wedge*, code 13).
- **lecture gmsh** — `ops/mesher/gmsh.rs` : associer le **code gmsh**
  (`element_type_from_gmsh`) ; prisme = code 6.
- **mailleurs producteurs** — un type volumique se fabrique typiquement par
  extrusion : `ops/mesher/sweep.rs` engendre PENTA6 depuis un TRI3 (via
  `extrude` et `sweep_solid`, TRI3 → PENTA6 comme QUA4 → HEX8).

## 8. Tests

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
| facettes orientées + nombre de coins | `ops/mesher/orient.rs` | `orient`, `skin`, `border` |
| domaine de référence (centre, appartenance, projection) | `ops/geom/locate.rs`, `ops/geom/project.rs` | `locate_points`, `project_points` |
| primitive de rendu + faces/arêtes | `viz/mesh_draw.rs` | la visualisation |
| rendu interpolé (subdivision) | `viz/subdivide.rs` | la visualisation colorée |
| code VTK / gmsh | `ops/export/vtk.rs`, `ops/mesher/gmsh.rs` | les entrées/sorties |
| tests d'invariants | `tests/` / doctests | le merge |

Comme pour [ajouter une physique](../ajouter-une-physique.md), tout le code
**générique** (assemblage, Jacobien, viz hors `match`) reste inchangé.

> **Repère pratique.** Il n'est pas nécessaire de retrouver ces fichiers à la
> main : ajoutez la variante à l'énumération et `cargo build` énumère lui-même
> tous les `match` devenus non exhaustifs. Le tableau ci-dessus dit *quoi*
> écrire dans chacun ; le compilateur dit *où*. Les deux dernières lignes ont
> précisément été découvertes ainsi en ajoutant [PYRA5](../elements/pyra5.md).
