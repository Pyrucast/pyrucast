# Ajouter un élément fini

Ce chapitre liste **tous les points de code à toucher** pour ajouter un type
d'élément. Comme pour [ajouter une physique](../ajouter-une-physique.md), le
coût est **O(1) fichier** : il ne dépend pas du nombre d'éléments déjà en
place.

## Le principe en une phrase

L'énum [`ElementType`](../mesh.md) ne sert qu'au **stockage** et à la
**sérialisation** ; **tout le comportement** vit dans une struct par élément
(sous `src/atoms/element_kind/`) qui implémente le trait `ElementKind`. Un
unique point de dispatch, `ElementType::as_kind()`, relie les deux. Le code
générique — `skin`, `orient`, `border`, les localisateurs, l'export, le
rendu — ne fait **jamais** de `match` par variante.

```text
ElementType  (enum : stockage + sérialisation bincode + nom cast3m)
├── POI1, SEG2, TRI3, …, HEX27
├── ALL          -> &'static [ElementType]   ← la seule énumération
└── as_kind(&self) -> &'static dyn ElementKind   ← l'unique match

ElementKind  (trait : le comportement d'un type d'élément)
├── Identité (requis)
│   ├── element_type / ref_nodes / reversal_permutation / corner_count
│   └── nodes_per_cell, topological_dim, is_quadratic  (fournis, tirés de ref_nodes)
├── Topologie de référence (défaut : rien)
│   └── facets -> &[Facet] / edges -> &[[usize; 2]]
├── Domaine de référence (requis)
│   └── ref_centroid / ref_measure / contains_ref / clamp_ref
├── Interpolation (requis)
│   ├── degree -> Option<Interpolation>
│   ├── shape_into / dshape_into        (formes sans allocation)
│   └── shape / dshape                  (fournis : surcouche allouante)
├── Quadrature
│   ├── gauss -> (Vec, Vec)             (requis)
│   └── reduced                         (fourni : centroïde × mesure)
├── Échange (requis)
│   └── vtk_code / gmsh_code / gmsh_permutation (défaut : None)
└── Familles (défaut : None)
    └── quadratic / linear_parent / split_into
```

## Les étapes

Ajouter un élément se réduit à **trois** gestes :

1. **`src/atoms/element_kind/<mon_element>.rs`** (nouveau) — une struct unité
   et son `impl ElementKind`. Calquer sur `tri3.rs` (cas linéaire le plus
   court), `tri6.rs` (quadratique, qui délègue son domaine à son parent
   linéaire) ou `pyra5.rs` (le cas difficile : fonctions de forme
   rationnelles et quadrature conique).
2. **`src/atoms/element_kind/mod.rs`** — un `mod <mon_element>;` et un bras
   dans `as_kind()`.
3. **`src/atoms/element_type.rs`** — une variante dans l'énum, son rustdoc,
   et une entrée dans `ElementType::ALL`.

Rien d'autre. En particulier :

- **aucun wrapper PyO3** : `ElementType` traverse la frontière en **chaîne**,
  et `from_name` se déduit de `ALL` + `name()` ;
- `nodes_per_cell`, `topological_dim` et `from_name` sur `ElementType`
  délèguent au trait, et n'ont donc pas de bras à compléter ;
- `skin`, `orient`, `border`, `convert`, `to_quadratic`, l'export VTK, la
  lecture gmsh, le rendu et la subdivision colorée sont **génériques** et ne
  changent pas.

Reste à écrire la doc : une fiche `book/src/elements/<nom>.md`, une entrée
dans `SUMMARY.md`, une ligne dans les deux tableaux de
[`elements/index.md`](../elements/index.md), et une ligne dans la table de
correspondance gmsh d'[`operateurs/maillage.md`](../operateurs/maillage.md).

## Ce que le trait demande, et pourquoi

### L'élément de référence

`ref_nodes()` est **la donnée racine** : les coordonnées de chaque nœud dans
le repère \\( \xi \\), dans l'ordre local. Le nombre de nœuds et la dimension
topologique s'en déduisent, et c'est la vérité contre laquelle tous les tests
d'invariants recoupent les autres tables.

Fixer d'abord la convention — repère, numérotation locale, orientation (CCW
pour les faces) — et la documenter dans le rustdoc de la variante.

> **La convention des nœuds milieux.** Tout le code s'appuie dessus : **les
> coins d'abord, puis un nœud milieu par arête, dans l'ordre de `edges()`**.
> Le milieu de l'arête `k` est donc toujours à l'indice local
> `corner_count() + k`. `QUA9` et `HEX27` ajoutent leurs centres de face et de
> volume après. C'est ce qui permet à `to_quadratic`, au fil de fer et aux
> facettes quadratiques de se déduire d'une seule table d'arêtes.

### Les facettes

`facets()` rend les facettes orientées vers l'extérieur — les arêtes d'une
surface, les faces d'un volume. **Une facette est un élément à part entière** :
une face de `TET10` est un `TRI6`, une face de `HEX27` est un `QUA9`. Le champ
`nodes` porte donc les nœuds milieux, et `Facet::corners()` restreint aux
coins — ce sur quoi deux mailles voisines s'accordent quel que soit leur degré,
donc ce qui sert de clé d'adjacence.

C'est cette seule table qui alimente `skin`, `orient`, le culling des faces
cachées et la subdivision du rendu.

### Le domaine de référence

`contains_ref` et `clamp_ref` décrivent le domaine et la projection dessus ;
`ref_centroid` en donne un point intérieur, qui sert **à la fois** de départ
au Newton d'inversion et de point de la quadrature réduite. `ref_measure` est
la mesure du domaine, à laquelle les poids de quadrature doivent sommer.

### L'interpolation

`degree()` déclare le degré Lagrange du type — un `TRI6` *est* quadratique, ce
n'est pas un choix. `shape_into` et `dshape_into` écrivent dans un tampon
fourni : c'est la forme à implémenter, parce que l'inversion de la géométrie
(`locate_points`, `project_points`) les appelle dans une boucle de Newton, où
une allocation par appel dominerait le coût. `shape`/`dshape` sont les
surcouches allouantes, fournies.

Astuce de validation : recouper les dérivées analytiques par différences
finies, ce que fait déjà `check_dshape_matches_fd` pour tous les types.

Le Jacobien, son déterminant (y compris le cas **manifold**
\\( d_s > d_r \\)) et les dérivées physiques \\( \partial N_i / \partial x \\)
sont **génériques** — rien à écrire.

### La quadrature

`gauss()` rend les points et poids de la règle par défaut, calibrée pour
intégrer exactement la matrice de masse Lagrange-1 sur un élément droit. La
règle **réduite** ne s'écrit pas : c'est le défaut du trait, un point au
centroïde portant toute la mesure.

## Les tests que vous obtenez gratuitement

`src/atoms/element_kind/mod.rs` porte une batterie d'invariants qui boucle sur
`ElementType::ALL` — un type neuf y entre le jour où il est déclaré, sans une
ligne de test à écrire :

- métadonnées cohérentes entre l'énum et le trait ;
- nœuds milieux exactement au milieu de leur arête ;
- arêtes bien formées, tout coin sur au moins une arête ;
- facettes de codimension 1, du même degré que leur parent, géométriquement
  cohérentes avec `ref_nodes` ;
- jeux de coins de facettes distincts ;
- centroïde intérieur, nœuds de référence dans le domaine, clamp ramenant
  dedans ;
- codes VTK et gmsh uniques, permutations gmsh bijectives ;
- couple linéaire ↔ quadratique réciproque, à coins et arêtes égaux ;
- somme des poids = mesure de référence ;
- partition de l'unité, somme des dérivées nulle, Kronecker aux nœuds,
  dérivées analytiques contre différences finies.

À ajouter à la main : un test de volume sur un élément droit de géométrie
connue (`containers/finite_element_space/mod.rs` en a un par type), et un
aller-retour de sérialisation si des données nouvelles apparaissent.

## Pourquoi garder l'énum

Même raison que pour les physiques : `bincode` n'est pas auto-descriptif, et
`typetag` — nécessaire pour sérialiser un `Box<dyn ElementKind>` — ne le
supporte pas. On perdrait la persistance. L'énum donne en prime
l'exhaustivité au compilateur sur `as_kind()`. Le trait n'est jamais
sérialisé : `as_kind()` rend un `&'static`, reconstruit à la volée depuis la
variante.

## Ce que ça a coûté, et rapporté

Avant ce découpage, le savoir d'un élément était réparti sur **une vingtaine
de fichiers** et recopié jusqu'à quatre fois : les facettes en trois
exemplaires, les nœuds de référence en quatre, le centroïde en trois. Huit de
ces tables retombaient sur `_ => None`, donc s'oubliaient sans un mot du
compilateur — et trois bugs en vivaient (`skin` muet sur `PYRA5` et sur tout
élément quadratique, un défaut de rendu silencieusement faux, une liste de
test à taille figée).

Ce chapitre listait alors un tableau « quel fichier pour quoi ». Il n'en a
plus besoin : il n'y a **plus de liste de fichiers à parcourir**, seulement un
trait à remplir.
