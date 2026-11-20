# Ajouter un élément fini

Ce chapitre liste **tous les points de code à toucher** pour ajouter un type
d'élément. Comme pour [ajouter une physique](../ajouter-une-physique.md), le
coût est **O(1) fichier** : il ne dépend pas du nombre d'éléments déjà en
place.

## Le principe en une phrase

L'énum [`ElementType`](../mesh.md) ne sert qu'au **stockage** et à la
**sérialisation** ; **tout le comportement** vit dans une struct par élément
(sous `src/atoms/element_kind/`) qui implémente le trait `ElementKind`. Le
point de dispatch, `ElementType::as_kind()`, relie les deux. Le code
générique — `skin`, `orient`, les localisateurs, l'export, le rendu — ne fait
**jamais** de `match` par variante. Seuls les opérateurs qui n'acceptent que
certains types (`border`, `sweep`, `extrude`, `revolve`, `pave_volume`,
`triangulate_surface`, `regularize`) les nomment, pour dire lesquels.

L'énum garde quatre `match` qui doublent des données du trait —
`nodes_per_cell`, `topological_dim`, `name` et `reversal_permutation` — et un
test de la batterie vérifie qu'ils disent la même chose que lui.

```text
ElementType  (enum : stockage + sérialisation bincode + nom "TRI3")
├── POI1, SEG2, TRI3, QUA4, TET4, PENTA6, PYRA5, HEX8,
│   SEG3, TRI6, QUA8, TET10, PENTA15, HEX20, QUA9, HEX27
├── ALL          -> &'static [ElementType]   ← la liste des types
├── nodes_per_cell / topological_dim / name / reversal_permutation
│                                            ← un match chacun
└── as_kind(self) -> &'static dyn ElementKind   ← le match vers le trait

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
│   └── vtk_code / gmsh_code / gmsh_permutation / med_permutation
│                                         (permutations : identité par défaut)
└── Familles (défaut : None)
    └── quadratic / linear_parent / split_into
```

## Les étapes

Ajouter un élément touche **trois** fichiers :

1. **`src/atoms/element_kind/<mon_element>.rs`** (nouveau) — une struct unité
   et son `impl ElementKind`. Calquer sur `tri3.rs` (cas linéaire le plus
   court), `tri6.rs` (quadratique, qui délègue son domaine à son parent
   linéaire) ou `pyra5.rs` (le cas difficile : fonctions de forme
   rationnelles et quadrature conique).
2. **`src/atoms/element_kind/mod.rs`** — un `mod <mon_element>;` et un bras
   dans `as_kind()`.
3. **`src/atoms/element_type.rs`** — une variante dans l'énum, son rustdoc,
   une entrée dans `ElementType::ALL`, un bras dans chacun des quatre `match`
   (`nodes_per_cell`, `topological_dim`, `name`, `reversal_permutation`), et le
   type dans le test `all_lists_every_variant_exactly_once` (son bras, et le
   nombre de types qu'il compte).

Le compilateur signale un bras manquant dans chaque `match`, y compris celui du
test. Une entrée oubliée dans `ALL`, elle, n'est attrapée que par le compte du
test (`ALL.len()`), qu'il faut donc relever. Pour le reste :

- **aucun wrapper PyO3** : `ElementType` traverse la frontière en **chaîne**,
  et `from_name` se déduit de `ALL` et de `name()` ;
- `skin`, `orient`, `convert`, `to_quadratic`, l'export VTK, la lecture gmsh,
  l'échange MED, le rendu et la subdivision colorée sont **génériques** et ne
  changent pas — à condition que `gmsh_permutation` et `med_permutation` disent
  comment ces formats numérotent les nœuds de la maille ;
- un opérateur qui nomme les types qu'il accepte (`border`, `sweep`,
  `extrude`, `revolve`, `pave_volume`, `triangulate_surface`, `regularize`)
  refuse le nouveau type avec un message, tant qu'on ne l'y a pas ajouté. Pour un volume, la numérotation MED se vérifie contre medcoupling :
  `tests/python/test_medcoupling.py` exige un volume positif et des nœuds
  milieux au milieu des arêtes qu'il en déduit ; ajouter le type à sa table
  `REFERENCE`.

Reste à écrire la doc : une fiche `book/src/elements/<nom>.md`, une entrée
dans `SUMMARY.md`, une ligne dans le catalogue
d'[`elements/index.md`](../elements/index.md), et une ligne dans la table de
correspondance gmsh d'[`algorithmes/maillage-echanges.md`](../algorithmes/maillage-echanges.md#types-déléments-reconnus).

Les deux figures de la fiche (élément de référence, quadrature) sont dessinées
par `script/elements_figures.py` : y déclarer l'élément dans `ELEMENTS` (sa
règle de quadrature, et ses faces s'il est 3-D). Les nœuds sont lus dans la
table de la fiche. Les SVG ne sont pas commités : `check_doc`, `build.sh` et la
publication du book lancent le script avant `mdbook build`, et le dossier
`book/src/elements/img/`, ignoré par git, est vidé puis réécrit à chaque fois.
Avant un `mdbook serve`, lancer soi-même `python script/elements_figures.py`.
`tests/python/test_doc_elements_figures.py` échoue si la règle dessinée diffère
de celle de `pyrucast`.

## Ce que le trait demande, et pourquoi

### L'élément de référence

`ref_nodes()` est **la donnée racine** : les coordonnées de chaque nœud dans
le repère \\( \xi \\), dans l'ordre local. Le nombre de nœuds et la dimension
topologique s'en déduisent, et c'est la référence contre laquelle les tests de
cohérence recoupent les autres tables.

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

C'est cette seule table qui alimente `skin`, `orient`, l'élimination des faces
cachées au rendu et la subdivision du rendu.

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

Les dérivées analytiques sont recoupées par différences finies, pour tous les
types, par `check_dshape_matches_fd`.

Le Jacobien, son déterminant (y compris le cas d'une **variété** plongée,
\\( d_s > d_r \\)) et les dérivées physiques \\( \partial N_i / \partial x \\)
sont **génériques** — rien à écrire.

### La quadrature

`gauss()` rend les points et poids de la règle par défaut, calibrée pour
intégrer exactement la matrice de masse Lagrange-1 sur un élément droit. La
règle **réduite** ne s'écrit pas : c'est le défaut du trait, un point au
centroïde portant toute la mesure.

## Les tests que vous obtenez gratuitement

`src/atoms/element_kind/mod.rs` porte une batterie de tests de cohérence qui
boucle sur `ElementType::ALL` — un type neuf y entre dès qu'il est déclaré,
sans une ligne de test à écrire :

- métadonnées cohérentes entre l'énum et le trait ;
- nœuds milieux exactement au milieu de leur arête ;
- arêtes bien formées, tout coin sur au moins une arête ;
- facettes de codimension 1, du même degré que leur parent, géométriquement
  cohérentes avec `ref_nodes` ;
- jeux de coins de facettes distincts ;
- centroïde intérieur, nœuds de référence dans le domaine, clamp ramenant
  dedans ;
- codes VTK et gmsh uniques, permutations gmsh et MED bijectives ;
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

## To-do : ajouter un élément fini

`<NOM>` est la variante (`PYRA5`), `<nom>` le fichier (`pyra5.rs`), `<Nom>` la
struct (`Pyra5`).

**Code**

- [ ] Fixer la convention de l'élément : repère, numérotation locale,
  orientation.
- [ ] Créer le fichier de l'élément et y définir la struct unité `<Nom>`
  (`src/atoms/element_kind/<nom>.rs`).
- [ ] Implémenter `ElementKind` pour `<Nom>` : nœuds et domaine de référence,
  fonctions de forme et leurs dérivées, quadrature, codes VTK et gmsh ; selon
  le cas facettes, arêtes, permutations gmsh et MED, parent linéaire ou
  version quadratique (`src/atoms/element_kind/<nom>.rs`).
- [ ] Déclarer le module (`mod <nom>;`) et brancher la variante dans
  `as_kind()` (`src/atoms/element_kind/mod.rs`).
- [ ] Ajouter la variante `<NOM>` à `enum ElementType`, documenter sa
  convention dans son rustdoc, et l'ajouter à `ElementType::ALL`
  (`src/atoms/element_type.rs`).
- [ ] Compléter les quatre `match` d'`ElementType` : `nodes_per_cell`,
  `topological_dim`, `name`, `reversal_permutation`
  (`src/atoms/element_type.rs`).
- [ ] Mettre à jour le test `all_lists_every_variant_exactly_once` : ajouter
  le bras, relever le compte `ALL.len()` (`src/atoms/element_type.rs`).
- [ ] Ajouter le type aux opérateurs qui doivent l'accepter : `border`,
  `sweep`, `extrude`, `revolve`, `pave_volume`, `triangulate_surface`,
  `regularize` (`src/ops/mesh/`).
- [ ] Écrire un test de volume sur un élément droit de géométrie connue
  (`src/containers/finite_element_space/mod.rs`).

**Documentation**

- [ ] Écrire la fiche de l'élément et l'ajouter au sommaire
  (`book/src/elements/<nom>.md`, `book/src/SUMMARY.md`).
- [ ] Ajouter sa ligne au catalogue, et le type aux physiques qui l'acceptent
  dans la table « Ce que chaque physique pose sur quels éléments »
  (`book/src/elements/index.md`).
- [ ] Ajouter son code gmsh à la table « Types d'éléments reconnus »
  (`book/src/algorithmes/maillage-echanges.md`).
- [ ] Déclarer l'élément dans `ELEMENTS` : règle de quadrature, et faces s'il
  est 3-D (`script/elements_figures.py`).

**Vérification**

- [ ] Pour un volume : ajouter le type à la table `REFERENCE`, puis lancer les
  tests MED (`tests/python/test_medcoupling.py`, `script/check_medcoupling.sh`).
- [ ] Formater (`cargo fmt`, `ruff format .`), puis lancer toutes les
  vérifications (`script/check_all.sh`).
