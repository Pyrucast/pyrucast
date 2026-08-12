# pyrucast — Feuille de route

Ce document dit **où en est le projet** et **ce qui pourrait venir ensuite**. La
première partie est un état des lieux daté ; la seconde est une liste de pistes
**non arbitrées** — elles seront précisées, ordonnées ou abandonnées plus tard.

État arrêté au **5 août 2026** (v0.2.1).

## Philosophie

- Librairie élément fini : cœur Rust + API Python, inspirée des principes de cast3m.
- Code **simple, maintenable, éditable par un humain non expert**.
- Dépendances externes réduites au strict nécessaire — **accord explicite requis avant tout ajout**.

## Décisions d'architecture verrouillées

| Sujet | Décision |
|---|---|
| Mémoire | Store central à handles : slab + indices générationnels + comptage de références + swap disque. **Global au processus**, adressé par fonctions de module (`insert` / `read` / `write`) — pas d'objet `Session` à passer. |
| Séparation des couches | `containers/` (structures) ⊥ `ops/` (opérateurs) ⊥ `py/` (binding). Un module d'`ops/` porte le nom du **conteneur qu'il produit**. Les règles complètes sont dans `CONVENTIONS.md`. |
| Primitives géométriques | `nalgebra` (vecteurs et matrices de petite taille — géométrie, maillage, visualisation), `nalgebra-sparse` pour le stockage creux. |
| Algèbre linéaire (solveur) | **LU creuse directe `faer`**, multithreadée, avec cache de factorisation sur la `Matrix` (*factoriser une fois, résoudre souvent*). `SolveMethod` est le point d'extension pour un autre back-end. |
| Sérialisation | `serde` + `bincode` via un trait `Persist` **unique**, partagé entre swap disque et sauvegarde/relecture fichier. |
| Parallélisme | `rayon`, **toujours actif**, porté *au-dessus* des noyaux de physique : un noyau ne voit ni rayon, ni le store, ni un verrou. |
| Binding Python | `pyo3` + `maturin`, *mixed layout* : extension plate privée `_pyrucast` + couche Python pure qui la range en sous-modules. |
| Documentation | `mdbook` (théorie + doctests) + rustdoc, publiés sur GitHub Pages. |
| Algorithmes non-linéaires / transitoires | **Orchestrés en Python**, pas en Rust (voir plus bas). |
| Méthode | Largeur d'abord : toutes les structures + bindings + doc/tests avant le numérique lourd. |

Deux décisions ont été **révisées en cours de route**, et le sont ici
définitivement : le solveur n'est pas une implémentation maison derrière un
trait `LinearSolver` (c'est `faer`), et le store n'est pas exposé via une
`Session` (il est global).

### Dépendances approuvées (socle figé)

Toujours liées : `serde`, `bincode`, `nalgebra`, `nalgebra-sparse`, `faer` (LU creux du solveur), `rayon` (parallélisme), `parking_lot` (verrous du store), `paste` (macros d'agrégat). Optionnelles, derrière une feature : `pyo3`, `pyo3-stub-gen`, et la visualisation `plotters`, `winit`, `softbuffer`. Outillage : `maturin`, `mdbook`, `ruff`, `criterion`. Tout autre ajout = nouvelle demande explicite.

### Definition of Done par objet

1. Struct Rust vivant dans le Store (adressable par `Handle<T>`)
2. `Debug` (structurel) + `Display` (résumé façon listing cast3m)
3. Tests unitaires Rust + doctests sur tout l'API public
4. Binding PyO3 : `__repr__` → `Debug`, `__str__` → `Display`
5. Tests Python (pytest)
6. Chapitre mdbook (théorie + API)

Un objet n'est terminé que si ces 6 points sont verts.

---

# Ce qui est fait et disponible

En chiffres : **82 600 lignes de Rust**, 1 046 tests unitaires Rust (+ 20
fichiers de tests d'intégration), 447 tests Python, 27 doctests, 75 pages de
book, 26 exemples, 6 scripts de formation, **100 fonctions exposées en Python**.

## Socle mémoire

Store à handles générationnels : slab par type, free-list, recyclage, détection
des handles périmés (`StaleHandle`). Refcount **à deux niveaux** — les slots
d'un côté, les nœuds d'une `Coords` de l'autre, avec `gc()`. Verrou par objet
(`RwLock` de cellule), guards possédés permettant la lecture **en place**.
Swap disque `Resident / OnDisk / Free`, transparent vis-à-vis de `Drop`.
`compact()` rend la mémoire de queue.

## Conteneurs et atomes

Les **sept agrégats** — `Mesh`, `FiniteElementSpace`, `NodeField`,
`ElementField`, `Model`, `Matrix`, `Evolution` — chacun avec sa vue `Sub*`,
`len` / `[i]` / `|`. Plus `Coords` (le magasin de coordonnées, jeux multiples,
repère axisymétrique) et les atomes : `Node`, `Cell`, `Element`, `ElementType`,
`Point*` / `Vector*`, `Band`, `RgbColor`.

## Éléments finis — 16 types

`POI1` ; linéaires `SEG2`, `TRI3`, `QUA4`, `TET4`, `PYRA5`, `PENTA6`, `HEX8`
(Lagrange-1) ; quadratiques `SEG3`, `TRI6`, `QUA8`, `QUA9`, `TET10`, `PENTA15`,
`HEX20`, `HEX27` (Lagrange-2, sérendipité ou complets). Fonctions de forme et
dérivées, jacobien y compris le cas *manifold*, quadratures `Gauss` et
`Reduced` — plus la quadrature conique Gauss × Jacobi de la pyramide.
Axisymétrie portée par `Coords` et intégrée dans la seule mesure `det_j_w`.

Chaque type tient dans **un fichier** `atoms/element_kind/<nom>.rs` implémentant
le trait `ElementKind` — nœuds de référence, facettes, arêtes, domaine,
interpolation, quadrature, codes VTK/gmsh, familles. Un unique `match`,
`ElementType::as_kind()`, relie l'énum au comportement, sur le modèle de
`SubModel::as_kind()` côté physiques : ajouter un élément coûte un fichier et
deux variantes, et aucun consommateur générique ne change.

## Physiques — 17 sous-modèles

Thermique : `HeatConduction`, `Convection` (échange de surface / film),
`Radiation` (rayonnement à l'infini `σε(T⁴ − T_∞⁴)` : rigidité linéarisée autour
de `T_∞`, résidu exact, tangente cohérente validée par différences finies ;
première physique à déclarer **deux** natures, `[Thermal, Radiation]`).
Diffusion : `Fick` (concentration `c` / flux `j`, nature `Diffusion` propre),
`InterfaceTransfer` (échange `h(c₁ − c₂)` entre deux maillages non conformes en
nœuds, variante thermique comprise).
Mécanique : `Truss`, `Elasticity` (contraintes/déformations planes,
axisymétrique, 3-D), `Plasticity` (la **loi d'écoulement en attribut** : von Mises parfaite ou à
écrouissage isotrope, Drucker-Prager non associé avec traitement du sommet,
Ottosen à quatre paramètres intégrée par plan sécant ; puis les lois
**dépendantes du temps** — fluages de Norton, Lemaitre et Blackburn,
viscoplasticité de Chaboche et sa variante endommageable de Lemaitre-Chaboche,
qui erronent en l'absence de `dt` ; tangentes toutes confrontées à une différence
finie des forces internes), `Mazars` (endommagement), `Timoshenko`,
`Frame` (portique 2-D), `Frame3d`, `FollowerPressure` (charge dont la direction
tourne avec la surface, bâtie sur les tangentes déformées et non sur Nanson —
`I + ∇_s u` n'est pas un gradient de transformation sur une variété). Contraintes : `Dirichlet`, `Mpc`,
`Embedded` (baignage), `Contact` (nœud-surface, unilatéral).
Dilatation thermique non couplée (`thermal_strain`, `alpha` en composante
matériau facultative).

**Symétrie matériau** (`MaterialSymmetry`, `src/models/symmetry.rs`) : axe
orthogonal à l'hypothèse cinématique, partagé par `Elasticity`,
`HeatConduction` et `Fick` — isotrope (défaut, inchangé), orthotrope,
anisotrope. Le repère d'orthotropie est donné par des **vecteurs** portés par le
champ matériau (`V1X/V1Y`, plus `V1Z` et `V2*` en 3-D), comme
`MATE 'DIRECTION' V1 V2` de Cast3M. La rotation du tenseur d'élasticité passe
par l'ordre 4 plutôt que par une matrice de Bond, ce qui supprime toute
convention d'indices ; l'isotropie court-circuite ce chemin et garde ses nombres
exacts.

Le coût d'ajout d'une physique est **O(1) fichier** : une struct + un
`impl SubModelKind`, deux lignes de câblage.

## Assemblage

Trois formes de contribution (`Contribution`) : `Computed` (intégrée à la volée
et dispersée dans le CSR), `Literal` (valeurs déjà remplies — Dirichlet, MPC) et
`Coupling` (bloc **inter-maillages**, lignes sur un maillage et colonnes sur un
autre ; son scatter est séquentiel, un coloriage sur une seule connectivité ne
prouvant plus la disjonction).

Quatre genres de matrice derrière une machinerie unique (`MatrixKind`) :
raideur / conductivité, masse / capacité, raideur géométrique, tangente
cohérente — plus `lump`. Motif creux mémoïsé **par genre** sur le `Model`,
matrices élémentaires calculées en parallèle et dispersées dans le CSR par
**coloration des cellules**, sans matérialiser de COO. Éléments
multi-quadrature (Timoshenko) sur le même chemin. Forces internes `∫ Bᵀσ`
(Cast3M `BSIG`) et divergence par le même driver de scatter nodal.

## Solveur

LU creuse directe (`faer`), factorisation mise en cache sur la `Matrix`.
Trois voies : **Lagrange** (`solve`, système augmenté), **élimination /
condensation** (`solve_eliminate`), **active-set unilatéral**
(`solve_unilateral`). Sortie sur supports de blocs (handles POI1 réutilisés,
champs soustractibles).

## Maillage

Primitives : `line`, `circle`, `arc`, `transfinite`, `points`.
Balayages : `sweep`, `extrude` (TRI3 → PENTA6, QUA4 → HEX8), `revolve`,
`sweep_solid`.
Mailleurs libres : `triangulate_surface` (Delaunay contraint + Ruppert, trous
compris, TRI3/QUA4, contours 2-D et 3-D planaires) et `triangulate_volume`
(prédicats exacts, enveloppe gelée, récupération, raffinement et lissage
anti-*slivers*).
Mailleurs frontaux : `pave_surface` (quadrangles) et `pave_volume` (couche
limite HEX8 + raccord PYRA5 + cœur TET4).
Topologie et transformations : `skin`, `border`, `orient`, `consolidate`,
`merge_nodes`, `to_quadratic`, `convert`, `to_poi1`, `translate`, `rotate`,
symétries, sélections géométriques (sphère, plan, cylindre, cône, tore, ligne).
Entrées/sorties : lecture **gmsh** (MSH 2.2 et 4.1, ASCII et binaire), export
**VTK**.

## Champs et opérateurs

Cinématique (`gradient`, `deformation`, `beam_deformation`,
`frame_deformation`), comportement (`behavior`), matériaux
(`material_field`, `interp_to_gauss`), positions, `restrict`, `flux`,
`divergence`, `internal_forces`, masques et arithmétique de champ,
réductions (`integral`, `xtx`, `xty`), requêtes géométriques
(`locate_points`, `project_points`, `contact_gaps`), `Evolution` (valeur
tabulée interpolée). 168 fonctions libres dans `ops/`.

## Parallélisme

`rayon` toujours actif, politique de grain centralisée
(`parallel::MIN_PARALLEL_LEN`). Drivers `models::kernel` au-dessus de noyaux
purs et séquentiels ; zéro-copie par guards tenus pendant toute la région
parallèle. Déterminisme **bit-à-bit** pour les opérateurs write-once et les
réductions ; déterminisme par coloration (non bit-à-bit) pour l'assemblage et
les scatters nodaux ; le solveur fait exception (back-end faer).

## API Python

*Mixed layout* : extension plate privée `_pyrucast`, rangée par la couche
Python pure en sous-modules nommés d'après le conteneur produit
(`pyrucast.mesh`, `pyrucast.element_field`, `pyrucast.matrix`, …), les
conteneurs et atomes restant au top-level. Stub `.pyi` versionné.
Interruption coopérative (`Ctrl+C`) via le trait `Cancel`.
Orchestration non-linéaire en Python pur : `pyrucast.thermomechanics`
(pas-à-pas thermo→méca), Newton modifié accéléré par Anderson dans les
exemples.

## Visualisation

Rendu CPU `plotters` (PNG/SVG) et fenêtre interactive `winit`/`softbuffer` :
maillages, champs coloriés **par élément** (jamais moyennés entre éléments),
rendu interpolé par subdivision, courbes, axes, gizmo.

## Outillage

`script/check.sh` (la passe complète, à brancher en CI), `build.sh` / `dev.sh`
et leurs équivalents PowerShell, `run_examples.sh`, `set_new_version.sh`,
`scaling.sh`. CI GitHub Actions : publication du book et de la rustdoc sur
Pages, release multi-OS (Linux, Windows, macOS).

---

# Pistes futures

Rien de ce qui suit n'est arbitré : ni l'ordre, ni le périmètre, ni même le
fait de le faire. Ces points sont ceux que l'état des lieux a fait apparaître
comme **manquants**, à préciser plus tard.

## Transitoire et intégration en temps

Le manque fonctionnel le plus visible. La masse et le *lumping* existent ; ce
qui manque est le **pilotage** : schéma d'intégration en temps (θ-méthode,
Newmark), et la couche Python qui l'orchestre — l'équivalent d'un `PASAPAS`.
S'y rattache l'**advection** (`ADVE`), seule brique Cast3M encore absente côté
matrices.

## Solveur et performance

- **Renumérotation** — `Coords` porte déjà une permutation optionnelle séparant
  l'ordre solveur de l'identité, mais personne ne la calcule : une réduction de
  bande/profil (type Cuthill–McKee) reste à écrire, avec l'invariance des
  résultats en test.
- **Méthodes itératives** et **factorisation de Cholesky** pour les matrices
  symétriques — le drapeau de symétrie existe déjà sur la `Matrix`, et
  `SolveMethod` est le point d'extension prévu.
- **Passe performance** sur gros maillage, avec les benchs (`benches/parallel.rs`,
  `benches/geom.rs`, `script/scaling.sh`) comme instrument.
- **Allocateur global** — voir ci-dessous.

### Allocateur global : reprendre les défauts de page des gros champs

**Le problème.** Un opérateur qui produit un champ rend un conteneur **neuf**.
Sur un maillage sérieux, ce conteneur est énorme : `behavior::integrate` sur
3,61 M de QUA4 en axisymétrique rend 462 Mo (4 points de Gauss × 4 composantes
× 8 octets). Au-delà de son seuil, glibc sert un tel bloc par `mmap` et le rend
par `munmap` au `Drop`. La mémoire revient donc au noyau à chaque appel, et
l'appel suivant la **refault page par page** : 112 812 pages de 4 Ko, une par
page, une fois par appel.

Mesuré (`perf stat`, grille 1900×1900) : **11,9 M de défauts de page** pour une
centaine d'appels — exactement le compte théorique, donc ni fuite ni gaspillage,
simplement le tarif d'allouer un demi-gigaoctet neuf. Chaque défaut coûte
**1,07 µs** (entrée noyau, allocation d'une page, remise à zéro de 4 Ko, mise à
jour de la table), soit **120 ms par appel, ~15 % d'un appel de 819 ms**. La
bande passante effective tombe à 560 Mo/s : l'opération n'est pas limitée par le
débit mémoire mais par la latence de remise en service des pages.

Ça ne concerne pas que les benchs : une boucle de Newton ou un transitoire
rappelle `integrate` à chaque itération, et paie donc à chaque itération.

**La solution.** Remplacer l'allocateur global par un allocateur à arènes
(`mimalloc` ou `jemalloc`), en une ligne dans `lib.rs` :

```rust
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

**Pourquoi ça aide.** Un allocateur à arènes **conserve** les blocs libérés au
lieu de les rendre au noyau, et les recycle au tour suivant. Les pages restent
donc mappées : on ne les faulte qu'au premier appel. Vérifié en relevant
`MALLOC_MMAP_THRESHOLD_`, qui produit le même effet sur glibc — **11,9 M → 402 k
défauts, soit 30×**, à nombre d'instructions rigoureusement constant (3,506 G vs
3,511 G). C'est bien du temps noyau supprimé, pas du calcul déplacé.

L'intérêt dépasse `integrate` : l'assemblage et les matrices allouent bien plus
gros, et profiteraient du même recyclage. C'est aussi portable macOS et Windows,
là où le réglage `mallopt` équivalent ne vaudrait que pour glibc.

**Ce qu'il en coûte.** Une dépendance de plus au socle, et une empreinte
résidente plus haute puisque les arènes sont conservées — à surveiller vis-à-vis
du swap, qui existe précisément pour tenir la mémoire. À mesurer avant/après sur
`benches/` **et** sur la RSS, pas seulement sur le temps.

**L'alternative, plus profonde.** Une forme `integrate_into(&mut champ, …)` qui
réutilise le champ du tour précédent supprimerait l'allocation elle-même, pas
seulement ses défauts. Mais elle élargit le seam `Domain::integrate_behavior`,
alors que le contrat veut qu'un auteur de physique n'écrive que
`integrate_point` — chantier de conception à part entière.

## Sauvegarde et reprise

La persistance n'est faite qu'à moitié : le trait `Persist` sert au swap, mais
la **sauvegarde du graphe d'objets** n'existe pas. Il manque le remappage des
handles, le conteneur fichier versionné (en-tête magique + numéro de format) et
l'API Python `save` / `load`. Le format binaire visé est portable Linux ↔
Windows : entiers little-endian normalisés, `usize` sur 64 bits, `f64`
IEEE-754, aucun chemin ni séparateur dépendant de l'OS dans le *payload*.

## Swap : du manuel à l'automatique

Le swap fonctionne, mais **rien ne le déclenche tout seul** : `swap_out(&h)` est
manuel, objet par objet, et réservé au Rust. C'est à l'appelant de décider quoi
évincer et quand — autant dire que personne ne le fait. Deux marches :

1. **exposer le swap côté Python**, pour qu'un script puisse déclencher une
   éviction ;
2. **évincer quand c'est nécessaire** — un budget RAM, une priorité par slot et
   une date de dernier usage, et le store se déleste seul de ce qui est
   transitoire. C'est l'évolution **B** ci-dessous, et c'est la marche qui rend
   le swap réellement utile : tant qu'il faut le demander à la main, il ne sert
   qu'aux cas qu'on a vus venir.

Le swap étant déjà transparent vis-à-vis de `Drop` et adossé au même `Persist`
que la sauvegarde, la brique manquante est **la politique**, pas le mécanisme.

## Qualité de maillage

Le domaine le plus avancé du code, et le moins planifié. Le point ouvert est
documenté dans `triangulate_volume.rs` : la **récupération d'arêtes** de
l'enveloppe est inachevée — une arête bloquée reste tributaire des bascules,
sinon de `allow_surface_nodes`. S'y ajoutent les mesures de qualité et leur
suivi dans le temps.

## Évolutions mémoire (conditionnelles)

Le store actuel est la fondation : on **ne le remplace pas**. Trois évolutions
sont identifiées, chacune **déclenchée par une mesure**, pas par anticipation —
aucune mesure ne les a encore justifiées. Détails et arbitrages dans
[book/src/memory-model.md](book/src/memory-model.md).

1. **A. Indirection + compactage déplaçant** — table `id → slot_idx`, les slots
   vivants se déplacent pour combler les trous, les handles restent valides.
   ~100 lignes, API publique inchangée. *Déclencheur* : hauts plateaux mémoire
   sur cycles intensifs de création/destruction.
2. **B. Swap annoté + éviction LRU** — priorité par slot (`Pinned` / `Working` /
   `Scratch`) et `last_used`, éviction automatique sous budget RAM. C'est la
   seconde marche de la piste *Swap* ci-dessus.
   *Déclencheur* : le swap manuel devient insuffisant.
3. **C. Arènes par génération** — jeune génération collectée souvent, vieille
   rarement. À reconsidérer seulement quand A et B auront montré leurs limites.

---

# Annexe — algorithmes non-linéaires et transitoires : orchestration Python

Les schémas non-linéaires (boucle de **Newton**) et l'**intégration en temps**
ne sont **pas** codés en Rust : ils sont **pilotés côté Python**, par des
fonctions qui composent les opérateurs du cœur.

C'est le modèle de **Cast3M**, où ces algorithmes ne sont pas des opérateurs
natifs mais des **procédures GIBIANE** (`PASAPAS`, `UNPAS`, `TRANSNON`…)
enchaînant les opérateurs de base (`RIGI`, `KTAN`, `BSIG`, `RESO`, `COMP`,
`EXCO`…). Ici, le langage d'orchestration est **Python** au lieu de GIBIANE :

- **le cœur Rust fournit les briques** — assemblage (`stiffness`, `mass`,
  `geometric`, `tangent`), résolution linéaire (`solve`, `solve_eliminate`,
  `solve_unilateral`), intégration du comportement (`behavior`), forces
  internes (`internal_forces`), opérations de champ ;
- **Python assemble l'algorithme** — boucle de Newton (résidu, tangente,
  incrément, test de convergence) et schéma en temps sont des fonctions Python
  appelant ces briques.

C'est déjà le cas en pratique : `pyrucast.thermomechanics` déroule un
pas-à-pas thermo→mécanique, et les exemples déroulent un Newton modifié
accéléré par Anderson. Un équivalent de `pasapas` / `unpas` / `transnon` sera
donc une **bibliothèque Python** livrée avec le binding, pas un opérateur Rust
(cf. la colonne « Équivalent pyrucast » de `opérateur_castem.csv`).
