# pyrucast — Feuille de route de développement

## Philosophie

- Librairie élément fini : cœur Rust + API Python, inspirée des principes de cast3m.
- Code **simple, maintenable, éditable par un humain non expert**.
- Dépendances externes réduites au strict nécessaire — **accord explicite requis avant tout ajout**.

## Décisions d'architecture verrouillées

| Sujet | Décision |
|---|---|
| Mémoire | Store central à handles : arène + indices générationnels + comptage de références + swap disque. Exposé via `Session`. |
| Primitives géométriques | `nalgebra` (vecteurs et matrices de petite taille — géométrie, maillage, visualisation). |
| Algèbre linéaire (solveur) | Implémentation maison, derrière un trait `LinearSolver` enfichable (backend externe possible plus tard, sur accord). |
| Sérialisation | `serde` + `bincode` via un trait `Persist` **unique**, partagé entre swap disque et sauvegarde/relecture fichier. |
| Binding Python | `pyo3` + `maturin`, incrémental objet par objet. |
| Documentation | `mdbook` (théorie + doctests). |
| Méthode | Largeur d'abord : toutes les structures + bindings + doc/tests avant le numérique lourd. |

### Dépendances approuvées (socle figé)

`pyo3`, `maturin`, `mdbook`, `serde`, `bincode`, `nalgebra`. Visualisation (optionnelle) : `plotters`, `winit`, `softbuffer`. Tout autre ajout = nouvelle demande explicite.

## Persistance : swap et sauvegarde mutualisés (portable Linux ↔ Windows)

Un **seul** trait `Persist` (sérialisation `serde` + `bincode`) sert de socle commun :

- **Swap** : sérialise un objet (slot) isolé pour libérer la RAM, rechargé à l'accès. Orchestré par le Store (indexation des slots, politique d'éviction).
- **Sauvegarde / reprise** : sérialise le **graphe** d'objets d'une `Session` (un Mesh référence une Configuration, un Model un FE space…) avec remappage des handles, dans un conteneur versionné.

Format **portable Linux ↔ Windows** :

- bincode : entiers little-endian normalisés (indépendant de l'endianness hôte), `usize` encodé sur 64 bits (portable 32/64-bit), `f64` IEEE-754.
- Aucune donnée dépendante de l'OS dans le payload (pas de chemins absolus ni de séparateurs OS ; les slots sont des identifiants opaques).
- Conteneur fichier = en-tête magique + numéro de version de format + payload `Persist`. La version protège l'évolution du format.

Cœur partagé = `Persist` par objet. Ce qui diffère : le swap orchestre slot par slot ; la sauvegarde orchestre le graphe complet + un manifeste.

## Definition of Done par objet (largeur d'abord + binding incrémental)

1. Struct Rust vivant dans le Store (adressable par `Handle<T>`)
2. `Debug` (structurel) + `Display` (résumé façon listing cast3m)
3. Tests unitaires Rust + doctests sur tout l'API public
4. Binding PyO3 : `__repr__` → `Debug`, `__str__` → `Display`
5. Tests Python (pytest)
6. Chapitre mdbook (théorie + API)

Un objet n'est terminé que si ces 6 points sont verts.

## Phases

### Phase 0 — Fondations projet & conventions
- Crate `pyrucast` (cdylib + rlib), `pyo3` / `maturin`, venv Python.
- Conventions : `PyrucastError` + `Result`, Display vs Debug, trait `Persist`.
- Harnais : `cargo test`, `cargo test --doc`, `pytest`, `mdbook build` / `mdbook test`.
- Squelette mdbook (introduction, conventions).

### Phase 1 — Le Store (cœur, prérequis absolu)
- `Handle<T>` générationnel ; Drop intelligent décrémentant le refcount (libération automatique).
- Slab + free-list + générations (détection des handles périmés).
- Refcount + recyclage automatique des slots à 0.
- Fragmentation : réemploi via free-list + `compact()`.
- Swap disque : états `Resident / OnDisk / Free`, `swap_out` / `swap_in`, crochet de politique d'éviction ; conteneur portable partagé avec la sauvegarde.
- `Session` : possède le Store (`Arc`), exposé en Python.
- Décision ouverte : `Session` explicite vs store global.

### Phase 2 — Tous les objets (structures + bindings, sans numérique lourd)
Dans l'ordre de dépendance, chacun selon la Definition of Done :
1. **Configuration** — jeux de coordonnées, bascule de jeu actif, création/suppression de nœuds à chaud. Séparer identité (id interne stable) et ordre solveur (permutation rechargeable). Décision ouverte : politique de suppression d'un nœud encore référencé par un champ.
2. **Node** — accesseur utilisateur (handle vers Configuration).
3. **Mesh / SubMesh / `ElementType` (enum)** — un sous-maillage par type ; POI1 = liste de nœuds.
4. **NodeField** — valeurs sur maillage POI1, multi-composantes nommées.
5. **FiniteElementSpace** — maillage + formulation EF (éléments de référence, fonctions de forme, points de Gauss : données).
6. **ElementField** — valeurs par point de Gauss × composante.
7. **Model** — modèle physique (élasticité, plasticité, thermique…) sur un FE space.
8. **Matrix** — matrice creuse maison (COO/CSR), construction manuelle, point d'extension assemblage.

### Phase 3 — Numérique maison (assemblage & résolution)
Fonctions de forme & dérivées, quadrature, matrices élémentaires, assemblage `Model → Matrix`, solveurs (Gradient Conjugué puis factorisation directe) derrière `LinearSolver`, cas de validation comparé à l'analytique.

### Phase 4 — Optimisation de la renumérotation
Réduction de bande/profil maison (type Cuthill–McKee) branchée sur la permutation solveur ; invariance des résultats testée.

### Phase 5 — Parallélisme à mémoire partagée
Store thread-safe (verrous shardés / `RwLock` par slot) ; assemblage et solveur parallèles. Décision ouverte : `std::thread` scopé vs `rayon`.

### Phase 6 — Durcissement & finition

#### Gestion mémoire — évolutions (séquence A → B → C, déclenchées par les mesures)

Le store actuel est la fondation : on **ne le remplace pas**. On y ajoute, **quand le besoin se mesure**, les évolutions suivantes :

1. **A. Indirection + compactage déplaçant.** Table `id → slot_idx` ; on déplace les slots vivants pour combler les trous au milieu du `Vec`, les handles restent valides (ils référencent l'`id` logique, pas le slot physique). ~100 lignes, API publique inchangée.
   - *Déclencheur* : hauts plateaux mémoire observés sur cycles intensifs de création/destruction.
   - *Bénéfice* : résout la fragmentation interne — c'est l'approche historique de cast3m.

2. **B. Swap annoté + éviction LRU.** Chaque slot porte une priorité (`Pinned` / `Working` / `Scratch`) et un `last_used` ; éviction automatique sous budget RAM. S'appuie sur la composition des `Handle` pour la structure « qui référence quoi ».
   - *Déclencheur* : le swap manuel devient insuffisant (typiquement sur les premiers solveurs grosse échelle).
   - *Bénéfice* : swap intelligent façon cast3m moderne, sans appels manuels à `swap_out`.

3. **C. Arènes par génération** *(à arbitrer)*. Jeune génération (objets transitoires, collectée souvent) + vieille génération (Configuration, Mesh, collectée rarement), style GC générationnel.
   - *À reconsidérer* uniquement quand A et B auront révélé leurs propres limites.

Détails et tradeoffs : [book/src/memory-model.md](book/src/memory-model.md) (section *Limites connues et évolution prévue*).

#### Autres travaux de Phase 6
- Réglage des paramètres de compactage et d'éviction.
- Benchmarks (high-water mark mémoire, aller-retour swap, assemblage gros maillage).
- Complétion mdbook (chapitres restants, exemples, galerie).
- Passe performance générale.

## Décisions ouvertes (non bloquantes)

- Phase 1 : `Session` explicite vs store global.
- Phase 2 : politique de suppression d'un nœud encore référencé par un champ.
- Phase 5 : `std::thread` scopé vs `rayon`.
