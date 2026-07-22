# Compilation et tests

Ce chapitre couvre l'installation **complète pour développer** sur pyrucast :
tests Rust et Python, doctests, *features* Cargo, génération du stub `.pyi`,
documentation. Pour le **simple usage en Python** (quatre commandes), voir
[Installation et démarrage rapide](installation.md).

## Prérequis

| Outil | Version | Rôle |
|---|---|---|
| Rust (via `rustup`) | stable récent, édition 2021 | Compilation du cœur |
| Python | ≥ 3.9 (3.13 testé) | API Python et `maturin` |
| `mdbook` | ≥ 0.4 | Génération de cette documentation |
| `mdbook-mermaid` | ≥ 0.14 | Rendu des graphes (ex. [graphe des dépendances](developper/arborescence.md#graphe-des-dépendances-externes)) |

Système (uniquement pour **builds avec l'API Python**, voir features plus bas) :

- **Linux** : installer les en-têtes Python — `python3-dev` (Debian/Ubuntu) ou
  `python3-devel` (Fedora/RHEL). `pyo3` en a besoin pour l'édition de liens.
- **Windows** : l'installateur officiel de Python inclut déjà les en-têtes.

> **Build Rust pur.** Par défaut (`cargo build` / `cargo test`, sans feature),
> le crate ne compile **ni `pyo3` ni `libpython`** : ni Python ni `python3-dev`
> ne sont requis. Voir [« Usage en Rust pur »](#usage-en-rust-pur) ci-dessous.

## Mise en place du venv et des outils Python

Le venv (`.venv` à la racine) héberge `maturin` et `pytest`, et sert
d'environnement cible à `maturin develop`.

### Linux / macOS (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip maturin pytest
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip maturin pytest
```

> **Important.** Pour les builds **avec l'API Python** (`maturin`, ou `cargo`
> avec `--features python-api`/`stub-gen`), `pyo3` cherche l'interpréteur via
> `VIRTUAL_ENV` : **activez toujours le venv** avant d'invoquer ces commandes.
> Un `cargo build`/`cargo test` **pur** (sans feature) n'a pas cette contrainte
> — il ne touche pas à Python.

## Compilation et tests, étape par étape

Les commandes suivantes supposent le venv activé. Elles sont identiques sous
Windows et Linux.

```text
cargo build                # cœur Rust pur (rlib + cdylib), sans pyo3
cargo test                 # tests unitaires + intégration + doctests (Rust pur)
cargo test --doc           # doctests explicitement
maturin develop            # construit et installe le module Python dans le venv
python -m pytest           # tests Python
cargo doc --no-deps --lib  # référence API Rust (rustdoc) — voir ci-dessous
mdbook build book          # génère cette documentation HTML
mdbook test book           # exécute le code testable de la documentation
```

`maturin develop` installe le module en mode *editable* dans
`.venv/Lib/site-packages/pyrucast/` (Windows) ou
`.venv/lib/python3.X/site-packages/pyrucast/` (Linux/macOS). Toute modification
du Rust est intégrée au prochain `maturin develop`.

### Documentation Rust (rustdoc)

`cargo doc --no-deps --lib` génère la **référence API Rust** à partir des
commentaires `///`. Le résultat est dans `target/doc/pyrucast/index.html`.

- `--no-deps` exclut la doc des dépendances (`pyo3`, `serde`, `bincode`,
  `nalgebra`) — gain de temps majeur.
- `--lib` limite à la crate `pyrucast` (sans les tests d'intégration).
- `--open` ouvre la page dans le navigateur à la fin.

Rustdoc et mdbook sont complémentaires : rustdoc couvre la **référence par
item** ; ce mdbook couvre les **principes, l'architecture et les exemples
transverses**.

La référence rustdoc est **publiée à côté du book** sur Codeberg Pages, à
<https://gauthier.codeberg.page/pyrucast/rust/> (lien également présent en tête
de l'[introduction](introduction.md)). Sa génération et son déploiement sont
décrits dans [« Publication automatique du book »](#publication-automatique-du-book)
ci-dessous.

## Features Cargo

Plusieurs morceaux sont gardés derrière des *features* Cargo : ce qu'on
n'active pas n'est ni compilé, ni embarqué. Toutes optionnelles, elles
s'activent en cumul (`--features a,b`).

Sans aucune feature, le crate est **du Rust pur** : `pyo3` n'est même pas
compilé (dépendance optionnelle), donc aucun lien à `libpython`.

| Feature | Apport | Implique | Quand l'activer |
|---|---|---|---|
| `python-api` | tire la dépendance `pyo3` + compile le code des `#[pyclass]`/`#[pyfunction]` (toute l'API Python) | `dep:pyo3` | rarement à la main ; activée automatiquement par `extension-module` et `stub-gen`. La désactiver (défaut) donne un crate Rust pur. |
| `extension-module` | `python-api` + dit à `pyo3` de **ne pas** se lier à `libpython` (l'interpréteur hôte la fournit au chargement du `.so`) | `python-api`, `pyo3/extension-module` | systématique pour `maturin develop` / `maturin build`. |
| `viz` | export PNG/SVG (rendu CPU via `plotters`) | — | scripts headless, captures pour la doc, CI. |
| `viz-interactive` | fenêtre interactive `winit`/`softbuffer` (souris, gizmo) | `viz` | environnement graphique disponible. |
| `stub-gen` | `python-api` + binaire `stub_gen` qui produit `pyrucast.pyi` | `python-api`, `pyo3-stub-gen` | après modification des bindings, pour rafraîchir le stub vu par les IDE. |

> **`extension-module` vs `stub-gen`.** `extension-module` produit un `.so`
> chargé par Python : `pyo3` se passe alors du link à `libpython`. À l'inverse,
> `stub-gen` produit un **binaire exécutable** : il faut `libpython` linkée
> normalement, donc on n'active **pas** `extension-module` ce jour-là. Les deux
> ne sont jamais activées ensemble.

### Usage en Rust pur

pyrucast s'utilise comme **bibliothèque Rust** sans rien de Python. Comme
`pyo3` est optionnel, une dépendance par défaut ne compile ni le binding ni
`libpython` :

```toml
# Cargo.toml d'un projet Rust tiers
[dependencies]
pyrucast = { path = "…", default-features = false }   # Rust pur, pas de pyo3
```

```rust,ignore
use pyrucast::containers::mesh::ElementType;
use pyrucast::ops::mesher;

let mesh = mesher::pave_surface(&contour, ElementType::TRI3, Some(1.0))?;
```

Tout le cœur (`containers`, `ops`, `interrupt`, `store`, …) est disponible et
PyO3-free. Seule la couche `py` (les `#[pyclass]`) demande `python-api`. Pour
interrompre un calcul long depuis du Rust pur, voir
[Interrompre une fonction](developper/interrompre-une-fonction.md).

### Génération du stub Python (`.pyi`)

Les IDE (Pylance/Pyright, PyCharm) ne savent pas inspecter un `.so` compilé :
sans stub, les complétions tombent sur `Any`. Le binaire `stub_gen` lit les
annotations `///` + les macros `#[gen_stub_*]` et écrit un `pyrucast.pyi`
complet à la racine :

```sh
# Activer le venv comme d'habitude (le binaire link à libpython).
source .venv/bin/activate
cargo run --bin stub_gen --features stub-gen
```

Régénérer `pyrucast.pyi` à chaque changement de signature Python (nouvelle
classe, paramètre, docstring `///`). Le fichier est **versionné** dans le repo,
utilisable tel quel par les IDE.

## Scripts « tout-en-un »

Le dossier `script/` contient deux niveaux d'automatisation. Tous activent (ou
créent) le venv automatiquement et s'arrêtent à la première erreur.

### `script/check.sh` — vérification rapide (CI)

Enchaîne les vérifications de non-régression :

```bash
bash script/check.sh
```

Successivement : `cargo test`, `cargo test --doc`, `cargo test --features viz`,
`cargo build --features viz-interactive`, `maturin develop`, `pytest`,
`cargo doc --no-deps --lib`, `mdbook build`, `mdbook test`. C'est le script à
brancher en intégration continue.

### `script/build.sh` / `script/build.ps1` — build complet + documentation

Le script « tout faire de bout en bout », **Linux/macOS** (`build.sh`, bash) et
**Windows** (`build.ps1`, PowerShell) :

```bash
bash script/build.sh                         # Linux / macOS
```

```powershell
.\script\build.ps1                           # Windows
# si les scripts sont bloqués :
powershell -ExecutionPolicy Bypass -File .\script\build.ps1
```

Il déroule, dans l'ordre :

1. **Vérification des prérequis** — `cargo`, `python` (≥ 3.9), création/activation
   du venv, installation de `maturin` et `pytest`, installation de `mdbook` si
   absent (`cargo install mdbook`).
2. **Compilation + tests** — `cargo build`, `cargo test` (unitaires +
   intégration + doctests), `cargo test --doc`, `cargo test --features viz`.
3. **Module Python avec visu interactive** — `maturin develop --release
   --features extension-module,viz-interactive`, puis `pytest`.
4. **Documentation** — `cargo doc` (référence Rust), régénération du stub
   `pyrucast.pyi`, `mdbook build` + `mdbook test`, et un export **pydoc HTML**
   de l'API Python (`target/python-doc/pyrucast.html`).
5. **Vérification finale** — le module s'importe et la visualisation est bien
   compilée (`Mesh.plot` présent) ; sinon le script échoue.
6. **Résumé** — emplacements des trois documentations (book, rustdoc, pydoc) et
   les commandes pour **ouvrir le livre** et **lancer une fenêtre interactive**.

À la fin, la librairie Python est installée dans le venv **avec la
visualisation interactive** : il suffit d'activer le venv et d'appeler
`mesh.plot()` (`save=None` ouvre la fenêtre — cf. [Visualisation](visualization.md)).

## Publication automatique du book

Le book est publié sur **Codeberg Pages** à l'adresse
<https://gauthier.codeberg.page/pyrucast/>, et la **référence rustdoc** à côté,
sous <https://gauthier.codeberg.page/pyrucast/rust/>.

La publication est automatisée par un workflow **Forgejo Actions**
(`.forgejo/workflows/pages.yml`) : à chaque push sur `master` qui touche au
book ou aux sources Rust, le workflow build le book (`mdbook build book`) **et**
la doc Rust (`cargo doc --no-deps --lib`), puis pousse sur la branche `pages` un
site combiné — le book à la racine, la rustdoc sous `rust/` — branche que
Codeberg Pages sert directement. Le workflow peut aussi être déclenché à la main
(*workflow_dispatch*) depuis l'onglet *Actions*.

En attendant qu'un runner soit disponible, la publication peut se faire **à la
main** avec `script/publish-book.sh` : il build le book **et** la rustdoc, puis
pousse le site combiné sur la branche `pages` avec tes propres identifiants git
(aucun token CI requis).

Prérequis à configurer une fois côté Codeberg (interface web), pour la
publication **automatique** via Forgejo Actions :

1. **Activer les Actions** : Settings → *Units* (Overview) → cocher *Actions*.
2. **Un runner** avec le label `docker` (les runners partagés Codeberg étant
   limités, on enregistre généralement son propre *Forgejo Runner*).
3. **Le secret `DEPLOY_TOKEN`** : Settings → Actions → Secrets, contenant un
   *Access Token* Codeberg avec le scope `write:repository` (utilisé pour
   pousser la branche `pages`).

## Dépannage rapide

- *`error: failed to run the Python interpreter at ...`* lors d'un
  `cargo build` : le venv n'est pas activé, ou un `VIRTUAL_ENV` obsolète pointe
  vers un chemin invalide. Réactivez le venv du projet.
- *`No module named 'pyrucast'`* lors de `pytest` : `maturin develop` n'a pas
  déposé le module. Vérifier que `VIRTUAL_ENV` est défini, puis relancer
  `maturin develop`.
- *`mdbook: command not found`* : installer `mdbook` via `cargo install mdbook`
  ou un binaire publié.
- *`The "mermaid" preprocessor exited unsuccessfully`* (ou graphes affichés en
  bloc de code brut) : installer le préprocesseur via `cargo install
  mdbook-mermaid`. Il doit être sur le `PATH` au moment de `mdbook build`.
