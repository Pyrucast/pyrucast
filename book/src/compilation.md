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
| `ruff` | récent | Format du Python (`ruff format`), vérifié par `check.sh` |
| `mdbook` | ≥ 0.4 | Génération de cette documentation |
| `mdbook-mermaid` | ≥ 0.14 | Rendu des graphes (ex. [graphe des dépendances](developper/arborescence.md#graphe-des-dépendances-externes)) |

`cargo fmt` et `clippy` viennent avec la toolchain (`rustup component add
rustfmt clippy` si besoin).

Système (uniquement pour **builds avec l'API Python**, voir features plus bas) :

- **Linux** : installer les en-têtes Python — `python3-dev` (Debian/Ubuntu) ou
  `python3-devel` (Fedora/RHEL). `pyo3` en a besoin pour l'édition de liens.
- **Windows** : l'installateur officiel de Python inclut déjà les en-têtes.

> **Build Rust pur.** Par défaut (`cargo build` / `cargo test`, sans feature),
> le crate ne compile **ni `pyo3` ni `libpython`** : ni Python ni `python3-dev`
> ne sont requis. Voir [« Usage en Rust pur »](#usage-en-rust-pur) ci-dessous.

## Mise en place du venv et des outils Python

Le venv (`.venv` à la racine) héberge `maturin`, `pytest` et `ruff`, et sert
d'environnement cible à `maturin develop`.

### Linux / macOS (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip maturin pytest ruff
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip maturin pytest ruff
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
cargo fmt                  # format Rust — à passer AVANT toute vérification
ruff format .              # format Python — idem
cargo build                # cœur Rust pur (rlib + cdylib), sans pyo3
cargo test                 # tests unitaires + intégration + doctests (Rust pur)
cargo test --doc           # doctests explicitement
maturin develop            # construit et installe le module Python dans le venv
python -m pytest           # tests Python
cargo doc --no-deps --lib  # référence API Rust (rustdoc) — voir ci-dessous
mdbook build book          # génère cette documentation HTML
mdbook test book           # exécute le code testable de la documentation
```

Le projet étant un *mixed layout*, `maturin develop` fait deux choses : il
dépose l'**extension compilée** dans `python/pyrucast/` (à côté des `.py`), et
installe le paquet en mode *editable* dans le venv — un simple
`site-packages/pyrucast.pth` pointant vers `python/`, pas une copie. Toute
modification du Rust est donc intégrée au prochain `maturin develop`, et une
modification des seuls `.py` est visible immédiatement.

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

La référence rustdoc est **publiée à côté du book** sur GitHub Pages, à
<https://pyrucast.github.io/pyrucast/rust/> (lien également présent en tête
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
| `stub-gen` | `python-api` + binaire `stub_gen` qui produit le stub `.pyi` | `python-api`, `pyo3-stub-gen` | après modification des bindings, pour rafraîchir le stub vu par les IDE. |

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
use pyrucast::atoms::ElementType;
use pyrucast::ops::mesh;

let mesh = mesh::triangulate_surface(&contour, ElementType::TRI3, Some(1.0))?;
```

Tout le cœur (`containers`, `ops`, `interrupt`, `handle`, …) est disponible et
PyO3-free. Seule la couche `py` (les `#[pyclass]`) demande `python-api`. Pour
interrompre un calcul long depuis du Rust pur, voir
[Interrompre une fonction](developper/interrompre-une-fonction.md).

### Génération du stub Python (`.pyi`)

Les IDE (Pylance/Pyright, PyCharm) ne savent pas inspecter un `.so` compilé :
sans stub, les complétions tombent sur `Any`. Le binaire `stub_gen` lit les
annotations `///` + les macros `#[gen_stub_*]` et écrit un stub complet :

```sh
# Activer le venv comme d'habitude (le binaire link à libpython).
source .venv/bin/activate
cargo run --bin stub_gen --features stub-gen
```

Le fichier produit est **`python/pyrucast/_pyrucast/__init__.pyi`** — le stub de
l'extension compilée, à côté du paquet Python (*mixed layout*). Son chemin n'est
pas choisi par le binaire : `pyo3-stub-gen` le dérive de `python-source` +
`module-name` dans `pyproject.toml`.

Régénérer le stub à chaque changement de signature Python (nouvelle classe,
paramètre, docstring `///`). Le fichier est **versionné** dans le repo,
utilisable tel quel par les IDE.

#### Les dunders polymorphes des agrégats

Une exception : `__getitem__`, `__or__` et `__ror__` des agrégats (`Mesh`,
`Model`, `NodeField`, …) prennent et rendent un `PyAny`, dont le générateur ne
peut déduire que `typing.Any` — et `mesh[0].` ne proposerait alors plus rien
dans l'IDE. Ces méthodes vivent donc dans des blocs `#[pymethods]`
**volontairement non décorés** par `gen_stub_pymethods`, invisibles au
générateur, et leurs entrées du `.pyi` sont écrites à la main en syntaxe Python
dans les deux littéraux passés à `impl_aggregate_pymethods!` (voir
`src/py/mesh.rs`) : surcharges `@overload` typées et docstrings propres à chaque
agrégat, ce qui permet aussi de dire au bon endroit ce que `|` fait vraiment sur
ce type-là. Le nom de classe y désigne le **type Rust** (`class PyMesh:`), et
les types renvoyés passent par le marqueur `pyo3_stub_gen.RustType[...]`.

Ces blocs non décorés sont **fermés** : une méthode qu'on y ajouterait
disparaîtrait silencieusement du stub. Toute nouvelle méthode d'agrégat va dans
les blocs décorés ; une méthode polymorphe de plus demande d'étendre le littéral
correspondant, sur les sept sites d'appel de la macro.

Le même patron — bloc fermé + `submit!` adjacent — sert partout où une méthode
écrite à la main est polymorphe : `SubNodeField.__getitem__`,
`SubElementField.__getitem__`, `Node.__or__`/`__ror__`, `Matrix.__mul__`. Il
sert aussi à corriger une **différence de vocabulaire** : `pyo3` regroupe les
comparaisons sous `__richcmp__`, qui n'existe pas côté Python — le stub des
champs déclare donc à la main `__ge__`/`__gt__`/`__le__`/`__lt__`, qui rendent
un masque 0/1 et non un booléen.

## Scripts « tout-en-un »

Le dossier `script/` contient deux niveaux d'automatisation. Tous activent (ou
créent) le venv automatiquement et s'arrêtent à la première erreur.

Autour d'eux gravitent quelques utilitaires : `dev.sh` / `dev.ps1` (build
minimal — module Python en release avec la visu interactive + stub, rien
d'autre), `run_examples.sh` (exemples et formation de bout en bout, appelé par
`check.sh`), `set_new_version.sh` (passe de version : tout vérifier sans
warning, reporter le numéro dans `Cargo.toml` / `pyproject.toml`, commit + tag —
il ne pousse rien), `scaling.sh` (mesure de montée en charge du parallélisme) et
`generate-formation-figures.sh` (régénère les SVG de la formation, qui sont des
artefacts commités).

### `script/check.sh` — vérification rapide (CI)

Enchaîne les vérifications de non-régression :

```bash
bash script/check.sh
```

Successivement : `cargo fmt --check`, `ruff format --check .`, `cargo test`,
`cargo test --doc`, `cargo test --features viz`,
`cargo build --features viz-interactive`,
`maturin develop --features extension-module,viz`, `pytest`,
`script/run_examples.sh`, `cargo doc --no-deps --lib` (avec
`RUSTDOCFLAGS="-D warnings"`), `mdbook build`, `mdbook test`. C'est le script à
brancher en intégration continue.

Deux points valent d'être notés :

- les **deux vérifications de format viennent en tête** — le script *vérifie*,
  il ne formate pas. Passer `cargo fmt` et `ruff format .` **avant** de le
  lancer, sinon il s'arrête au premier pas ;
- `run_examples.sh` rejoue les **exemples et les scripts de formation** de bout
  en bout. Ce sont des chaînes de calcul complètes : elles attrapent ce que les
  tests unitaires laissent passer, typiquement une méthode renommée dont plus
  personne ne se sert.

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
   `python/pyrucast/_pyrucast/__init__.pyi`, `mdbook build` + `mdbook test`, et
   un export **pydoc HTML** de l'API Python
   (`target/python-doc/pyrucast.html`).
5. **Vérification finale** — le module s'importe et la visualisation est bien
   compilée (`Mesh.plot` présent) ; sinon le script échoue.
6. **Résumé** — emplacements des trois documentations (book, rustdoc, pydoc) et
   les commandes pour **ouvrir le livre** et **lancer une fenêtre interactive**.

À la fin, la librairie Python est installée dans le venv **avec la
visualisation interactive** : il suffit d'activer le venv et d'appeler
`mesh.plot()` (`save=None` ouvre la fenêtre — cf. [Visualisation](visualization.md)).

## Publication automatique du book

Le book est publié sur **GitHub Pages** à l'adresse
<https://pyrucast.github.io/pyrucast/>, et la **référence rustdoc** à côté,
sous <https://pyrucast.github.io/pyrucast/rust/>.

La publication est automatisée par un workflow **GitHub Actions**
(`.github/workflows/pages.yml`) : à chaque push sur `master` qui touche au
book ou aux sources Rust, le workflow build le book (`mdbook build book`) **et**
la doc Rust (`cargo doc --no-deps --lib`), assemble un site combiné — le book à
la racine, la rustdoc sous `rust/` — et le déploie directement sur GitHub Pages
via `actions/deploy-pages` (pas de branche `pages`, pas de token à gérer : le
déploiement utilise les permissions natives du workflow). Il peut aussi être
déclenché à la main (*workflow_dispatch*) depuis l'onglet *Actions*.

Prérequis à configurer une fois côté GitHub (interface web) :

1. **Settings → Pages → Build and deployment → Source** : sélectionner
   *GitHub Actions*.

Les runners GitHub sont hébergés et gratuits sur dépôt public — pas de runner
à enregistrer soi-même, contrairement à Codeberg.

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
