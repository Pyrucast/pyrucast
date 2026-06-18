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

Système :

- **Linux** : installer les en-têtes Python — `python3-dev` (Debian/Ubuntu) ou
  `python3-devel` (Fedora/RHEL). `pyo3` en a besoin pour l'édition de liens.
- **Windows** : l'installateur officiel de Python inclut déjà les en-têtes.

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

> **Important.** `pyo3` cherche l'interpréteur Python via la variable
> d'environnement `VIRTUAL_ENV`. **Activez toujours le venv** avant d'invoquer
> `cargo` ou `maturin`, sinon `cargo build` peut échouer en tentant d'utiliser
> un interpréteur introuvable.

## Compilation et tests, étape par étape

Les commandes suivantes supposent le venv activé. Elles sont identiques sous
Windows et Linux.

```text
cargo build                # cœur Rust (rlib + cdylib)
cargo test                 # tests unitaires + intégration + doctests
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

## Features Cargo

Plusieurs morceaux sont gardés derrière des *features* Cargo : ce qu'on
n'active pas n'est ni compilé, ni embarqué. Toutes optionnelles, elles
s'activent en cumul (`--features a,b`).

| Feature | Apport | Implique | Quand l'activer |
|---|---|---|---|
| `python-api` | code des `#[pyclass]` (toutes les classes Python) | — | rarement à la main ; activée automatiquement par `extension-module` et `stub-gen`. Utile pour `cargo test` quand on veut compiler les bindings sans link spécial à libpython. |
| `extension-module` | `python-api` + dit à `pyo3` de **ne pas** se lier à `libpython` (l'interpréteur hôte la fournit au chargement du `.so`) | `python-api`, `pyo3/extension-module` | systématique pour `maturin develop` / `maturin build`. |
| `viz` | export PNG/SVG (rendu CPU via `plotters`) | — | scripts headless, captures pour la doc, CI. |
| `viz-interactive` | fenêtre interactive `winit`/`softbuffer` (souris, gizmo) | `viz` | environnement graphique disponible. |
| `stub-gen` | `python-api` + binaire `stub_gen` qui produit `pyrucast.pyi` | `python-api`, `pyo3-stub-gen` | après modification des bindings, pour rafraîchir le stub vu par les IDE. |

> **`extension-module` vs `stub-gen`.** `extension-module` produit un `.so`
> chargé par Python : `pyo3` se passe alors du link à `libpython`. À l'inverse,
> `stub-gen` produit un **binaire exécutable** : il faut `libpython` linkée
> normalement, donc on n'active **pas** `extension-module` ce jour-là. Les deux
> ne sont jamais activées ensemble.

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

## Dépannage rapide

- *`error: failed to run the Python interpreter at ...`* lors d'un
  `cargo build` : le venv n'est pas activé, ou un `VIRTUAL_ENV` obsolète pointe
  vers un chemin invalide. Réactivez le venv du projet.
- *`No module named 'pyrucast'`* lors de `pytest` : `maturin develop` n'a pas
  déposé le module. Vérifier que `VIRTUAL_ENV` est défini, puis relancer
  `maturin develop`.
- *`mdbook: command not found`* : installer `mdbook` via `cargo install mdbook`
  ou un binaire publié.
