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
| `ruff` | récent | Format du Python (`ruff format`), vérifié par `check_format` |
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
{{#include ../../tests/doc_conteneurs.rs:mailler_en_rust}}
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
d'autre), `run_examples.sh` / `run_examples.ps1` (exemples et formation de bout
en bout, appelés par `check_examples`), `set_new_version.sh` (passe de version :
tout vérifier sans warning, reporter le numéro dans `Cargo.toml` /
`pyproject.toml`, commit + tag — il ne pousse rien), `scaling.sh` (mesure de
montée en charge du parallélisme) et `generate-formation-figures.sh` (régénère
les SVG de la formation, qui sont des artefacts commités).

### `script/check_*.sh` — vérifications, en bloc ou à la carte

La vérification est découpée en **cinq blocs indépendants**, chacun lançable
seul, plus deux enchaîneurs. Chaque bloc existe en bash (`.sh`, Linux/macOS) et
en PowerShell (`.ps1`, Windows), au comportement identique.

Le tableau ci-dessous donne, commande par commande, ce qu'elle vérifie, où
vivent les tests qu'elle exécute, ce qu'elle coûte, et quels scripts la
lancent. Les durées sont mesurées **à chaud** — arbre déjà compilé, ce qui est
le cas courant ; un premier tour après un changement de features paie en plus
la recompilation.

| commande | ce qu'elle teste | où vivent les tests | temps | quick | fmt | rust | py | ex | doc | all | version |
|---|---|---|--:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `cargo fmt --check` | le Rust est formaté | — | 1 s | ✓ | ✓ | | | | | ✓ | ✓ |
| `ruff format --check .` | le Python est formaté | — | 1 s | ✓ | ✓ | | | | | ✓ | ✓ |
| `cargo check --all-targets` | le build **Rust pur** compile encore | — | <1 s | ✓ | | ✓ | | | | ✓ | ✓ |
| `cargo test --features viz` | 1036 tests + 893 doctests | `#[cfg(test)]` dans 123 fichiers de `src/`, 42 fichiers `tests/*.rs`, commentaires `///` et `//!` | 154 s | ✓ | | ✓ | | | | ✓ | ✓ |
| `cargo build --features viz-interactive` | la fenêtre winit compile | aucun test — 960 lignes que rien n'exerce sans écran | 14 s | ✓ | | ✓ | | | | ✓ | ✓ |
| `maturin develop --features extension-module,viz` | l'extension Python s'installe | — | 20 s | | | | ✓ | | | ✓ | ✓ |
| `python -m pytest` | l'API Python, unité par unité | 73 fichiers `tests/python/test_*.py`, dont 13 `test_doc_*.py` qui sont aussi les sources des exemples du book | 32 s | | | | ✓ | | | ✓ | ✓ |
| `script/run_examples.sh` | des chaînes de calcul **de bout en bout** | 30 `examples/*.py`, 3 `examples/*.rs`, 6 `formation/*.py` | 80 s | | | | | ✓ | | ✓ | ✓ |
| `cargo doc --no-deps --lib` (`-D warnings`) | la rustdoc n'a aucun lien cassé | — | 1 s | | | | | | ✓ | ✓ | ✓ |
| `python script/doc_lint.py` | les **cinq garde-fous** du book et des exemples | `book/src/**.md`, la rustdoc, le module Python installé | 28 s | | | | | | ✓ | ✓ | ✓ |
| `mdbook build book` | le book se rend | `book/src/**` | 1 s | | | | | | ✓ | ✓ | ✓ |
| `cargo clippy` × 4 jeux de features | aucun avertissement, `-D warnings` | — | 105 s | | | | | | | | ✓ |

Et les totaux, dans le même régime :

| script | contenu | temps |
|---|---|--:|
| `check_format` | formatage seul | 2 s |
| `check_rust` | cœur Rust | 169 s |
| `check_python` | liaison et tests Python | 64 s |
| `check_examples` | exemples et formation | 80 s |
| `check_doc` | rustdoc, garde-fous, book | 50 s |
| **`check_quick`** | **formatage + Rust — la boucle de commit** | **~3 min** |
| `check_all` | les cinq blocs | ~6 min |
| `set_new_version` | `check_all` + les quatre clippy | ~8 min |

Deux remarques que ces chiffres appellent.

**Les doctests ne coûtent presque plus rien**, alors qu'ils représentaient 44 %
de la suite entière. Le crate est passé à l'**edition 2024**, qui les fusionne
en un seul binaire au lieu d'en compiler un par exemple : 893 doctests sont
passés de 232 s à 17 s. Ce qui domine aujourd'hui le bloc Rust, ce n'est plus
l'exécution des tests (37 s) mais la **compilation des 42 fichiers de tests
d'intégration**.

**Les doctests tournent sous `viz`, jamais sous `viz-interactive`.** La couche
interactive tire 58 crates de plus — winit, Wayland, X11 — et chaque binaire de
test les lie. Elle est donc *compilée sans être testée*, ce qui suffit : elle ne
porte aucun test qu'un écran ne soit nécessaire pour exercer.

```bash
bash script/check_quick.sh   # la boucle de commit : formatage + Rust
bash script/check_doc.sh     # je viens de toucher à la doc
bash script/check_all.sh     # la passe complète, à brancher en CI
```

```powershell
.\script\check_doc.ps1        # Windows
.\script\check_all.ps1
```

`check_all` enchaîne les cinq **dans cet ordre**, qui n'est pas indifférent : le
formatage d'abord (il échoue en une seconde), le cœur Rust ensuite, puis Python
— qui *(ré)installe* l'extension dont les exemples ont besoin —, les exemples,
et la documentation en dernier, la plus lente. Un bloc lancé seul qui a besoin
du module compilé le dit plutôt que d'échouer obscurément.

`script/check.sh` reste comme alias de `check_all.sh`. `check_quick` est le
raccourci du cas courant : il ne remplace pas `check_all`, il le précède — les
exemples, Python et la doc restent à passer avant de pousser.

`set_new_version.sh` **appelle** `check_all` plutôt que de recopier ses pas :
la copie précédente avait divergé, et ne lançait ni les garde-fous de
documentation ni les exemples au moment précis où l'on pose un tag.

Trois points valent d'être notés :

- les **deux vérifications de format viennent en tête** — le script *vérifie*,
  il ne formate pas. Passer `cargo fmt` et `ruff format .` **avant** de le
  lancer, sinon il s'arrête au premier pas ;
- `run_examples` rejoue les **exemples et les scripts de formation** de bout
  en bout. Ce sont des chaînes de calcul complètes : elles attrapent ce que les
  tests unitaires laissent passer, typiquement une méthode renommée dont plus
  personne ne se sert ;
- **ne jamais piper `check_all`** (`| tail`, `| grep`) : le code de retour
  devient celui du dernier maillon du tube, et un échec passe pour un succès.

### Les extraits du book sont du code exécuté

Aucune page du book ne possède de code : **tout bloc `rust` ou `python` est un
`{{#include}}`** pointant un test, un exemple ou une source. Le code affiché
est donc exécuté par `check_rust`, `check_python` ou `check_examples`, et il ne
peut plus diverger de l'API.

`check_doc` ne compile donc rien lui-même. Il lance `script/doc_lint.py`, quatre
vérifications de texte :

| garde-fou | ce qu'il tient |
|---|---|
| `includes` | chaque `{{#include}}` résout : fichier, ancre, texte non vide |
| `fences` | aucune page ne possède de code |
| `symboles` | la prose ne cite aucun symbole disparu |
| `doctests` | cliquet : la couverture des doctests ne peut que monter |

Le premier est le plus important, parce que le mécanisme porteur n'est gardé par
rien d'autre : une ancre inexistante rend un bloc **vide**, avec un code de
retour 0 et sans un mot.

`mdbook test` a été retiré de la chaîne : tous les blocs Rust du book sont
`rust,ignore` — c'est ce que le mécanisme d'inclusion impose —, si bien qu'il ne
compilait rien. Un pas vert qui ne regarde rien coûte plus cher que son absence.

La règle complète, et le choix entre doctest, test d'intégration et exemple, est
page [Documentation et tests](developper/documentation-et-tests.md).

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
   `python/pyrucast/_pyrucast/__init__.pyi`, `mdbook build`, et
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
