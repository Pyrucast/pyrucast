# Compilation et tests

Ce chapitre couvre l'installation **complète pour développer** sur pyrucast :
tests Rust et Python, doctests, *features* Cargo, génération du stub `.pyi`,
documentation. Pour le **simple usage en Python** (quatre commandes), voir
[Installation et démarrage rapide](installation.md).

## Prérequis

| Outil | Version | Rôle |
|---|---|---|
| Rust (via `rustup`) | ≥ 1.89 (`rust-version` du crate), édition 2024 | Compilation du cœur |
| Python | ≥ 3.11 (3.13 testé) | API Python et `maturin` |
| `ruff` | 0.16.4 (la version épinglée en CI) | Format et règles du Python (`ruff format`, `ruff check`), vérifiés par `check_format` |
| `mdbook` | 0.5.4 (la version épinglée en CI) | Génération de cette documentation |
| `mdbook-mermaid` | 0.17.1 (celle qui va avec mdBook 0.5) | Rendu des graphes (ex. [graphe des dépendances](developper/arborescence.md#graphe-des-dépendances-externes)) |

`cargo fmt` et `clippy` viennent avec la toolchain (`rustup component add
rustfmt clippy` si besoin).

Système (uniquement pour **builds avec l'API Python**, voir features plus bas) :

- **Linux** : installer les en-têtes Python — `python3-dev` (Debian/Ubuntu) ou
  `python3-devel` (Fedora/RHEL). `pyo3` en a besoin pour l'édition de liens.
- **Windows** : l'installateur officiel de Python inclut déjà les en-têtes.

Pour la feature `viz` sous Linux (donc pour `check_rust`, qui la compile) :
les en-têtes de fontconfig et freetype, que `plotters` lie pour le rendu du
texte — `libfontconfig1-dev` et `libfreetype6-dev` sous Debian/Ubuntu.

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
pip install --upgrade pip maturin pytest ruff==0.16.4
```

`ruff` est épinglé, comme en CI : ses règles par défaut et son formatage changent
d'une version à l'autre, et `check_format` les vérifie tous deux.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip maturin pytest ruff==0.16.4
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
python script/elements_figures.py  # figures des fiches d'éléments, non commitées
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
| `python-api` | tire les dépendances `pyo3` et `pyrucast-macros` + compile le code des `#[pyclass]`/`#[pyfunction]` (toute l'API Python) | `dep:pyo3`, `dep:pyrucast-macros` | rarement à la main ; activée automatiquement par `extension-module` et `stub-gen`. La désactiver (défaut) donne un crate Rust pur. |
| `extension-module` | `python-api` + dit à `pyo3` de **ne pas** se lier à `libpython` (l'interpréteur hôte la fournit au chargement du `.so`) | `python-api`, `pyo3/extension-module`, `spill` | systématique pour `maturin develop` / `maturin build`. |
| `spill` | allocateur qui place les grosses allocations dans des fichiers, sous Unix, quand `PYRUCAST_SPILL_DIR` est défini (voir [Calculs plus gros que la RAM](algorithmes/solveur.md#calculs-plus-gros-que-la-ram)) | `dep:libc` | activée par `extension-module` ; un programme Rust qui a son propre allocateur la laisse éteinte. |
| `viz` | export PNG/SVG/SVGZ (rendu CPU via `plotters`, compression `flate2`) | `dep:plotters`, `dep:flate2` | scripts sans écran, captures pour la doc, CI. |
| `viz-interactive` | fenêtre interactive `winit`/`softbuffer` (souris, repère d'orientation) | `viz`, `dep:winit`, `dep:softbuffer` | environnement graphique disponible. |
| `stub-gen` | `python-api` + binaire `stub_gen` qui produit le stub `.pyi` | `python-api`, `dep:pyo3-stub-gen`, `viz` | après modification des bindings, pour rafraîchir le stub vu par les IDE. |
| `abi3` | compile contre l'API stable de Python (≥ 3.11) : une seule roue pour tous les interpréteurs pris en charge | `pyo3/abi3-py311` | la CI de publication seulement. |

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

Tout le cœur (`containers`, `ops`, `models`, `archive`, `interrupt`, `handle`)
est disponible sans PyO3. Seule la couche `py` (les `#[pyclass]`) demande
`python-api`. Pour
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
`SubElementField.__getitem__`, `Node.__or__`/`__ror__`, `Matrix.__mul__`, et
les sommes `__add__`/`__sub__` de `Matrix` et `SubMatrix` (plus le `__neg__` du
bloc, logé dans le même bloc fermé). Il
sert aussi à corriger une **différence de vocabulaire** : `pyo3` regroupe les
comparaisons sous `__richcmp__`, qui n'existe pas côté Python — le stub des
champs déclare donc à la main `__ge__`/`__gt__`/`__le__`/`__lt__`, qui rendent
un masque 0/1 et non un booléen.

## Scripts « tout-en-un »

Le dossier `script/` contient deux niveaux d'automatisation. Tous activent (ou
créent) le venv automatiquement et s'arrêtent à la première erreur.

Autour d'eux gravitent quelques utilitaires : `dev.sh` / `dev.ps1` (build
minimal — module Python en release avec la visu interactive + stub, rien
d'autre), `run_examples.sh` / `run_examples.ps1` (exemples et tutoriel de bout
en bout, appelés par `check_examples`), `set_new_version.sh` (passe de version :
`check_all`, `check_clippy` et `cargo package`, puis reporter le numéro dans
`Cargo.toml` — seule déclaration de version du projet —, commit + tag : il ne
pousse rien),
`publish_macros.sh` (vérifie et publie `pyrucast-macros` sur crates.io, le seul
paquet que la CI ne publie pas),
`scaling.sh` (mesure de montée en charge du parallélisme),
`generate-tutoriel-figures.sh` (régénère les SVG du tutoriel, qui sont des
artefacts commités : ils demandent pyrucast compilé avec `viz`) et
`elements_figures.py` (dessine les SVG des fiches d'éléments, en Python pur ;
ils ne sont pas commités, et `check_doc`, `build` et la publication du book le
lancent avant `mdbook build`).

### `script/check_*.sh` — vérifications, en bloc ou à la carte

La vérification est découpée en **blocs indépendants**, chacun lançable seul :
les cinq que `check_all` enchaîne (`check_format`, `check_rust`,
`check_python`, `check_examples`, `check_doc`), et trois qu'il ne lance pas
(`check_clippy`, `check_gmsh`, `check_medcoupling`). Deux enchaîneurs les
regroupent, `check_all` et `check_quick`. Chaque script existe en bash (`.sh`,
Linux/macOS) et en PowerShell (`.ps1`, Windows), au comportement identique.

Le tableau ci-dessous donne, commande par commande, ce qu'elle vérifie, où
vivent les tests qu'elle exécute, ce qu'elle coûte, et quels scripts la
lancent. Les durées sont mesurées **après la modification d'un fichier de
`src/`**, le cas courant de la boucle de travail : les dépendances sont déjà
compilées, pyrucast et ses binaires de test sont recompilés. Un premier tour
après un changement de dépendances paie en plus leur compilation.

| commande | ce qu'elle teste | où vivent les tests | temps | quick | fmt | rust | py | ex | doc | all | version |
|---|---|---|--:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `cargo fmt --all --check` | le Rust est formaté | — | 3 s | ✓ | ✓ | | | | | ✓ | ✓ |
| `ruff format --check .` | le Python est formaté | — | 1 s | ✓ | ✓ | | | | | ✓ | ✓ |
| `ruff check .` | le Python passe les règles par défaut de ruff 0.16.4 (exceptions dans `pyproject.toml`, une par motif) | — | <1 s | ✓ | ✓ | | | | | ✓ | ✓ |
| `cargo clippy --workspace --all-targets` (`-D warnings`) | le build **Rust pur** compile, sans avertissement clippy — dont un `Result` qui ne peut pas échouer (`unnecessary_wraps`, déclaré dans `Cargo.toml`) | — | 8 s | ✓ | | ✓ | | | | ✓ | ✓ |
| `cargo test --workspace --features viz` | 1274 tests unitaires (dont 3 de `macros/`), 272 tests d'intégration, 1003 doctests | `#[cfg(test)]` dans 144 fichiers de `src/` et 1 de `macros/src/`, 47 fichiers `tests/*.rs`, commentaires `///` et `//!` | 276 s | ✓ | | ✓ | | | | ✓ | ✓ |
| `cargo build --features viz-interactive` | la fenêtre winit compile | aucun test — 959 lignes que rien n'exerce sans écran | 49 s | ✓ | | ✓ | | | | ✓ | ✓ |
| `cargo test --features spill --test spill` | l'allocateur `spill` place les grosses allocations dans des fichiers (Unix) | `tests/spill.rs`, 3 tests | 136 s | ✓ | | ✓ | | | | ✓ | ✓ |
| `maturin develop --release --features extension-module,viz` | l'extension Python s'installe, compilée en release comme la roue | — | 96 s | | | | ✓ | | | ✓ | ✓ |
| `python -m pytest` | l'API Python, unité par unité (763 tests, hors gmsh et medcoupling) | 100 fichiers `tests/python/test_*.py`, dont 27 `test_doc_*.py` qui sont aussi les sources des exemples du book | 78 s | | | | ✓ | | | ✓ | ✓ |
| `script/run_examples.sh` | des chaînes de calcul **de bout en bout** | 30 `examples/*.py`, 11 `tutoriel/*.py` | 69 s | | | | | ✓ | | ✓ | ✓ |
| `cargo doc --no-deps --lib` (`-D warnings`) | la rustdoc n'a aucun lien cassé | — | 15 s | | | | | | ✓ | ✓ | ✓ |
| `python script/elements_figures.py` | les figures des fiches d'éléments se dessinent, dans `book/src/elements/img/` vidé puis réécrit | — | <1 s | | | | | | ✓ | ✓ | ✓ |
| `mdbook build book` | le book se rend | `book/src/**` | 5 s | | | | | | ✓ | ✓ | ✓ |
| `python script/doc_lint.py` | les **neuf garde-fous** du book et des exemples | `book/src/**.md`, le book construit, la rustdoc, le module Python installé | 53 s | | | | | | ✓ | ✓ | ✓ |
| `cargo clippy --workspace --all-targets` × 4 jeux de features | aucun avertissement, `-D warnings` | — | 56 s | | | | | | | | ✓ |

Et les totaux, dans le même régime :

| script | contenu | temps |
|---|---|--:|
| `check_format` | formatage seul | 4 s |
| `check_rust` | cœur Rust | 469 s |
| `check_python` | liaison et tests Python | 174 s |
| `check_examples` | exemples et tutoriel | 69 s |
| `check_doc` | rustdoc, garde-fous, book | 73 s |
| `check_clippy` | quatre jeux de features, `-D warnings` | 56 s |
| `check_gmsh` | l'interface gmsh, 14 tests (exige `pip install gmsh`) | 5 s |
| `check_medcoupling` | l'interface MED, 37 tests (exige `pip install medcoupling`) | 2 s |
| **`check_quick`** | **formatage + Rust — la boucle de commit** | **~8 min** |
| `check_all` | les cinq blocs | ~13 min |
| `set_new_version` | `check_all` + `check_clippy`, puis `cargo package` (non chronométré) | ~14 min et plus |

`check_gmsh` ne fait pas partie de `check_all`, pour la même raison que
`check_clippy` n'en fait pas partie : on ne l'impose pas à la boucle
quotidienne. Un développeur qui ne touche pas à l'import gmsh n'a donc pas à
installer gmsh — les tests concernés se sautent proprement. On le lance en
touchant à cette interface, et le job `verify` de la CI le lance toujours.
Sous Linux, la roue `gmsh` embarque un `libgmsh` lié à OpenGL : il faut aussi
les paquets système `libglu1-mesa` et `libopengl0`.

`check_medcoupling` suit la même règle pour l'interface MED. medcoupling n'est
publié que pour Linux x86_64 et Windows : sur macOS, ce bloc ne peut pas
tourner.

Deux remarques que ces chiffres appellent.

**Le bloc Rust est surtout de la compilation.** Sur les 276 s de
`cargo test --workspace --features viz`, l'exécution n'en prend que 27 :
20 s de tests unitaires, 5 s de tests d'intégration, 1,5 s de doctests. Le
reste compile la bibliothèque, ses 47 binaires de tests d'intégration et les
doctests — que l'édition 2024 fusionne en un seul binaire au lieu d'en compiler
un par exemple. `tests/spill.rs` fait exception : ses 3 tests s'exécutent en
90 s.

**Les doctests tournent sous `viz`, jamais sous `viz-interactive`.** La couche
interactive tire 62 crates de plus — winit, Wayland, X11 — et chaque binaire de
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
formatage d'abord (il échoue en quelques secondes), le cœur Rust ensuite, puis
Python — qui *(ré)installe* l'extension dont les exemples ont besoin —, les
exemples, et la documentation en dernier, dont les garde-fous lisent le module
installé. Un bloc lancé seul qui a besoin
du module compilé le dit plutôt que d'échouer obscurément.

`script/check.sh` est un alias de `check_all.sh`. `check_quick` est le
raccourci du cas courant : il ne remplace pas `check_all`, il le précède — les
exemples, Python et la doc restent à passer avant de pousser.

`set_new_version.sh` **appelle** `check_all` plutôt que de recopier ses pas :
une copie pourrait diverger, et ne plus lancer les garde-fous de documentation
ou les exemples au moment précis où l'on pose un tag.

Trois points valent d'être notés :

- les **deux vérifications de format viennent en tête** — le script *vérifie*,
  il ne formate pas. Passer `cargo fmt` et `ruff format .` **avant** de le
  lancer, sinon il s'arrête au premier pas ;
- `run_examples` rejoue les **exemples et les scripts du tutoriel** de bout
  en bout. Ce sont des chaînes de calcul complètes : elles attrapent ce que les
  tests unitaires laissent passer, typiquement une méthode renommée dont plus
  personne ne se sert ;
- **ne jamais piper `check_all`** (`| tail`, `| grep`) : le code de retour
  devient celui du dernier maillon du tube, et un échec passe pour un succès.

### Les extraits du book sont du code exécuté

Aucune page du book ne possède de code : **tout bloc `rust` ou `python` est un
`{{#include}}`** pointant un test, un exemple ou une source. Le code affiché
est donc exécuté par `check_rust`, `check_python` ou `check_examples`, et il ne
peut pas diverger de l'API.

`check_doc` ne compile donc rien lui-même. Il construit la rustdoc et le book,
puis lance `script/doc_lint.py`, neuf vérifications de texte :

| garde-fou | ce qu'il tient |
|---|---|
| `includes` | chaque `{{#include}}` résout : fichier, ancre, texte non vide |
| `fences` | aucune page ne possède de code |
| `symboles` | la prose ne cite aucun symbole disparu |
| `doctests` | registre : tout item public porte un exemple, et la liste des exceptions ne peut que diminuer |
| `api-python` | registre : toute entrée Python est citée par un exemple exécuté |
| `equations` | chaque équation arrive intacte à MathJax |
| `reference` | toute fonction publique est rendue une fois dans la référence et listée une fois dans le sommaire ; les pages construites la contiennent (mdBook saute sans rien dire un préprocesseur qui ne démarre pas, par exemple si `python` désigne un Python 2) |
| `stub` | le stub commité dit ce que dit le module installé : noms, arguments, docstrings |
| `docstrings` | registre : une docstring suit la règle d'écriture |

Le premier est le plus important, parce que le mécanisme porteur n'est gardé par
rien d'autre : une ancre inexistante rend un bloc **vide**, avec un code de
retour 0 et sans un mot.

`mdbook test` n'est pas dans la chaîne : tous les blocs Rust du book sont
`rust,ignore` — c'est ce que le mécanisme d'inclusion impose —, si bien qu'il ne
compilerait rien. Un pas vert qui ne regarde rien coûte plus cher que son
absence.

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

1. **Vérification des prérequis** — `cargo`, `python` (≥ 3.11), création/activation
   du venv, installation de `maturin` et `pytest`, installation de `mdbook` si
   absent (`cargo install mdbook`).
2. **Compilation + tests** — `cargo build`, `cargo test` (unitaires +
   intégration + doctests), `cargo test --doc`, `cargo test --features viz`.
3. **Module Python avec visu interactive** — `maturin develop --release
   --features extension-module,viz-interactive`, puis `pytest`.
4. **Documentation** — `cargo doc` (référence Rust), régénération du stub
   `python/pyrucast/_pyrucast/__init__.pyi`, figures des fiches d'éléments
   (`script/elements_figures.py`), `mdbook build`, et un export **pydoc HTML**
   de l'API Python (`target/python-doc/pyrucast.html`).
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
book ou aux sources Rust, le workflow dessine les figures des fiches d'éléments
(`python script/elements_figures.py`), build le book (`mdbook build book`) **et**
la doc Rust (`cargo doc --no-deps --lib`), assemble un site combiné — le book à
la racine, la rustdoc sous `rust/` — et le déploie directement sur GitHub Pages
via `actions/deploy-pages` (pas de branche `pages`, pas de token à gérer : le
déploiement utilise les permissions natives du workflow). Il peut aussi être
déclenché à la main (*workflow_dispatch*) depuis l'onglet *Actions*.

Prérequis à configurer une fois côté GitHub (interface web) :

1. **Settings → Pages → Build and deployment → Source** : sélectionner
   *GitHub Actions*.

Les runners GitHub sont hébergés et gratuits sur dépôt public — pas de runner
à enregistrer soi-même.

## Journal des versions et publication

Poser un tag `vX.Y.Z` déclenche `.github/workflows/release.yml`, qui publie le
crate sur crates.io, les wheels et la sdist sur PyPI, puis crée la **Release
GitHub** dont les notes sont écrites automatiquement à partir des messages de
commit.

### La convention de message de commit

C'est la seule obligation que cette automatisation ajoute. Un sujet s'écrit
`type(portée): sujet`, avec un `!` avant le deux-points quand le changement
casse l'API :

| Préfixe | Section du journal |
|---|---|
| `type(portée)!:` — n'importe quel type suivi de `!` | Ruptures d'API, en tête |
| `feat` | Nouveautés |
| `fix` | Corrections |
| `perf` | Performances |
| `refactor` | Remaniements |
| `docs` | Documentation |
| `test` | Tests |
| `build`, `ci`, `chore` | Compilation et outillage |
| `style` | Style |
| tout le reste | Divers |

Le **sujet est repris tel quel** dans les notes : il est écrit une fois, au
moment du commit, et jamais retouché ensuite. Le corps du message, lui, n'est
jamais publié — il reste pour qui lit `git log`. Deux cas particuliers sont
traités par la configuration : `chore: version X.Y.Z` est sauté, et les commits
`docs(rustdoc)` sont remplacés par une seule ligne qui en donne le nombre, sans
quoi ils noieraient tout le reste.

### git-cliff n'est pas un prérequis

Le tri est fait par [git-cliff](https://git-cliff.org/), piloté par `cliff.toml`
à la racine. **Rien à installer** : ni pour développer, ni pour publier. L'outil
est téléchargé par le workflow, dans une version épinglée, comme mdBook l'est
pour le book. Le tableau des prérequis en tête de page ne gagne donc aucune
ligne.

Pour relire les notes avant de poser le tag, si on le souhaite :

```sh
cargo install git-cliff
git-cliff --unreleased --tag vX.Y.Z   # ce que la prochaine Release dira
```

Et si les notes publiées déplaisent, la Release s'édite dans l'interface
GitHub : ni workflow à rejouer, ni tag à déplacer.

### Une seule déclaration de version

`Cargo.toml` porte la version, et lui seul. `pyproject.toml` la lit de là
(`dynamic = ["version"]`), et `pyrucast.__version__` en vient
(`src/lib.rs`, `env!("CARGO_PKG_VERSION")`). Le crate, la wheel et le module
importé descendent donc du même champ : ils ne *peuvent pas* diverger.

Le workflow refuse de démarrer si `pyproject.toml` redéclare une version — une
version statique l'emporterait sur `Cargo.toml` chez maturin et partirait seule
sur PyPI.

De là la règle qui rend la page `releases/` lisible :

> **La Release GitHub existe si et seulement si crates.io et PyPI portent
> réellement cette version, et si les artefacts joints portent les étiquettes de
> compatibilité attendues.**

Le dernier job constate au lieu de supposer : il interroge les deux registres
jusqu'à les y trouver, et vérifie sur les noms de fichiers qu'il y a bien trois
wheels `cp311-abi3` et une sdist. `cargo publish` comme l'action PyPI savent
sauter une version déjà présente ; ce silence est utile pour rejouer un job, et
dangereux partout ailleurs.

### Le tag est le seul point de contrôle

Les push sur `master` ne sont **pas vérifiés** : la vérification se fait là où
elle est irréversible, au tag. Le job `verify` précède tout le reste et lance
les cinq blocs de `check_all`, `check_gmsh` et `check_medcoupling` (après
`check_python`, qui a construit le module), puis `check_clippy` —
`set_new_version.sh` lance en local les cinq blocs de `check_all` et
`check_clippy`.

Un bloc par étape nommée, et non `check_all.sh` en une fois : l'interface
Actions montre alors laquelle rougit sans qu'il faille dérouler tout le journal.
Ce sont les mêmes scripts, appelés un à un plutôt que par leur boucle — et une
étape vérifie que la liste des cinq blocs de la CI est celle de
`check_all.sh`, faute de quoi un sixième bloc ne serait jamais lancé au moment
de publier.

Un seul job, parce que l'ordre contraint : `doc_lint.py` importe `pyrucast`,
donc `check_doc` exige que `check_python` ait construit le module.

Le job reconstitue l'environnement de développement — toolchain Rust, en-têtes
fontconfig, freetype et expat, les bibliothèques OpenGL qu'exige `gmsh`, mdBook
et son préprocesseur mermaid, un venv avec maturin, pytest, ruff, gmsh et
medcoupling. Compter 12 à 15 minutes avec le cache Cargo (les cinq
dernières publications), davantage quand ce cache est à reconstruire.

`set_new_version.sh` garde malgré tout son quart d'heure : c'est l'échec
**rapide**, avant que le tag existe. Découvrir la panne après coup obligerait à
déplacer un tag, ou à brûler un numéro.

### `pyrucast-macros` se publie à la main, et **avant**

Le workspace porte deux paquets, et `release.yml` n'en publie qu'un : le paquet
racine. `pyrucast` dépend pourtant de `pyrucast-macros` en `path` **et** en
version, donc `cargo package` comme `cargo publish` cherchent ce numéro dans
l'index de crates.io. Tant qu'il n'y est pas, les deux échouent sur
`no matching package named pyrucast-macros found` — y compris
`set_new_version.sh`, à son étape `cargo package`, avant même de poser le tag.

L'outillage de macros bouge bien plus lentement que la bibliothèque, et sa
version est délibérément décorrélée de la sienne : le publier à chaque tag
n'aurait aucun sens. Il se publie donc à la main, à chaque fois qu'il change :

```bash
bash script/publish_macros.sh          # publie la version courante
bash script/publish_macros.sh 0.1.1    # la passe à 0.1.1, puis publie
```

Le script vérifie la crate seule (format, clippy `-D warnings`, tests, rustdoc)
puis son **consommateur** — une macro procédurale ne prouve rien tant qu'un
appelant ne l'a pas expansée —, empaquette, fait un `--dry-run`, et ne publie
qu'après confirmation. Avec un numéro, il le reporte des deux côtés : dans
`macros/Cargo.toml` et dans la dépendance de `Cargo.toml`, qui ne peuvent pas
diverger sans que la publication cesse de débloquer quoi que ce soit. Il attend
enfin que l'index serve la version, faute de quoi le `cargo package` suivant
échouerait sur le même `no matching package`.

### On construit tout avant de publier

L'ordre des jobs compte autant que l'attestation finale. **On construit tout
avant de publier quoi que ce soit** : `crates-io` attend la sdist et les wheels,
parce que `cargo publish` est une porte à sens unique et qu'un numéro ne se
republie jamais. Publier le crate pendant qu'une cible de la matrice échoue
laisserait sur crates.io une version que PyPI ne recevrait jamais.

### Ce que porte chaque distribution

`pip install pyrucast` prend une wheel quand il en existe une pour la
plateforme, et retombe sinon sur la sdist, qu'il compile. Les deux ne portent
pas la même chose :

| | Plateformes | Features compilées | Prérequis |
|---|---|---|---|
| **wheel** `cp311-abi3` | Linux x86_64 (manylinux2014), Windows x86_64, macOS universal2 | `extension-module`, `spill`, `viz`, `viz-interactive`, `abi3` | aucun — Python ≥ 3.11 |
| **sdist** `.tar.gz` | toutes les autres (ARM, musl, BSD) | `extension-module`, `spill` | rustup, et la compilation du crate |

Il n'y a pas de wheel Linux aarch64 : `viz` lie fontconfig et freetype, et
`pkg-config` refuse par principe de fonctionner en compilation croisée. La
construire supposerait d'émuler un conteneur aarch64 — trois minutes de
compilation x86_64 en deviennent quinze à trente.

Une installation depuis la sdist n'a donc **pas la visualisation** :
`mesh.plot()` n'existe pas. C'est assumé — compiler `viz` exigerait fontconfig
et freetype sur la machine cible — mais ce n'est pas silencieux :

{{#include ../../tests/python/test_smoke.py:features}}

`__features__` liste ce que *ce* binaire porte : `('python-api',
'extension-module', 'viz', 'viz-interactive', 'abi3', 'spill')` sur une wheel
publiée, `('python-api', 'extension-module', 'spill')` depuis la sdist. Le test
qui précède ne se contente pas de lire la liste — un second vérifie qu'elle
**dit vrai** plutôt que d'être recopiée à la main : la présence de `viz` doit
équivaloir à celle de `Mesh.plot`.

> **Pré-versions.** Pour un tag `v0.4.0-rc1`, crates.io garde `0.4.0-rc1` là où
> PyPI normalise en `0.4.0rc1` : le workflow rapproche les deux formes et refuse
> celles qu'il ne sait pas superposer. `test_version_exposed`, lui, n'accepte
> que `X.Y.Z` : une pré-version le fait échouer.

## Dépannage rapide

- *`error: failed to run the Python interpreter at ...`* lors d'un
  `cargo build` : le venv n'est pas activé, ou un `VIRTUAL_ENV` obsolète pointe
  vers un chemin invalide. Réactivez le venv du projet.
- *`No module named 'pyrucast'`* lors de `pytest` : `maturin develop` n'a pas
  déposé le module. Vérifier que `VIRTUAL_ENV` est défini, puis relancer
  `maturin develop`.
- *`mdbook: command not found`* : installer `mdbook` via `cargo install --locked
  mdbook --version 0.5.4` ou un binaire publié.
- *`The "mermaid" preprocessor exited unsuccessfully`* (ou graphes affichés en
  bloc de code brut) : installer le préprocesseur via `cargo install --locked
  mdbook-mermaid --version 0.17.1`. Il doit être sur le `PATH` au moment de
  `mdbook build`, et aller avec mdBook : la 0.17 parle le protocole de
  mdBook 0.5, la 0.16 celui de la 0.4.
