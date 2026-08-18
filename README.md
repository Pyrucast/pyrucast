# pyrucast

Bibliothèque d'éléments finis en Rust, inspirée de cast3m, exposée à Python —
et utilisable telle quelle en **Rust pur** (`pyo3` est une dépendance
optionnelle : un build par défaut ne tire ni `pyo3` ni `libpython`).

- **Documentation complète** : le book mdbook dans [`book/`](book/) (architecture,
  modèle mémoire, maillage, champs, physiques…) — `mdbook build book` pour le HTML.
- **Référence API Rust** : `cargo doc --no-deps --lib --open`.

## Prérequis

| Outil | Version | Installation |
|---|---|---|
| Rust | stable (édition 2021) | [`rustup`](https://rustup.rs) |
| Python | ≥ 3.9 | *API Python uniquement* — inutile en Rust pur |
| En-têtes Python | — | *API Python uniquement* — **Linux** : `python3-dev` (Debian/Ubuntu) ou `python3-devel` (Fedora/RHEL). Windows : inclus dans l'installateur officiel. |

## Usage en Rust pur (sans Python)

```toml
[dependencies]
pyrucast = { git = "…", default-features = false }   # pas de pyo3, pas de libpython
```

```rust
use pyrucast::atoms::ElementType;
let mesh = pyrucast::ops::mesh::triangulate_surface(&contour, ElementType::TRI3, Some(1.0))?;
```

`cargo build` / `cargo test` (sans feature) compilent le cœur en Rust pur —
ni Python ni venv requis. L'API Python (`#[pyclass]`) vit derrière la feature
`python-api`, activée automatiquement par `maturin`.

## Compilation dans un venv (API Python)

Pour les builds **avec l'API Python** (`maturin`, ou `cargo --features
python-api`), `pyo3` localise l'interpréteur via `VIRTUAL_ENV` : **activez
toujours le venv avant `cargo` ou `maturin`**, sinon la compilation échoue avec
`error: failed to run the Python interpreter at ...`. (Un `cargo build` pur n'a
pas cette contrainte.)

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip maturin
maturin develop --release
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip maturin
maturin develop --release
```

`maturin develop` compile le module et l'installe dans le venv. Vérification :

```bash
python -c "import pyrucast; c = pyrucast.Coords(2); print(c)"
```

Après toute modification du Rust, relancer simplement `maturin develop --release`.

## Développement

```bash
pip install pytest ruff     # en plus de maturin, dans le venv
cargo build                 # cœur Rust pur (sans pyo3 ; pas besoin du venv)
cargo test                  # tests unitaires + intégration + doctests (Rust pur)
maturin develop && python -m pytest   # tests Python
cargo fmt && ruff format .  # formatage standard (Rust + Python)
bash script/check_all.sh    # enchaîne toutes les vérifications (formatage inclus)
bash script/check_doc.sh    # ou un seul bloc : format / rust / python / examples / doc
```

Le chapitre [Compilation et tests](book/src/compilation.md) du book détaille les
features Cargo (`viz`, `viz-interactive`, `stub-gen`…), la génération du stub
`python/pyrucast/_pyrucast/__init__.pyi` et le dépannage courant.

## Licence

Distribué sous licence **[Mozilla Public License 2.0](LICENSE)** (MPL-2.0).

C'est un copyleft *au fichier* : toute modification d'un fichier source de
pyrucast doit être republiée sous MPL, mais on peut librement construire du code
propriétaire **au-dessus** de la bibliothèque et l'y combiner. La MPL est par
ailleurs compatible GPL/CeCILL (« Secondary Licenses », §3.3).
