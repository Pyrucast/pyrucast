# pyrucast

Bibliothèque d'éléments finis en Rust, inspirée de cast3m, exposée à Python.

- **Documentation complète** : le book mdbook dans [`book/`](book/) (architecture,
  modèle mémoire, maillage, champs, physiques…) — `mdbook build book` pour le HTML.
- **Référence API Rust** : `cargo doc --no-deps --lib --open`.

## Prérequis

| Outil | Version | Installation |
|---|---|---|
| Rust | stable (édition 2021) | [`rustup`](https://rustup.rs) |
| Python | ≥ 3.9 | python.org ou gestionnaire de paquets |
| En-têtes Python | — | **Linux uniquement** : `python3-dev` (Debian/Ubuntu) ou `python3-devel` (Fedora/RHEL). Windows : inclus dans l'installateur officiel. |

## Compilation dans un venv

`pyo3` localise l'interpréteur Python via la variable `VIRTUAL_ENV` :
**activez toujours le venv avant `cargo` ou `maturin`**, sinon la compilation
échoue avec `error: failed to run the Python interpreter at ...`.

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
pip install pytest          # en plus de maturin, dans le venv
cargo build                 # cœur Rust seul (le venv doit être activé)
cargo test                  # tests unitaires + intégration + doctests
maturin develop && python -m pytest   # tests Python
bash script/check.sh        # enchaîne toutes les vérifications
```

Le chapitre [Compilation et tests](book/src/compilation.md) du book détaille les
features Cargo (`viz`, `viz-interactive`, `stub-gen`…), la génération du stub
`pyrucast.pyi` et le dépannage courant.
