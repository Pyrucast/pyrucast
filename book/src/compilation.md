# Compilation et tests

Ce chapitre décrit la construction de pyrucast depuis les sources, sur Windows et Linux.

- **Vous voulez juste utiliser la librairie en Python** → suivez la section [Démarrage rapide](#démarrage-rapide--utiliser-pyrucast-en-python) ci-dessous (~4 commandes).
- **Vous voulez développer dessus** (tests, doc, modifications du code Rust) → voir [Installation détaillée](#prérequis) plus bas. Tout est aussi scriptable en une commande (`scripts/check.ps1` ou `scripts/check.sh`).

## Démarrage rapide — utiliser pyrucast en Python

### Pré-requis

- Rust stable, installé via [`rustup`](https://rustup.rs).
- Python ≥ 3.9.
- **Linux uniquement** : en-têtes Python — `python3-dev` (Debian/Ubuntu) ou `python3-devel` (Fedora/RHEL).

### Compilation et installation

À la racine du dépôt cloné, dans un terminal ouvert sur le projet.

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip maturin
maturin develop --release
```

#### Linux / macOS (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip maturin
maturin develop --release
```

L'option `--release` compile en mode optimisé : recommandé pour tout usage réel (un build de debug est typiquement 10× plus lent à l'exécution).

### Vérification immédiate

```bash
python -c "import pyrucast; c = pyrucast.Configuration(2); n = c.add_node([0.0, 0.0]); print(c); print(n)"
```

Sortie attendue :

```text
Configuration: dim=2, sets=1 (active="default"), nodes=1 (0 collected), permutation: identity
<Node #0>
```

Le module `pyrucast` est installé dans le venv. Tant que le venv est activé (`Activate.ps1` sur Windows, `source .venv/bin/activate` sur Linux), `import pyrucast` fonctionne dans n'importe quel script Python.

### Premier exemple complet

Un script minimal qui crée une `Configuration` 2D, deux nœuds, un sous-maillage et un maillage :

```python
import pyrucast

c = pyrucast.Configuration(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])

sm = pyrucast.SubMesh(c, "POI1")     # sous-maillage = liste de nœuds
sm.add_cell([a.id])
sm.add_cell([b.id])

mesh = pyrucast.Mesh(c)
mesh.add_submesh(sm)

print(c)
print(mesh)
```

### Premier exemple complet en Rust

L'exemple équivalent au script Python ci-dessus, compilable en tant que binary ou test d'intégration :

```rust,ignore
use pyrucast::configuration::Configuration;
use pyrucast::element_type::ElementType;
use pyrucast::mesh::{Mesh, SubMesh};
use pyrucast::node::Node;
use pyrucast::store::insert;

fn main() {
    let cfg = insert(Configuration::new(2).unwrap());
    let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
    let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();

    let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
    sm.add_cell(&[a.id()]).unwrap();
    sm.add_cell(&[b.id()]).unwrap();

    let sm_h = insert(sm);
    let mut mesh = Mesh::new(cfg);
    mesh.add_submesh(sm_h).unwrap();

    println!("{}", mesh); // Mesh: 1 submesh(es), 2 cell(s) total
}
```

### Recompilation après modification du Rust

```bash
maturin develop --release
```

Le module Python du venv est remplacé en place ; pas besoin de réinstaller. Pas non plus besoin de relancer le venv tant qu'il reste activé.

---

## Installation détaillée (développement, tests, documentation)

Cette section couvre l'installation complète pour contribuer à pyrucast : tests Rust et Python, doctests, génération de la documentation théorique.

## Prérequis

| Outil | Version | Rôle |
|---|---|---|
| Rust (via `rustup`) | stable récent, édition 2021 | Compilation du cœur |
| Python | ≥ 3.9 (3.13 testé) | API Python et `maturin` |
| `mdbook` | ≥ 0.4 | Génération de cette documentation |

Système :

- **Linux** : installer les en-têtes Python — `python3-dev` (Debian/Ubuntu) ou `python3-devel` (Fedora/RHEL). `pyo3` en a besoin pour l'édition de liens.
- **Windows** : l'installateur officiel de Python inclut déjà les en-têtes ; aucune étape supplémentaire.

## Mise en place du venv et des outils Python

Le venv (`.venv` à la racine du projet) héberge `maturin` et `pytest`, et sert d'environnement cible à `maturin develop`.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip maturin pytest
```

### Linux / macOS (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip maturin pytest
```

> **Important.** `pyo3` cherche l'interpréteur Python via la variable d'environnement `VIRTUAL_ENV`. **Activez toujours le venv** avant d'invoquer `cargo` ou `maturin`, sinon `cargo build` peut échouer en tentant d'utiliser un interpréteur introuvable.

## Compilation et tests, étape par étape

Les commandes suivantes supposent le venv activé. Elles sont identiques sous Windows et Linux.

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

`maturin develop` installe le module en mode *editable* dans `.venv/Lib/site-packages/pyrucast/` (Windows) ou `.venv/lib/python3.X/site-packages/pyrucast/` (Linux/macOS). Toute modification du Rust est intégrée au prochain `maturin develop`.

### Documentation Rust (rustdoc)

`cargo doc --no-deps --lib` génère la **référence API Rust** à partir des commentaires `///` du code source. Le résultat est écrit dans `target/doc/pyrucast/index.html`.

- `--no-deps` exclut la doc des dépendances (`pyo3`, `serde`, `bincode`) — gain de temps majeur.
- `--lib` limite à la crate `pyrucast` (sans les tests d'intégration).
- Ajouter `--open` ouvre la page dans le navigateur par défaut à la fin de la génération.

Rustdoc et mdbook sont complémentaires : rustdoc couvre la **référence par item** (modules, types, fonctions) ; le présent mdbook couvre les **principes, l'architecture et les exemples transverses**.

## Script « tout-en-un »

Pour enchaîner toutes les vérifications dans l'ordre :

### Windows

```powershell
.\scripts\check.ps1
```

### Linux / macOS

```bash
bash scripts/check.sh
```

Les deux scripts activent le venv, puis exécutent successivement : `cargo test`, `cargo test --doc`, `maturin develop`, `pytest`, `cargo doc --no-deps --lib`, `mdbook build`, `mdbook test`. Toute commande qui échoue interrompt le script.

## Dépannage rapide

- *`error: failed to run the Python interpreter at ...`* lors d'un `cargo build` : le venv n'est pas activé, ou un `VIRTUAL_ENV` obsolète pointe vers un chemin invalide. Réactivez le venv du projet.
- *`No module named 'pyrucast'`* lors de `pytest` : `maturin develop` n'a pas déposé le module dans le venv. Vérifier que `VIRTUAL_ENV` est défini, puis relancer `maturin develop`.
- *`mdbook: command not found`* : installer `mdbook` via `cargo install mdbook` ou un binaire publié.
