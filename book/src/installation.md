# Installation et démarrage rapide

Cette page suffit pour **utiliser pyrucast en Python** : quelques commandes,
un premier script, et de quoi vérifier que tout fonctionne. Pour développer
sur la librairie (tests Rust, doctests, génération de la documentation,
*features* Cargo), voir [Compilation et tests](compilation.md).

> **Rust pur ?** pyrucast s'utilise aussi comme **bibliothèque Rust sans
> Python** : `pyo3` y est une dépendance optionnelle, donc un build par défaut
> ne tire ni `pyo3` ni `libpython`. Voir
> [Usage en Rust pur](compilation.md#usage-en-rust-pur).

## Prérequis

- **Rust** stable, installé via [`rustup`](https://rustup.rs).
- **Python** ≥ 3.9 (pour l'API Python ; inutile en [Rust pur](compilation.md#usage-en-rust-pur)).
- **Linux uniquement** : les en-têtes Python — `python3-dev` (Debian/Ubuntu)
  ou `python3-devel` (Fedora/RHEL). `pyo3` en a besoin pour l'édition de
  liens ; sur Windows l'installateur officiel les inclut déjà.

## Compilation et installation

À la racine du dépôt cloné. `pyo3` localise l'interpréteur Python via la
variable `VIRTUAL_ENV` : **activez toujours le venv** avant `cargo` ou
`maturin`, sinon la compilation échoue avec `error: failed to run the Python
interpreter at ...`.

### Linux / macOS (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip maturin
maturin develop --release --features extension-module,viz-interactive
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip maturin
maturin develop --release --features extension-module,viz-interactive
```

`maturin develop` compile le module Rust et l'installe dans le venv (mode
*editable*). L'option `--release` compile en optimisé : recommandé pour tout
usage réel (un build debug est typiquement 10× plus lent à l'exécution).
Après toute modification du Rust, relancer simplement `maturin develop
--release` — pas besoin de réinstaller ni de réactiver le venv tant qu'il
reste actif. Les options  --features extension-module,viz-interactive 
permettent d'activer la visualisation interactive.

## Vérification immédiate

```bash
python -c "import pyrucast; c = pyrucast.Coords(2); n = c.add_node([0.0, 0.0]); print(c); print(n)"
```

Sortie attendue :

```text
Coords: dim=2, configs=1 (active="default"), nodes=1 (0 collected), permutation: identity
<Node #0>
```

## Premier script

Un exemple minimal — une `Coords` 2D, deux nœuds, un maillage POI1 :

```python
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])

mesh = pyrucast.Mesh(c, "SEG2")  # un sous-maillage
mesh.unit().add_cell([a, b])
mesh.plot()
print(c)
print(mesh)  # Mesh: 1 submesh(es), 2 cell(s) total
mesh.dump()
```

À partir d'ici, le chapitre [Introduction](introduction.md) présente le
modèle d'objets, et la section [Objets](objets.md) détaille chaque brique
(coordonnées, maillage, champs, modèle physique…). Les chapitres
[Conduction thermique](thermique.md) et [Mécanique](mecanique.md) déroulent
des problèmes complets de bout en bout.

## Aller plus loin

- **Référence API Rust** : `cargo doc --no-deps --lib --open`.
- **Exemples Python** complets et exécutables : dossier `examples/` du dépôt
  (thermique, treillis, élasticité, poutres…).
- **Développer sur pyrucast** (tests, doc, *features*) :
  [Compilation et tests](compilation.md).
- **Tout construire d'un coup** (prérequis + compilation + tests + les trois
  documentations + module avec visu interactive) : `bash script/build.sh`
  (Linux/macOS) ou `.\script\build.ps1` (Windows) — voir
  [Scripts « tout-en-un »](compilation.md#scripts--tout-en-un-).
