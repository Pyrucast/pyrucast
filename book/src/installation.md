# Installation et démarrage rapide

Cette page suffit pour **utiliser pyrucast en Python** : l'installer, vérifier
qu'il fonctionne, écrire un premier script. Deux voies : la **version
précompilée**, sans rien compiler, ou la **compilation** depuis les sources.
Pour développer sur la librairie (tests Rust, doctests, génération de la
documentation, *features* Cargo), voir [Compilation et tests](compilation.md).

> **Rust pur ?** pyrucast s'utilise aussi comme **bibliothèque Rust sans
> Python** : `pyo3` y est une dépendance optionnelle, donc un build par défaut
> ne tire ni `pyo3` ni `libpython`. Voir
> [Usage en Rust pur](compilation.md#usage-en-rust-pur).

## Version précompilée

```bash
pip install pyrucast
```

et pour passer à la dernière version :

```bash
pip install --upgrade pyrucast
```

PyPI distribue pyrucast en *wheels* — des paquets **précompilés**, prêts à
installer — pour trois plateformes :

| Système | Processeur |
|---|---|
| Linux, glibc ≥ 2.17 (*manylinux2014*) | x86_64 |
| Windows 64 bits | x86_64 |
| macOS ≥ 10.12 (Intel), ≥ 11 (Apple Silicon) | x86_64 et arm64 (*universal2*) |

Seul prérequis : **Python ≥ 3.11**. Ni Rust ni compilateur. La wheel comprend
la visualisation (`plot`), fenêtre interactive incluse.

Sur une autre plateforme (Linux ARM, musl, BSD…), `pip` ne trouve pas de wheel
et compile les sources : il faut alors Rust (voir ci-dessous), et le module
obtenu n'a **pas** la visualisation — voir
[Ce que porte chaque distribution](compilation.md#ce-que-porte-chaque-distribution).

## Compilation

Pour suivre la version en développement, ou pour compiler avec la
visualisation sur une plateforme sans wheel.

### Prérequis

- **Rust** stable, installé via [`rustup`](https://rustup.rs).
- **Python** ≥ 3.11.
- **Linux uniquement** : la visualisation lie fontconfig et freetype, il faut
  leurs en-têtes — `pkg-config libfontconfig1-dev libfreetype6-dev`
  (Debian/Ubuntu) ou `pkg-config fontconfig-devel freetype-devel`
  (Fedora/RHEL). Windows et macOS passent par les polices du système.

### Compiler et installer

À la racine du dépôt cloné
(`git clone https://github.com/Pyrucast/pyrucast`), dans un environnement
virtuel : `maturin develop` refuse de s'en passer (« Couldn't find a
virtualenv or conda environment »).

#### Linux / macOS (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip maturin
maturin develop --release --features extension-module,viz-interactive
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip maturin
maturin develop --release --features extension-module,viz-interactive
```

`maturin develop` compile le module Rust et l'installe dans le venv (mode
*editable*). `--release` compile en optimisé, ce qu'il faut pour tout usage
réel : un build de débogage est bien plus lent à l'exécution. Les features
font le reste : `extension-module` fait du binaire un module Python, et
`viz-interactive` ajoute la visualisation, fenêtre interactive comprise (elle
inclut `viz`, l'export PNG/SVG).

Après toute modification du Rust, relancer **la même commande** : sans
`--features …viz-interactive`, le module recompilé perdrait la visualisation.

## Vérifier l'installation

```bash
python -c "import pyrucast; c = pyrucast.Coords(2); n = c.add_node([0.0, 0.0]); print(c); print(n)"
```

Sortie attendue :

```text
Coords: dim=2, configs=1 (active="default"), nodes=1 (0 collected), permutation: identity
<Node #0>
```

`pyrucast.__features__` dit ce que porte le module installé : `viz` y figure
quand la visualisation est compilée.

## Premier script

Une ligne de 10 éléments entre deux nœuds, par la **fonction libre** `line` du
module `pyrucast.mesh` :

```python
{{#include ../../tests/python/test_doc_demarrage.py:premier_script}}
```

Les opérateurs de pyrucast sont des fonctions libres, rangées par ce qu'elles
produisent : tout ce qui rend un maillage est dans `pyrucast.mesh`. Quand une
fonction prend en premier l'objet qu'elle transforme et rend un nouvel objet,
elle existe en règle générale **aussi en méthode** de cet objet — les deux
écritures sont équivalentes :

```python
{{#include ../../tests/python/test_doc_demarrage.py:methode}}
```

La méthode sert à **composer**. Pour la bordure d'une plaque tirée de la ligne,
les fonctions libres s'imbriquent et se lisent de l'intérieur ; les méthodes
s'enchaînent et se lisent dans l'ordre des opérations :

```python
{{#include ../../tests/python/test_doc_demarrage.py:composition}}
```

La règle complète — quelles fonctions ont leur méthode, et pourquoi — est
dans [Conventions](conventions.md#le-verbe-exposé-aussi-en-méthode).

Enfin, pour voir la plaque (une fenêtre interactive s'ouvre, s'il y a un
écran) :

```python
{{#include ../../tests/python/test_doc_demarrage.py:apercu}}
```

À partir d'ici, le chapitre [Introduction](introduction.md) présente le
modèle d'objets, et la section [Objets](objets.md) détaille chaque brique
(coordonnées, maillage, champs, modèle physique…). Les chapitres
[Conduction thermique](thermique.md) et [Mécanique](mecanique.md) déroulent
des problèmes complets de bout en bout.

## Aller plus loin

- **Formation** : un parcours guidé, du maillage au calcul thermique et
  mécanique — [Formation débutant](formation/debutant.md).
- **Référence API Rust** : publiée sur
  <https://pyrucast.github.io/pyrucast/rust/>, ou en local avec
  `cargo doc --no-deps --lib --open`.
- **Exemples Python** complets et exécutables : dossier `examples/` du dépôt
  (thermique, treillis, élasticité, poutres…).
- **Développer sur pyrucast** (tests, doc, *features*) :
  [Compilation et tests](compilation.md).
- **Tout construire d'un coup** (prérequis + compilation + tests + les trois
  documentations + module avec visu interactive) : `bash script/build.sh`
  (Linux/macOS) ou `.\script\build.ps1` (Windows) — voir
  [Scripts « tout-en-un »](compilation.md#scripts--tout-en-un-).
