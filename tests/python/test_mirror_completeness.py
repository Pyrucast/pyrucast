"""Garde-fou du miroir Rust → Python.

`CONVENTIONS.md` § « Règle Rust → Python : miroir 1:1 » : toute fonction libre
d'`ops` est exposée à Python, sous le même nom, dans le sous-module du même
module Rust. La seule asymétrie tolérée est la **non-exposition des
constructeurs `Sub*`**.

Ce test lit la **surface publique** de chaque module d'`ops` — les `pub use`
et les `pub fn` de son fichier racine, la liste que le développeur tient déjà
pour Rust — et vérifie que chaque nom se retrouve dans le sous-module Python
correspondant. C'est le trou par lequel
`ops::element_field::frame_deformation` est restée sans binding : le garde-fou
des méthodes (`test_method_exposure.py`) lit le **stub**, donc il ne peut pas
voir une fonction qui n'y est pas. Les deux tests sont complémentaires — l'un
garde la projection Rust → Python, l'autre la projection fonction → méthode.

Une fonction volontairement non exposée s'écrit dans `RUST_ONLY`, avec sa
raison.
"""

import pathlib
import re

import pyrucast

ROOT = pathlib.Path(__file__).resolve().parents[2]
OPS = ROOT / "src" / "ops"

# Modules Rust dont les fonctions n'ont pas (encore) de sous-module Python.
NO_PYTHON_MODULE = {
    "geom": "requêtes géométriques encore internes (locate/project/nearest)",
}

# Fonctions Rust délibérément non exposées, avec la raison.
RUST_ONLY = {
    # Variantes `*_cancellable` : côté Python, l'interruption est branchée dans
    # le wrapper de la fonction nominale, pas exposée comme une fonction à part.
    "pave_surface_cancellable": "l'interruption est câblée dans le wrapper nominal",
    "pave_volume_cancellable": "l'interruption est câblée dans le wrapper nominal",
    "triangulate_surface_cancellable": "l'interruption est câblée dans le wrapper nominal",
    "triangulate_volume_cancellable": "l'interruption est câblée dans le wrapper nominal",
    # Détails d'implémentation partagés, pas des opérateurs.
    "check_unique_component_per_support": "garde interne de la fusion, pas un opérateur",
    "assemble_kind": "moteur commun des assembleurs, pas une opération d'usage",
    "select_sub_cells": "les vues `Sub*` passent par le dispatch de `select`",
    "select_sub_nodes": "les vues `Sub*` passent par le dispatch de `select`",
    "mask_sub_cells": "les vues `Sub*` passent par le dispatch de `mask`",
    "mask_sub_nodes": "les vues `Sub*` passent par le dispatch de `mask`",
    "mask_cells": "dispatch par type dans `field.mask`",
    "mask_nodes": "dispatch par type dans `field.mask`",
    "select_cells": "dispatch par type dans `mesh.select`",
    "select_nodes": "dispatch par type dans `mesh.select`",
    "integral_element": "dispatch par type dans `measure.integral`",
    "consolidate": "exposé sous le nom court dans son sous-module Python",
    "set": "exposé sous le nom court dans `pyrucast.coords`",
    "solve": "exposé par variante (`solve`, `solve_eliminate`, `solve_unilateral`)",
    "Band": "type de valeur, transporté par les arguments `ge`/`gt`/`le`/`lt`",
    "FluxDensity": "type de valeur, transporté par l'argument `density`",
    "Location": "type de retour de `geom`, non exposé",
    "Projection": "type de retour de `geom`, non exposé",
}


# `solver` est le seul module dont les points d'entrée vivent dans les
# sous-modules (un `solve` par back-end) sans être ré-exportés à la racine : le
# balayage ci-dessous ne peut pas les voir, on les nomme donc explicitement.
SOLVER_ENTRY_POINTS = ["solve", "solve_eliminate", "solve_unilateral"]


def module_roots():
    """Le fichier racine de chaque module d'`ops` — dossier ou fichier seul.

    Les deux formes existent (`ops/mesh/mod.rs` et `ops/matrix.rs`) : les
    oublier, c'est rendre le garde-fou aveugle à un module entier.
    """
    for d in sorted(OPS.iterdir()):
        if d.is_dir():
            yield d.name, d / "mod.rs"
        elif d.suffix == ".rs" and d.stem not in ("mod", "coloring", "scatter"):
            yield d.stem, d


def rust_exports():
    """(module, nom) de la surface publique de chaque module d'`ops`.

    Deux sources, parce que les deux sont utilisées dans le dépôt : les
    ré-exports `pub use sous_module::…` et les `pub fn` déclarées directement
    dans le fichier racine du module.
    """
    for module, root in module_roots():
        text = root.read_text()
        for line in text.splitlines():
            m = re.match(r"pub use \w+::(?:\{(.+)\}|(\w+));", line.strip())
            if m:
                names = m.group(1) or m.group(2)
                for name in (n.strip() for n in names.split(",")):
                    if name and name[0].islower():
                        yield module, name
                continue
            m = re.match(r"pub fn (\w+)", line)
            if m:
                yield module, m.group(1)
        if module == "solver":
            for name in SOLVER_ENTRY_POINTS:
                yield module, name


def test_every_rust_operator_has_a_python_binding():
    missing = []
    for module, name in rust_exports():
        if module in NO_PYTHON_MODULE or name in RUST_ONLY:
            continue
        py_module = getattr(pyrucast, module, None)
        if py_module is None:
            missing.append(f"pyrucast.{module} — le sous-module n'existe pas")
        elif not hasattr(py_module, name):
            missing.append(f"pyrucast.{module}.{name}  (← ops::{module}::{name})")
    assert not missing, "opérateurs Rust sans binding Python :\n  " + "\n  ".join(
        missing
    )


def test_rust_only_entries_are_documented_and_real():
    """Une dérogation doit porter une raison et viser un nom qui existe."""
    names = {name for _, name in rust_exports()}
    for fn, reason in RUST_ONLY.items():
        assert reason.strip(), f"{fn} : dérogation sans raison écrite"
    stale = [fn for fn in RUST_ONLY if fn not in names and fn[0].islower()]
    assert not stale, f"dérogations périmées, ces fonctions n'existent plus : {stale}"


def test_frame_deformation_is_reachable():
    """Régression nommée : c'est la fonction qui a motivé ce test."""
    assert hasattr(pyrucast.element_field, "frame_deformation")
