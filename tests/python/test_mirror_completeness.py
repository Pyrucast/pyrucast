"""Garde-fou du miroir Rust → Python.

`CONVENTIONS.md` § « Règle Rust → Python : miroir 1:1 » : toute fonction libre
d'`ops` est exposée à Python, sous le même nom, dans le sous-module du même
module Rust. La seule asymétrie tolérée est la **non-exposition des
constructeurs `Sub*`**.

Ce test lit la **surface publique** de chaque module d'`ops` — les `pub use`
et les `pub fn` de son fichier racine, la liste que le développeur tient déjà
pour Rust — et vérifie que chaque nom se retrouve dans le sous-module Python
correspondant. C'est le trou par lequel
`ops::element_field::beam_deformation` est restée sans binding : le garde-fou
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
    # Exception assumée : `locate_points` et `project_points` sont les
    # primitives géométriques sous `model.embedded` et `model.contact`.
    # L'utilisateur Python obtient leur résultat sous forme de contrainte
    # assemblable, ce qui est le niveau utile ; les exposer suppose de décider
    # comment traduire `Location` et `Projection`, ce qui reste à trancher.
    "geom": "primitives internes des contraintes embedded / contact",
}

# Fonctions Rust délibérément non exposées, avec la raison.
RUST_ONLY = {
    # Variantes `*_cancellable` : côté Python, l'interruption est branchée dans
    # le wrapper de la fonction nominale, pas exposée comme une fonction à part.
    "grid_surface_cancellable": "l'interruption est câblée dans le wrapper nominal",
    "grid_surface2_cancellable": "l'interruption est câblée dans le wrapper nominal",
    "pave_surface_cancellable": "l'interruption est câblée dans le wrapper nominal",
    "pave_volume_cancellable": "l'interruption est câblée dans le wrapper nominal",
    "triangulate_surface_cancellable": "l'interruption est câblée dans le wrapper nominal",
    "triangulate_volume_cancellable": "l'interruption est câblée dans le wrapper nominal",
    # Détails d'implémentation partagés, pas des opérateurs.
    "check_unique_component_per_support": "garde interne de la fusion, pas un opérateur",
    "assemble_kind": "moteur commun des assembleurs, pas une opération d'usage",
    "select_sub_cells": "les vues `Sub*` passent par le dispatch de `select`",
    "select_sub_nodes": "les vues `Sub*` passent par le dispatch de `select`",
    "mask_sub": "les vues `Sub*` passent par le dispatch de `mask`",
    # Les six écritures VTK typées sont derrière l'unique `export.export_vtk`,
    # qui choisit selon ce qu'on lui passe (maillage, champ nodal, champ par
    # éléments) et selon la présence d'un chemin.
    "write_vtk_mesh": "dispatch dans `export.export_vtk`",
    "write_vtk_node_field": "dispatch dans `export.export_vtk`",
    "write_vtk_element_field": "dispatch dans `export.export_vtk`",
    "vtk_mesh_string": "variante « vers une chaîne », non exposée",
    "vtk_node_field_string": "variante « vers une chaîne », non exposée",
    "vtk_element_field_string": "variante « vers une chaîne », non exposée",
    # Les variantes `*_with_symmetry` / `*_with_law` de `ops::model` : Rust passe
    # une enum (`MaterialSymmetry`, `PlasticLaw`, `DamageLaw`), Python n'expose
    # pas ces enums. Le pli est différent des deux côtés — un mot-clé `symmetry=`
    # pour la symétrie, une fonction par loi pour les lois (voir PYTHON_ONLY) —
    # mais aucune opération ne manque.
    "heat_conduction_with_symmetry": "replié dans `model.heat_conduction(fes, symmetry=…)`",
    "fick_with_symmetry": "replié dans `model.fick(fes, espèce, symmetry=…)`",
    "elasticity_with_symmetry": "replié dans `model.elasticity(fes, model, symmetry=…)`",
    "plasticity_with_law": "déplié en une fonction Python par loi (`drucker_prager`, `creep_norton`…)",
    "damage_with_law": "déplié en une fonction Python par loi (`damage_tc`, `gurson`…)",
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
    # `nearest_node` a quitté `ops::geom` : ce n'est pas un opérateur mais une
    # méthode de `Mesh`, des deux côtés (mono-conteneur, vue dérivée).
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
        # `pub use sous_module::{a, b};` — éventuellement sur plusieurs lignes,
        # ce que rustfmt fait dès que la liste dépasse la largeur. Un balayage
        # ligne à ligne les rate en silence : c'est ainsi que toute la famille
        # `points_*` est restée invisible à ce test.
        # `pub use sous_module::…` mais aussi `pub use crate::chemin::…` : un
        # opérateur peut être défini ailleurs et seulement ré-exporté ici, ce
        # qu'un chemin à un seul segment ne couvrait pas — le garde-fou perdait
        # alors l'opérateur **en silence**.
        for m in re.finditer(r"pub use (?:\w+::)+(?:\{(.*?)\}|(\w+));", text, re.S):
            names = m.group(1) or m.group(2)
            for name in (n.strip() for n in names.split(",")):
                if name and name[0].islower():
                    yield module, name
        for m in re.finditer(r"^pub fn (\w+)", text, re.M):
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


# Fonctions Python sans fonction libre Rust homonyme, avec la raison. Le sens
# Rust → Python ne suffit pas : `filter_components` et `rename_component` sont
# restées longtemps en fonction libre Python alors que Rust n'avait que la
# méthode — l'asymétrie que la convention interdit, invisible au balayage
# ci-dessus. Les deux ont depuis été retirées.
PYTHON_ONLY = {
    "mask_node": "nom plat de `node_field.mask` (namespace `_pyrucast` plat)",
    "mask_element": "nom plat de `element_field.mask`",
    "consolidate_mesh": "nom plat de `mesh.consolidate`",
    "consolidate_node": "nom plat de `node_field.consolidate`",
    "consolidate_element": "nom plat de `element_field.consolidate`",
    "set_positions": "nom plat de `coords.set`",
    "select": "dispatch par type sur `mesh::select_nodes` / `select_cells`",
    "integral": "dispatch par type sur `measure::integral` / `integral_element`",
    "integrate_behavior": "nom qualifié de `element_field::behavior::integrate`",
    "solve_eliminate": "nom qualifié de `solver::eliminate::solve`",
    "solve_unilateral": "nom qualifié de `solver::unilateral::solve`",
    "export_vtk": "nom qualifié de `export::vtk::write`",
    "xtx": "primitive du trait `Field`, exposée en opérateur de réduction",
    "xty": "primitive du trait `Field`, exposée en opérateur de réduction",
    # La seule entrée qui ne soit pas un simple renommage : `from_gmsh` a besoin
    # d'un interpréteur CPython vivant portant le module `gmsh`, ce que Rust ne
    # peut pas avoir. Elle n'invente d'ailleurs aucune opération — elle va
    # chercher les tableaux du modèle courant et les passe à l'opérateur Rust
    # `mesh::from_gmsh_arrays`, qui, lui, est un miroir strict.
    "from_gmsh": "lit le modèle gmsh vivant : exige l'interpréteur, donc sans jumeau Rust",
}


def python_free_functions():
    """(module, nom) des fonctions libres exposées par les sous-modules Python."""
    for module, _ in module_roots():
        py_module = getattr(pyrucast, module, None)
        if py_module is None:
            continue
        for name in getattr(py_module, "__all__", []):
            yield module, name


def test_every_python_function_has_a_rust_operator():
    """Le miroir dans l'autre sens : pas de fonction Python sans opérateur Rust.

    C'est le sens que le premier balayage ne couvre pas, et par lequel
    `filter_components` / `rename_component` ont survécu en double forme.
    """
    rust = {name for _, name in rust_exports()}
    orphans = [
        f"pyrucast.{module}.{name}"
        for module, name in python_free_functions()
        if name not in rust and name not in PYTHON_ONLY
    ]
    assert not orphans, (
        "fonctions Python sans fonction libre Rust — soit l'opérateur manque "
        "côté Rust, soit c'est une méthode déguisée en fonction :\n  "
        + "\n  ".join(orphans)
    )


def test_python_only_entries_are_documented_and_real():
    """Une dérogation doit porter une raison **et** rester nécessaire.

    La seconde moitié manquait : onze entrées ont survécu à l'apparition de
    leur jumelle Rust sans que rien ne le signale. Une dérogation périmée est
    un raisonnement qu'on croit encore valable.
    """
    # Trois noms existent des deux côtés sans que la dérogation soit périmée :
    # la fonction Python y répartit sur **plusieurs** fonctions Rust, dont une
    # porte le même nom. Le nom coïncide, l'opération non.
    dispatchers = {"integral", "solve_eliminate", "solve_unilateral"}
    rust = {name for _, name in rust_exports()}
    for fn, reason in PYTHON_ONLY.items():
        assert reason.strip(), f"{fn} : dérogation sans raison écrite"
        if fn in dispatchers:
            continue
        assert fn not in rust, (
            f"{fn} : dérogation périmée — Rust expose désormais ce nom, "
            "la retirer de PYTHON_ONLY"
        )


def test_beam_deformation_is_reachable():
    """Régression nommée : c'est la fonction qui a motivé ce test."""
    assert hasattr(pyrucast.element_field, "beam_deformation")
