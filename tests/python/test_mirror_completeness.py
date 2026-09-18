"""Guard rail of the Rust → Python mirror.

`CONVENTIONS.md` § "Rust → Python rule: a 1:1 mirror": every free function of
`ops` is exposed to Python, under the same name, in the sub-module of the same
Rust module. The only tolerated asymmetry is the **non-exposure of the `Sub*`
constructors**.

This test reads the **public surface** of each module of `ops` — the `pub use`
and the `pub fn` of its root file, the list the developer already keeps for
Rust — and checks that each name is found again in the matching Python
sub-module. That is the hole through which
`ops::element_field::beam_deformation` was left without a binding: the guard
rail of the methods (`test_method_exposure.py`) reads the **stub**, so it cannot
see a function that is not in it. The two tests are complementary — one guards
the Rust → Python projection, the other the function → method projection.

A function deliberately left unexposed is written down in `RUST_ONLY`, with its
reason.
"""

import pathlib
import re

import pyrucast

ROOT = pathlib.Path(__file__).resolve().parents[2]
OPS = ROOT / "src" / "ops"

# Rust modules whose functions have no Python sub-module (yet).
NO_PYTHON_MODULE = {
    # An accepted exception: `locate_points` and `project_points` are the
    # geometric primitives beneath `model.embedded` and `model.contact`. The
    # Python user gets their result as an assemblable constraint, which is the
    # useful level; exposing them means deciding how to translate `Location` and
    # `Projection`, which is still to be settled.
    "geom": "internal primitives of the embedded / contact constraints",
}

# Rust functions deliberately left unexposed, with the reason.
RUST_ONLY = {
    # The `*_cancellable` variants: on the Python side, interruption is wired
    # into the wrapper of the nominal function, not exposed as a function of
    # its own.
    "grid_surface_cancellable": "interruption is wired into the nominal wrapper",
    "grid_surface2_cancellable": "interruption is wired into the nominal wrapper",
    "pave_surface_cancellable": "interruption is wired into the nominal wrapper",
    "pave_volume_cancellable": "interruption is wired into the nominal wrapper",
    "triangulate_surface_cancellable": "interruption is wired into the nominal wrapper",
    "triangulate_volume_cancellable": "interruption is wired into the nominal wrapper",
    # Shared implementation details, not operators.
    "check_unique_component_per_support": "internal guard of the merge, not an operator",
    "assemble_kind": "common engine of the assemblers, not an operation of use",
    "select_sub_cells": "the `Sub*` views go through the dispatch of `select`",
    "select_sub_nodes": "the `Sub*` views go through the dispatch of `select`",
    "mask_sub": "the `Sub*` views go through the dispatch of `mask`",
    # The six typed VTK writers sit behind the single `export.export_vtk`, which
    # chooses according to what it is handed (mesh, nodal field, element field)
    # and according to the presence of a path.
    "write_vtk_mesh": "dispatched inside `export.export_vtk`",
    "write_vtk_node_field": "dispatched inside `export.export_vtk`",
    "write_vtk_element_field": "dispatched inside `export.export_vtk`",
    "vtk_mesh_string": '"to a string" variant, not exposed',
    "vtk_node_field_string": '"to a string" variant, not exposed',
    "vtk_element_field_string": '"to a string" variant, not exposed',
    # The `*_with_symmetry` / `*_with_law` variants of `ops::model`: Rust passes
    # an enum (`MaterialSymmetry`, `PlasticLaw`, `DamageLaw`), Python does not
    # expose those enums. The fold differs on each side — a `symmetry=` keyword
    # for the symmetry, one function per law for the laws (see PYTHON_ONLY) —
    # but no operation is missing.
    "heat_conduction_with_symmetry": "folded into `model.heat_conduction(fes, symmetry=…)`",
    "fick_with_symmetry": "folded into `model.fick(fes, species, symmetry=…)`",
    "elasticity_with_symmetry": "folded into `model.elasticity(fes, model, symmetry=…)`",
    "plasticity_with_law": "unfolded into one Python function per law (`drucker_prager`, `creep_norton`…)",
    "damage_with_law": "unfolded into one Python function per law (`damage_tc`, `gurson`…)",
    "select_cells": "dispatched by type inside `mesh.select`",
    "select_nodes": "dispatched by type inside `mesh.select`",
    "integral_element": "dispatched by type inside `measure.integral`",
    "consolidate": "exposed under the short name in its own Python sub-module",
    "set": "exposed under the short name in `pyrucast.coords`",
    "solve": "exposed per variant (`solve`, `solve_eliminate`, `solve_unilateral`)",
    "Band": "value type, carried by the `ge`/`gt`/`le`/`lt` arguments",
    "FluxDensity": "value type, carried by the `density` argument",
    "Location": "return type of `geom`, not exposed",
    "Projection": "return type of `geom`, not exposed",
    # `nearest_node` has left `ops::geom`: it is not an operator but a method of
    # `Mesh`, on both sides (single container, derived view).
}


# `solver` is the only module whose entry points live in the sub-modules (one
# `solve` per back-end) without being re-exported at the root: the sweep below
# cannot see them, so they are named explicitly.
SOLVER_ENTRY_POINTS = ["solve", "solve_eliminate", "solve_unilateral"]


def module_roots():
    """The root file of each module of `ops` — a directory or a lone file.

    Both forms exist (`ops/mesh/mod.rs` and `ops/matrix.rs`): forgetting them
    means blinding the guard rail to a whole module.
    """
    for d in sorted(OPS.iterdir()):
        if d.is_dir():
            yield d.name, d / "mod.rs"
        elif d.suffix == ".rs" and d.stem not in ("mod", "coloring", "scatter"):
            yield d.stem, d


def rust_exports():
    """(module, name) of the public surface of each module of `ops`.

    Two sources, because both are used in the repository: the `pub use
    sub_module::…` re-exports and the `pub fn` declared directly in the root
    file of the module.
    """
    for module, root in module_roots():
        text = root.read_text()
        # `pub use sub_module::{a, b};` — possibly spread over several lines,
        # which rustfmt does as soon as the list exceeds the width. A line-by-line
        # sweep misses those silently: that is how the whole `points_*` family
        # stayed invisible to this test.
        # `pub use sub_module::…` but also `pub use crate::path::…`: an operator
        # may be defined elsewhere and only re-exported here, which a
        # single-segment path did not cover — the guard rail then lost the
        # operator **silently**.
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
            missing.append(f"pyrucast.{module} — the sub-module does not exist")
        elif not hasattr(py_module, name):
            missing.append(f"pyrucast.{module}.{name}  (← ops::{module}::{name})")
    assert not missing, "Rust operators without a Python binding:\n  " + "\n  ".join(
        missing
    )


def test_rust_only_entries_are_documented_and_real():
    """An exemption must carry a reason and point at a name that exists."""
    names = {name for _, name in rust_exports()}
    for fn, reason in RUST_ONLY.items():
        assert reason.strip(), f"{fn}: exemption with no written reason"
    stale = [fn for fn in RUST_ONLY if fn not in names and fn[0].islower()]
    assert not stale, f"stale exemptions, these functions no longer exist: {stale}"


# Python functions with no Rust free function of the same name, with the reason.
# The Rust → Python direction is not enough: `filter_components` and
# `rename_component` stayed for a long time as Python free functions while Rust
# only had the method — the very asymmetry the convention forbids, invisible to
# the sweep above. Both have since been removed.
PYTHON_ONLY = {
    "mask_node": "flat name of `node_field.mask` (flat `_pyrucast` namespace)",
    "mask_element": "flat name of `element_field.mask`",
    "consolidate_mesh": "flat name of `mesh.consolidate`",
    "consolidate_node": "flat name of `node_field.consolidate`",
    "consolidate_element": "flat name of `element_field.consolidate`",
    "set_positions": "flat name of `coords.set`",
    "select": "dispatched by type onto `mesh::select_nodes` / `select_cells`",
    "integral": "dispatched by type onto `measure::integral` / `integral_element`",
    "integrate_behavior": "qualified name of `element_field::behavior::integrate`",
    "solve_eliminate": "qualified name of `solver::eliminate::solve`",
    "solve_unilateral": "qualified name of `solver::unilateral::solve`",
    "export_vtk": "qualified name of `export::vtk::write`",
    "xtx": "primitive of the `Field` trait, exposed as a reduction operator",
    "xty": "primitive of the `Field` trait, exposed as a reduction operator",
    # The only entry that is not a plain renaming: `from_gmsh` needs a live
    # CPython interpreter carrying the `gmsh` module, which Rust cannot have. It
    # invents no operation either — it fetches the arrays of the current model
    # and passes them to the Rust operator `mesh::from_gmsh_arrays`, which is
    # itself a strict mirror.
    "from_gmsh": "reads the live gmsh model: requires the interpreter, hence no Rust twin",
}


def python_free_functions():
    """(module, name) of the free functions exposed by the Python sub-modules."""
    for module, _ in module_roots():
        py_module = getattr(pyrucast, module, None)
        if py_module is None:
            continue
        for name in getattr(py_module, "__all__", []):
            yield module, name


def test_every_python_function_has_a_rust_operator():
    """The mirror the other way round: no Python function without a Rust operator.

    That is the direction the first sweep does not cover, and through which
    `filter_components` / `rename_component` survived in a double form.
    """
    rust = {name for _, name in rust_exports()}
    orphans = [
        f"pyrucast.{module}.{name}"
        for module, name in python_free_functions()
        if name not in rust and name not in PYTHON_ONLY
    ]
    assert not orphans, (
        "Python functions with no Rust free function — either the operator is "
        "missing on the Rust side, or this is a method disguised as a "
        "function:\n  " + "\n  ".join(orphans)
    )


def test_python_only_entries_are_documented_and_real():
    """An exemption must carry a reason **and** remain necessary.

    The second half was missing: eleven entries survived the appearance of their
    Rust twin without anything flagging it. A stale exemption is a piece of
    reasoning one still believes to be valid.
    """
    # Three names exist on both sides without the exemption being stale: there
    # the Python function spreads over **several** Rust functions, one of which
    # carries the same name. The name coincides, the operation does not.
    dispatchers = {"integral", "solve_eliminate", "solve_unilateral"}
    rust = {name for _, name in rust_exports()}
    for fn, reason in PYTHON_ONLY.items():
        assert reason.strip(), f"{fn}: exemption with no written reason"
        if fn in dispatchers:
            continue
        assert fn not in rust, (
            f"{fn}: stale exemption — Rust now exposes this name, "
            "remove it from PYTHON_ONLY"
        )


def test_beam_deformation_is_reachable():
    """A named regression: this is the function that motivated this test."""
    assert hasattr(pyrucast.element_field, "beam_deformation")
