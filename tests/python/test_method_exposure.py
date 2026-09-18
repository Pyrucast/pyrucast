"""Guard for the rule "the verb also exposed as a method".

`CONVENTIONS.md` § "The verb also exposed as a method": a free function is
**also** a method of its subject if (1) its first argument is the subject,
(2) it returns a container, (3) it makes sense for every instance of the type.
This test reads the stub — hence no list of functions to keep by hand — and
checks that the projection is complete: every function meeting (1) and (2)
carries a method, unless it appears below with its reason.

Adding a function here demands a written reason. That is the price of the
exception, and it is deliberate.
"""

import pathlib
import re

import pyrucast

CONTAINERS = {
    "Mesh",
    "SubMesh",
    "FiniteElementSpace",
    "SubFiniteElementSpace",
    "Model",
    "SubModel",
    "Matrix",
    "SubMatrix",
    "NodeField",
    "SubNodeField",
    "ElementField",
    "SubElementField",
    "Evolution",
    "SubEvolution",
}

# The four field flavours, for the polymorphic operators (`typing.Any`).
FIELDS = ["NodeField", "SubNodeField", "ElementField", "SubElementField"]

# Methods that do not carry **their free function's** documentation — each
# with the reason that forces it.
DOC_PROPRE = {
    "select": (
        "polymorphic operator: the method is born of one function per flavour, "
        "documented for that receiver alone, not of the free function returning `Any`"
    ),
    "mask": "same — the product is determined by the receiver's flavour",
}

# The "See …" pointers the stub is still allowed to carry: **none**. A pointer is
# dead text in the `.pyi` the IDEs read, so a method written by hand — even
# `merge_nodes`, whose `Py<Self>` receiver keeps it out of `#[py_op]` — copies
# its free function's documentation instead of pointing at it.
DOC_PROPRE_DANS_LE_STUB = []

# The binding's sources, for checking the pointers' targets.
OPS_RS = pathlib.Path(__file__).resolve().parents[2] / "src" / "py" / "ops"

# Free function -> method name, when the name changes because the method must
# carry the qualifier the module supplied to the function.
RENAMED = {
    "stiffness": "stiffness_matrix",
    "mass": "mass_matrix",
    "geometric": "geometric_matrix",
    "tangent": "tangent_matrix",
    "sub_material_field": "material_field",
    "consolidate_mesh": "consolidate",
    "consolidate_node": "consolidate",
    "consolidate_element": "consolidate",
    # Noms plats de l'extension : `_pyrucast` étant plat, `node_field.mask` et
    # `element_field.mask` register there under distinct names.
    "mask_node": "mask",
    "mask_element": "mask",
}

# Whole Python modules without a method, with the reason. The waiver holds by
# **construction** — it holds for every function of the module, those to come
# included — and that is what sets it apart from a name-by-name exclusion.
NO_METHOD_MODULES = {
    "model": (
        "condition (1): the first argument is the **support** the model covers "
        "(the FE space, or the meshes a constraint relates), not a subject being "
        "transformed. `fes.heat_conduction()` would have every FE space promise "
        "the catalogue's 28 physics."
    ),
}

# Without a method, with the reason. Condition (3) unless stated otherwise.
NO_METHOD = {
    "deformation": "requires displacement components u_x/u_y/u_z",
    "beam_deformation": "exige déplacements + rotations",
    "shell_deformation": "requires the six shell DOFs, and the formulation",
    "thermal_strain": "requires a temperature, and alpha in the material",
    "merge": "symmetric — `a | b` is already its form",
    "psca": "symmetric — order does not matter",
}

PYI = pathlib.Path(pyrucast.__file__).parent / "_pyrucast" / "__init__.pyi"


def free_functions():
    """(name, subject type, return type) of the stub's free functions."""
    for name, args, ret in re.findall(
        r"^def (\w+)\((.*?)\) -> ([^:]+):", PYI.read_text(), re.M
    ):
        parts = [a.strip() for a in args.split(",") if a.strip()]
        if not parts:
            continue
        m = re.match(r"\w+:\s*(?:typing\.Any|([\w.]+))", parts[0])
        if not m:
            continue
        first = m.group(1) or "Any"
        ret = ret.strip()
        if first not in CONTAINERS and first != "Any":
            continue
        if ret not in CONTAINERS and ret != "typing.Any":
            continue
        yield name, first, ret


def subjects(first):
    """A subject's concrete types — four for a polymorphic operator."""
    return FIELDS if first == "Any" else [first]


def excluded_by_module():
    """The names covered by a whole-module waiver."""
    return {
        name
        for module in NO_METHOD_MODULES
        for name in getattr(pyrucast, module).__all__
    }


def test_every_eligible_operator_is_also_a_method():
    by_module = excluded_by_module()
    missing = []
    for name, first, _ret in free_functions():
        if name in NO_METHOD or name in by_module:
            continue
        method = RENAMED.get(name, name)
        for cls in subjects(first):
            if not hasattr(getattr(pyrucast, cls), method):
                missing.append(f"{cls}.{method}  (← {name})")
    assert not missing, "opérateurs éligibles sans méthode :\n  " + "\n  ".join(missing)


def test_exclusions_are_documented_and_real():
    """Every exclusion must carry a reason and target an existing function."""
    names = {name for name, _, _ in free_functions()}
    for fn, reason in NO_METHOD.items():
        assert reason.strip(), f"{fn}: exclusion without a written reason"
        assert fn in names, f"{fn} : exclusion périmée, la fonction n'existe plus"


def test_module_exclusions_are_documented_and_real():
    """A module waiver must target a living module that needs it.

    "Needs it" = at least one of its functions would be eligible under
    conditions (1) and (2) without it. A module where no function is eligible
    any more would see its waiver become noise, and this test reports it.
    """
    eligible = {name for name, _, _ in free_functions()}
    for module, reason in NO_METHOD_MODULES.items():
        assert reason.strip(), f"{module}: waiver without a written reason"
        names = set(getattr(pyrucast, module).__all__)
        assert names, f"{module}: waiver on an empty module"
        assert names & eligible, (
            f"{module}: stale waiver, none of its functions is eligible "
            "for the projection into a method any more"
        )


def test_renames_point_to_existing_functions():
    names = {name for name, _, _ in free_functions()}
    stale = [fn for fn in RENAMED if fn not in names]
    assert not stale, f"renommages périmés : {stale}"


def test_methods_carry_the_doc_of_their_function():
    """A derived method displays its free function's documentation.

    That is what `#[py_op]` and `impl_field_transform_pymethod!` won: the text
    is written once, on the function, and **copied** onto the method — so
    `help()` and the IDEs' stub show it in full. A "See …" pointer coming back
    here would be a regression of the displayed help, invisible otherwise since
    autrement puisque le code compilerait très bien.

    The test starts from the stub, hence from no hand-kept list: it confronts
    every free function with the method of the same, or renamed, name.
    """
    manquantes = []
    for nom, first, _ret in free_functions():
        methode = RENAMED.get(nom, nom)
        if methode in DOC_PROPRE:
            continue
        attendu = getattr(pyrucast._pyrucast, nom).__doc__
        if not attendu:
            continue
        for cls in subjects(first):
            porte = getattr(getattr(pyrucast, cls), methode, None)
            if porte is None or porte.__doc__ == attendu:
                continue
            manquantes.append(f"{cls}.{methode}  (← {nom}) : {porte.__doc__!r:.60}")
    assert not manquantes, (
        "methods that do not display their function's doc:\n  "
        + "\n  ".join(manquantes)
    )


def test_no_pointer_survives_in_the_stub():
    """No docstring of the stub boils down to "See …".

    That is what Pylance and PyCharm read: a pointer there is dead text, for
    want of a link mechanism. The count is bounded by the hand-written methods
    alone, and it can only shrink.
    """
    restants = re.findall(r"See `pyrucast\.[\w.]+`", PYI.read_text())
    assert len(restants) <= len(DOC_PROPRE_DANS_LE_STUB), (
        f"{len(restants)} pointers in the stub, {len(DOC_PROPRE_DANS_LE_STUB)} "
        f"expected at most: {sorted(set(restants))}"
    )


def test_every_pointer_aims_at_a_living_target():
    """A pointer "See `pyrucast.X.Y`" must target something that exists.

    Nothing checked it, and twelve pointers pointed into the void for months:
    `field.mask` after the verb moved to `node_field` and `element_field`,
    `field.filter_components` and `field.rename_component`, which never had a
    free function at all. The code compiled, the tests passed, and the user read
    the name of a non-existent function on hover.
    """
    morts = []
    for source in sorted(OPS_RS.glob("*.rs")):
        for ligne, texte in enumerate(source.read_text().split("\n"), 1):
            m = re.search(r"/// See `pyrucast\.(\w+)\.(\w+)`\.", texte)
            if not m:
                continue
            module, verbe = m.groups()
            if getattr(getattr(pyrucast, module, None), verbe, None) is None:
                morts.append(f"{source.name}:{ligne} → pyrucast.{module}.{verbe}")
    assert not morts, "pointeurs vers une cible inexistante :\n  " + "\n  ".join(morts)


def test_chaining_actually_works():
    """Un cas concret de bout en bout — la raison d'être de la règle."""
    c = pyrucast.Coords(dim=2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    d = c.add_node([0.0, 1.0])
    e = c.add_node([1.0, 1.0])
    quad = pyrucast.mesh.sweep(
        pyrucast.mesh.line(a, b, 1), pyrucast.mesh.line(d, e, 1), 1
    )
    # méthode, puis méthode, puis méthode — la forme libre ferait trois appels
    # calls to read from the inside out.
    contour = quad.border().consolidate()
    assert contour.cell_count() == 4

    xs = quad.positions(["X"])
    droite = xs.select(ge=1.0)  # champ nodal → Mesh : le module suit la sortie
    assert droite.cell_count() == 2  # two POI1, the nodes at x = 1

    # renommage et filtrage de composantes, eux aussi chaînables
    assert xs.rename_component("X", "abscisse").components() == ["abscisse"]
