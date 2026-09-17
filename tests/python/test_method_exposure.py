"""Garde-fou de la règle « le verbe exposé aussi en méthode ».

`CONVENTIONS.md` § « Le verbe exposé aussi en méthode » : une fonction libre
est **aussi** une méthode de son sujet si (1) son premier argument est le
sujet, (2) elle rend un conteneur, (3) elle a un sens pour toute instance du
type. Ce test lit le stub — donc aucune liste de fonctions à tenir à la main —
et vérifie que la projection est complète : toute fonction qui remplit (1) et
(2) porte une méthode, sauf si elle figure ci-dessous avec sa raison.

Ajouter une fonction ici demande une raison écrite. C'est le prix de
l'exception, et c'est voulu.
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

# Les quatre saveurs de champ, pour les opérateurs polymorphes (`typing.Any`).
FIELDS = ["NodeField", "SubNodeField", "ElementField", "SubElementField"]

# Méthodes qui ne portent pas la documentation de **leur fonction libre** —
# chacune avec la raison qui l'y oblige.
DOC_PROPRE = {
    "select": (
        "opérateur polymorphe : la méthode naît d'une fonction par saveur, "
        "documentée pour ce seul receveur, et non de la fonction libre à `Any`"
    ),
    "mask": "idem — le produit est déterminé par la saveur du receveur",
    "merge_nodes": (
        "receveur `Py<Self>` et non `PyRef` : avec `in_place` elle rend l'objet "
        "lui-même, pas une vue empruntée — écrite à la main"
    ),
}

# Les pointeurs « Voir … » que le stub a encore le droit de porter : ceux des
# méthodes écrites à la main, et rien d'autre. Ce compte ne peut que décroître.
DOC_PROPRE_DANS_LE_STUB = ["merge_nodes"]

# Les sources du binding, pour le contrôle des cibles de pointeurs.
OPS_RS = pathlib.Path(__file__).resolve().parents[2] / "src" / "py" / "ops"

# Fonction libre -> nom de la méthode, quand le nom change parce que la méthode
# doit porter le qualificatif que le module fournissait à la fonction.
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
    # `element_field.mask` s'y enregistrent sous des noms distincts.
    "mask_node": "mask",
    "mask_element": "mask",
}

# Modules Python entiers sans méthode, avec la raison. La dérogation vaut par
# **construction** — elle tient pour chaque fonction du module, celles à venir
# comprises — et c'est ce qui la distingue d'une exclusion nom par nom.
NO_METHOD_MODULES = {
    "model": (
        "condition (1) : le premier argument est le **support** que le modèle "
        "recouvre (l'espace EF, ou les maillages que relie une contrainte), pas "
        "un sujet qu'on transforme. `fes.heat_conduction()` ferait promettre à "
        "tout espace EF les 28 physiques du catalogue."
    ),
}

# Sans méthode, avec la raison. Condition (3) sauf mention contraire.
NO_METHOD = {
    "deformation": "exige des composantes de déplacement u_x/u_y/u_z",
    "beam_deformation": "exige déplacements + rotations",
    "shell_deformation": "exige les six DDL de coque, et la formulation",
    "thermal_strain": "exige une température, et alpha dans le matériau",
    "merge": "symétrique — `a | b` est déjà sa forme",
    "psca": "symétrique — l'ordre ne compte pas",
}

PYI = pathlib.Path(pyrucast.__file__).parent / "_pyrucast" / "__init__.pyi"


def free_functions():
    """(nom, type du sujet, type de retour) des fonctions libres du stub."""
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
    """Les types concrets d'un sujet — quatre pour un opérateur polymorphe."""
    return FIELDS if first == "Any" else [first]


def excluded_by_module():
    """Les noms couverts par une dérogation de module entier."""
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
    """Toute exclusion doit porter une raison et viser une fonction existante."""
    names = {name for name, _, _ in free_functions()}
    for fn, reason in NO_METHOD.items():
        assert reason.strip(), f"{fn} : exclusion sans raison écrite"
        assert fn in names, f"{fn} : exclusion périmée, la fonction n'existe plus"


def test_module_exclusions_are_documented_and_real():
    """Une dérogation de module doit viser un module vivant qui en a besoin.

    « Qui en a besoin » = au moins une de ses fonctions serait éligible aux
    conditions (1) et (2) sans elle. Un module dont plus aucune fonction ne
    l'est verrait sa dérogation devenir du bruit, et ce test la signale.
    """
    eligible = {name for name, _, _ in free_functions()}
    for module, reason in NO_METHOD_MODULES.items():
        assert reason.strip(), f"{module} : dérogation sans raison écrite"
        names = set(getattr(pyrucast, module).__all__)
        assert names, f"{module} : dérogation sur un module vide"
        assert names & eligible, (
            f"{module} : dérogation périmée, plus aucune de ses fonctions "
            "n'est éligible à la projection en méthode"
        )


def test_renames_point_to_existing_functions():
    names = {name for name, _, _ in free_functions()}
    stale = [fn for fn in RENAMED if fn not in names]
    assert not stale, f"renommages périmés : {stale}"


def test_methods_carry_the_doc_of_their_function():
    """Une méthode dérivée affiche la documentation de sa fonction libre.

    C'est l'acquis de `#[py_op]` et de `py_field_unary!` : le texte est écrit
    une fois, sur la fonction, et **recopié** sur la méthode — donc `help()`
    comme le stub des IDE le montrent en entier. Un pointeur « Voir … » qui
    reviendrait ici serait une régression de l'aide affichée, invisible
    autrement puisque le code compilerait très bien.

    Le test part du stub, donc d'aucune liste tenue à la main : il confronte
    chaque fonction libre à la méthode de même nom, ou de nom renommé.
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
        "méthodes qui n'affichent pas la doc de leur fonction :\n  "
        + "\n  ".join(manquantes)
    )


def test_no_pointer_survives_in_the_stub():
    """Aucune docstring du stub ne se réduit à « Voir … ».

    C'est ce que lisent Pylance et PyCharm : un pointeur y est du texte mort,
    faute de mécanisme de lien. Le compte est borné par les seules méthodes
    écrites à la main, et il ne peut que décroître.
    """
    restants = re.findall(r"Voir `pyrucast\.[\w.]+`", PYI.read_text())
    assert len(restants) <= len(DOC_PROPRE_DANS_LE_STUB), (
        f"{len(restants)} pointeurs dans le stub, {len(DOC_PROPRE_DANS_LE_STUB)} "
        f"attendus au plus : {sorted(set(restants))}"
    )


def test_every_pointer_aims_at_a_living_target():
    """Un pointeur « Voir `pyrucast.X.Y` » doit viser quelque chose qui existe.

    Rien ne le vérifiait, et douze pointeurs ont pointé dans le vide pendant
    des mois : `field.mask` après le déménagement du verbe vers `node_field` et
    `element_field`, `field.filter_components` et `field.rename_component` qui
    n'ont jamais eu de fonction libre. Le code compilait, les tests passaient,
    et l'utilisateur lisait au survol le nom d'une fonction inexistante.
    """
    morts = []
    for source in sorted(OPS_RS.glob("*.rs")):
        for ligne, texte in enumerate(source.read_text().split("\n"), 1):
            m = re.search(r"/// Voir `pyrucast\.(\w+)\.(\w+)`\.", texte)
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
    # imbriqués à lire de l'intérieur vers l'extérieur.
    contour = quad.border().consolidate()
    assert contour.cell_count() == 4

    xs = quad.positions(["X"])
    droite = xs.select(ge=1.0)  # champ nodal → Mesh : le module suit la sortie
    assert droite.cell_count() == 2  # deux POI1, les nœuds de x = 1

    # renommage et filtrage de composantes, eux aussi chaînables
    assert xs.rename_component("X", "abscisse").components() == ["abscisse"]
