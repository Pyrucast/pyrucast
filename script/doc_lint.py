#!/usr/bin/env python3
"""Documentation guards — five checks, no compilation.

Called by `script/check_doc.{sh,ps1}`, after `cargo doc` (whose output it
reads). The rules it enforces are written in `CONVENTIONS.md`, part
"Documentation et tests", and told in
`book/src/developper/documentation-et-tests.md`.

1. `includes`   — every `{{#include}}` of the book resolves: file, anchor, and
                  a non-empty text. This is the most important guard, because
                  nothing else guards the carrying mechanism: a missing anchor
                  yields an **empty** block, exit code 0, without a word.

2. `fences`     — no page owns code: every `rust`/`python` block contains an
                  include, without exception.
3. `symboles`   — the prose cites no vanished symbol. Nothing else covers a
                  paragraph, and that is where three of the nine errors found
                  in August 2026 were.
4. `doctests`   — Rust coverage ratchet: a new public item carries its
                  example, and the existing debt can only shrink.
5. `api-python` — the Python counterpart, at a looser granularity: every
                  public entry is **cited** by an executed example of the book
                  (`tests/python/test_doc_*.py`). Python docstrings being
                  written in the `///` of `src/py/`, one doctest per item
                  would cost a home-made collector; citation gives the same
                  non-regression guarantee for twenty lines.

Every waiver ledger carries a **reason** and its hygiene test, which fails on
a stale entry — the pattern of
`tests/python/test_mirror_completeness.py`, reused rather than reinvented.

    python script/doc_lint.py            # everything
    python script/doc_lint.py includes   # a single check
    python script/doc_lint.py --ratchet         # rewrites doc_coverage.txt
    python script/doc_lint.py --ratchet-python  # rewrites python_coverage.txt
"""

from __future__ import annotations

import ast
import inspect
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book" / "src"
LEDGER = ROOT / "script" / "doc_coverage.txt"
LEDGER_PY = ROOT / "script" / "python_coverage.txt"
EXEMPLES_PY = ROOT / "tests" / "python"

# ── Registres ───────────────────────────────────────────────────────────────


# Migration debt: pages that still own copied code. **This ledger can only
# shrink** — hygiene refuses a page exceeding its count, and demands that a
# page down to zero be removed. Nothing is ever added to it.
DETTE_MIGRATION = {}

# Symbols cited in prose that the audit must not try to resolve.
SYMBOLES_TOLERES = {
    "T::union_subs": "T is a type parameter, not a concrete type",
    "ops::mesher": "historical name, cited to tell the 2026-08-03 renaming",
    "SubMesh::connectivity": "pub(crate): the Parallelism page describes the internal machinery",
    "SubMatrix::add_entry_with": "pub(crate): same page, the zero-copy twin of a public method",
    "SubNodeField::nodes_with": "pub(crate): same page, the zero-copy twin of a public method",
    "Type::membre": "metavariable of the Documentation et tests page",
    "module::verbe": "metavariable: the marker of delegation methods",
    "Coords::acquire": "wrong name, cited to tell what the guard found",
}

# Roots that do not come from the crate: the right-hand side is not checked.
CRATES_EXTERNES = {
    "std",
    "core",
    "alloc",
    "nalgebra",
    "nalgebra_sparse",
    "serde",
    "bincode",
    "faer",
    "rayon",
    "parking_lot",
    "paste",
    "pyo3",
    "pyo3_stub_gen",
    "ctrlc",
    "plotters",
    "winit",
    "softbuffer",
    "criterion",
}
TYPES_EXTERNES = {
    "Vec",
    "String",
    "Option",
    "Result",
    "Box",
    "Arc",
    "Rc",
    "HashMap",
    "HashSet",
    "BTreeMap",
    "Path",
    "PathBuf",
    "Duration",
    "Instant",
    "AtomicBool",
    "AtomicUsize",
    "RwLock",
    "Mutex",
    "Python",
    "PyResult",
    "DMatrix",
    "CsrMatrix",
    "CscMatrix",
    "CooMatrix",
}

# ── Utilitaires ─────────────────────────────────────────────────────────────


def pages():
    """The book's pages, path relative to book/src, sorted."""
    return sorted(p for p in BOOK.rglob("*.md"))


def rel(p: Path) -> str:
    return str(p.relative_to(BOOK)).replace("\\", "/")


def fences(text: str):
    """The fenced blocks: (language, lines, opening line number)."""
    out, opened, buf, start = [], None, [], 0
    for i, line in enumerate(text.split("\n"), 1):
        if opened is None:
            m = re.match(r"^```(\w[\w,-]*)\s*$", line)
            if m:
                opened, buf, start = m.group(1), [], i
        elif line.startswith("```"):
            out.append((opened, buf, start))
            opened = None
        else:
            buf.append(line)
    return out


def inline_codes(text: str):
    """The `inline code` spans, outside fenced blocks.

    Restricting the symbol audit to these spans is what rules out the false
    positives — file names (`pyrucast.pth`), domains (`pyrucast.github.io`)
    and everyday sentences that contain a `::` by accident.
    """
    without_blocks = re.sub(r"```.*?```", "", text, flags=re.S)
    return re.findall(r"`([^`\n]+)`", without_blocks)


# ── 1. Include resolution ───────────────────────────────────────────────────


def check_includes():
    erreurs = []
    for p in pages():
        text = p.read_text()
        for m in re.finditer(
            r"\{\{#(?:rustdoc_)?include\s+([^}:]+?)(?::([^}]+))?\}\}", text
        ):
            chemin, ancre = m.group(1).strip(), (m.group(2) or "").strip()
            ligne = text[: m.start()].count("\n") + 1
            cible = (p.parent / chemin).resolve()
            ou = f"{rel(p)}:{ligne}"
            if not cible.exists():
                erreurs.append(f"{ou}: file not found — {chemin}")
                continue
            source = cible.read_text(errors="ignore")
            if not ancre:
                if not source.strip():
                    erreurs.append(f"{ou}: the included file is empty — {chemin}")
                continue
            debut = re.search(rf"ANCHOR:\s*{re.escape(ancre)}\s*$", source, re.M)
            fin = re.search(rf"ANCHOR_END:\s*{re.escape(ancre)}\s*$", source, re.M)
            if not debut or not fin:
                manque = "ANCHOR" if not debut else "ANCHOR_END"
                erreurs.append(f'{ou}: {manque} "{ancre}" missing from {chemin}')
                continue
            corps = source[debut.end() : fin.start()]
            if not corps.strip():
                erreurs.append(f'{ou}: anchor "{ancre}" of {chemin} is empty')
    return erreurs


# ── 2. Lint de clôtures ─────────────────────────────────────────────────────


def blocs_ecrits_a_la_main(text: str) -> int:
    return sum(
        1
        for langue, corps, _ in fences(text)
        if (langue.startswith("rust") or langue.startswith("python"))
        and not any("{{#include" in ligne for ligne in corps)
    )


def check_fences():
    erreurs, vus = [], {}
    for p in pages():
        nom = rel(p)
        n = blocs_ecrits_a_la_main(p.read_text())
        vus[nom] = n
        if not n:
            continue
        budget = DETTE_MIGRATION.get(nom)
        if budget is None:
            erreurs.append(
                f"{nom}: {n} hand-written code block(s). "
                "No page owns code: write a test or an example, "
                "l'encadrer d'ANCHOR, et l'inclure (CONVENTIONS.md, règle 1)."
            )
        elif n > budget:
            erreurs.append(
                f"{nom}: {n} hand-written blocks, the debt allows only "
                f"{budget}. This ledger can only shrink."
            )
    # Hygiene: no stale entry in the ledger.
    for nom, budget in DETTE_MIGRATION.items():
        if nom not in vus:
            erreurs.append(f'DETTE_MIGRATION: "{nom}" no longer exists, remove it')
        elif vus[nom] == 0:
            erreurs.append(
                f'DETTE_MIGRATION: "{nom}" is migrated (0 block), remove it from the ledger'
            )
        elif vus[nom] < budget:
            erreurs.append(
                f'DETTE_MIGRATION: "{nom}" went down to {vus[nom]} blocks, '
                f"the ledger announces {budget} — update it."
            )
    return erreurs


# ── 3. Symboles cités en prose ──────────────────────────────────────────────


def variantes_denum(source: str):
    """The variants of every `enum` of the file, by brace counting.

    Serves as a **fallback** for the types rustdoc does not document
    (`pub(crate)`): exact membership itself comes from
    [`membres_par_type`], which reads the rustdoc.
    """
    noms = set()
    for m in re.finditer(r"\benum\s+[A-Z][A-Za-z0-9_]*[^{]*\{", source):
        profondeur, i = 1, m.end()
        while i < len(source) and profondeur:
            profondeur += (source[i] == "{") - (source[i] == "}")
            i += 1
        for ligne in source[m.end() : i].split("\n"):
            v = re.match(r"\s*([A-Z][A-Za-z0-9_]*)", ligne)
            if v:
                noms.add(v.group(1))
    return noms


def membres_par_type():
    """For every public type, the exact set of its members.

    Read from the rustdoc (`id="method.*"`, `variant.*`, `associatedconstant.*`),
    the only source that truly knows membership — a home-made parser would say
    a name exists *somewhere*, which would let a `Physics::Gauss` through.
    Trait methods count: `mesh.clone()` is a
    citation légitime.
    """
    carte = {}
    for p in DOC.rglob("*.html"):
        m = re.match(r"(?:struct|enum|trait)\.([A-Za-z0-9_]+)\.html$", p.name)
        if not m:
            continue
        h = p.read_text(errors="ignore")
        carte.setdefault(m.group(1), set()).update(
            re.findall(
                r'id="(?:method|tymethod|variant|associatedconstant)\.([A-Za-z0-9_]+)"',
                h,
            )
        )
    return carte


def symboles_du_crate():
    """Names of modules, functions, types and members defined in src/."""
    modules, fonctions, types, membres = set(), set(), set(), set()
    for p in (ROOT / "src").rglob("*.rs"):
        s = p.read_text(errors="ignore")
        modules |= set(re.findall(r"pub mod\s+([a-z_0-9]+)", s))
        fonctions |= set(re.findall(r"\bfn\s+([a-z_][a-z0-9_]*)", s))
        types |= set(
            re.findall(r"\b(?:struct|enum|trait|type)\s+([A-Z][A-Za-z0-9_]*)", s)
        )
        membres |= variantes_denum(s)
        membres |= set(re.findall(r"\bconst\s+([A-Z][A-Z0-9_]*)", s))
    return modules, fonctions, types, membres


def check_symboles():
    erreurs = []
    if not (DOC / "all.html").exists():
        return ["target/doc absent — lancer `cargo doc --no-deps --lib` d'abord"]
    modules, fonctions, types, membres = symboles_du_crate()
    modules.add("pyrucast")  # the crate root, as written in Rust
    par_type = membres_par_type()
    try:
        import pyrucast
    except ImportError:
        return ["pyrucast is not importable — run script/check_python.sh first"]

    utilises = set()
    for p in pages():
        for extrait in inline_codes(p.read_text()):
            # The whole path, not two segments: `ops::matrix::stiffness` cut
            # into pairs would leave the **last** segment unchecked — that is,
            # the name that moves most often.
            for m in re.finditer(
                r"\b(?:[A-Za-z_][A-Za-z0-9_]*::)+[A-Za-z_][A-Za-z0-9_]*\b", extrait
            ):
                chemin = m.group(0)
                if chemin in SYMBOLES_TOLERES:
                    utilises.add(chemin)
                    continue
                segments = chemin.split("::")
                if segments[0] in CRATES_EXTERNES or segments[0] in TYPES_EXTERNES:
                    continue
                if any(s.endswith("_") for s in segments):
                    continue  # truncated citation of the `points_…` kind
                faute = None
                for i, s in enumerate(segments):
                    dernier = i == len(segments) - 1
                    if s in TYPES_EXTERNES or s in CRATES_EXTERNES:
                        break
                    if s in par_type and not dernier:
                        # Public type: rustdoc gives the exact list of its
                        # members, no need to settle for a "this name
                        # existe quelque part ».
                        suivant = segments[i + 1]
                        if suivant not in par_type[s]:
                            faute = f"{s} n'a pas de membre « {suivant} »"
                        break
                    if s[0].isupper():
                        connu = s in types or (i > 0 and s in membres)
                        if not connu:
                            faute = f"nom inconnu ({s})"
                    elif dernier:
                        if s not in fonctions and s not in modules:
                            faute = f"fonction inconnue ({s})"
                    elif s not in modules:
                        # The exact shape of the August bug:
                        # `assemble::stiffness` survived the renaming
                        faute = f"module inconnu ({s})"
                    if faute:
                        break
                if faute:
                    erreurs.append(f'{rel(p)}: {faute} in "{chemin}"')
            for m in re.finditer(
                r"\bpyrucast\.([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b", extrait
            ):
                module, verbe = m.group(1), m.group(2)
                cle = f"pyrucast.{module}.{verbe}"
                if cle in SYMBOLES_TOLERES:
                    utilises.add(cle)
                    continue
                if verbe.endswith("_"):
                    continue  # truncated citation of the `points_…` kind
                objet = getattr(pyrucast, module, None)
                if objet is None:
                    erreurs.append(f"{rel(p)} : module Python inconnu — {cle}")
                elif not hasattr(objet, verbe):
                    erreurs.append(f"{rel(p)} : verbe Python inconnu — {cle}")
    for cle, raison in SYMBOLES_TOLERES.items():
        if not raison.strip():
            erreurs.append(f'SYMBOLES_TOLERES: "{cle}" without a written reason')
        elif cle not in utilises:
            erreurs.append(f'SYMBOLES_TOLERES: "{cle}" is no longer cited, remove it')
    return erreurs


# ── 4. Doctest coverage ratchet ─────────────────────────────────────────────

DOC = ROOT / "target" / "doc" / "pyrucast"


def delegations():
    """The **pure delegation** methods — those whose whole documentation is
    "see [the free function]".

    Elles n'ont aucune logique : elles appellent l'opérateur, receveur compris.
    The free function is the canonical form and carries the
    documentation (`CONVENTIONS.md`, « Le verbe exposé aussi en méthode ») :
    demanding an example of them would duplicate its own, and give a second
    text to let age.

    The criterion is that **intentional marker**, not the location: these
    blocks live in `src/ops/**/methods.rs` but also at the bottom of
    `src/ops/matrix.rs`, and some are macro-produced on `impl $T`. Searching
    en laissait passer la moitié.
    """
    noms = set()
    for p in (ROOT / "src").rglob("*.rs"):
        lignes = p.read_text(errors="ignore").split("\n")
        for i, ligne in enumerate(lignes):
            m = re.match(r"\s+pub fn ([a-z_0-9]+)", ligne)
            if not m:
                continue
            doc, k = [], i - 1
            while k >= 0 and lignes[k].strip().startswith("///"):
                doc.insert(0, lignes[k].strip())
                k -= 1
            if (
                len(doc) == 1
                and re.search(r"[Vv]oir \[", doc[0])
                and "fn@crate::ops::" in doc[0]
            ):
                noms.add(m.group(1))
    return noms


def api_publique():
    """The public set in rustdoc's sense: free items + methods.

    Free items come from `all.html`, which does not list methods; those are
    read on each type's page, in the `implementations` section alone — trait
    impls (`Debug`, `Clone`…) have no
    porter d'exemple.

    Everything is named by **full path**: thirteen types of the crate are
    homonyms (`Facet`, `Grid`, `Interpolation`…), and a short key would
    confondrait.
    """
    tous = (DOC / "all.html").read_text(errors="ignore")
    # `all.html` also lists what is **re-exported from another crate**: the
    # `pub use rayon::prelude::*` of `parallel` puts thirteen traits there
    # that are not ours. Their rustdoc page has no "source" link to
    # `src/pyrucast/` — that is what sets them apart from ours. Demanding an
    # example on `rayon::ParallelIterator` would amount to documenting rayon.
    libres = set()
    for href, nom in re.findall(r'<li><a href="([^"]+)">([^<]+)</a></li>', tous):
        page = DOC / href
        # The "source" link is relative: its `../` depth depends on the
        # page's. So we look for the segment, not the whole path.
        if page.is_file() and "/src/pyrucast/" not in page.read_text(errors="ignore"):
            continue
        libres.add(nom)
    methodes = set()
    for p in DOC.rglob("*.html"):
        m = re.match(r"(?:struct|enum|trait)\.([A-Za-z0-9_]+)\.html$", p.name)
        if not m:
            continue
        chemin = "::".join([*p.relative_to(DOC).parts[:-1], m.group(1)])
        h = p.read_text(errors="ignore")
        i = h.find('id="implementations"')
        if i < 0:
            continue
        # Stop at the first block that is no longer ours: trait impls, but
        # also the **methods inherited through `Deref`** (`Objects`
        # dereferences to `BTreeMap`). Demanding an example on
        # serait réclamer de documenter la bibliothèque standard.
        #
        # **Generic** and **synthetic** impls count just the same: a type
        # without a single hand-written trait impl has no
        # `trait-implementations` section, and without them the bound was
        # missing — hence the `from_subset`, `into_either` and `vzip` of
        # nalgebra and either that sat in the ledger as if they were ours.
        bornes = [
            x
            for x in (
                h.find('id="trait-implementations"'),
                h.find('id="deref-methods'),
                h.find('id="blanket-implementations"'),
                h.find('id="synthetic-implementations"'),
            )
            if x > i
        ]
        segment = h[i : min(bornes)] if bornes else h[i:]
        for nom in set(
            re.findall(r'id="(?:method|associatedconstant)\.([A-Za-z0-9_]+)"', segment)
        ):
            methodes.add(f"{chemin}::{nom}")
    # Delegations are documented by their target, not by themselves.
    deleguees = delegations()
    methodes = {m for m in methodes if m.split("::")[-1] not in deleguees}
    return libres, methodes


def items_documentes(publics):
    """The items carrying an example, per `cargo test --doc -- --list`.

    The path rustdoc gives a doctest is that of the module where the `impl`
    lives, not that of the type: `ops::matrix::Matrix::assemble` designates a
    method of `containers::matrix::Matrix`. Hence the fallback on the
    `Type::method` suffix, accepted only if it designates a single candidate.
    """
    sortie = subprocess.run(
        ["cargo", "test", "--doc", "--", "--list"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout
    par_suffixe = {}
    for item in publics:
        par_suffixe.setdefault("::".join(item.split("::")[-2:]), []).append(item)
    # A **re-export** only removes segments: `ops::mesh::triangulation`
    # réexporte `…::triangulation::cdt::delaunay_2d` sous
    # `…::triangulation::delaunay_2d`. The public path is therefore a
    # subsequence of the path rustdoc gives the doctest. That is narrower
    # than comparing the last segment alone, which would confuse the `new`s.
    par_dernier = {}
    for item in publics:
        par_dernier.setdefault(item.split("::")[-1], []).append(item)

    def sous_suite(court, long):
        it = iter(long)
        return all(seg in it for seg in court)

    documentes, ambigus = set(), []
    for m in re.finditer(r"^\S+ - (\S+) \(line \d+\): test$", sortie, re.M):
        # rustdoc names the doctest of a parameterized `impl` with its
        # parameters — `CellGeom<'a>::det_j_w` — where the type's page is
        # called `CellGeom`. Without this normalization, a documented item
        chemin = re.sub(r"<[^>]*>", "", m.group(1))
        if chemin in publics:
            documentes.add(chemin)
            continue
        candidats = par_suffixe.get("::".join(chemin.split("::")[-2:]), [])
        if not candidats:
            segments = chemin.split("::")
            candidats = [
                item
                for item in par_dernier.get(segments[-1], [])
                if sous_suite(item.split("::"), segments)
            ]
        if len(candidats) == 1:
            documentes.add(candidats[0])
        elif len(candidats) > 1:
            ambigus.append(f"{chemin} → {', '.join(sorted(candidats))}")
    return documentes, ambigus


def check_doctests(ratchet=False):
    if not (DOC / "all.html").exists():
        return ["target/doc absent — lancer `cargo doc --no-deps --lib` d'abord"]
    libres, methodes = api_publique()
    publics = libres | methodes
    documentes, ambigus = items_documentes(publics)
    if ambigus:
        return [
            f"doctest impossible à rattacher, plusieurs items candidats : {a}"
            for a in ambigus
        ]
    sans_exemple = publics - documentes

    if ratchet:
        LEDGER.write_text(
            "# Public items without an executable example — ratchet ledger.\n"
            "# It can only SHRINK: a new public item carries its example\n"
            "# (CONVENTIONS.md, règle 2), et un item documenté sort d'ici.\n"
            "# Régénérer : python script/doc_lint.py --ratchet\n"
            + "".join(f"{n}\n" for n in sorted(sans_exemple))
        )
        print(
            f"ledger rewritten: {len(sans_exemple)} items without an example, out of {len(publics)}"
        )
        return []

    if not LEDGER.exists():
        return [
            f"{LEDGER.name} missing — create it with `python script/doc_lint.py --ratchet`"
        ]
    connus = {
        l.strip()
        for l in LEDGER.read_text().splitlines()
        if l.strip() and not l.startswith("#")
    }
    erreurs = []
    for item in sorted(sans_exemple - connus):
        erreurs.append(
            f"{item}: public item without an executable example in its documentation "
            "(CONVENTIONS.md, rule 2). Add a doctest — `ignore` is forbidden."
        )
    for item in sorted(connus & documentes):
        erreurs.append(
            f"{item} : porte désormais un exemple — le retirer de "
            f"script/{LEDGER.name} (the ledger must keep only the real debt)."
        )
    for item in sorted(connus - publics):
        erreurs.append(
            f"{item}: no longer a public item — remove it from script/{LEDGER.name}."
        )
    return erreurs


# ── 5. Is the Python API cited by an executed example? ──────────────────────


def api_python():
    """The installed module's public surface, in three families.

    Deduplicated by **identity**: a class is reachable as `pyrucast.Coords`
    and as `pyrucast.coords.Coords`. Methods that **take an operator's name**
    are ruled out — they are the delegations (`mesh.skin()` for
    `pyrucast.mesh.skin(mesh)`), exempted on the Rust side for the same
    reason: the target carries the example.
    """
    import pyrucast

    classes, libres = {}, {}
    modules = [pyrucast] + [
        getattr(pyrucast, n)
        for n in dir(pyrucast)
        if inspect.ismodule(getattr(pyrucast, n, None))
    ]
    for mod in modules:
        court = mod.__name__.split(".")[-1]
        prefixe = "pyrucast" if mod is pyrucast else f"pyrucast.{court}"
        for nom in dir(mod):
            if nom.startswith("_"):
                continue
            obj = getattr(mod, nom)
            if inspect.isclass(obj):
                classes.setdefault(id(obj), (nom, obj))
            # Prefer the façade module to the compiled extension module.
            elif callable(obj) and (id(obj) not in libres or court != "_pyrucast"):
                libres[id(obj)] = f"{prefixe}.{nom}"

    verbes = {n.split(".")[-1] for n in libres.values()}
    methodes = {}
    for nom_classe, classe in classes.values():
        for m in dir(classe):
            if m.startswith("_") or not callable(getattr(classe, m, None)):
                continue
            if m not in verbes:  # une délégation : sa cible porte l'exemple
                methodes.setdefault(m, set()).add(nom_classe)
    return set(libres.values()), {n for n, _ in classes.values()}, methodes


def noms_cites():
    """Every identifier and attribute the book's examples write.

    Read by AST rather than by regular expression: a name in a comment or in
    a string does not count.
    """
    vus = set()
    for fichier in sorted(EXEMPLES_PY.glob("test_doc_*.py")):
        for noeud in ast.walk(ast.parse(fichier.read_text(encoding="utf-8"))):
            if isinstance(noeud, ast.Attribute):
                vus.add(noeud.attr)
            elif isinstance(noeud, ast.Name):
                vus.add(noeud.id)
    return vus


def check_api_python(ratchet=False):
    """Citation ratchet: every public Python entry appears in an executed
    example of the book.

    **What this proves, and what it does not.** The granularity is the
    *name*, not the call: `.get(` cited once holds for the `get` of every
    container. That is assumed — those methods are the common grammar of the
    aggregates, and one citation does cover the notion. What the guard
    guarantees is narrower than the Rust ratchet and enough for what is asked
    of it: **no public entry is missing from the examples**, and a new entry
    cannot get in without being shown.

    A class counts as cited as soon as one of its methods is: the Python of a
    idiomatique écrit `mesh.unit()`, jamais `SubMesh(...)`, et exiger le nom
    `Sub*` would amount to demanding a turn of phrase nobody writes.
    """
    try:
        fonctions, classes, methodes = api_python()
    except ImportError:
        return ["module `pyrucast` non installé — lancer `maturin develop` d'abord"]

    vus = noms_cites()
    absents = {f for f in fonctions if f.split(".")[-1] not in vus}
    absents |= {m for m in methodes if m not in vus}
    absents |= {
        c for c in classes if c not in vus and not (methodes_de(c, methodes) & vus)
    }

    if ratchet:
        LEDGER_PY.write_text(
            "# Public Python entries no example of the book cites.\n"
            "# Ratchet ledger: it can only SHRINK.\n"
            "# Régénérer : python script/doc_lint.py --ratchet-python\n"
            + "".join(f"{n}\n" for n in sorted(absents)),
            encoding="utf-8",
        )
        return []

    connus = set()
    if LEDGER_PY.exists():
        connus = {
            l.strip()
            for l in LEDGER_PY.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")
        }

    erreurs = []
    for nom in sorted(absents - connus):
        erreurs.append(
            f"{nom}: public Python entry no example of the book cites "
            f"(CONVENTIONS.md, rule 2). Add it to a `tests/python/test_doc_*.py`, "
            f"from which the page presenting it will pull it."
        )
    for nom in sorted(connus - absents):
        erreurs.append(
            f"{nom} : désormais cité — le retirer de script/python_coverage.txt "
            f"(the ledger must keep only the real debt)."
        )
    return erreurs


def methodes_de(nom_classe, methodes):
    """The method names proper to `nom_classe`."""
    return {m for m, classes in methodes.items() if nom_classe in classes}


# ── Point d'entrée ──────────────────────────────────────────────────────────

VERIFICATIONS = {
    "includes": ("resolution of the book's includes", check_includes),
    "fences": ("no page owns code", check_fences),
    "symboles": ("symbols cited in prose", check_symboles),
    "doctests": ("doctest coverage ratchet", check_doctests),
    "api-python": ("Python API citation ratchet", check_api_python),
}


def main(argv):
    if "--ratchet-python" in argv:
        erreurs = check_api_python(ratchet=True)
        for e in erreurs:
            print(f"    {e}", file=sys.stderr)
        return 1 if erreurs else 0
    if "--ratchet" in argv:
        erreurs = check_doctests(ratchet=True)
        for e in erreurs:
            print(f"    {e}", file=sys.stderr)
        return 1 if erreurs else 0
    demandees = [a for a in argv if not a.startswith("-")] or list(VERIFICATIONS)
    inconnues = [d for d in demandees if d not in VERIFICATIONS]
    if inconnues:
        print(f"vérification inconnue : {', '.join(inconnues)}", file=sys.stderr)
        print(f"disponibles : {', '.join(VERIFICATIONS)}", file=sys.stderr)
        return 2

    total = 0
    for nom in demandees:
        libelle, fonction = VERIFICATIONS[nom]
        erreurs = fonction()
        if erreurs:
            total += len(erreurs)
            print(f"\n✗ {nom} — {libelle}", file=sys.stderr)
            for e in erreurs:
                print(f"    {e}", file=sys.stderr)
        else:
            print(f"✓ {nom} — {libelle}")
    if total:
        print(f"\n{total} problème(s) de documentation.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
