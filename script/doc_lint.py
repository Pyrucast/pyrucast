#!/usr/bin/env python3
"""Garde-fous de la documentation — quatre vérifications, aucune compilation.

Appelé par `script/check_doc.{sh,ps1}`, après `cargo doc` (dont il lit la
sortie). Les règles qu'il fait respecter sont écrites dans `CONVENTIONS.md`,
partie « Documentation et tests », et racontées dans
`book/src/developper/documentation-et-tests.md`.

1. `includes`   — chaque `{{#include}}` du book résout : fichier, ancre, et un
                  texte non vide. C'est le garde-fou le plus important, parce
                  que le mécanisme porteur n'est gardé par rien d'autre : une
                  ancre inexistante rend un bloc **vide**, code de retour 0,
                  sans un mot.
2. `fences`     — aucune page ne possède de code : tout bloc `rust`/`python`
                  contient un include, sans exception.
3. `symboles`   — la prose ne cite pas de symbole disparu. Rien d'autre ne
                  couvre un paragraphe, et c'est là qu'étaient trois des neuf
                  erreurs trouvées en août 2026.
4. `doctests`   — cliquet de couverture : un item public nouveau porte son
                  exemple, et la dette existante ne peut que décroître.

Chaque registre de dérogations porte une **raison** et son test d'hygiène, qui
échoue sur une entrée périmée — le motif de
`tests/python/test_mirror_completeness.py`, réutilisé et non réinventé.

    python script/doc_lint.py            # tout
    python script/doc_lint.py includes   # une seule vérification
    python script/doc_lint.py --ratchet  # réécrit script/doc_coverage.txt
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book" / "src"
LEDGER = ROOT / "script" / "doc_coverage.txt"

# ── Registres ───────────────────────────────────────────────────────────────


# Dette de migration : pages qui possèdent encore du code recopié. **Ce
# registre ne peut que décroître** — l'hygiène refuse qu'une page dépasse son
# compte, et exige qu'une page tombée à zéro soit retirée. Rien ne s'y ajoute.
DETTE_MIGRATION = {}

# Symboles cités en prose que l'audit ne doit pas chercher à résoudre.
SYMBOLES_TOLERES = {
    "T::union_subs": "T est un paramètre de type, pas un type concret",
    "ops::mesher": "nom historique, cité pour raconter le renommage du 2026-08-03",
    "SubMesh::connectivity": "pub(crate) : la page Parallélisme décrit la machinerie interne",
    "Type::membre": "métavariable de la page Documentation et tests",
    "Coords::acquire": "nom fautif, cité pour raconter ce que le garde-fou a trouvé",
}

# Racines qui ne viennent pas du crate : la partie droite ne s'y vérifie pas.
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
    """Les pages du book, chemin relatif à book/src, triées."""
    return sorted(p for p in BOOK.rglob("*.md"))


def rel(p: Path) -> str:
    return str(p.relative_to(BOOK)).replace("\\", "/")


def fences(text: str):
    """Les blocs délimités : (langue, lignes, numéro de la ligne d'ouverture)."""
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
    """Les passages en `code inline`, hors blocs délimités.

    Restreindre l'audit des symboles à ces passages est ce qui écarte les faux
    positifs — noms de fichiers (`pyrucast.pth`), domaines (`pyrucast.github.io`)
    et phrases courantes qui contiennent un `::` par accident.
    """
    without_blocks = re.sub(r"```.*?```", "", text, flags=re.S)
    return re.findall(r"`([^`\n]+)`", without_blocks)


# ── 1. Résolution des includes ──────────────────────────────────────────────


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
                erreurs.append(f"{ou} : fichier introuvable — {chemin}")
                continue
            source = cible.read_text(errors="ignore")
            if not ancre:
                if not source.strip():
                    erreurs.append(f"{ou} : le fichier inclus est vide — {chemin}")
                continue
            debut = re.search(rf"ANCHOR:\s*{re.escape(ancre)}\s*$", source, re.M)
            fin = re.search(rf"ANCHOR_END:\s*{re.escape(ancre)}\s*$", source, re.M)
            if not debut or not fin:
                manque = "ANCHOR" if not debut else "ANCHOR_END"
                erreurs.append(f"{ou} : {manque} « {ancre} » absente de {chemin}")
                continue
            corps = source[debut.end() : fin.start()]
            if not corps.strip():
                erreurs.append(f"{ou} : l'ancre « {ancre} » de {chemin} est vide")
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
                f"{nom} : {n} bloc(s) de code écrit(s) à la main. "
                "Aucune page ne possède de code : écrire un test ou un exemple, "
                "l'encadrer d'ANCHOR, et l'inclure (CONVENTIONS.md, règle 1)."
            )
        elif n > budget:
            erreurs.append(
                f"{nom} : {n} blocs écrits à la main, la dette n'en autorise que "
                f"{budget}. Ce registre ne peut que décroître."
            )
    # Hygiène : pas d'entrée périmée dans le registre.
    for nom, budget in DETTE_MIGRATION.items():
        if nom not in vus:
            erreurs.append(f"DETTE_MIGRATION : « {nom} » n'existe plus, la retirer")
        elif vus[nom] == 0:
            erreurs.append(
                f"DETTE_MIGRATION : « {nom} » est migrée (0 bloc), la retirer du registre"
            )
        elif vus[nom] < budget:
            erreurs.append(
                f"DETTE_MIGRATION : « {nom} » est descendue à {vus[nom]} blocs, "
                f"le registre en annonce {budget} — mettre à jour."
            )
    return erreurs


# ── 3. Symboles cités en prose ──────────────────────────────────────────────


def variantes_denum(source: str):
    """Les variantes de chaque `enum` du fichier, par comptage d'accolades.

    Sert de **repli** pour les types que la rustdoc ne documente pas
    (`pub(crate)`) : l'appartenance exacte, elle, vient de
    [`membres_par_type`], qui lit la rustdoc.
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
    """Pour chaque type public, l'ensemble exact de ses membres.

    Lu dans la rustdoc (`id="method.*"`, `variant.*`, `associatedconstant.*`),
    qui est la seule source qui connaisse vraiment l'appartenance — un parseur
    maison dirait qu'un nom existe *quelque part*, ce qui laisserait passer un
    `Physics::Gauss`. Les méthodes de traits comptent : `mesh.clone()` est une
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
    """Noms de modules, fonctions, types et membres définis dans src/."""
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
    modules.add("pyrucast")  # la racine du crate, telle qu'on l'écrit en Rust
    par_type = membres_par_type()
    try:
        import pyrucast
    except ImportError:
        return ["pyrucast n'est pas importable — lancer script/check_python.sh d'abord"]

    utilises = set()
    for p in pages():
        for extrait in inline_codes(p.read_text()):
            # Le chemin entier, pas deux segments : `ops::matrix::stiffness`
            # découpé en paires laisserait le **dernier** segment sans
            # vérification — c'est-à-dire le nom qui bouge le plus souvent.
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
                    continue  # citation tronquée du genre `points_…`
                faute = None
                for i, s in enumerate(segments):
                    dernier = i == len(segments) - 1
                    if s in TYPES_EXTERNES or s in CRATES_EXTERNES:
                        break
                    if s in par_type and not dernier:
                        # Type public : rustdoc donne la liste exacte de ses
                        # membres, on n'a plus à se contenter d'un « ce nom
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
                        # La forme exacte du bug d'août : `assemble::stiffness`
                        # a survécu au renommage parce que rien ne lisait la prose.
                        faute = f"module inconnu ({s})"
                    if faute:
                        break
                if faute:
                    erreurs.append(f"{rel(p)} : {faute} dans « {chemin} »")
            for m in re.finditer(
                r"\bpyrucast\.([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b", extrait
            ):
                module, verbe = m.group(1), m.group(2)
                cle = f"pyrucast.{module}.{verbe}"
                if cle in SYMBOLES_TOLERES:
                    utilises.add(cle)
                    continue
                if verbe.endswith("_"):
                    continue  # citation tronquée du genre `points_…`
                objet = getattr(pyrucast, module, None)
                if objet is None:
                    erreurs.append(f"{rel(p)} : module Python inconnu — {cle}")
                elif not hasattr(objet, verbe):
                    erreurs.append(f"{rel(p)} : verbe Python inconnu — {cle}")
    for cle, raison in SYMBOLES_TOLERES.items():
        if not raison.strip():
            erreurs.append(f"SYMBOLES_TOLERES : « {cle} » sans raison écrite")
        elif cle not in utilises:
            erreurs.append(f"SYMBOLES_TOLERES : « {cle} » n'est plus cité, le retirer")
    return erreurs


# ── 4. Cliquet de couverture des doctests ───────────────────────────────────

DOC = ROOT / "target" / "doc" / "pyrucast"


def delegations():
    """Les méthodes de pure délégation, `Type::verbe`.

    Elles vivent dans `src/ops/**/methods.rs`, ne contiennent aucune logique et
    appellent la fonction libre, receveur compris. C'est la **fonction libre**
    qui est la forme canonique et qui porte la documentation
    (`CONVENTIONS.md`, « Le verbe exposé aussi en méthode ») : leur réclamer un
    exemple dupliquerait le sien, et alourdirait de six cents lignes des
    fichiers dont tout l'objet est de tenir en une ligne par verbe.
    """
    noms = set()
    for p in (ROOT / "src" / "ops").rglob("methods.rs"):
        source = p.read_text(errors="ignore")
        typ = None
        for ligne in source.split("\n"):
            m = re.match(r"impl(?:<[^>]*>)? ([A-Za-z0-9_]+)", ligne)
            if m:
                typ = m.group(1)
                continue
            m = re.match(r"\s+pub fn ([a-z_0-9]+)", ligne)
            if m and typ:
                noms.add(f"{typ}::{m.group(1)}")
    return noms


def api_publique():
    """L'ensemble public au sens de rustdoc : items libres + méthodes.

    Les items libres viennent de `all.html`, qui ne liste pas les méthodes ;
    celles-ci se lisent sur la page de chaque type, dans la seule section
    `implementations` — les impls de traits (`Debug`, `Clone`…) n'ont pas à
    porter d'exemple.

    Tout est nommé par **chemin complet** : treize types du crate sont
    homonymes (`Facet`, `Grid`, `Interpolation`…), et une clé courte les
    confondrait.
    """
    tous = (DOC / "all.html").read_text(errors="ignore")
    libres = set(re.findall(r'<li><a href="[^"]+">([^<]+)</a></li>', tous))
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
        # S'arrêter au premier bloc qui n'est plus le nôtre : les impls de
        # traits, mais aussi les **méthodes héritées par `Deref`** (`Objects`
        # déréférence vers `BTreeMap`). Exiger un exemple sur `BTreeMap::range`
        # serait réclamer de documenter la bibliothèque standard.
        bornes = [
            x
            for x in (h.find('id="trait-implementations"'), h.find('id="deref-methods'))
            if x > i
        ]
        segment = h[i : min(bornes)] if bornes else h[i:]
        for nom in set(
            re.findall(r'id="(?:method|associatedconstant)\.([A-Za-z0-9_]+)"', segment)
        ):
            methodes.add(f"{chemin}::{nom}")
    # Les délégations sont documentées par leur cible, pas par elles-mêmes.
    deleguees = delegations()
    methodes = {m for m in methodes if "::".join(m.split("::")[-2:]) not in deleguees}
    return libres, methodes


def items_documentes(publics):
    """Les items portant un exemple, d'après `cargo test --doc -- --list`.

    Le chemin que rustdoc donne à un doctest est celui du module où vit
    l'`impl`, pas celui du type : `ops::matrix::Matrix::assemble` désigne une
    méthode de `containers::matrix::Matrix`. D'où le repli sur le suffixe
    `Type::methode`, accepté seulement s'il ne désigne qu'un candidat.
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

    documentes, ambigus = set(), []
    for m in re.finditer(r"^\S+ - (\S+) \(line \d+\): test$", sortie, re.M):
        chemin = m.group(1)
        if chemin in publics:
            documentes.add(chemin)
            continue
        candidats = par_suffixe.get("::".join(chemin.split("::")[-2:]), [])
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
            "# Items publics sans exemple exécutable — registre du cliquet.\n"
            "# Il ne peut que RÉTRÉCIR : un item public nouveau porte son exemple\n"
            "# (CONVENTIONS.md, règle 2), et un item documenté sort d'ici.\n"
            "# Régénérer : python script/doc_lint.py --ratchet\n"
            + "".join(f"{n}\n" for n in sorted(sans_exemple))
        )
        print(
            f"registre réécrit : {len(sans_exemple)} items sans exemple, sur {len(publics)}"
        )
        return []

    if not LEDGER.exists():
        return [
            f"{LEDGER.name} absent — le créer avec `python script/doc_lint.py --ratchet`"
        ]
    connus = {
        l.strip()
        for l in LEDGER.read_text().splitlines()
        if l.strip() and not l.startswith("#")
    }
    erreurs = []
    for item in sorted(sans_exemple - connus):
        erreurs.append(
            f"{item} : item public sans exemple exécutable dans sa documentation "
            "(CONVENTIONS.md, règle 2). Ajouter un doctest — `ignore` est proscrit."
        )
    for item in sorted(connus & documentes):
        erreurs.append(
            f"{item} : porte désormais un exemple — le retirer de "
            f"script/{LEDGER.name} (le registre ne doit garder que la dette réelle)."
        )
    for item in sorted(connus - publics):
        erreurs.append(
            f"{item} : n'est plus un item public — le retirer de script/{LEDGER.name}."
        )
    return erreurs


# ── Point d'entrée ──────────────────────────────────────────────────────────

VERIFICATIONS = {
    "includes": ("résolution des includes du book", check_includes),
    "fences": ("aucune page ne possède de code", check_fences),
    "symboles": ("symboles cités en prose", check_symboles),
    "doctests": ("cliquet de couverture des doctests", check_doctests),
}


def main(argv):
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
