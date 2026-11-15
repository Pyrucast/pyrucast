"""The book's operator reference, generated from the Python docstrings.

One script, two uses:

- an **mdBook preprocessor** (declared in `book/book.toml`, ahead of `links`):
  it replaces three directives in the pages, so nothing generated is ever
  committed;
- a **module** `script/doc_lint.py` imports for its guards (`audit`).

The directives:

- `{{#pymodule matrix}}` — the reference of a module: its docstring, then one
  entry per public function (prototype, docstring, twin methods, algorithm
  pages, example);
- `{{#pysummary matrix.stiffness matrix.mass}}` — a table, one row per name:
  short prototype linked to the entry, first line of the docstring;
- `{{#pyalgo mesh.regularize mesh.cleanup}}` — at the top of an algorithm
  page, or of the section of a page that tells the algorithm: declares the
  functions it explains. The place gets links to their entries, and each
  entry gets a link back that lands on the directive, labelled by the page
  and the section it stands in.

A directive preceded by a backslash is left alone (and loses the backslash),
as mdBook does for `\\{{#include}}`.

Everything is read **by AST**, never by importing the compiled module:
`python/pyrucast/<module>.py` gives `__all__` and where each name comes from,
the stub `_pyrucast/__init__.pyi` gives typed prototypes and docstrings, and
the few functions written in Python are read from their own file. Building the
book therefore needs neither the extension module nor any dependency.

Two naming conventions tie the rest together:

- **example** — an anchor named `<module>-<function>` (`matrix-stiffness`) in
  any `tests/python/test_doc_*.py` is shown under the entry. The hyphen sets
  it apart from the free anchors, which use underscores;
- **algorithm** — declared by the page itself, through `{{#pyalgo}}`.

An unknown name is an **error** and fails the build: a directive never renders
as an empty block.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book" / "src"

# The module holding what `pyrucast` exposes at top level (`save`, `load`…).
ROOT_MODULE = "pyrucast"
# The sections a docstring may carry, in this order.
SECTIONS = ("Parameters", "Returns", "Raises", "Notes", "See also")
# Above this width a prototype is written one parameter per line.
SIGNATURE_WIDTH = 88
# Longest first line a docstring may have: it is a row of the summary table.
SUMMARY_WIDTH = 80

DIRECTIVE = re.compile(r"(\\)?\{\{#(pymodule|pysummary|pyalgo)\b([^}]*)\}\}")
ANCHOR = re.compile(r"ANCHOR:\s*([\w-]+)\s*$", re.MULTILINE)
INLINE_CODE = re.compile(r"(?<![\[`])`([^`\n]+)`(?!`)")
INTRA_LINK = re.compile(r"\[`([^`\n]+)`\](?![(\[])")
HEADING = re.compile(r"(#{1,6})\s+(.+?)\s*$")
HEADING_ID = re.compile(r"\s*\{#[\w-]+\}$")
SYMBOL = re.compile(r"(pyrucast\.)?(?:(\w+)\.)?(\w+)(\(.*\))?")


# ── The API, read by AST ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Entry:
    """One public function, as the reference shows it."""

    module: str  # "matrix"; `ROOT_MODULE` for a top-level function
    name: str  # "stiffness" — the public name, not the stub's
    signature: str  # full typed prototype, qualified, wrapped when long
    short: str  # "stiffness(model, materials) → Matrix"
    summary: str  # first paragraph of the docstring, on one line
    body: str  # the rest of the docstring (Markdown)
    params: tuple[str, ...]  # parameter names, in order
    methods: tuple[str, ...]  # twin methods: ("Model.mass_matrix",)
    example: Path | None  # the test_doc_*.py file carrying the anchor
    doc: str  # the whole docstring, as written

    @property
    def key(self) -> str:
        """The public name a directive writes: `matrix.stiffness`, `save`."""
        return self.name if self.module == ROOT_MODULE else f"{self.module}.{self.name}"

    @property
    def anchor(self) -> str:
        """Heading id of the entry, and name of its example anchor."""
        return f"{self.module}-{self.name}"


@dataclass
class Api:
    entries: dict[str, Entry] = field(default_factory=dict)
    # module → (docstring, public function names in `__all__` order)
    modules: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
    # example anchors of the `<module>-<function>` form → file
    anchors: dict[str, Path] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class _OptionalToUnion(ast.NodeTransformer):
    """`typing.Optional[X]` → `X | None`, as a user would write it."""

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if ast.unparse(node.value).split(".")[-1] == "Optional":
            return ast.BinOp(left=node.slice, op=ast.BitOr(), right=ast.Constant(None))
        return node


def _type(annotation) -> str:
    if annotation is None:
        return ""
    tree = _OptionalToUnion().visit(copy.deepcopy(annotation))
    return re.sub(r"\b(?:builtins|typing)\.", "", ast.unparse(tree))


def _parameters(args: ast.arguments):
    """`(name, rendered text, required)` per parameter, with the `/` and `*`
    markers as nameless items."""

    def one(arg, default, prefix=""):
        kind = _type(arg.annotation)
        text = prefix + arg.arg + (f": {kind}" if kind else "")
        if default is not None:
            text += (" = " if kind else "=") + ast.unparse(default)
        return (arg.arg, text, default is None and not prefix)

    out = []
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for i, (arg, default) in enumerate(zip(positional, defaults)):
        out.append(one(arg, default))
        if args.posonlyargs and i == len(args.posonlyargs) - 1:
            out.append(("", "/", False))
    if args.vararg:
        out.append(one(args.vararg, None, "*"))
    elif args.kwonlyargs:
        out.append(("", "*", False))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        out.append(one(arg, default))
    if args.kwarg:
        out.append(one(args.kwarg, None, "**"))
    return out


def _signature(qualified: str, node: ast.FunctionDef) -> str:
    texts = [text for _, text, _ in _parameters(node.args)]
    returns = _type(node.returns)
    tail = f" -> {returns}" if returns else ""
    flat = f"{qualified}({', '.join(texts)}){tail}"
    if len(flat) <= SIGNATURE_WIDTH:
        return flat
    return "\n".join([f"{qualified}("] + [f"    {t}," for t in texts] + [f"){tail}"])


def _short(name: str, node: ast.FunctionDef) -> str:
    """Required parameters by name, `…` standing for the optional ones."""
    params = _parameters(node.args)
    shown = [n for n, _, required in params if required]
    if any(n and not required for n, _, required in params):
        shown.append("…")
    returns = _type(node.returns)
    return f"{name}({', '.join(shown)})" + (f" → {returns}" if returns else "")


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _method_twins(stub: Path) -> dict[str, list[str]]:
    """Docstring → the methods carrying it. A method exposing an operator on
    its subject is generated with the operator's own docstring, so equal text
    is what identifies the pair."""
    twins: dict[str, list[str]] = {}
    for cls in ast.parse(stub.read_text(encoding="utf-8")).body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                doc = ast.get_docstring(node)
                if doc:
                    twins.setdefault(doc, []).append(f"{cls.name}.{node.name}")
    return twins


def _example_anchors(root: Path, errors: list[str]) -> dict[str, Path]:
    """The example anchors of the `<module>-<function>` form, and their file.

    An anchor without its end, or with nothing in between, is an error here:
    mdBook would render it as an empty block without a word."""
    anchors: dict[str, Path] = {}
    for path in sorted((root / "tests" / "python").glob("test_doc_*.py")):
        source = path.read_text(encoding="utf-8")
        for start in ANCHOR.finditer(source):
            name = start.group(1)
            if "-" not in name:
                continue
            if name in anchors:
                errors.append(
                    f'example anchor "{name}" is defined twice: '
                    f"{anchors[name].name} and {path.name}"
                )
            anchors[name] = path
            end = re.search(
                rf"ANCHOR_END:\s*{re.escape(name)}\s*$",
                source[start.end() :],
                re.MULTILINE,
            )
            if not end:
                errors.append(f'{path.name}: ANCHOR_END "{name}" missing')
            elif not source[start.end() : start.end() + end.start()].strip("# \n"):
                errors.append(f'{path.name}: example anchor "{name}" is empty')
    return anchors


def load_api(root: Path = ROOT) -> Api:
    """The public functions of `pyrucast` and of its verb modules."""
    api = Api()
    package = root / "python" / "pyrucast"
    stub = package / "_pyrucast" / "__init__.pyi"
    sources = {"_pyrucast": _functions(stub)}
    twins = _method_twins(stub)
    api.anchors = _example_anchors(root, api.errors)

    for path in sorted(package.glob("*.py")):
        is_root = path.name == "__init__.py"
        if path.name.startswith("_") and not is_root:
            continue
        module = ROOT_MODULE if is_root else path.stem
        tree = ast.parse(path.read_text(encoding="utf-8"))
        origin = {}  # public name → (private module, name there)
        exported: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                for alias in node.names:
                    origin[alias.asname or alias.name] = (node.module, alias.name)
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
            ):
                exported = ast.literal_eval(node.value)

        names = []
        for name in exported:
            if name not in origin:
                # At top level, `__all__` also lists the verb modules.
                if not is_root:
                    api.errors.append(
                        f"pyrucast.{module}.{name}: listed in __all__ but not imported"
                    )
                continue
            source, original = origin[name]
            if source not in sources:
                sources[source] = _functions(package / f"{source}.py")
            node = sources[source].get(original)
            if node is None:
                # At top level, the containers are classes: not operators.
                if not is_root:
                    api.errors.append(
                        f"pyrucast.{module}.{name}: no function `{original}` in {source}"
                    )
                continue
            doc = ast.get_docstring(node) or ""
            head, _, body = doc.partition("\n\n")
            qualified = "pyrucast." + ("" if is_root else f"{module}.") + name
            entry = Entry(
                module=module,
                name=name,
                signature=_signature(qualified, node),
                short=_short(name, node),
                summary=" ".join(head.split()),
                body=body.strip("\n"),
                params=tuple(n for n, _, _ in _parameters(node.args) if n),
                methods=tuple(twins.get(doc, ())),
                example=api.anchors.get(f"{module}-{name}"),
                doc=doc,
            )
            api.entries[entry.key] = entry
            names.append(name)
        if names:
            api.modules[module] = (ast.get_docstring(tree) or "", names)
    return api


# ── The book, as far as the directives see it ───────────────────────────────


@dataclass
class Page:
    path: str  # relative to book/src, POSIX: "reference/matrix.md"
    title: str
    content: str


@dataclass
class Layout:
    """Where things live, read off the directives before any expansion."""

    homes: dict[str, list[str]] = field(default_factory=dict)  # module → pages
    listed: dict[str, list[str]] = field(default_factory=dict)  # key → pages
    algorithms: dict[str, list[Algorithm]] = field(default_factory=dict)  # key → …


@dataclass(frozen=True)
class Algorithm:
    """One `{{#pyalgo}}`: the place an entry of the reference links to."""

    page: Page
    fragment: str  # id laid where the directive stands: "pyalgo-2"
    section: str  # heading the directive stands under, "" under the page title

    @property
    def label(self) -> str:
        return (
            f"{self.page.title} — {self.section}" if self.section else self.page.title
        )


def _directives(text: str):
    for m in DIRECTIVE.finditer(text):
        if not m.group(1):
            yield m.group(2), m.group(3).split(), m.start()


def _section(text: str, position: int) -> str:
    """Heading of the section `position` falls in — none under the page title."""
    section, fenced = "", False
    for line in text[:position].split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            heading = HEADING.match(line)
            if heading:
                level, title = heading.groups()
                section = "" if level == "#" else HEADING_ID.sub("", title)
    return section


def read_layout(pages: list[Page]) -> Layout:
    layout = Layout()
    for page in pages:
        rank = 0
        for kind, names, position in _directives(page.content):
            if kind == "pyalgo":
                rank += 1
                place = Algorithm(
                    page, f"pyalgo-{rank}", _section(page.content, position)
                )
            for name in names:
                if kind == "pymodule":
                    layout.homes.setdefault(name, []).append(page.path)
                elif kind == "pysummary":
                    layout.listed.setdefault(name, []).append(page.path)
                else:
                    layout.algorithms.setdefault(name, []).append(place)
    return layout


def _relative(target: str, page: Page) -> str:
    return os.path.relpath(target, os.path.dirname(page.path) or ".").replace(
        os.sep, "/"
    )


class Renderer:
    def __init__(self, api: Api, layout: Layout, root: Path = ROOT):
        self.api, self.layout, self.root = api, layout, root
        self.errors: list[str] = []

    # ── links ──

    def link(self, entry: Entry, page: Page) -> str | None:
        """Markdown target of `entry` seen from `page`, or `None` when no page
        renders its module."""
        homes = self.layout.homes.get(entry.module)
        if not homes:
            return None
        here = homes[0] == page.path
        return ("" if here else _relative(homes[0], page)) + f"#{entry.anchor}"

    def _resolve(self, span: str, entry: Entry | None, explicit: bool) -> Entry | None:
        """The entry an `inline code` span names, if it names one."""
        m = SYMBOL.fullmatch(span)
        if not m:
            return None
        prefix, module, name, call = m.groups()
        if module:
            return self.api.entries.get(f"{module}.{name}")
        if prefix:
            return self.api.entries.get(name)
        if entry is None or name == entry.name:
            return None
        # A bare name is first of all a parameter of the function at hand.
        if name in entry.params and not (call or explicit):
            return None
        scope = "" if entry.module == ROOT_MODULE else f"{entry.module}."
        return self.api.entries.get(scope + name)

    def prose(self, text: str, page: Page, entry: Entry | None = None) -> str:
        """Docstring Markdown made fit for a page: section titles become bold
        labels, and a name between backticks that designates a public function
        becomes a link to its entry."""

        def linked(span: str, explicit: bool) -> str:
            target = self._resolve(span, entry, explicit)
            url = self.link(target, page) if target else None
            return f"[`{span}`]({url})" if url else f"`{span}`"

        out, fenced = [], False
        for line in text.split("\n"):
            if line.lstrip().startswith("```"):
                fenced = not fenced
            elif not fenced:
                title = re.fullmatch(r"#+\s*(.+?)\s*", line)
                if title and title.group(1) in SECTIONS:
                    line = f"**{title.group(1)}**"
                else:
                    line = INTRA_LINK.sub(lambda m: linked(m.group(1), True), line)
                    line = INLINE_CODE.sub(lambda m: linked(m.group(1), False), line)
            out.append(line)
        return "\n".join(out)

    # ── blocks ──

    def table(self, entries: list[Entry], page: Page) -> str:
        rows = ["| Prototype | Description |", "|---|---|"]
        for e in entries:
            scope = "" if e.module == ROOT_MODULE else f"{e.module}."
            proto = f"`{scope}{e.short}`".replace("|", "\\|")
            url = self.link(e, page)
            if url is None:
                self.errors.append(
                    f"{page.path}: no page renders the module of `{e.key}` "
                    f"(missing {{{{#pymodule {e.module}}}}})"
                )
                url = ""
            summary = self.prose(e.summary, page, e).replace("|", "\\|")
            rows.append(f"| [{proto}]({url}) | {summary} |")
        return "\n".join(rows)

    def entry(self, e: Entry, page: Page) -> str:
        parts = [
            f"## `{e.name}` {{#{e.anchor}}}",
            f"```python\n{e.signature}\n```",
            self.prose(e.summary, page, e),
        ]
        if e.body:
            parts.append(self.prose(e.body, page, e))
        if e.methods:
            methods = ", ".join(f"`{m}`" for m in e.methods)
            parts.append(f"**Also a method:** {methods}.")
        algorithms = self.layout.algorithms.get(e.key, [])
        if algorithms:
            links = ", ".join(
                f"[{a.label}]({_relative(a.page.path, page)}#{a.fragment})"
                for a in algorithms
            )
            parts.append(f"**Algorithm:** {links}.")
        if e.example:
            source = os.path.relpath(
                e.example, (self.root / "book" / "src" / page.path).parent
            )
            include = (
                "{{#include " + source.replace(os.sep, "/") + f":{e.anchor}" + "}}"
            )
            parts.append(f"**Example**\n\n```python\n{include}\n```")
        return "\n\n".join(parts)

    def module(self, name: str, page: Page) -> str:
        doc, names = self.api.modules[name]
        scope = "" if name == ROOT_MODULE else f"{name}."
        entries = [self.api.entries[scope + n] for n in names]
        parts = [self.prose(doc, page)] if doc and name != ROOT_MODULE else []
        parts.append(self.table(entries, page))
        parts += [self.entry(e, page) for e in entries]
        return "\n\n".join(parts)

    def algorithm(self, entries: list[Entry], page: Page, rank: int) -> str:
        """The line a `{{#pyalgo}}` becomes. It carries the id the entries of
        the reference link to: they land on the section, not on the page."""
        links = []
        for e in entries:
            url = self.link(e, page)
            links.append(f"[`{e.key}`]({url})" if url else f"`{e.key}`")
        return (
            f'<a id="pyalgo-{rank}"></a>Fonctions décrites : ' + ", ".join(links) + "."
        )

    # ── a page ──

    def expand(self, page: Page) -> str:
        rank = 0  # of the `pyalgo` at hand in the page, as `read_layout` counts

        def replace(m: re.Match) -> str:
            nonlocal rank
            if m.group(1):  # escaped: shown as written
                return m.group(0)[1:]
            kind, names = m.group(2), m.group(3).split()
            rank += kind == "pyalgo"
            if not names:
                self.errors.append(f"{page.path}: {{{{#{kind}}}}} names nothing")
                return ""
            known = self.api.modules if kind == "pymodule" else self.api.entries
            unknown = [n for n in names if n not in known]
            if unknown:
                what = "module" if kind == "pymodule" else "function"
                self.errors.append(
                    f"{page.path}: {{{{#{kind}}}}} names unknown {what}(s): "
                    + ", ".join(unknown)
                )
                return ""
            if kind == "pymodule":
                return "\n\n".join(self.module(n, page) for n in names)
            entries = [self.api.entries[n] for n in names]
            if kind == "pysummary":
                return self.table(entries, page)
            return self.algorithm(entries, page, rank)

        return DIRECTIVE.sub(replace, page.content)


# ── Guard: what `doc_lint.py reference` checks ──────────────────────────────


def _title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
    return m.group(1) if m else fallback


def pages_on_disk(root: Path = ROOT) -> list[Page]:
    book = root / "book" / "src"
    pages = []
    for path in sorted(book.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(book)).replace(os.sep, "/")
        pages.append(Page(rel, _title(text, path.stem), text))
    return pages


def audit(root: Path = ROOT) -> list[str]:
    """Every public function is rendered once and listed once; every name a
    directive or an example anchor writes designates a function that exists."""
    api = load_api(root)
    pages = pages_on_disk(root)
    layout = read_layout(pages)
    renderer = Renderer(api, layout, root)
    for page in pages:
        renderer.expand(page)
    errors = list(api.errors) + renderer.errors

    for module in api.modules:
        homes = layout.homes.get(module, [])
        if len(homes) != 1:
            where = ", ".join(homes) or "no page"
            errors.append(
                f"module `{module}`: rendered by {where} — expected exactly one "
                f"{{{{#pymodule {module}}}}}"
            )
    for key in api.entries:
        listed = layout.listed.get(key, [])
        if len(listed) != 1:
            where = (
                f"{len(listed)} times ({', '.join(listed)})" if listed else "nowhere"
            )
            errors.append(
                f"`{key}`: listed {where} by a {{{{#pysummary}}}} — expected once"
            )
    wanted = {e.anchor for e in api.entries.values()}
    for anchor, path in api.anchors.items():
        if anchor not in wanted:
            errors.append(
                f'{path.name}: anchor "{anchor}" has the <module>-<function> form '
                "but names no public function"
            )
    return errors


# ── mdBook preprocessor ─────────────────────────────────────────────────────


def _chapters(node):
    """Every chapter of the book handed over by mdBook, whatever the nesting."""
    if isinstance(node, dict):
        if "content" in node and "name" in node:
            yield node
        for value in node.values():
            yield from _chapters(value)
    elif isinstance(node, list):
        for value in node:
            yield from _chapters(value)


def main(argv: list[str]) -> int:
    if argv[:1] == ["supports"]:
        return 0  # plain Markdown out: every renderer is served
    _context, book = json.load(sys.stdin)
    chapters = [c for c in _chapters(book) if c.get("path")]
    pages = [
        Page(c["path"].replace(os.sep, "/"), c["name"], c["content"]) for c in chapters
    ]
    api = load_api()
    renderer = Renderer(api, read_layout(pages))
    for chapter, page in zip(chapters, pages):
        chapter["content"] = renderer.expand(page)
    errors = api.errors + renderer.errors
    if errors:
        for error in errors:
            print(f"book_pyapi: {error}", file=sys.stderr)
        return 1
    json.dump(book, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
