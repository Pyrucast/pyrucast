#!/usr/bin/env python3
"""Reference-element figures of the book — pure Python, no dependency.

Writes, for each element of `ELEMENTS`, three SVG files into
`book/src/elements/img/`:

- `<name>-reference.svg`  : the reference element, nodes numbered and coloured
                            by role, axes;
- `<name>-quadrature.svg` : the `"GAUSS"` rule next to the `"REDUCED"` rule;
- `<name>-thumb.svg`      : a small picture for the catalogue of `elements/index.md`.

Like the tutorial figures, the SVGs are **committed artefacts**: `mdbook build`
never regenerates them. After touching this script or a node table, run

    python script/elements_figures.py            # rewrite the SVGs
    python script/elements_figures.py --check    # fail if they are stale

**Single source of truth.** The node coordinates are read from the table of the
element's own page (`book/src/elements/<name>.md`, section « Élément de
référence »): the figure cannot disagree with the page. The quadrature rules are
recomputed here from their closed form; `tests/python/test_doc_elements_figures.py`
compares them with the tables `pyrucast` returns (`gauss_xi`, `gauss_weight`).

3-D elements are drawn in a fixed oblique projection, hidden edges dashed.
Visibility is computed (outward normal · direction to the viewer), so any convex
polyhedron works once its faces are listed.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "book" / "src" / "elements"
OUT = PAGES / "img"

# ── Palette: mid-tones that read on the light *and* the dark mdBook themes,
#    because an SVG loaded through <img> cannot follow the theme: every figure
#    carries its own white card.
INK = "#26323f"
MUTED = "#6b7785"
FILL = "#eef3f8"
FACE_TINT = {2: "#f6f9fc", 0: "#e2eaf3", 1: "#d2dde9"}  # by dominant normal axis
ROLE_COLOR = {
    "sommet": "#1f5a99",
    "extrémité": "#1f5a99",
    "milieu": "#c8691a",
    "milieu d'arête": "#c8691a",
    "centre de face": "#2a8a5c",
    "centre": "#8a3fa8",
}
GAUSS_COLOR = "#c0392b"
FONT = "Helvetica, Arial, sans-serif"

# Oblique projection: ξ to the right, ζ up, η receding at `angle` degrees with
# length ratio `k` (right-handed). Default: up-right, as for the cube. An element
# whose faces would hide each other under that view (tetrahedron, prism) takes
# its own `oblique=(k, angle)`.
OBLIQUE = (0.55, 30.0)

# ── Reading the node table of a page ────────────────────────────────────────

MATH = re.compile(r"^\\\\\(\s*(.*?)\s*\\\\\)$")  # a cell that is only \\( … \\)


def parse_number(text: str) -> float:
    """`-1`, `+1`, `0`, `\\tfrac12`, `\\tfrac{1}{3}`, `-\\tfrac12` → float."""
    t = text.strip().replace(" ", "")
    sign = -1.0 if t.startswith("-") else 1.0
    t = t.lstrip("+-")
    m = re.fullmatch(r"\\tfrac(\d)(\d)", t) or re.fullmatch(r"\\tfrac\{(\d+)\}\{(\d+)\}", t)
    if m:
        return sign * int(m.group(1)) / int(m.group(2))
    return sign * float(t)


def read_nodes(page: Path) -> list[tuple[float, ...]]:
    """Coordinates of the reference nodes, in the order of the page's table.

    A row may carry several nodes side by side (`| 0 | … | | 3 | … |`), and a node
    may be followed by extra cells (a role, the edge it sits on): the coordinates
    are the **last** pure-math cell between one node index and the next.
    """
    nodes: dict[int, tuple[float, ...]] = {}
    in_section = False
    for line in page.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            in_section = line.startswith("## Élément de référence")
        if not in_section or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        starts = [k for k, c in enumerate(cells) if re.fullmatch(r"\d+", c)]
        for a, b in zip(starts, starts[1:] + [len(cells)]):
            maths = [MATH.match(c) for c in cells[a + 1 : b]]
            maths = [m.group(1) for m in maths if m]
            if not maths:
                continue
            coords = maths[-1]
            if coords.startswith("("):
                coords = coords[1:-1]
            nodes[int(cells[a])] = tuple(parse_number(c) for c in coords.split(","))
    if not nodes or sorted(nodes) != list(range(len(nodes))):
        raise SystemExit(f"{page}: node table not found or not numbered 0..n-1 ({sorted(nodes)})")
    return [nodes[i] for i in range(len(nodes))]


# ── Quadrature rules (closed forms) ─────────────────────────────────────────

GL2 = [(-1 / math.sqrt(3), 1.0), (1 / math.sqrt(3), 1.0)]
GL3 = [(-math.sqrt(0.6), 5 / 9), (0.0, 8 / 9), (math.sqrt(0.6), 5 / 9)]

Rule = list  # of ((ξ, …), w)


def tensor(dim: int, line=GL3) -> Rule:
    pts: Rule = [((), 1.0)]
    for _ in range(dim):
        pts = [(p + (x,), w * wx) for p, w in pts for x, wx in line]
    return pts


def on_unit_interval(line) -> list[tuple[float, float]]:
    """A rule of [-1, 1] carried to [0, 1]."""
    return [((1 + x) / 2, w / 2) for x, w in line]


def prism(tri: Rule, line) -> Rule:
    return [(p + (z,), w * wz) for p, w in tri for z, wz in on_unit_interval(line)]


def dunavant6() -> Rule:
    out = []
    for b, w in [(0.445948490915965, 0.111690794839005), (0.091576213509771, 0.054975871827661)]:
        a = 1 - 2 * b
        out += [((b, b), w), ((a, b), w), ((b, a), w)]
    return out


def hammer3() -> Rule:
    return [((0.5, 0.0), 1 / 6), ((0.5, 0.5), 1 / 6), ((0.0, 0.5), 1 / 6)]


def hammer4() -> Rule:
    a, b = (5 - math.sqrt(5)) / 20, (5 + 3 * math.sqrt(5)) / 20
    return [((a, a, a), 1 / 24), ((b, a, a), 1 / 24), ((a, b, a), 1 / 24), ((a, a, b), 1 / 24)]


def distinct_permutations(t: tuple) -> list[tuple]:
    from itertools import permutations

    return sorted(set(permutations(t)))


def keast11() -> Rule:
    """Barycentric families (L0, L1, L2, L3); the point is (L1, L2, L3)."""
    d, c = 0.399403576166799, 0.100596423833201
    out: Rule = [((0.25, 0.25, 0.25), -74 / 5625)]
    out += [(L[1:], 343 / 45000) for L in distinct_permutations((11 / 14, 1 / 14, 1 / 14, 1 / 14))]
    out += [(L[1:], 56 / 2250) for L in distinct_permutations((d, d, c, c))]
    return out


def pyramid_conic8() -> Rule:
    """2×2 Gauss-Legendre on (a, b) crossed with a Gauss-Jacobi rule in ζ for (1-ζ)²."""
    z1, z2 = 1 / 3 - math.sqrt(10) / 15, 1 / 3 + math.sqrt(10) / 15
    w1 = (1 / 12 - z2 / 3) / (z1 - z2)
    out = []
    for z, wz in ((z1, w1), (z2, 1 / 3 - w1)):
        for a, wa in GL2:
            for b, wb in GL2:
                out.append(((a * (1 - z), b * (1 - z), z), wa * wb * wz))
    return out


# ── Element descriptions ────────────────────────────────────────────────────


@dataclass
class Element:
    name: str
    title: str  # « TRI6 — triangle quadratique »
    dim: int
    scale: float  # pixels per reference unit
    gauss: list
    reduced: tuple
    oblique: tuple = OBLIQUE
    n_vertices: int | None = None  # simplex: the first nodes are the vertices
    outline: list[int] = field(default_factory=list)  # 2-D polygon, vertex indices
    faces: list[list[int]] = field(default_factory=list)  # 3-D, vertex indices
    axis_len: float = 1.3  # 2-D axes length, in reference units
    nodes: list[tuple[float, ...]] = field(default_factory=list)

    @property
    def quadratic(self) -> bool:
        """Has nodes beyond the vertices (the interpolation is `"LAGRANGE2"`)."""
        return any(self.role(i) not in ("sommet", "extrémité") for i in range(len(self.nodes)))

    def viewer(self) -> tuple[float, float, float]:
        """Direction from the scene towards the viewer, in reference coordinates."""
        k, a = self.oblique[0], math.radians(self.oblique[1])
        return (k * math.cos(a), -1.0, k * math.sin(a))

    def role(self, i: int) -> str:
        if self.n_vertices is not None:  # simplex, prism, pyramid: vertices come first
            return "sommet" if i < self.n_vertices else "milieu d'arête"
        zeros = sum(abs(c) < 1e-12 for c in self.nodes[i])
        if self.dim == 1:
            return "extrémité" if zeros == 0 else "milieu"
        if zeros == 0:
            return "sommet"
        if zeros == 1:
            return "milieu d'arête"
        if zeros == self.dim:
            return "centre"
        return "centre de face"


HEX_FACES = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
TET_FACES = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]
PYRA_FACES = [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
PRISM_FACES = [[0, 2, 1], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]]
# Views chosen so that two or three faces show: a plain default hides the slanted
# face of a tetrahedron or a prism behind the others.
TET_VIEW = (0.9, 35.0)
PRISM_VIEW = (0.9, 150.0)

ELEMENTS = [
    Element("seg2", "SEG2 — segment linéaire", 1, 150.0,
            tensor(1, GL2), ((0.0,), 2.0), axis_len=1.35),
    Element("tri3", "TRI3 — triangle linéaire", 2, 220.0,
            hammer3(), ((1 / 3, 1 / 3), 0.5), n_vertices=3, outline=[0, 1, 2], axis_len=1.25),
    Element("qua4", "QUA4 — quadrangle bilinéaire", 2, 105.0,
            tensor(2, GL2), ((0.0, 0.0), 4.0), outline=[0, 1, 2, 3], axis_len=1.5),
    Element("tet4", "TET4 — tétraèdre linéaire", 3, 170.0,
            hammer4(), ((0.25, 0.25, 0.25), 1 / 6), n_vertices=4, faces=TET_FACES, oblique=TET_VIEW),
    Element("pyra5", "PYRA5 — pyramide linéaire", 3, 92.0,
            pyramid_conic8(), ((0.0, 0.0, 0.25), 4 / 3), n_vertices=5, faces=PYRA_FACES),
    Element("penta6", "PENTA6 — prisme linéaire", 3, 150.0,
            prism(hammer3(), GL2), ((1 / 3, 1 / 3, 0.5), 0.5), n_vertices=6,
            faces=PRISM_FACES, oblique=PRISM_VIEW),
    Element("hex8", "HEX8 — hexaèdre trilinéaire", 3, 78.0,
            tensor(3, GL2), ((0.0, 0.0, 0.0), 8.0), faces=HEX_FACES),
    Element("seg3", "SEG3 — segment quadratique", 1, 150.0,
            tensor(1), ((0.0,), 2.0), axis_len=1.35),
    Element("tri6", "TRI6 — triangle quadratique", 2, 220.0,
            dunavant6(), ((1 / 3, 1 / 3), 0.5),
            n_vertices=3, outline=[0, 1, 2], axis_len=1.25),
    Element("qua8", "QUA8 — quadrangle sérendipité", 2, 105.0,
            tensor(2), ((0.0, 0.0), 4.0), outline=[0, 1, 2, 3], axis_len=1.5),
    Element("qua9", "QUA9 — quadrangle biquadratique", 2, 105.0,
            tensor(2), ((0.0, 0.0), 4.0), outline=[0, 1, 2, 3], axis_len=1.5),
    Element("tet10", "TET10 — tétraèdre quadratique", 3, 190.0,
            keast11(), ((0.25, 0.25, 0.25), 1 / 6), n_vertices=4, faces=TET_FACES,
            oblique=TET_VIEW),
    Element("penta15", "PENTA15 — prisme sérendipité", 3, 170.0,
            prism(dunavant6(), GL3), ((1 / 3, 1 / 3, 0.5), 0.5), n_vertices=6,
            faces=PRISM_FACES, oblique=PRISM_VIEW),
    Element("hex20", "HEX20 — hexaèdre sérendipité", 3, 78.0,
            tensor(3), ((0.0, 0.0, 0.0), 8.0), faces=HEX_FACES),
    Element("hex27", "HEX27 — hexaèdre triquadratique", 3, 88.0,
            tensor(3), ((0.0, 0.0, 0.0), 8.0), faces=HEX_FACES),
]

# ── SVG plumbing ────────────────────────────────────────────────────────────


def f(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".") if abs(x) >= 0.05 else "0"


def proj(el: Element, p: tuple[float, ...]) -> tuple[float, float]:
    s = el.scale
    if el.dim == 1:
        return (s * p[0], 0.0)
    if el.dim == 2:
        return (s * p[0], -s * p[1])
    x, y, z = p
    k, a = el.oblique[0], math.radians(el.oblique[1])
    return (s * (x + k * math.cos(a) * y), -s * (z + k * math.sin(a) * y))


def dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


@dataclass
class Geometry:
    """Visibility of the faces, edges and nodes of a 3-D element."""

    visible_faces: list[int]
    face_normals: list[tuple[float, float, float]]
    edges: list[tuple[int, int, bool]]  # (a, b, hidden)
    node_hidden: list[bool]
    node_inner: list[bool]  # on no face: the centre of a volume


def analyse_3d(el: Element) -> Geometry:
    verts = [el.nodes[i] for i in range(len(el.nodes)) if el.role(i) == "sommet"]
    centre = tuple(sum(v[k] for v in verts) / len(verts) for k in range(3))
    normals, planes = [], []
    for face in el.faces:
        p0, p1, p2 = (el.nodes[face[i]] for i in range(3))
        n = cross(sub(p1, p0), sub(p2, p0))
        norm = math.sqrt(dot(n, n))
        n = tuple(c / norm for c in n)
        if dot(n, sub(p0, centre)) < 0:
            n = tuple(-c for c in n)
        normals.append(n)
        planes.append(p0)
    visible = [dot(n, el.viewer()) > 1e-9 for n in normals]

    def faces_of(p) -> list[int]:
        return [k for k in range(len(el.faces)) if abs(dot(normals[k], sub(p, planes[k]))) < 1e-9]

    seen, edges = set(), []
    for k, face in enumerate(el.faces):
        for a, b in zip(face, face[1:] + face[:1]):
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            adjacent = [j for j, other in enumerate(el.faces) if a in other and b in other]
            edges.append((a, b, not any(visible[j] for j in adjacent)))
    node_faces = [faces_of(p) for p in el.nodes]
    return Geometry(
        visible_faces=[k for k in range(len(el.faces)) if visible[k]],
        face_normals=normals,
        edges=edges,
        node_hidden=[bool(fs) and not any(visible[k] for k in fs) for fs in node_faces],
        node_inner=[not fs for fs in node_faces],
    )


def svg_open(w: float, h: float, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{f(w)}" height="{f(h)}" '
        f'viewBox="0 0 {f(w)} {f(h)}" role="img" font-family="{FONT}">',
        f"<title>{title}</title>",
        f"<desc>{desc}</desc>",
        "<defs>"
        f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>'
        "</defs>",
        f'<rect x="0.5" y="0.5" width="{f(w - 1)}" height="{f(h - 1)}" rx="8" fill="#ffffff" '
        'stroke="#d5dbe3"/>',
    ]


def text(x, y, s, size=13, anchor="middle", color=INK, weight="normal", style="normal") -> str:
    return (
        f'<text x="{f(x)}" y="{f(y)}" font-size="{size}" text-anchor="{anchor}" fill="{color}" '
        f'font-weight="{weight}" font-style="{style}">{s}</text>'
    )


def axis_label(x, y, s) -> str:
    return text(x, y, s, size=15, color=MUTED, style="italic")


# ── Drawing an element (reference outline + nodes) ──────────────────────────


def outline_svg(el: Element, geo: Geometry | None, faint: bool = False) -> list[str]:
    """Faces and edges. `faint` is for the quadrature figure, where the element is a backdrop."""
    out = []
    stroke = "#9aa7b6" if faint else INK
    width = 1.2 if faint else 1.6
    pt = lambda i: proj(el, el.nodes[i])  # noqa: E731
    if el.dim == 1:
        xs = [pt(i)[0] for i in range(len(el.nodes))]
        out.append(
            f'<line x1="{f(min(xs))}" y1="0" x2="{f(max(xs))}" y2="0" stroke="{stroke}" '
            f'stroke-width="{3 if not faint else 2}" stroke-linecap="round"/>'
        )
        return out
    if el.dim == 2:
        d = " ".join(f"{f(pt(i)[0])},{f(pt(i)[1])}" for i in el.outline)
        out.append(
            f'<polygon points="{d}" fill="{"#f6f9fc" if faint else FILL}" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-linejoin="round"/>'
        )
        return out
    assert geo is not None
    for k in geo.visible_faces:
        n = geo.face_normals[k]
        tint = FACE_TINT[max(range(3), key=lambda a: abs(n[a]))]
        d = " ".join(f"{f(pt(i)[0])},{f(pt(i)[1])}" for i in el.faces[k])
        out.append(f'<polygon points="{d}" fill="{"#f6f9fc" if faint else tint}" stroke="none"/>')
    for hidden in (True, False):
        for a, b, h in geo.edges:
            if h != hidden:
                continue
            (x1, y1), (x2, y2) = pt(a), pt(b)
            dash = ' stroke-dasharray="4 4" stroke-opacity=".6"' if hidden else ""
            out.append(
                f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" stroke="{stroke}" '
                f'stroke-width="{width if not hidden else 1.1}" stroke-linecap="round"{dash}/>'
            )
    return out


def nodes_svg(el: Element, geo: Geometry | None) -> list[str]:
    out = []
    order = sorted(range(len(el.nodes)), key=lambda i: bool(geo and not geo.node_hidden[i]))
    for i in order:  # hidden nodes first, so a visible one never sits under them
        x, y = proj(el, el.nodes[i])
        color = ROLE_COLOR[el.role(i)]
        hidden = bool(geo and geo.node_hidden[i])
        inner = bool(geo and geo.node_inner[i])
        extra = ' stroke-dasharray="3 2" stroke-opacity=".65"' if hidden or inner else ""
        out.append(
            f'<circle cx="{f(x)}" cy="{f(y)}" r="10" fill="#ffffff" stroke="{color}" '
            f'stroke-width="2.4"{extra}/>'
        )
        out.append(text(x, y + 3.8, str(i), size=11, color=MUTED if hidden else INK, weight="bold"))
    return out


def axes_2d(el: Element) -> list[str]:
    L = el.axis_len * el.scale
    out = []
    for (dx, dy), label in list(zip(((1, 0), (0, -1)), "ξη"))[: el.dim]:
        x, y = dx * L, dy * L
        # a segment lies on its own axis: the arrow starts at its far end
        x0 = max(proj(el, n)[0] for n in el.nodes) if el.dim == 1 else 0
        out.append(
            f'<line x1="{f(x0)}" y1="0" x2="{f(x)}" y2="{f(y)}" stroke="{MUTED}" stroke-width="1.2" '
            f'marker-end="url(#ah)"/>'
        )
        out.append(axis_label(x + dx * 12, y + dy * 12 + 5, label))
    return out


def triad_3d(el: Element, ox: float, oy: float, length: float = 38.0) -> list[str]:
    out = []
    for axis, label in enumerate("ξηζ"):
        unit = tuple(1.0 if k == axis else 0.0 for k in range(3))
        ux, uy = proj(Element(**{**el.__dict__, "scale": 1.0}), unit)
        n = math.hypot(ux, uy)
        ux, uy = ux / n, uy / n
        x, y = ox + ux * length, oy + uy * length
        out.append(
            f'<line x1="{f(ox)}" y1="{f(oy)}" x2="{f(x)}" y2="{f(y)}" stroke="{MUTED}" '
            f'stroke-width="1.2" marker-end="url(#ah)"/>'
        )
        out.append(axis_label(x + ux * 11, y + uy * 11 + 5, label))
    return out


def bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def legend(roles: list[str], x: float, y: float) -> list[str]:
    out, cursor = [], x
    for role in roles:
        out.append(
            f'<circle cx="{f(cursor + 5)}" cy="{f(y - 4)}" r="5" fill="#ffffff" '
            f'stroke="{ROLE_COLOR[role]}" stroke-width="2.4"/>'
        )
        out.append(text(cursor + 15, y, role, size=12, anchor="start", color=MUTED))
        cursor += 15 + 6.6 * len(role) + 18
    return out


def alt_reference(el: Element) -> str:
    roles = ", ".join(r for r in ROLE_COLOR if any(el.role(i) == r for i in range(len(el.nodes))))
    n = len(el.nodes)
    numbered = "numérotés 0 et 1" if n == 2 else f"numérotés de 0 à {n - 1}"
    axes = {1: "axe ξ", 2: "axes ξ, η", 3: "axes ξ, η, ζ"}[el.dim]
    return f"Élément de référence {el.title.split(' ')[0]} : {n} nœuds {numbered} ({roles}), {axes}."


def figure_reference(el: Element) -> str:
    geo = analyse_3d(el) if el.dim == 3 else None
    pts = [proj(el, p) for p in el.nodes]
    if el.dim >= 2:
        pts += [(el.axis_len * el.scale + 22, 0), (0, -el.axis_len * el.scale - 22), (-6, 6)]
    if el.dim == 1:
        pts += [(el.axis_len * el.scale + 22, 0), (0, -34), (0, 34)]
    x0, y0, x1, y1 = bbox(pts)
    margin = 26
    left = margin + (100 if el.dim == 3 else 0)
    roles = [r for r in ROLE_COLOR if any(el.role(i) == r for i in range(len(el.nodes)))]
    legend_w = sum(15 + 6.6 * len(r) + 18 for r in roles) - 18
    width = max(x1 - x0 + left + margin, legend_w + 2 * margin)
    height = y1 - y0 + 2 * margin + 30
    tx, ty = left - x0 + (width - (x1 - x0 + left + margin)) / 2, margin - y0
    out = svg_open(width, height, el.title, alt_reference(el))
    out.append(f'<g transform="translate({f(tx)},{f(ty)})">')
    out += outline_svg(el, geo)
    if el.dim <= 2:
        out += axes_2d(el)  # over the element: the origin matters (QUA8 centre)
    out += nodes_svg(el, geo)
    if el.dim == 3:
        out += triad_3d(el, x0 - 74, y1 - 4, length=34.0)
    out.append("</g>")
    out += legend(roles, (width - legend_w) / 2, height - 16)
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ── Quadrature figure ───────────────────────────────────────────────────────


def panel_quadrature(el: Element, geo, points, title, subtitle, disc: bool) -> tuple[list[str], tuple]:
    """One panel drawn around the origin; returns its svg and bounding box."""
    out = outline_svg(el, geo, faint=True)
    for i, p in enumerate(el.nodes):  # nodes as small ticks, unnumbered
        x, y = proj(el, p)
        out.append(f'<rect x="{f(x - 2.5)}" y="{f(y - 2.5)}" width="5" height="5" fill="#ffffff" '
                   f'stroke="#9aa7b6" stroke-width="1"/>')
    wmax = max(abs(w) for _, w in points)
    far_first = sorted(points, key=lambda pw: dot(pw[0], el.viewer()) if el.dim == 3 else 0.0)
    rmax = 9.5 if el.dim == 2 else 7.0
    for p, w in far_first:
        x, y = proj(el, p)
        if disc:
            r = rmax * math.sqrt(abs(w) / wmax)
            if w < 0:  # Keast's centre weight: a ring, not a disc
                out.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="{f(r)}" fill="#ffffff" '
                           f'stroke="{GAUSS_COLOR}" stroke-width="2.4"/>')
            else:
                out.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="{f(r)}" fill="{GAUSS_COLOR}" '
                           f'fill-opacity=".85" stroke="#ffffff" stroke-width="1.2"/>')
        else:
            d = 8
            out.append(
                f'<path d="M{f(x - d)},{f(y - d)} L{f(x + d)},{f(y + d)} M{f(x - d)},{f(y + d)} '
                f'L{f(x + d)},{f(y - d)}" stroke="{GAUSS_COLOR}" stroke-width="3" stroke-linecap="round"/>'
            )
    pts = [proj(el, p) for p in el.nodes]
    if el.dim == 1:
        pts += [(0, -rmax), (0, rmax)]
    return out, bbox(pts)


def figure_quadrature(el: Element) -> str:
    geo = analyse_3d(el) if el.dim == 3 else None
    short = el.title.split(" ")[0]
    panels = [
        (el.gauss, f'"GAUSS"', f"{len(el.gauss)} points", True),
        ([el.reduced], f'"REDUCED"', "1 point", False),
    ]
    drawn = [panel_quadrature(el, geo, *pn[:1], pn[1], pn[2], pn[3]) for pn in panels]
    boxes = [b for _, b in drawn]
    pad, gap, head, foot = 30, 40, 38, 44
    pw = max(b[2] - b[0] for b in boxes) + 2 * pad
    ph = max(b[3] - b[1] for b in boxes) + 2 * pad
    width = 2 * pw + gap
    height = head + ph + foot
    desc = (f"Règles d'intégration du {short} : {len(el.gauss)} points de Gauss "
            f"(disques d'aire proportionnelle au poids) et le point unique de la règle réduite.")
    out = svg_open(width, height, f"{short} — quadrature", desc)
    for k, ((svg, (x0, y0, x1, y1)), pn) in enumerate(zip(drawn, panels)):
        cx = k * (pw + gap) + pw / 2
        tx = cx - (x0 + x1) / 2
        ty = head + ph / 2 - (y0 + y1) / 2
        out.append(text(cx, 26, f"{pn[1]} · {pn[2]}", size=14, weight="bold"))
        out.append(f'<g transform="translate({f(tx)},{f(ty)})">')
        out += svg
        out.append("</g>")
    note = "disque : aire proportionnelle au poids"
    if any(w < 0 for _, w in el.gauss):
        note += " · cercle : poids négatif"
    out.append(text(width / 2, height - 16, note + " · carré : nœud de l'élément",
                    size=12, color=MUTED))
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ── Thumbnail (catalogue of `elements/index.md`) ────────────────────────────


def figure_thumb(el: Element, box: float = 104.0) -> str:
    """The element alone: outline and unnumbered nodes, no axes, no legend."""
    unit = Element(**{**el.__dict__, "scale": 1.0})
    x0, y0, x1, y1 = bbox([proj(unit, p) for p in el.nodes])
    el = Element(**{**el.__dict__, "scale": (box - 24) / max(x1 - x0, y1 - y0, 0.5 if el.dim == 1 else 1e-9)})
    geo = analyse_3d(el) if el.dim == 3 else None
    x0, y0, x1, y1 = bbox([proj(el, p) for p in el.nodes])
    width, height = box, box * 0.8 if el.dim == 1 else box
    tx, ty = (width - (x0 + x1)) / 2, (height - (y0 + y1)) / 2
    out = svg_open(width, height, el.title, alt_reference(el))
    out.append(f'<g transform="translate({f(tx)},{f(ty)})">')
    out += outline_svg(el, geo)
    for i, p in enumerate(el.nodes):
        x, y = proj(el, p)
        hidden = bool(geo and geo.node_hidden[i])
        extra = ' fill-opacity=".45"' if hidden else ""
        out.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="3.6" fill="{ROLE_COLOR[el.role(i)]}"{extra}/>')
    out.append("</g></svg>")
    return "\n".join(out) + "\n"


# ── Driver ──────────────────────────────────────────────────────────────────


def load(el: Element) -> Element:
    el.nodes = read_nodes(PAGES / f"{el.name}.md")
    return el


def render_all() -> dict[str, str]:
    files = {}
    for el in map(load, ELEMENTS):
        files[f"{el.name}-reference.svg"] = figure_reference(el)
        files[f"{el.name}-quadrature.svg"] = figure_quadrature(el)
        files[f"{el.name}-thumb.svg"] = figure_thumb(el)
    return files


def main(argv: list[str]) -> int:
    files = render_all()
    if "--check" in argv:
        stale = [n for n, s in files.items()
                 if not (OUT / n).exists() or (OUT / n).read_text(encoding="utf-8") != s]
        if stale:
            print("stale figures (run `python script/elements_figures.py`):", *stale, sep="\n  ")
            return 1
        print(f"OK : {len(files)} element figures up to date.")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for name, svg in files.items():
        (OUT / name).write_text(svg, encoding="utf-8")
    print(f"{len(files)} figures written to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
