# ANCHOR: script
"""Compare les quatre mailleurs surfaciques sur une même forme.
/ Compares the four surface meshers on one and the same shape.

FR — Chaque figure montre le **même contour** maillé quatre fois : une seule
construction, puis trois copies obtenues par `translate`, de sorte qu'aucun
mailleur ne bénéficie d'une discrétisation de bord différente des autres. La
disposition est toujours la même — en haut `triangulate_surface` et
`pave_surface`, en bas `grid_surface` et `grid_surface2`.

FR — La mesure de qualité est la *mean ratio* du pire coin, la même pour les
triangles et pour les quadrangles : 1 pour un coin droit à côtés égaux, 0 pour
un coin plat, négatif pour une maille retournée.

EN — Every figure shows the **same contour** meshed four times: built once,
then copied three times with `translate`, so no mesher gets a boundary
discretisation the others did not have. The layout is always the same — top
`triangulate_surface` and `pave_surface`, bottom `grid_surface` and
`grid_surface2`.

EN — Quality is the worst corner's *mean ratio*, one measure for triangles and
quadrangles alike: 1 for a square corner with equal sides, 0 for a flat corner,
negative for an inverted cell.

Lancement / Running ::

    maturin develop --features extension-module,viz
    python examples/comparer_mailleurs_2d.py

    # Figures du livre / book figures (book/src/img/) :
    # PYRUCAST_IMG_DIR=book/src/img python examples/comparer_mailleurs_2d.py
"""

import math
import os
import tempfile

import pyrucast as pc

# FR — Répertoire des figures. Le tracé sort en PNG et non en SVG : sur un
# maillage de plusieurs milliers de mailles, un vectoriel pèse dix fois plus
# et met le navigateur à genoux pour un rendu identique à cette échelle.
# EN — Figure directory. Figures are PNG rather than SVG: on a mesh of several
# thousand cells a vector file weighs ten times as much and brings the browser
# to its knees, for a rendering identical at this scale.
OUT = os.environ.get("PYRUCAST_IMG_DIR", tempfile.gettempdir())

# FR — Vue de face, plan XY. EN — Face-on view of the XY plane.
VUE = (90.0, 90.0, 1.0)

# FR — Les quatre mailleurs, dans l'ordre où la figure les range.
# EN — The four meshers, in the order the figure lays them out.
MAILLEURS = [
    ("triangulate_surface", lambda c, h: pc.mesh.triangulate_surface(c, "TRI3", h)),
    ("pave_surface", lambda c, h: pc.mesh.pave_surface(c, "QUA4", h)),
    ("grid_surface", lambda c, h: pc.mesh.grid_surface(c, "QUA4", h)),
    ("grid_surface2", lambda c, h: pc.mesh.grid_surface2(c, "QUA4", h)),
]


# ANCHOR: qualite
def qualite(sommets):
    """Mean ratio du pire coin. / Worst corner's mean ratio."""
    pire, k = 1.0, len(sommets)
    for i in range(k):
        u = (
            sommets[(i + 1) % k][0] - sommets[i][0],
            sommets[(i + 1) % k][1] - sommets[i][1],
        )
        v = (sommets[i - 1][0] - sommets[i][0], sommets[i - 1][1] - sommets[i][1])
        nu, nv = math.hypot(*u), math.hypot(*v)
        pire = min(pire, 2 * (u[0] * v[1] - u[1] * v[0]) / (nu * nu + nv * nv))
    return pire


# ANCHOR_END: qualite


def contour_de(angles, taille):
    """Un contour SEG2 fermé passant par `angles`, au pas `taille`.
    / A closed SEG2 contour through `angles`, at step `taille`."""
    coords = pc.Coords(2)
    noeuds = [coords.add_node(list(p)) for p in angles]
    contour = None
    for (a, b), (na, nb) in zip(
        zip(angles, angles[1:] + angles[:1]), zip(noeuds, noeuds[1:] + noeuds[:1])
    ):
        seg = pc.mesh.line(na, nb, max(1, round(math.dist(a, b) / taille)))
        contour = seg if contour is None else contour | seg
    return pc.mesh.chain(pc.mesh.consolidate(contour))


# ANCHOR: comparer
def comparer(nom, contour, taille, fichier, noeuds=True):
    """Maille `contour` par les quatre méthodes et trace la figure.
    / Meshes `contour` with all four methods and draws the figure.

    FR — Le contour n'est construit qu'une fois : ses trois copies viennent de
    `translate`, qui rend un maillage neuf dans les mêmes `Coords` — c'est ce
    qui permet de les réunir sur une seule figure.
    """
    points = [n.position() for sub in contour for cell in sub for n in cell]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    dx, dy = (max(xs) - min(xs)) * 1.15, (max(ys) - min(ys)) * 1.25
    coins = [(0.0, 0.0), (dx, 0.0), (0.0, -dy), (dx, -dy)]

    tout, lignes = None, []
    for (titre, mailler), coin in zip(MAILLEURS, coins):
        copie = pc.mesh.translate(contour, list(coin))
        maillage = mailler(copie, taille)
        qs = sorted(
            qualite([n.position() for n in cell]) for sub in maillage for cell in sub
        )
        comptes = dict(zip(maillage.element_types(), maillage.cell_counts()))
        lignes.append(
            f"{titre:20s} {len(qs):6d} mailles, {comptes.get('TRI3', 0):5d} tri, "
            f"pire {qs[0]:.3f}, p5 {qs[len(qs) // 20]:.3f}"
        )
        for sub in copie:
            sub.face_color = (20, 90, 200)
        bloc = maillage | copie
        if noeuds:
            points_rouges = pc.mesh.to_poi1(copie)
            for sub in points_rouges:
                sub.face_color = (220, 30, 30)
            bloc = bloc | points_rouges
        tout = bloc if tout is None else tout | bloc

    print(f"\n{nom}   (taille visée {taille:g})")
    for ligne in lignes:
        print(f"   {ligne}")
    tout.plot(
        view=VUE,
        show_axes=False,
        save=os.path.join(OUT, fichier),
        title=f"{nom} — h={taille:g} — haut : triangulate, pave — bas : grid, grid2",
    )


# ANCHOR_END: comparer


# ── Les formes / The shapes ─────────────────────────────────────────────────

# FR — Le profil crénelé : deux tours pleine hauteur et sept créneaux entre
# elles. Décliné en deux versions dont seule la base change, ce qui isole
# l'effet de la discrétisation du contour.
HAUT, LONG = 0.3, 0.6
NIVEAUX = [40, 3, 6, 4, 6, 4, 6, 3, 40]
U, V = LONG / len(NIVEAUX), HAUT / max(NIVEAUX)
PROFIL = []
for i, niveau in enumerate(NIVEAUX):
    PROFIL += [[i * U, niveau * V], [(i + 1) * U, niveau * V]]


def carre_arrondi(cote=1.0, taille=0.05):
    """Un carré dont un angle est arrondi au rayon d'un demi-côté.
    / A square with one corner rounded to half-side radius."""
    r = cote / 2.0
    centre = (cote - r, cote - r)
    n = max(1, round((math.pi * r / 2.0) / taille))
    arc = [
        (
            centre[0] + r * math.cos(k / n * math.pi / 2),
            centre[1] + r * math.sin(k / n * math.pi / 2),
        )
        for k in range(1, n + 1)
    ]
    return [(0.0, 0.0), (cote, 0.0), (cote, cote - r)] + arc + [(0.0, cote)]


def cercle(rayon=1.0, cotes=60):
    return [
        (rayon * math.cos(i / cotes * math.tau), rayon * math.sin(i / cotes * math.tau))
        for i in range(cotes)
    ]


FORMES = [
    # nom, angles, taille visée, fichier, nœuds du contour visibles
    (
        "Rectangle 1 × 0,63",
        [(0, 0), (1, 0), (1, 0.63), (0, 0.63)],
        0.1,
        "rectangle",
        True,
    ),
    (
        "Plaque, marche pile sur la grille",
        [(0, 0), (1, 0), (1, 1), (0.5, 1), (0.5, 0.6), (0, 0.6)],
        0.1,
        "plaque-sur-grille",
        True,
    ),
    (
        "Plaque à marche (0,53 ; 0,61)",
        [(0, 0), (1, 0), (1, 1), (0.53, 1), (0.53, 0.61), (0, 0.61)],
        0.1,
        "plaque-hors-grille",
        True,
    ),
    (
        "L à cotes quelconques",
        [(0, 0), (1.03, 0), (1.03, 0.47), (0.41, 0.47), (0.41, 0.98), (0, 0.98)],
        0.1,
        "l-quelconque",
        True,
    ),
    (
        "L étiré à 1,02",
        [(0, 0), (1.03, 0), (1.03, 0.47), (0.41, 0.47), (0.41, 1.02), (0, 1.02)],
        0.1,
        "l-102",
        True,
    ),
    (
        "L étiré à 1,10",
        [(0, 0), (1.03, 0), (1.03, 0.47), (0.41, 0.47), (0.41, 1.10), (0, 1.10)],
        0.1,
        "l-110",
        True,
    ),
    (
        "Créneau, base coupée sous chaque barre",
        [[i * U, 0.0] for i in range(len(NIVEAUX) + 1)] + PROFIL[::-1],
        V * 1.5,
        "creneau-base-coupee",
        False,
    ),
    (
        "Créneau, base d'un seul tenant",
        [[0.0, 0.0], [LONG, 0.0]] + PROFIL[::-1],
        V * 1.5,
        "creneau-base-entiere",
        False,
    ),
    (
        "Maison, toit et encoche de porte",
        [
            (0.0, 0.0),
            (0.38, 0.0),
            (0.38, 0.40),
            (0.62, 0.40),
            (0.62, 0.0),
            (1.0, 0.0),
            (1.0, 0.80),
            (0.50, 1.20),
            (0.0, 0.80),
        ],
        0.05,
        "maison",
        True,
    ),
    (
        "Carré, un angle arrondi au demi-côté",
        carre_arrondi(),
        0.05,
        "carre-arrondi",
        True,
    ),
    ("Cercle R = 1", cercle(), 0.05, "cercle", False),
]


def main():
    for nom, angles, taille, fichier, noeuds in FORMES:
        comparer(
            nom,
            contour_de(angles, taille),
            taille,
            f"mailleurs-2d-{fichier}.png",
            noeuds,
        )


if __name__ == "__main__":
    main()
