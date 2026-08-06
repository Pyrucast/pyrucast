"""Contour SEG2 d'un profil crénelé, à taille de maille unique.

La forme : deux tours pleine hauteur aux extrémités, et entre elles une
série de créneaux bas qui montent et descendent. Neuf « barres » de même
largeur, dont la hauteur est donnée en unités verticales par ``LEVELS``.

Le maillage est paramétré par un seul entier, ``N_MIN`` : le nombre
d'éléments posés sur le plus petit segment du contour. Il fixe la taille
de maille visée ``h``, et tous les autres segments sont découpés en
``round(longueur / h)`` éléments — d'où des SEG2 quasiment tous de même
longueur sur tout le tour.

Lancer :

    PYO3_PYTHON=/usr/bin/python3.13 \
        maturin develop --features extension-module
    python examples/contour_creneau.py
"""

import pyrucast as pc

# Encombrement de la forme.
HEIGHT = 0.3
LENGTH = 0.6

# Le seul paramètre de maillage : éléments sur le plus petit segment.
N_MIN = 4

# Hauteur de chacune des neuf barres, en unités de HEIGHT / 40. Les deux
# tours montent à 40 (soit HEIGHT), les créneaux oscillent entre 3 et 6.
LEVELS = [40, 3, 6, 4, 6, 4, 6, 3, 40]

U = LENGTH / len(LEVELS)  # largeur d'une barre
V = HEIGHT / max(LEVELS)  # unité verticale


def corners() -> list[list[float]]:
    """Les angles du contour, dans le sens trigonométrique.

    Le tour part de l'origine, longe la base vers la droite, remonte la
    tour de droite, puis redescend l'escalier des créneaux vers la gauche
    avant de refermer sur l'origine.
    """
    # Le profil supérieur, de gauche à droite : deux points par barre,
    # ce qui donne directement l'escalier (plat, montée, plat, ...).
    top = []
    for i, level in enumerate(LEVELS):
        top.append([i * U, level * V])
        top.append([(i + 1) * U, level * V])

    # La base est coupée sous chaque barre, et pas d'un seul tenant. Rien ne
    # l'impose au paveur frontal, mais `grid_surface` a besoin que les nœuds
    # de la base tombent sur les colonnes que les créneaux imposent à sa
    # grille : d'un seul tenant elle vaudrait LENGTH / round(LENGTH / h)
    # = 0,00375, contre U / round(U / h) = 0,0037037 au-dessus, et pas un
    # nœud ne serait partagé.
    base = [[i * U, 0.0] for i in range(len(LEVELS) + 1)]

    # Base de gauche à droite, puis le profil parcouru à l'envers. Le
    # premier point n'est pas répété à la fin : c'est `pairs` qui referme.
    return base + top[::-1]


def pairs(points):
    """Les couples de points consécutifs, le dernier refermant le tour."""
    return list(zip(points, points[1:] + points[:1]))


def length(a, b):
    return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5


def main() -> None:
    coords = pc.Coords(2)
    points = corners()

    # Un nœud par angle, créé une seule fois : `line` les réutilise, donc
    # les segments voisins se raccordent sans nœud en double.
    nodes = [coords.add_node(p) for p in points]

    # La taille de maille visée découle du plus petit segment.
    sides = pairs(points)
    h = min(length(a, b) for a, b in sides) / N_MIN

    contour = None
    for (a, b), (na, nb) in zip(sides, pairs(nodes)):
        n_elems = max(N_MIN, round(length(a, b) / h))
        seg = pc.mesh.line(na, nb, n_elems)
        contour = seg if contour is None else contour | seg

    contour = pc.mesh.chain(pc.mesh.consolidate(contour))

    print(f"angles       : {len(points)}")
    print(f"taille visée : {h:.6f}")
    print(f"contour      : {contour.cell_count()} SEG2, {coords.node_count()} nœuds")

    # Contrôle : longueur réelle des éléments, du plus court au plus long.
    lengths = sorted(
        length(*(node.position() for node in cell)) for sub in contour for cell in sub
    )
    print(f"longueurs    : {lengths[0]:.6f} … {lengths[-1]:.6f}")

    # Remplissage en quadrangles, à la même taille visée que les SEG2.
    # `pave_surface` avance un front depuis le contour ; ses rangées se
    # rencontrent quelque part à l'intérieur, et c'est cette ligne-là qui
    # porte les mailles aplaties et les triangles résiduels.
    front = pc.mesh.pave_surface(contour, "QUA4", h)
    print(f"pave_surface : {dict(zip(front.element_types(), front.cell_counts()))}")

    # `grid_surface` pose une grille au lieu d'un front. La forme est
    # rectilinéaire et le contour a été découpé pour elle, donc la grille
    # atteint le bord partout : il ne reste aucune bande à paver.
    grid = pc.mesh.grid_surface(contour, "QUA4", h)
    print(f"grid_surface : {dict(zip(grid.element_types(), grid.cell_counts()))}")


if __name__ == "__main__":
    main()
