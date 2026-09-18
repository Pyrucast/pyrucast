"""SEG2 contour of a crenellated profile, at a single element size.

The shape: two full-height towers at the ends, and between them a series of
low crenellations going up and down. Nine "bars" of equal width, whose height
is given in vertical units by ``LEVELS``.

The mesh is parameterized by a single integer, ``N_MIN``: the number of
elements laid on the contour's smallest segment. It sets the targeted element
size ``h``, and every other segment is cut into ``round(length / h)``
elements — hence SEG2 of nearly equal length all the way round.


Lancer :

    PYO3_PYTHON=/usr/bin/python3.13 \
        maturin develop --features extension-module
    python examples/contour_creneau.py
"""

import pyrucast as pc

# Encombrement de la forme.
HEIGHT = 0.3
LENGTH = 0.6

# The only meshing parameter: elements on the smallest segment.
N_MIN = 4

# Height of each of the nine bars, in units of HEIGHT / 40. The two towers
# rise to 40 (that is, HEIGHT), the crenellations swing between 3 and 6.
LEVELS = [40, 3, 6, 4, 6, 4, 6, 3, 40]

U = LENGTH / len(LEVELS)  # width of one bar
V = HEIGHT / max(LEVELS)  # unité verticale


def corners() -> list[list[float]]:
    """The contour's corners, counter-clockwise.

    Le tour part de l'origine, longe la base vers la droite, remonte la
    right tower, then walks the crenellation staircase back down to the left
    before closing on the origin.
    """
    # The upper profile, left to right: two points per bar, which gives the
    # staircase directly (flat, rise, flat, ...).
    top = []
    for i, level in enumerate(LEVELS):
        top.append([i * U, level * V])
        top.append([(i + 1) * U, level * V])

    # The base is cut under each bar, not in one piece. Nothing forces that on
    # the front paver, but `grid_surface` needs the base's nodes to fall on the
    # columns the crenellations impose on its grid: in one piece it would be
    # LENGTH / round(LENGTH / h) = 0.00375, against U / round(U / h) = 0.0037037
    # above, and not one node would be shared.

    base = [[i * U, 0.0] for i in range(len(LEVELS) + 1)]

    # Base from left to right, then the profile walked backwards. The first point
    # is not repeated at the end: `pairs` is what closes the loop.
    return base + top[::-1]


def pairs(points):
    """The pairs of consecutive points, the last one closing the loop."""
    return list(zip(points, points[1:] + points[:1]))


def length(a, b):
    return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5


def main() -> None:
    coords = pc.Coords(2)
    points = corners()

    # One node per corner, created once only: `line` reuses them, so neighbouring
    # segments join without a duplicate node.
    nodes = [coords.add_node(p) for p in points]

    # The targeted element size follows from the smallest segment.
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

    # Check: the elements' real length, from shortest to longest.
    lengths = sorted(
        length(*(node.position() for node in cell)) for sub in contour for cell in sub
    )
    print(f"longueurs    : {lengths[0]:.6f} … {lengths[-1]:.6f}")

    # Quadrangle filling, at the same targeted size as the SEG2.
    # `pave_surface` advances a front from the contour; its rows meet somewhere
    # inside, and that line is the one carrying the flattened cells and the
    # residual triangles.
    front = pc.mesh.pave_surface(contour, "QUA4", h)
    print(f"pave_surface : {dict(zip(front.element_types(), front.cell_counts()))}")

    # `grid_surface` lays a grid instead of a front. The shape is rectilinear and
    # the contour was cut for it, so the grid reaches the border everywhere: no
    # band is left to pave.
    grid = pc.mesh.grid_surface(contour, "QUA4", h)
    print(f"grid_surface : {dict(zip(grid.element_types(), grid.cell_counts()))}")


if __name__ == "__main__":
    main()
