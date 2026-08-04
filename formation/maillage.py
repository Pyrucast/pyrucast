# ANCHOR: script
"""Formation débutant — 1. Maillage. / Beginner training — 1. Meshing.

FR — Construit la pièce qui sert de fil rouge : une **chape percée**, plaque
rectangulaire terminée par un demi-disque et trouée en son centre, maillée de
quatre façons successives — contour, surface, volume, puis la même pièce en
**structuré**.

FR — Cotes : 0.30 × 0.10 m, demi-disque de rayon 0.05 m, trou de rayon
0.035 m, épaisseur 0.02 m. La pièce est plane dans **XZ** et son épaisseur est
portée par **Y** : la géométrie est donc 3D (`Coords(3)`) dès le départ, ce qui
évite d'avoir à la relever au moment de passer au volume.

FR — Deux familles de mailleurs sont comparées : le **non structuré**
(`pyrucast.mesh.triangulate_surface`), où l'on donne un contour fermé et une
**taille de maille** cible, et le **structuré** (`pyrucast.mesh.extrude`,
`pyrucast.mesh.sweep`), où l'on impose un **nombre d'éléments** et où le
maillage a la topologie d'une grille. Le détail pas à pas est dans le livre,
page « Maillage ».

EN — Builds the part used as the guiding thread: a **pierced lug**, a
rectangular plate capped by a half-disc and holed at its centre, meshed in four
successive ways — contour, surface, volume, then the same part **structured**.

EN — Dimensions: 0.30 × 0.10 m, half-disc of radius 0.05 m, hole of radius
0.035 m, thickness 0.02 m. The part lies in the **XZ** plane and its thickness
runs along **Y**: the geometry is 3-D (`Coords(3)`) from the start, so nothing
has to be lifted when moving on to the volume.

EN — Two families of meshers are compared: **unstructured**
(`pyrucast.mesh.triangulate_surface`), where you hand in a closed contour and
a target **element size**, and **structured** (`pyrucast.mesh.extrude`,
`pyrucast.mesh.sweep`), where you impose an **element count** and the mesh
has grid topology. The step-by-step walkthrough lives in the book's meshing
page.

Lancement / Running ::

    maturin develop --release
    python formation/maillage.py

    # Figures du livre / book figures (book/src/formation/img/) :
    # PYRUCAST_FORMATION_IMG_DIR=book/src/formation/img python formation/maillage.py
"""

import os

import pyrucast as pc

# ANCHOR: figures
# FR — Répertoire des figures : défini → export SVG, absent → fenêtre interactive.
# EN — Figure directory: set → SVG export, unset → interactive window.
OUT = os.environ.get("PYRUCAST_FORMATION_IMG_DIR")

# FR — Vue commune à toutes les figures : azimut, élévation, échelle.
# EN — View shared by every figure: azimuth, elevation, scale.
VUE = (-45, 25, 1.0)

# FR — Une couleur par face plane de la peau (recyclée s'il y en a plus).
# EN — One colour per flat skin face (recycled if there are more of them).
PALETTE = [(255, 0, 255), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]


def show(mesh: pc.Mesh, title: str, file: str, wireframe: bool = False) -> None:
    """FR — Trace `mesh` : fenêtre interactive, ou SVG si `OUT` est défini.

    EN — Plot `mesh`: interactive window, or SVG when `OUT` is set.
    """
    mesh.plot(
        view=VUE,
        title=title,
        wireframe=wireframe,
        save=os.path.join(OUT, file) if OUT else None,
    )


# ANCHOR_END: figures


# ── Géométrie / Geometry ──────────────────────────────────────────────────
# ANCHOR: coords
# FR — Un espace de coordonnées 3D, seul objet mutable de tout le script.
# EN — One 3-D coordinate space, the script's only mutable object.
coords = pc.Coords(3)
# ANCHOR_END: coords

# ANCHOR: parametres
# FR — Paramètres géométriques / EN — Geometrical parameters
LENGTH, HEIGHT = 0.30, 0.10  # m
HOLE_RADIUS = 0.035  # m
THICKNESS = 0.02  # m
# ANCHOR_END: parametres

# ANCHOR: points
# FR — Points guides : les coins, le centre du demi-disque et du trou, la pointe.
# EN — Guide points: the corners, the half-disc and hole centre, the tip.
p1 = coords.add_node([0.0, 0.0, 0.0])
p2 = coords.add_node([LENGTH, 0.0, 0.0])
p3 = coords.add_node([LENGTH + HEIGHT / 2.0, 0.0, HEIGHT / 2.0])
p4 = coords.add_node([LENGTH, 0.0, HEIGHT])
p5 = coords.add_node([0.0, 0.0, HEIGHT])
p6 = coords.add_node([LENGTH, 0.0, HEIGHT / 2.0])
# ANCHOR_END: points


# ── Contour fermé / Closed contour ────────────────────────────────────────
# ANCHOR: contour_bords
def border_plate_with_hole() -> pc.Mesh:
    # FR — Le contour se maille bord par bord : `line` droit, `arc` courbe.
    # EN — The contour is meshed edge by edge: `line` straight, `arc` curved.
    l12 = pc.mesh.line(p1, p2, 30)
    c23 = pc.mesh.arc(p2, p6, p3, 8)
    c34 = pc.mesh.arc(p3, p6, p4, 8)
    l45 = pc.mesh.line(p4, p5, 30)
    l51 = pc.mesh.line(p5, p1, 10)
    cex = l12 | c23 | c34 | l45 | l51
    # ANCHOR_END: contour_bords

    # ANCHOR: contour_consolidate
    # FR — `|` garde un sous-maillage par bord ; `consolidate_mesh` les fusionne.
    # EN — `|` keeps one submesh per edge; `consolidate_mesh` fuses them.
    cex = pc.mesh.consolidate(cex)
    # ANCHOR_END: contour_consolidate

    # ANCHOR: contour_trou
    # FR — Le trou : cercle centré sur p6, de normale −Y, en 32 segments.
    # EN — The hole: a circle centred on p6, with normal −Y, in 32 segments.
    cin = pc.mesh.circle(p6, [0, -1, 0], HOLE_RADIUS, 32)

    # FR — Deux sous-maillages : le contour extérieur, puis le trou.
    # EN — Two submeshes: the outer contour, then the hole.
    border = cex | cin
    show(border, "Contour de la plaque", "maillage-contour.svg")
    return border
    # ANCHOR_END: contour_trou


# ── Maillage non structuré / Unstructured mesh ────────────────────────────
# ANCHOR: deux_domaines
def unstructured_mesh(
    border: pc.Mesh,
) -> tuple[pc.Mesh, pc.Mesh, pc.Mesh, pc.Mesh]:
    # FR — Les deux boucles tournent dans le même sens : deux domaines pleins.
    # EN — Both loops wind the same way: two filled-in domains.
    plate_two_domains = pc.mesh.triangulate_surface(border, "TRI3", size=0.01)
    show(
        plate_two_domains,
        "Deux boucles CCW : le disque est rempli",
        "maillage-deux-domaines.svg",
    )
    # ANCHOR_END: deux_domaines

    # ANCHOR: invert_trou
    # FR — `invert` retourne la boucle du trou : le mailleur y voit un trou.
    # EN — `invert` flips the hole loop: the mesher then sees a genuine hole.
    border = border[:1] | pc.mesh.invert(border[1:])
    plate = pc.mesh.triangulate_surface(border, "TRI3", size=0.01)
    show(plate, "Plaque non structurée (TRI3)", "maillage-non-structure.svg")
    print(f"unstructured   : {plate.element_types()}, {plate.cell_count()} mailles")
    # ANCHOR_END: invert_trou

    # ANCHOR: extrude
    # FR — L'extrusion balaie la surface sur l'épaisseur : un TRI3 donne un PENTA6.
    # EN — Extrusion sweeps the surface through the thickness: TRI3 gives PENTA6.
    volume_extruded = pc.mesh.extrude(plate, [0, THICKNESS, 0], 2)
    show(
        volume_extruded,
        "Volume extrudé (toutes arêtes)",
        "maillage-volume-aretes.svg",
        wireframe=True,
    )
    show(
        volume_extruded,
        "Volume extrudé (faces cachées)",
        "maillage-volume-extrude.svg",
    )
    # ANCHOR_END: extrude

    # ANCHOR: skin
    # FR — `skin` extrait la peau et la découpe en faces planes, à 85° près.
    # EN — `skin` extracts the boundary and splits it into flat faces, at 85°.
    skin = pc.mesh.skin(volume_extruded, angle_deg=85)
    for i, face in enumerate(skin):
        face.face_color = PALETTE[i % len(PALETTE)]
    show(
        skin[1:-1],
        "Enveloppe QUA4 (faces planes)",
        "maillage-enveloppe-qua4.svg",
        wireframe=True,
    )
    # ANCHOR_END: skin

    # ANCHOR: convert_invert
    # FR — `convert` coupe chaque QUA4 en deux TRI3, `invert` sort les normales.
    # EN — `convert` splits each QUA4 into two TRI3, `invert` turns normals out.
    skin = pc.mesh.invert(pc.mesh.convert(skin, "TRI3"))
    show(
        skin[1:-1],
        "Enveloppe TRI3",
        "maillage-enveloppe-tri3.svg",
        wireframe=True,
    )
    # ANCHOR_END: convert_invert

    # ANCHOR: triangulate_volume
    # FR — Remplissage TET4 ; le mailleur peut redécouper l'enveloppe donnée.
    # EN — TET4 filling; the mesher may re-cut the envelope it was handed.
    volume_tetra = pc.mesh.triangulate_volume(skin, size=0.01, allow_surface_nodes=True)
    # ANCHOR_END: triangulate_volume

    # ANCHOR: noeuds_ajoutes
    # FR — Les nœuds ajoutés forment un second sous-maillage POI1, ici en rouge.
    # EN — The added nodes form a second POI1 submesh, shown here in red.
    volume_tetra[0].face_color = (0, 0, 0)
    for marqueur in volume_tetra[1:]:
        marqueur.face_color = (255, 0, 0)
    show(
        volume_tetra,
        "Volume non structuré triangulé (TET4)",
        "maillage-volume-tetra.svg",
        wireframe=True,
    )
    # ANCHOR_END: noeuds_ajoutes
    return plate, volume_extruded, skin, volume_tetra


# ── Maillage structuré / Structured mesh ──────────────────────────────────
# ANCHOR: comptes
def structured_mesh(plot: bool = True) -> tuple[pc.Mesh, pc.Mesh]:
    # FR — Plus de taille de maille : un nombre d'éléments par direction.
    # EN — No element size any more: an element count per direction.
    n15 = 10  # FR — éléments sur la hauteur / EN — elements through the height
    n12 = 20  # FR — éléments sur la longueur / EN — elements along the length
    # ANCHOR_END: comptes

    # ANCHOR: grille
    # FR — La grille : le bord gauche balayé par translation, un SEG2 → un QUA4.
    # EN — The grid: the left edge swept by translation, a SEG2 → a QUA4.
    l15 = pc.mesh.line(p1, p5, n15)
    x13 = LENGTH - (HEIGHT / 2.0)
    sr1 = pc.mesh.extrude(l15, [x13, 0, 0], n12)

    # FR — `plot=False` : les chapitres suivants importent le volume, sans figures.
    # EN — `plot=False`: later chapters import the volume, without any figure.
    if plot:
        show(sr1, "Grille structurée (QUA4)", "maillage-grille.svg")
    # ANCHOR_END: grille

    # ANCHOR: bord_droit
    # FR — Le bord droit de la grille ne se refabrique pas : il s'extrait.
    # EN — The grid's right edge is not rebuilt: it is extracted.
    border_sr1 = pc.mesh.border(sr1)
    right_nodes = pc.mesh.select(
        pc.node_field.positions(border_sr1, ["X"]), ge=x13 * (n12 - 0.5) / n12
    )
    l1213 = pc.mesh.elements_on(border_sr1, right_nodes, strict=True)
    # ANCHOR_END: bord_droit

    # ANCHOR: extremites
    # FR — Ses deux extrémités, la plus basse et la plus haute en Z.
    # EN — Its two ends, the lowest and the highest in Z.
    p13 = pc.mesh.select(
        pc.node_field.positions(l1213, ["Z"]), le=0.5 / n15 * HEIGHT
    ).node(0, 0, 0)
    p12 = pc.mesh.select(
        pc.node_field.positions(l1213, ["Z"]), ge=(n15 - 0.5) / n15 * HEIGHT
    ).node(0, 0, 0)
    # ANCHOR_END: extremites

    # ANCHOR: boucle_exterieure
    # FR — Boucle extérieure de la couronne : 10+10+5+10+5 = 40 segments.
    # EN — The ring's outer loop: 10+10+5+10+5 = 40 segments.
    cext = (
        pc.mesh.arc(p2, p6, p3, n15)
        | pc.mesh.arc(p3, p6, p4, n15)
        | pc.mesh.line(p4, p12, int(n15 / 2))
        | l1213
        | pc.mesh.line(p13, p2, int(n15 / 2))
    )
    cext = pc.mesh.consolidate(cext)
    # ANCHOR_END: boucle_exterieure

    # ANCHOR: boucle_interieure
    # FR — Boucle intérieure : le trou en quatre quarts, 40 segments aussi.
    # EN — Inner loop: the hole as four quarters, 40 segments as well.
    p14 = coords.add_node([LENGTH, 0.0, HEIGHT / 2.0 - HOLE_RADIUS])
    p15 = coords.add_node([LENGTH + HOLE_RADIUS, 0.0, HEIGHT / 2.0])
    p16 = coords.add_node([LENGTH, 0.0, HEIGHT / 2.0 + HOLE_RADIUS])
    p17 = coords.add_node([LENGTH - HOLE_RADIUS, 0.0, HEIGHT / 2.0])
    cin = (
        pc.mesh.arc(p14, p6, p15, n15)
        | pc.mesh.arc(p15, p6, p16, n15)
        | pc.mesh.arc(p16, p6, p17, n15)
        | pc.mesh.arc(p17, p6, p14, n15)
    )
    cin = pc.mesh.consolidate(cin)

    # FR — Les deux boucles, l'une bleue et l'autre rouge : découpages alignés.
    # EN — Both loops, one blue and one red: their cuttings line up.
    if plot:
        cext.unit().face_color, cin.unit().face_color = PALETTE[2], PALETTE[3]
        show(cext | cin, "Les deux boucles de la couronne", "maillage-boucles.svg")
    # ANCHOR_END: boucle_interieure

    # ANCHOR: sweep
    # FR — `sweep` relie les deux boucles par 3 couches de QUA4.
    # EN — `sweep` links both loops with 3 layers of QUA4.
    sh1 = pc.mesh.sweep(cext, cin, 3)
    if plot:
        show(sh1, "Couronne balayée (QUA4)", "maillage-couronne.svg")

    # FR — Grille et couronne partagent les nœuds du bord droit : `|` suffit.
    # EN — Grid and ring share the right edge's nodes: `|` is enough.
    grid = sr1 | sh1
    if plot:
        print(f"structured      : {grid.element_types()}, {grid.cell_count()} mailles")
        show(grid, "Plaque structurée (QUA4)", "maillage-structure.svg")
    # ANCHOR_END: sweep

    # ANCHOR: volume_hex8
    # FR — Même extrusion qu'en non structuré, mais un QUA4 donne un HEX8.
    # EN — Same extrusion as in the unstructured case, but a QUA4 gives an HEX8.
    volume_structure = pc.mesh.extrude(grid, [0, THICKNESS, 0], 2)
    if plot:
        show(
            volume_structure,
            "Volume structuré (HEX8)",
            "maillage-volume-structure.svg",
        )
    return grid, volume_structure
    # ANCHOR_END: volume_hex8


def main() -> None:
    contour = border_plate_with_hole()
    unstructured_mesh(contour)
    structured_mesh()
    if OUT:
        print(f"Figures written in {OUT}/")


if __name__ == "__main__":
    main()
# ANCHOR_END: script
