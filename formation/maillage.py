"""Formation débutant — 1. Maillage. / Beginner training — 1. Meshing.

FR — Construit la pièce qui sert de fil rouge : une **chape percée**, plaque
rectangulaire terminée par un demi-disque et trouée en son centre, maillée de
quatre façons successives — contour, surface, volume, puis la même pièce en
**structuré**. C'est l'équivalent pyrucast de la pièce « structure avec un
trou » de la formation Cast3M.

EN — Builds the part used as the guiding thread: a **pierced lug**, a
rectangular plate capped by a half-disc and holed at its centre, meshed in four
successive ways — contour, surface, volume, then the same part **structured**.
The pyrucast counterpart of the "structure with a hole" part of the Cast3M
training.

FR — Cotes : 0.30 × 0.10 m, demi-disque de rayon 0.05 m, trou de rayon
0.035 m, épaisseur 0.02 m. La pièce est plane dans **XZ** et son épaisseur est
portée par **Y** : la géométrie est donc 3D (`Coords(3)`) dès le départ, ce qui
évite d'avoir à la relever au moment de passer au volume.

EN — Dimensions: 0.30 × 0.10 m, half-disc of radius 0.05 m, hole of radius
0.035 m, thickness 0.02 m. The part lies in the **XZ** plane and its thickness
runs along **Y**: the geometry is 3-D (`Coords(3)`) from the start, so nothing
has to be lifted when moving on to the volume.

FR — Deux familles de mailleurs sont comparées, comme en Cast3M :

- **non structuré** (`pyrucast.mesher.triangulate_surface`) : on donne un
  contour fermé et une **taille de maille** cible, le mailleur place ses
  propres nœuds à l'intérieur ;
- **structuré** (`pyrucast.mesher.extrude`, `pyrucast.mesher.sweep`) : on
  impose un **nombre d'éléments**, et le maillage a la topologie d'une grille.

EN — Two families of meshers are compared, as in Cast3M:

- **unstructured** (`pyrucast.mesher.triangulate_surface`): you hand in a
  closed contour and a target **element size**, the mesher places its own nodes
  inside;
- **structured** (`pyrucast.mesher.extrude`, `pyrucast.mesher.sweep`): you
  impose an **element count**, and the mesh has grid topology.

Lancement / Running ::

    maturin develop --release
    python formation/maillage.py

    # Figures du livre / book figures (book/src/formation/img/) :
    # PYRUCAST_FORMATION_IMG_DIR=book/src/formation/img python formation/maillage.py
"""

import os

import pyrucast as pc

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
    """FR — Trace `mesh` : fenêtre interactive, ou SVG si le répertoire de
    figures est défini. Un seul appel par figure, donc un tracé du script = une
    image du livre.

    EN — Plot `mesh`: interactive window, or SVG when the figure directory is
    set. One call per figure, so one plot in the script = one image in the book.
    """
    mesh.plot(
        view=VUE,
        title=title,
        wireframe=wireframe,
        save=os.path.join(OUT, file) if OUT else None,
    )


# ── Géométrie / Geometry ──────────────────────────────────────────────────
# ANCHOR: geometrie
# FR — Options générales : un espace de coordonnées 3D, seul objet mutable du
# script — tous les mailleurs y déposent leurs nœuds.
# EN — General options: one 3-D coordinate space, the script's only mutable
# object — every mesher drops its nodes into it.
coords = pc.Coords(3)

# FR — Paramètres géométriques / EN — Geometrical parameters
LONGUEUR, HAUTEUR = 0.30, 0.10  # m
RAYON_TROU = 0.035  # m
EPAISSEUR = 0.02  # m

# FR — Points guides, les `POIN` de Cast3M. p1/p2/p4/p5 sont les coins du
# rectangle, p6 le centre du demi-disque **et** du trou, p3 la pointe. Rien
# d'autre n'est saisi à la main : tous les autres nœuds sortent d'un mailleur.
# EN — Guide points, the `POIN` of Cast3M. p1/p2/p4/p5 are the rectangle's
# corners, p6 the centre of both the half-disc and the hole, p3 the tip. Nothing
# else is entered by hand: every other node comes out of a mesher.
p1 = coords.add_node([0.0, 0.0, 0.0])
p2 = coords.add_node([LONGUEUR, 0.0, 0.0])
p3 = coords.add_node([LONGUEUR + HAUTEUR / 2.0, 0.0, HAUTEUR / 2.0])
p4 = coords.add_node([LONGUEUR, 0.0, HAUTEUR])
p5 = coords.add_node([0.0, 0.0, HAUTEUR])
p6 = coords.add_node([LONGUEUR, 0.0, HAUTEUR / 2.0])
# ANCHOR_END: geometrie


# ── Contour fermé / Closed contour ────────────────────────────────────────
# ANCHOR: contour
def contour_plaque_trouee() -> pc.Mesh:
    # FR — Le contour se maille **bord par bord**, chacun avec son mailleur
    # dédié : `line` pour les côtés droits, `arc` pour le demi-disque.
    # EN — The contour is meshed **edge by edge**, each with its own mesher:
    # `line` for the straight sides, `arc` for the half-disc.
    #
    # FR — Le nombre d'éléments par bord fixe la finesse du contour, et c'est le
    # seul endroit où l'on en décide : `triangulate_surface` respecte le contour
    # qu'on lui donne et ne le redécoupe jamais.
    # EN — The element count per edge sets the contour's fineness, and that is
    # the only place where it is decided: `triangulate_surface` honours the
    # contour it is handed and never re-cuts it.
    l12 = pc.mesher.line(p1, p2, 30)
    c23 = pc.mesher.arc(p2, p6, p3, 8)
    c34 = pc.mesher.arc(p3, p6, p4, 8)
    l45 = pc.mesher.line(p4, p5, 30)
    l51 = pc.mesher.line(p5, p1, 10)
    cex = l12 | c23 | c34 | l45 | l51

    # FR — `|` réunit les cinq bords dans un même maillage, mais chacun y garde
    # son sous-maillage. Or `triangulate_surface` exige qu'une boucle fermée
    # tienne dans **un seul** sous-maillage : `consolidate` fusionne les cinq
    # sans toucher à la connectivité (Cast3M : `cex = l12 ET c23 ET ...`).
    # EN — `|` brings the five edges into one mesh, but each keeps its own
    # submesh there. `triangulate_surface` requires a closed loop to live in
    # **one** submesh: `consolidate` fuses the five without touching
    # connectivity (Cast3M: `cex = l12 ET c23 ET ...`).
    cex = pc.consolidate(cex)

    # FR — Le trou, en une seule commande : cercle centré sur p6, de normale −Y
    # (donc dans le plan de la pièce), 32 segments.
    # EN — The hole, in a single command: circle centred on p6, with normal −Y
    # (hence in the part's plane), 32 segments.
    cin = pc.mesher.circle(p6, [0, -1, 0], RAYON_TROU, 32)

    # FR — Deux sous-maillages : le contour extérieur, puis le trou.
    # EN — Two submeshes: the outer contour, then the hole.
    contour = cex | cin
    show(contour, "Contour de la plaque", "maillage-contour.svg")
    return contour


# ANCHOR_END: contour


# ── Maillage non structuré / Unstructured mesh ────────────────────────────
# ANCHOR: non_structure
def maillage_non_structure(
    contour: pc.Mesh,
) -> tuple[pc.Mesh, pc.Mesh, pc.Mesh, pc.Mesh]:
    # FR — `triangulate_surface` lit **l'orientation** de chaque boucle : une
    # boucle antihoraire (CCW) est le bord extérieur d'un domaine, une boucle
    # horaire (CW) est un trou. Ici les deux tournent dans le même sens, donc le
    # mailleur voit deux domaines indépendants et **remplit le disque**.
    # EN — `triangulate_surface` reads each loop's **orientation**: a
    # counter-clockwise (CCW) loop is a domain's outer boundary, a clockwise
    # (CW) one is a hole. Here both wind the same way, so the mesher sees two
    # independent domains and **fills the disc in**.
    plaque_deux_domaines = pc.mesher.triangulate_surface(contour, "TRI3", size=0.01)
    show(
        plaque_deux_domaines,
        "Deux boucles CCW : le disque est rempli",
        "maillage-deux-domaines.svg",
    )

    # FR — `invert` retourne la boucle du trou (le sous-maillage 1 du contour) :
    # le mailleur y voit alors un vrai trou. `size` n'est qu'une taille
    # **cible**, le raffinement de Ruppert insère ses propres nœuds à
    # l'intérieur jusqu'à l'approcher.
    # EN — `invert` flips the hole loop (submesh 1 of the contour): the mesher
    # then sees a genuine hole. `size` is only a **target**, Ruppert refinement
    # inserts its own nodes inside until it gets close to it.
    contour = contour[:1] | pc.mesher.invert(contour[1:])
    plaque = pc.mesher.triangulate_surface(contour, "TRI3", size=0.01)
    show(plaque, "Plaque non structurée (TRI3)", "maillage-non-structure.svg")
    print(f"non structuré : {plaque.element_types()}, {plaque.cell_count()} mailles")
    # ANCHOR_END: non_structure

    # ANCHOR: volume
    # FR — Passage au volume : l'extrusion balaie la surface sur l'épaisseur, en
    # 2 couches. Un TRI3 balayé donne un prisme PENTA6.
    # EN — On to the volume: extrusion sweeps the surface through the thickness,
    # in 2 layers. A swept TRI3 gives a PENTA6 prism.
    volume_extrude = pc.mesher.extrude(plaque, [0, EPAISSEUR, 0], 2)
    show(
        volume_extrude,
        "Volume extrudé (toutes arêtes)",
        "maillage-volume-aretes.svg",
        wireframe=True,
    )
    show(
        volume_extrude,
        "Volume extrudé (faces cachées)",
        "maillage-volume-extrude.svg",
    )

    # FR — `skin` extrait la peau du volume et la **découpe en faces planes** :
    # deux facettes voisines restent dans la même face tant que leurs normales
    # diffèrent de moins de 85°. On obtient un sous-maillage par face — dessus,
    # dessous, chant du trou, chant extérieur — colorable indépendamment.
    # EN — `skin` extracts the volume's boundary and **splits it into flat
    # faces**: two neighbouring facets stay in the same face as long as their
    # normals differ by less than 85°. One submesh per face comes out — top,
    # bottom, hole wall, outer wall — each colourable on its own.
    skin = pc.mesher.skin(volume_extrude, angle_deg=85)
    for i, face in enumerate(skin):
        face.face_color = PALETTE[i % len(PALETTE)]
    show(
        skin[1:-1],
        "Enveloppe QUA4 (faces planes)",
        "maillage-enveloppe-qua4.svg",
        wireframe=True,
    )

    # FR — `triangulate_volume` n'accepte qu'une enveloppe **TRI3** dont les
    # normales sortent de la matière : `convert` coupe chaque QUA4 en deux TRI3
    # sans ajouter de nœud, `invert` retourne l'ensemble dans le bon sens.
    # EN — `triangulate_volume` only takes a **TRI3** envelope whose normals
    # point out of the material: `convert` splits each QUA4 into two TRI3
    # without adding a node, `invert` flips the whole thing the right way round.
    enveloppe = pc.mesher.invert(pc.mesher.convert(skin, "TRI3"))
    show(
        enveloppe[1:-1],
        "Enveloppe TRI3",
        "maillage-enveloppe-tri3.svg",
        wireframe=True,
    )

    # FR — Remplissage TET4 de l'enveloppe. `allow_surface_nodes=True` autorise
    # le mailleur à redécouper l'enveloppe là où il ne sait pas la respecter
    # telle quelle : la **forme** est conservée, mais la peau du résultat ne
    # coïncide plus maille pour maille avec celle qu'on a fournie.
    # EN — TET4 filling of the envelope. `allow_surface_nodes=True` lets the
    # mesher re-cut the envelope where it cannot honour it as handed in: the
    # **shape** is kept, but the result's skin no longer matches the supplied
    # one cell for cell.
    volume_tetra = pc.mesher.triangulate_volume(
        enveloppe, size=0.01, allow_surface_nodes=True
    )

    # FR — Le mailleur prévient sur stderr quand il a dû ajouter des nœuds, et
    # il les **nomme** : le résultat porte alors un SECOND sous-maillage, de
    # POI1, en plus des TET4 — d'où `element_types() == ['TET4', 'POI1']`. Piège
    # classique : tout ce qui parcourt les sous-maillages ou compte des mailles
    # doit prendre le TET4 seul. Ici on s'en sert pour montrer les nœuds ajoutés
    # (3 sur cette pièce) en rouge sur la silhouette noire des tétraèdres.
    # EN — The mesher warns on stderr when it had to add nodes, and it **names**
    # them: the result then carries a SECOND submesh, of POI1, next to the TET4
    # one — hence `element_types() == ['TET4', 'POI1']`. Classic trap: anything
    # walking the submeshes or counting cells must take the TET4 one alone. Here
    # we use it to show the added nodes (3 on this part) in red on the
    # tetrahedra's black silhouette.
    volume_tetra[0].face_color = (0, 0, 0)
    for marqueur in volume_tetra[1:]:
        marqueur.face_color = (255, 0, 0)
    show(
        volume_tetra,
        "Volume non structuré triangulé (TET4)",
        "maillage-volume-tetra.svg",
        wireframe=True,
    )
    # ANCHOR_END: volume
    return plaque, volume_extrude, enveloppe, volume_tetra


# ── Maillage structuré / Structured mesh ──────────────────────────────────
# ANCHOR: structure
def maillage_structure() -> tuple[pc.Mesh, pc.Mesh]:
    # FR — En structuré on ne donne plus une taille de maille mais un **nombre
    # d'éléments** par direction. La pièce est traitée en deux morceaux : une
    # grille rectangulaire à gauche, une couronne autour du trou à droite.
    # EN — Structured meshing takes an **element count** per direction rather
    # than an element size. The part is handled in two pieces: a rectangular
    # grid on the left, a ring around the hole on the right.
    n15 = 10  # FR — éléments sur la hauteur / EN — elements through the height
    n12 = 20  # FR — éléments sur la longueur / EN — elements along the length

    # FR — La grille : le bord gauche, balayé par translation. Un SEG2 extrudé
    # donne un QUA4, d'où une grille régulière n12 × n15.
    # EN — The grid: the left edge, swept by translation. An extruded SEG2 gives
    # a QUA4, hence a regular n12 × n15 grid.
    l15 = pc.mesher.line(p1, p5, n15)
    x13 = LONGUEUR - (HAUTEUR / 2.0)
    sr1 = pc.mesher.extrude(l15, [x13, 0, 0], n12)

    # FR — Il faut le bord **droit** de cette grille pour y raccorder la
    # couronne. On ne le refabrique pas avec `line` — ce serait une ligne
    # jumelle sans nœud commun, donc un maillage non conforme : on l'**extrait**.
    # `border` donne le contour de la grille, une sélection sur la coordonnée X
    # garde les nœuds de la dernière colonne, et `elements_on` remonte aux
    # segments dont **tous** les nœuds y sont (`strict=True`).
    # EN — The grid's **right** edge is needed to attach the ring to. It is not
    # rebuilt with `line` — that would be a twin line sharing no node, hence a
    # non-conforming mesh: it is **extracted**. `border` gives the grid's
    # boundary, a selection on the X coordinate keeps the last column's nodes,
    # and `elements_on` walks back to the segments **all** of whose nodes are in
    # it (`strict=True`).
    border_sr1 = pc.mesher.border(sr1)
    noeuds_droite = pc.field.select(
        pc.field.coordinates(border_sr1, ["X"]), ge=x13 * (n12 - 0.5) / n12
    )
    l1213 = pc.mesher.elements_on(border_sr1, noeuds_droite, strict=True)

    # FR — Ses deux extrémités, repérées elles aussi par coordonnée (la plus
    # basse et la plus haute en Z) : ce sont les points de raccord de la
    # couronne.
    # EN — Its two ends, located by coordinate as well (lowest and highest in
    # Z): these are the ring's attachment points.
    p13 = pc.field.select(
        pc.field.coordinates(l1213, ["Z"]), le=0.5 / n15 * HAUTEUR
    ).node(0, 0, 0)
    p12 = pc.field.select(
        pc.field.coordinates(l1213, ["Z"]), ge=(n15 - 0.5) / n15 * HAUTEUR
    ).node(0, 0, 0)

    # FR — Boucle extérieure de la couronne : le demi-disque, les deux tronçons
    # de bord restants et le bord droit de la grille — 10+10+5+10+5 = 40
    # segments, consolidés en une seule boucle fermée.
    # EN — The ring's outer loop: the half-disc, the two remaining edge pieces
    # and the grid's right edge — 10+10+5+10+5 = 40 segments, consolidated into
    # a single closed loop.
    cext = (
        pc.mesher.arc(p2, p6, p3, n15)
        | pc.mesher.arc(p3, p6, p4, n15)
        | pc.mesher.line(p4, p12, int(n15 / 2))
        | l1213
        | pc.mesher.line(p13, p2, int(n15 / 2))
    )
    cext = pc.consolidate(cext)

    # FR — Boucle intérieure : le trou, en quatre quarts de 10 segments, soit
    # **40 segments** comme la boucle extérieure — condition pour balayer l'une
    # vers l'autre. Les quatre points de départ sont posés explicitement pour
    # que le découpage du cercle s'aligne sur celui du contour extérieur.
    # EN — Inner loop: the hole, as four quarters of 10 segments, i.e. **40
    # segments** like the outer loop — the condition for sweeping one onto the
    # other. The four start points are placed explicitly so the circle's cutting
    # lines up with the outer contour's.
    p14 = coords.add_node([LONGUEUR, 0.0, HAUTEUR / 2.0 - RAYON_TROU])
    p15 = coords.add_node([LONGUEUR + RAYON_TROU, 0.0, HAUTEUR / 2.0])
    p16 = coords.add_node([LONGUEUR, 0.0, HAUTEUR / 2.0 + RAYON_TROU])
    p17 = coords.add_node([LONGUEUR - RAYON_TROU, 0.0, HAUTEUR / 2.0])
    cin = (
        pc.mesher.arc(p14, p6, p15, n15)
        | pc.mesher.arc(p15, p6, p16, n15)
        | pc.mesher.arc(p16, p6, p17, n15)
        | pc.mesher.arc(p17, p6, p14, n15)
    )
    cin = pc.consolidate(cin)

    # FR — `sweep` relie les deux boucles par 3 couches de QUA4 : c'est
    # l'équivalent de la surface réglée `REGL` de Cast3M, et le moyen d'obtenir
    # un maillage structuré propre autour d'un trou.
    # EN — `sweep` links both loops with 3 layers of QUA4: the counterpart of
    # Cast3M's ruled surface `REGL`, and the way to get a clean structured mesh
    # around a hole.
    sh1 = pc.mesher.sweep(cext, cin, 3)

    # FR — Grille et couronne partagent les nœuds du bord droit `l1213` : le
    # maillage est conforme, `|` suffit à les réunir.
    # EN — Grid and ring share the nodes of the right edge `l1213`: the mesh is
    # conforming, `|` is enough to bring them together.
    grille = sr1 | sh1
    show(grille, "Plaque structurée (QUA4)", "maillage-structure.svg")
    print(f"structuré      : {grille.element_types()}, {grille.cell_count()} mailles")

    # FR — Même extrusion que pour le non structuré, mais un QUA4 balayé donne
    # un hexaèdre HEX8 — le meilleur élément pour le calcul.
    # EN — Same extrusion as for the unstructured mesh, but a swept QUA4 gives
    # an HEX8 hexahedron — the best element to compute on.
    volume_structure = pc.mesher.extrude(grille, [0, EPAISSEUR, 0], 2)
    show(volume_structure, "Volume structuré (HEX8)", "maillage-volume-structure.svg")
    return grille, volume_structure


# ANCHOR_END: structure


def main() -> None:
    contour = contour_plaque_trouee()
    maillage_non_structure(contour)
    maillage_structure()
    if OUT:
        print(f"Figures écrites dans {OUT}/")


if __name__ == "__main__":
    main()
