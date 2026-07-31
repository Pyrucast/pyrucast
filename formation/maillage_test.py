"""Formation débutant — 1. Maillage.

Construit la géométrie utilisée dans toute la formation : une plaque
rectangulaire percée d'un trou circulaire (l'équivalent pyrucast de la pièce
« structure avec un trou » de la formation Cast3M). Deux mailleurs sont
comparés sur la même géométrie :

- **non structuré** (`pyrucast.mesher.triangulate_surface`) : contour fermé (loi
  extérieure + trou), remplissage par triangulation de Delaunay contrainte,
  taille de maille cible ;
- **structuré** (`pyrucast.mesher.sweep`) : balayage entre deux lignes
  opposées, nombre d'éléments imposé. pyrucast n'a pas encore l'équivalent de
  la surface réglée `REGL` de Cast3M pour mailler *proprement* autour d'un
  trou en structuré ; la version structurée ci-dessous reste donc une grille
  simple, sans trou.

Lancement ::

    maturin develop --release
    python formation/maillage.py

    # Pour régénérer les figures du livre (book/src/formation/img/) :
    # PYRUCAST_FORMATION_IMG_DIR=book/src/formation/img python formation/maillage.py
"""

import os
import tempfile

import pyrucast as pc

# ── Géométrie ─────────────────────────────────────────────────────────────
# ANCHOR: geometrie
# Options generales / general options
coords = pc.Coords(3)
# Parametres geometriques / geometrical parameters
LONGUEUR, HAUTEUR = 0.30, 0.10  # m
RAYON_TROU = 0.035  # m
EPAISSEUR = 0.02
CENTRE_TROU = (0.75 * LONGUEUR, HAUTEUR / 2.0)


def main() -> None:
    # Contour fermé (boucle extérieure + boucle du trou), en SEG2.
    # La boucle extérieure est construite bord par bord avec les mailleurs
    # dédiés (`line` par côté, `circle` pour le trou) — jamais de
    # nœud ou de maille ajoutés à la main. `triangulate_surface` exige qu'une boucle
    # fermée tienne dans **un seul** sous-maillage : `pyrucast.consolidate`
    # fusionne les quatre côtés (chacun son propre sous-maillage `SEG2`) en un
    # seul, sans changer leur connectivité (Cast3M : `cex = l12 ET c23 ET ...`
    # fait implicitement le même travail).
    p1 = coords.add_node([0.0, 0.0, 0.0])
    p2 = coords.add_node([LONGUEUR, 0.0, 0.0])
    p3 = coords.add_node([LONGUEUR + HAUTEUR / 2.0, 0.0, HAUTEUR / 2.0])
    p4 = coords.add_node([LONGUEUR, 0.0, HAUTEUR])
    p5 = coords.add_node([0.0, 0.0, HAUTEUR])
    p6 = coords.add_node([LONGUEUR, 0.0, HAUTEUR / 2.0])
    # Maillage du contour / Contour mesh
    # `triangulate_surface` (triangulation contrainte) redécoupe librement l'intérieur
    # comme les côtés selon `size` : pas besoin de pré-discrétiser
    # finement les côtés droits ici, un seul segment par côté suffit et laisse
    # le raffineur poser ses propres nœuds - une discrétisation fine d'entrée
    # (un nœud tous les 0.01 m) fait au contraire exploser le nombre
    # d'insertions de Steiner nécessaires pour un résultat identique.
    l12 = pc.mesher.line(p1, p2, 30)
    c23 = pc.mesher.arc(p2, p6, p3, 8)
    c34 = pc.mesher.arc(p3, p6, p4, 8)
    l45 = pc.mesher.line(p4, p5, 30)
    l51 = pc.mesher.line(p5, p1, 10)
    cex = l12 | c23 | c34 | l45 | l51
    cex = pc.consolidate(cex)
    cex.plot(view=(-45, 25, 1.0))

    # Maillage du cercle de centre p6
    cin = pc.mesher.circle(p6, [0, -1, 0], RAYON_TROU, 32)
    contour = cex | cin
    contour.plot(view=(-45, 25, 1.0))

    # ANCHOR_END: geometrie
    # ── Maillage non structuré (triangulation contrainte, avec trou) ───────────
    # ANCHOR: non_structure
    # `triangulate_surface` gère nativement le trou (boucle extérieure + boucle du
    # trou en une seule passe, Delaunay contraint + raffinement de Ruppert).
    # `size=0.005` : une taille cible plus petite (ou un contour plus finement
    # discrétisé en entrée) demande bien plus d'insertions de Steiner, jusqu'au
    # plafond du raffineur pour cette taille de contour.
    plaque = pc.mesher.triangulate_surface(contour, "TRI3", size=0.01)
    plaque.plot(view=(-45, 25, 1.0))

    contour = cex | pc.mesher.invert(cin)
    plaque = pc.mesher.triangulate_surface(contour, "TRI3", size=0.01)
    plaque.plot(view=(-45, 25, 1.0))
    print(f"non structuré : {plaque.element_types()}, {plaque.cell_count()} mailles")

    # Même contour, mailleur par pavage frontal : des quadrangles posés
    # directement, en rangées parallèles au bord. `all_quad=True` interdit
    # tout triangle résiduel, d'où une extrusion en HEX8 purs.
    pavee = pc.mesher.pave_surface(contour, "QUA4", size=0.01, all_quad=True)
    pavee.plot(view=(-45, 25, 1.0))
    print(f"pavé          : {pavee.element_types()}, {pavee.cell_count()} mailles")
    hexa = pc.mesher.extrude(pavee, [0, EPAISSEUR, 0], 2)
    print(f"extrudé       : {hexa.element_types()}, {hexa.cell_count()} mailles")

    volume = pc.mesher.extrude(plaque, [0, EPAISSEUR, 0], 2)
    volume.plot(view=(-45, 25, 1.0))
    skin = pc.mesher.skin(volume)
    skin[0].face_color = (255, 0, 0)
    skin = pc.mesher.convert(skin, "TRI3")
    skin.plot(view=(-45, 25, 1.0), wireframe=True)
    # La peau de la plaque ne rentre pas telle quelle dans une
    # tétraédrisation : `allow_surface_nodes` laisse le mailleur découper
    # l'enveloppe là où il en a besoin (la forme, elle, est conservée).
    volume2 = pc.mesher.triangulate_volume(skin, 0.5, allow_surface_nodes=True)
    volume2.plot(view=(45, 25, 1.0), wireframe=False)
    # ANCHOR_END: non_structure

    # ── Maillage structuré (grille QUA4, balayage entre deux bords) ────────────
    # ANCHOR: structure
    bas_gauche = coords.add_node([0.0, 0.0])
    haut_gauche = coords.add_node([0.0, HAUTEUR])
    bord_gauche = pc.mesher.line(bas_gauche, haut_gauche, 4)

    bas_droit = coords.add_node([LONGUEUR, 0.0])
    haut_droit = coords.add_node([LONGUEUR, HAUTEUR])
    bord_droit = pc.mesher.line(bas_droit, haut_droit, 4)

    grille = pc.mesher.sweep(bord_gauche, bord_droit, 12)
    return coords, grille

    # ANCHOR_END: structure

    _, grille = maillage_structure()
    print(f"structuré      : {grille.element_types()}, {grille.cell_count()} mailles")

    out = os.environ.get("PYRUCAST_FORMATION_IMG_DIR", tempfile.gettempdir())
    plaque.plot(save=os.path.join(out, "maillage-non-structure.svg"))
    grille.plot(save=os.path.join(out, "maillage-structure.svg"))
    print(f"Figures écrites dans {out}/")


if __name__ == "__main__":
    main()
