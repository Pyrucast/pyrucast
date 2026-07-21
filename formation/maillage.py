"""Formation débutant — 1. Maillage.

Construit la géométrie utilisée dans toute la formation : une plaque
rectangulaire percée d'un trou circulaire (l'équivalent pyrucast de la pièce
« structure avec un trou » de la formation Cast3M). Deux mailleurs sont
comparés sur la même géométrie :

- **non structuré** (`pyrucast.mesher.fill_surface`) : contour fermé (loi
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
LONGUEUR, HAUTEUR = 0.30, 0.10  # m
RAYON_TROU = 0.025  # m
CENTRE_TROU = (0.75 * LONGUEUR, HAUTEUR / 2.0)


def contour_plaque_trouee(coords: pc.Coords) -> pc.Mesh:
    """Contour fermé (boucle extérieure + boucle du trou), en SEG2.

    La boucle extérieure est construite bord par bord avec les mailleurs
    dédiés (`line` par côté, `circle_seg2` pour le trou) — jamais de
    nœud ou de maille ajoutés à la main. `fill_surface` exige qu'une boucle
    fermée tienne dans **un seul** sous-maillage : `pyrucast.consolidate`
    fusionne les quatre côtés (chacun son propre sous-maillage `SEG2`) en un
    seul, sans changer leur connectivité (Cast3M : `cex = l12 ET c23 ET ...`
    fait implicitement le même travail)."""
    p1 = coords.add_node([0.0, 0.0])
    p2 = coords.add_node([LONGUEUR, 0.0])
    p3 = coords.add_node([LONGUEUR, HAUTEUR])
    p4 = coords.add_node([0.0, HAUTEUR])

    bas = pc.mesher.line(p1, p2, 10)
    droit = pc.mesher.line(p2, p3, 4)
    haut = pc.mesher.line(p3, p4, 10)
    gauche = pc.mesher.line(p4, p1, 4)
    boucle_ext = pc.consolidate(bas | droit | haut | gauche)

    centre = coords.add_node(list(CENTRE_TROU))
    boucle_trou = pc.mesher.circle_seg2(centre, [0.0, 0.0, 1.0], RAYON_TROU, 16)

    return boucle_ext | boucle_trou


# ANCHOR_END: geometrie


# ── Maillage non structuré (triangulation contrainte, avec trou) ───────────
# ANCHOR: non_structure
def maillage_non_structure() -> tuple[pc.Coords, pc.Mesh]:
    coords = pc.Coords(2)
    contour = contour_plaque_trouee(coords)
    plaque = pc.mesher.fill_surface(contour, "TRI3", max_edge_length=0.02)
    return coords, plaque


# ANCHOR_END: non_structure


# ── Maillage structuré (grille QUA4, balayage entre deux bords) ────────────
# ANCHOR: structure
def maillage_structure() -> tuple[pc.Coords, pc.Mesh]:
    coords = pc.Coords(2)
    bas_gauche = coords.add_node([0.0, 0.0])
    haut_gauche = coords.add_node([0.0, HAUTEUR])
    bord_gauche = pc.mesher.line(bas_gauche, haut_gauche, 4)

    bas_droit = coords.add_node([LONGUEUR, 0.0])
    haut_droit = coords.add_node([LONGUEUR, HAUTEUR])
    bord_droit = pc.mesher.line(bas_droit, haut_droit, 4)

    grille = pc.mesher.sweep(bord_gauche, bord_droit, 12)
    return coords, grille


# ANCHOR_END: structure


def main() -> None:
    _, plaque = maillage_non_structure()
    print(f"non structuré : {plaque.element_types()}, {plaque.cell_count()} mailles")

    _, grille = maillage_structure()
    print(f"structuré      : {grille.element_types()}, {grille.cell_count()} mailles")

    out = os.environ.get("PYRUCAST_FORMATION_IMG_DIR", tempfile.gettempdir())
    plaque.plot(save=os.path.join(out, "maillage-non-structure.svg"))
    grille.plot(save=os.path.join(out, "maillage-structure.svg"))
    print(f"Figures écrites dans {out}/")


if __name__ == "__main__":
    main()
