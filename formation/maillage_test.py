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


def contour_plaque_trouee(coords: pc.Coords) -> pc.Mesh:
    """Contour fermé (boucle extérieure + boucle du trou), en SEG2.

    La boucle extérieure est construite bord par bord avec les mailleurs
    dédiés (`line` par côté, `circle` pour le trou) — jamais de
    nœud ou de maille ajoutés à la main. `fill_surface` exige qu'une boucle
    fermée tienne dans **un seul** sous-maillage : `pyrucast.consolidate`
    fusionne les quatre côtés (chacun son propre sous-maillage `SEG2`) en un
    seul, sans changer leur connectivité (Cast3M : `cex = l12 ET c23 ET ...`
    fait implicitement le même travail)."""
    p1 = coords.add_node([0.0, 0.0, 0.0])
    p2 = coords.add_node([LONGUEUR, 0.0, 0.0])
    p3 = coords.add_node([LONGUEUR + HAUTEUR / 2.0, 0.0, HAUTEUR / 2.0])
    p4 = coords.add_node([LONGUEUR, 0.0, HAUTEUR])
    p5 = coords.add_node([0.0, 0.0, HAUTEUR])
    p6 = coords.add_node([LONGUEUR, 0.0, HAUTEUR / 2.0])
    # Maillage du contour / Contour mesh
    # `fill_surface` (triangulation contrainte) redécoupe librement l'intérieur
    # comme les côtés selon `max_edge_length` : pas besoin de pré-discrétiser
    # finement les côtés droits ici, un seul segment par côté suffit et laisse
    # le raffineur poser ses propres nœuds - une discrétisation fine d'entrée
    # (un nœud tous les 0.01 m) fait au contraire exploser le nombre
    # d'insertions de Steiner nécessaires pour un résultat identique.
    l12 = pc.mesher.line(p1, p2, 1)
    c23 = pc.mesher.arc(p2, p6, p3, 6)
    c34 = pc.mesher.arc(p3, p6, p4, 6)
    l45 = pc.mesher.line(p4, p5, 1)
    l51 = pc.mesher.line(p5, p1, 1)
    cex = l12 | c23 | c34 | l45 | l51
    cex = pc.consolidate(cex)

    # Maillage du cercle de centre p6
    cin = pc.mesher.circle(p6, [0, -1, 0], RAYON_TROU, 10)
    return cex | cin


# ANCHOR_END: geometrie


# ── Maillage non structuré (triangulation contrainte, avec trou) ───────────
# ANCHOR: non_structure
def maillage_non_structure(contour) -> tuple[pc.Coords, pc.Mesh]:
    # `fill_surface` gère nativement le trou (boucle extérieure + boucle du
    # trou en une seule passe, Delaunay contraint) — contrairement à
    # `surface` (mailleur frontal) qui ne prend qu'une seule boucle pour
    # l'instant. `max_edge_length=0.025` : en dessous (contour plus finement
    # discrétisé en entrée, ou taille cible plus petite), le raffineur a
    # besoin de bien plus d'insertions de Steiner que son plafond n'en
    # autorise pour cette taille de contour.
    plaque = pc.mesher.fill_surface(contour, "TRI3", max_edge_length=0.025)
    # plaque.plot(view=(-45, 25, 1.0))
    return coords, plaque


def main() -> None:
    contour = contour_plaque_trouee(coords)
    # contour.plot(view=(-45, 25, 1.0))
    _, plaque = maillage_non_structure(contour)
    print(f"non structuré : {plaque.element_types()}, {plaque.cell_count()} mailles")


if __name__ == "__main__":
    main()
