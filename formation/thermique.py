"""Formation débutant — 2. Calcul thermique.

Reprend la plaque trouée de `formation/maillage.py` pour un calcul de
conduction stationnaire avec quatre types de sollicitations thermiques,
combinées comme dans la formation Cast3M :

- **température imposée** sur le bord du trou (Cast3M `BLOQ`/`DEPI`) ;
- **flux imposé** sur le bord gauche (Cast3M `FLUX`) ;
- **convection** (film, Robin) sur une partie du bord bas (Cast3M
  `MODE ... 'CONVECTION'` + `CONV`) ;
- **source volumique** sur une bande centrale (Cast3M `SOUR`).

Contrairement à Cast3M, les régions chargées ne doivent **pas partager de
nœuds** : `pyrucast` combine les seconds membres par union de champs (`|`),
qui exige des supports disjoints (à la différence de Cast3M, qui somme
silencieusement les contributions nodales de bords adjacents). D'où le
léger décalage des bords chargés dans la géométrie ci-dessous.

Limite connue par rapport à Cast3M : **pas de rayonnement** (pas de condition
de bord `εσ(T⁴∞ − T⁴)`) et **pas de terme transitoire** (pas de matrice de
capacité utilisée dans une boucle en temps) — seule la conduction
**stationnaire** est résolue ici. Voir la page thermique du livre pour le
détail de ces limites.

Lancement ::

    maturin develop --release
    python formation/thermique.py

    # Pour régénérer la figure du livre (book/src/formation/img/) :
    # PYRUCAST_FORMATION_IMG_DIR=book/src/formation/img python formation/thermique.py
"""

import os
import tempfile

import pyrucast as pc

# ── Géométrie et données ────────────────────────────────────────────────────
LONGUEUR, HAUTEUR = 0.30, 0.10  # m
RAYON_TROU = 0.025  # m
CENTRE_TROU = (0.75 * LONGUEUR, HAUTEUR / 2.0)

K_COND = 50.0  # W/m/K
FLUX_IMPOSE = -40_000.0  # W/m^2, bord gauche
H_CONV, T_EXT = 240.0, -80.0  # W/m^2/K, °C — convection, bord bas
SOURCE_VOLUMIQUE = 2_600_000.0  # W/m^3, bande centrale
T_IMPOSEE = 250.0  # °C, bord du trou


# ANCHOR: construction
def construire_plaque_trouee():
    """Plaque rectangulaire trouée, construite bord par bord avec les
    mailleurs dédiés (`line`, `circle` — jamais de nœud ou de
    maille ajoutés à la main). `pyrucast.consolidate` fusionne les quatre
    côtés en un seul sous-maillage, requis par `fill_surface` pour former
    une boucle fermée (Cast3M : `cex = l12 ET c23 ET ...` fait le même
    travail implicitement)."""
    coords = pc.Coords(2)
    p1 = coords.add_node([0.0, 0.0])
    p2 = coords.add_node([LONGUEUR, 0.0])
    p3 = coords.add_node([LONGUEUR, HAUTEUR])
    p4 = coords.add_node([0.0, HAUTEUR])

    bord_bas = pc.mesher.line(p1, p2, 10)
    bord_droit = pc.mesher.line(p2, p3, 4)
    bord_haut = pc.mesher.line(p3, p4, 10)
    bord_gauche = pc.mesher.line(p4, p1, 4)
    boucle_ext = pc.consolidate(bord_bas | bord_droit | bord_haut | bord_gauche)

    centre = coords.add_node(list(CENTRE_TROU))
    trou = pc.mesher.circle(centre, [0.0, 0.0, 1.0], RAYON_TROU, 16)

    contour = boucle_ext | trou
    plaque = pc.mesher.fill_surface(contour, "TRI3", max_edge_length=0.02)

    # Bord bas SANS le coin partagé avec le bord gauche (x = 0) : les charges
    # doivent rester sur des supports disjoints (voir plus bas). Sélection
    # par coordonnée, comme pour la bande centrale — pas d'index à la main.
    x = pc.field.coordinates(bord_bas, ["X"])
    noeuds_convection = pc.field.select(x, gt=1e-6)
    bord_bas_convection = pc.mesher.elements_on(
        bord_bas, noeuds_convection, strict=True
    )

    return coords, plaque, bord_bas_convection, bord_gauche, trou


# ANCHOR_END: construction


def main() -> None:
    coords, plaque, bord_bas, bord_gauche, trou = construire_plaque_trouee()

    fes = pc.FiniteElementSpace(plaque)
    bas_fes = pc.FiniteElementSpace(bord_bas)
    gauche_fes = pc.FiniteElementSpace(bord_gauche)

    # ANCHOR: modele
    # Conduction (volume) + convection (film, bord bas) + température imposée
    # (bord du trou) — cf. Model.heat_conduction / Model.convection /
    # Model.dirichlet, analogues de MODE 'THERMIQUE' 'CONDUCTION'/'CONVECTION'
    # et BLOQ.
    modele = pc.Model.heat_conduction(fes) | pc.Model.convection(bas_fes)

    trou_poi1 = pc.mesher.to_poi1(trou)
    multiplicateur_T = pc.mesher.translate(trou_poi1, [0.0, 0.0])
    modele = modele | pc.Model.dirichlet("T", "q", trou_poi1, multiplicateur_T)

    materiaux = pc.build.material_field(modele, [("k", K_COND), ("h", H_CONV)])
    # ANCHOR_END: modele

    # ANCHOR: chargements
    # Flux imposé (bord gauche) — Cast3M FLUX.
    flux_gauche = pc.assemble.flux(gauche_fes[0], FLUX_IMPOSE, "q")

    # Terme externe de la convection h·T_ext — même opérateur `flux`, sans
    # normale requise (Cast3M : partie externe du chargement de convection).
    charge_convection = pc.assemble.flux(bas_fes[0], H_CONV * T_EXT, "q")

    # Source volumique sur la bande centrale — sélection des nœuds internes
    # (ni sur le bord bas, ni sur le bord haut) puis des éléments qu'ils
    # supportent intégralement (Cast3M : ELEM 'APPUYE' 'STRICTEMENT').
    x = pc.field.coordinates(plaque, ["X"])
    noeuds_bande_x = pc.field.select(x, ge=0.35 * LONGUEUR, le=0.55 * LONGUEUR)
    y = pc.field.coordinates(noeuds_bande_x, ["Y"])
    noeuds_bande = pc.field.select(y, gt=1e-6, lt=HAUTEUR - 1e-6)
    elements_bande = pc.mesher.elements_on(plaque, noeuds_bande, strict=True)
    bande_fes = pc.FiniteElementSpace(elements_bande)
    charge_source = pc.assemble.flux(bande_fes[0], SOURCE_VOLUMIQUE, "q")

    # Température imposée (bord du trou) — Cast3M DEPI.
    temperature_imposee = pc.NodeField(multiplicateur_T, ["imposed_T"])
    temperature_imposee[0].add_to_component("imposed_T", T_IMPOSEE)

    # Union : les quatre régions chargées sont disjointes par construction.
    second_membre = (
        flux_gauche | charge_convection | charge_source | temperature_imposee
    )
    # ANCHOR_END: chargements

    # ANCHOR: resolution
    K = pc.assemble.stiffness(modele, materiaux)
    solution = pc.solver.solve(K, second_membre)
    # ANCHOR_END: resolution

    print(f"T min = {solution.min('T'):.1f} °C, T max = {solution.max('T'):.1f} °C")

    out = os.environ.get("PYRUCAST_FORMATION_IMG_DIR", tempfile.gettempdir())
    chemin = os.path.join(out, "thermique.svg")
    plaque.plot(save=chemin, field=solution, component="T", cmap="coolwarm", smooth=1)
    print(f"Champ de température écrit dans {chemin}")


if __name__ == "__main__":
    main()
