"""Formation débutant — 4. Mécanique non linéaire (plasticité).

Reprend la plaque trouée encastrée à gauche, avec l'effort de la masse
suspendue **monté progressivement** jusqu'à dépasser la limite élastique —
l'équivalent Python de la table `PASAPAS` de Cast3M (section 9 de la
formation) : ``pyrucast.thermomechanics.step_by_step`` orchestre la boucle
sur les pas de charge et, à chaque pas, un Newton **modifié** (rigidité
élastique, accéléré par l'accélération d'Anderson).

Il suffit de remplacer ``model.elasticity`` par ``Model.plasticity`` dans
`formation/mecanique.py` pour obtenir ce script — le même appel
``step_by_step`` gère la boucle non linéaire.

Point d'attention pyrucast, propre à cette version de la librairie : la
plasticité (`Model.plasticity`) ne consomme pas encore la composante
matériau optionnelle `alpha` (dilatation thermique, Cast3M `EPTH`) — le
couplage thermo-plastique de la section 9.2 de Cast3M (où `sigma_y` dépend
de la température) n'est donc pas repris ici.

Lancement ::

    maturin develop --release
    python formation/plasticite.py

    # Pour régénérer la figure du livre (book/src/formation/img/) :
    # PYRUCAST_FORMATION_IMG_DIR=book/src/formation/img python formation/plasticite.py
"""

import os
import tempfile

import pyrucast as pc

LONGUEUR, HAUTEUR = 0.30, 0.10  # m
RAYON_TROU = 0.025  # m
CENTRE_TROU = (0.75 * LONGUEUR, HAUTEUR / 2.0)

E, NU = 200e9, 0.3
# σy volontairement modeste : la géométrie et le chargement de cette
# formation ne sont pas à l'échelle d'un acier réel — ce qui compte est de
# faire apparaître une zone plastique en quelques pas, pas la réalité
# physique du matériau.
SIGMA_Y = 5e6
MASSE, G = 2500.0, 9.81
FACTEUR_CHARGE = 6.0  # multiplicateur de la masse suspendue, pour dépasser σy


def construire_plaque_trouee():
    """Même géométrie que `formation/mecanique.py`."""
    coords = pc.Coords(2)
    p1 = coords.add_node([0.0, 0.0])
    p2 = coords.add_node([LONGUEUR, 0.0])
    p3 = coords.add_node([LONGUEUR, HAUTEUR])
    p4 = coords.add_node([0.0, HAUTEUR])

    bas = pc.mesh.line(p1, p2, 10)
    droit = pc.mesh.line(p2, p3, 4)
    haut = pc.mesh.line(p3, p4, 10)
    bord_gauche = pc.mesh.line(p4, p1, 4)
    boucle_ext = pc.mesh.consolidate(bas | droit | haut | bord_gauche)

    centre = coords.add_node(list(CENTRE_TROU))
    trou = pc.mesh.circle(centre, [0.0, 0.0, 1.0], RAYON_TROU, 16)

    # Boucle extérieure CCW, trou horaire (CW) : orientation attendue par
    # `triangulate_surface` (le trou est inversé, `trou` reste utilisable ci-dessous).
    contour = boucle_ext | pc.mesh.invert(trou)
    plaque = pc.mesh.triangulate_surface(contour, "TRI3", size=0.02)

    y = pc.node_field.positions(trou, ["Y"])
    noeuds_bas_trou = pc.mesh.select(y, lt=CENTRE_TROU[1])
    arc_bas = pc.mesh.elements_on(trou, noeuds_bas_trou, strict=True)

    return coords, plaque, bord_gauche, arc_bas


def main() -> None:
    _coords, plaque, bord_gauche, arc_bas = construire_plaque_trouee()
    fes = pc.FiniteElementSpace(plaque)
    arc_fes = pc.FiniteElementSpace(arc_bas)

    # ANCHOR: modele_plastique
    encastrement = pc.mesh.to_poi1(bord_gauche)
    multiplicateur = pc.mesh.translate(encastrement, [0.0, 0.0])

    modele = pc.model.plasticity_perfect(fes, "plane_stress")
    modele = modele | pc.model.dirichlet("u_x", "f_x", encastrement, multiplicateur)
    modele = modele | pc.model.dirichlet("u_y", "f_y", encastrement, multiplicateur)
    # ANCHOR_END: modele_plastique

    # ANCHOR: chargement_evolution
    pression = -FACTEUR_CHARGE * MASSE * G / (2.0 * 3.14159265 * RAYON_TROU)
    modele = modele | pc.model.flux(arc_fes, "f_y", "mechanical")
    materiaux = pc.element_field.material_field(
        modele, [("E", E), ("nu", NU), ("sigma_y", SIGMA_Y), ("phi_f_y", pression)]
    )
    effort_final = pc.node_field.external_forces(modele, materiaux)
    charge = pc.Evolution(
        [(0.0, effort_final * 0.0), (1.0, effort_final)], out_of_range="clamp"
    )
    # ANCHOR_END: chargement_evolution

    # ANCHOR: pas_a_pas
    # DDL libres (hors encastrement) pour normer le résidu de Newton — sans
    # quoi les grandes réactions d'appui masquent la convergence réelle.
    x = pc.node_field.positions(plaque, ["X"])
    ddl_libres = pc.mesh.select(x, gt=1e-6)

    data = {
        "times": [0.0, 0.2, 0.4, 0.55, 0.7],  # pseudo-temps ∈ [0, 1]
        "model": modele,
        "loads": charge,
        "materials": materiaux,
        "free_mesh": ddl_libres,
        "max_newton": 200,
    }
    pc.thermomechanics.step_by_step(data)
    # ANCHOR_END: pas_a_pas

    print(f"{'t':>6} {'itérations':>11} {'anderson':>9} {'convergé':>9} {'p_max':>12}")
    for r in data["results"]:
        p_max = r["state"].max("p") if r["state"] is not None else 0.0
        print(
            f"{r['time']:>6.2f} {r['mech_iters']:>11} {r['mech_anderson']:>9} "
            f"{r['converged']!s:>9} {p_max:>12.3e}"
        )

    dernier = data["results"][-1]
    print(f"\nzone plastique développée : p_max = {dernier['state'].max('p'):.3e}")

    out = os.environ.get("PYRUCAST_FORMATION_IMG_DIR", tempfile.gettempdir())
    chemin = os.path.join(out, "plasticite.svg")
    plaque.plot(
        save=chemin, field=dernier["state"], component="p", cmap="viridis", smooth=0
    )
    print(f"Zone plastique (p) écrite dans {chemin}")


if __name__ == "__main__":
    main()
