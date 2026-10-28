"""Formation débutant — 4. Mécanique non linéaire (plasticité).

Picks the holed plate clamped on the left back up, with the hung mass's force
**ramped up** until it passes the elastic limit —
l'équivalent Python de la table `PASAPAS` de Cast3M (section 9 de la
formation) : ``pyrucast.stepping.step_by_step`` orchestre la boucle
over the load steps and, at each step, a **modified** Newton (elastic
stiffness, sped up by Anderson acceleration).

Replacing ``model.elasticity`` with ``Model.plasticity`` in
`formation/mecanique.py` is all it takes to get this script — the same call
``step_by_step`` gère la boucle non linéaire.

The thermo-plastic coupling of Cast3M's section 9.2 (`sigma_y` depending on
temperature) is not part of this script: ``step_by_step`` takes it through
``material_laws`` (see the book's "Calcul pas à pas" page).

Lancement ::

    maturin develop --release
    python formation/plasticite.py

    # To regenerate the book figure (book/src/formation/img/):
    # PYRUCAST_FORMATION_IMG_DIR=book/src/formation/img python formation/plasticite.py
"""

import os
import tempfile

import pyrucast as pc

LONGUEUR, HAUTEUR = 0.30, 0.10  # m
RAYON_TROU = 0.025  # m
CENTRE_TROU = (0.75 * LONGUEUR, HAUTEUR / 2.0)

E, NU = 200e9, 0.3
# σy deliberately modest: this training's geometry and loading are not at the
# scale of a real steel — what matters is to bring out a plastic zone in a
# few steps, not the material's physical reality.

SIGMA_Y = 5e6
MASSE, G = 2500.0, 9.81
FACTEUR_CHARGE = 6.0  # multiplier on the hung mass, to pass σy


def construire_plaque_trouee():
    """Same geometry as `formation/mecanique.py`."""
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

    # Outer loop CCW, hole clockwise (CW): the orientation
    # `triangulate_surface` expects (the hole is inverted, `trou` stays usable below).
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

    modele = pc.model.plasticity_perfect(fes, "plane_stress")
    modele = modele | pc.model.dirichlet(encastrement, modele, "u_x")
    modele = modele | pc.model.dirichlet(encastrement, modele, "u_y")
    # ANCHOR_END: modele_plastique

    # ANCHOR: chargement_evolution
    pression = -FACTEUR_CHARGE * MASSE * G / (2.0 * 3.14159265 * RAYON_TROU)
    # The pressure on the arc, as nodal forces: the loads of the computation.
    poussee = pc.model.flux(arc_fes, modele, "f_y")
    effort_final = pc.node_field.external_forces(
        poussee, pc.element_field.material_field(poussee, [("phi_f_y", pression)])
    )
    materiaux = pc.element_field.material_field(
        modele, [("E", E), ("nu", NU), ("sigma_y", SIGMA_Y)]
    )
    # Pseudo-time 1 is the nominal load; the ramp carries on past it.
    charge = pc.Evolution(
        [(0.0, effort_final * 0.0), (1.0, effort_final)], out_of_range="extrapolate"
    )
    # ANCHOR_END: chargement_evolution

    # ANCHOR: pas_a_pas
    data = {
        # Elastic over the first steps, plastic beyond — up to 15 % past the
        # nominal load.
        "times": [0.0, 0.2, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15],
        "model": modele,
        "loads": charge,
        "materials": materiaux,
    }
    pc.stepping.step_by_step(data)
    # ANCHOR_END: pas_a_pas

    print(f"{'t':>6} {'itérations':>11} {'anderson':>9} {'convergé':>9} {'p_max':>12}")
    for r in data["results"]:
        p_max = r["behaviour"].max("p") if r["behaviour"] is not None else 0.0
        print(
            f"{r['time']:>6.2f} {r['iterations']:>11} {r['accelerations']:>9} "
            f"{r['converged']!s:>9} {p_max:>12.3e}"
        )

    dernier = data["results"][-1]
    # What the ramp is for: elastic at first, a plastic zone at the end.
    assert data["results"][1]["behaviour"].max("p") == 0.0
    assert dernier["behaviour"].max("p") > 0.0
    print(f"\nzone plastique développée : p_max = {dernier['behaviour'].max('p'):.3e}")

    out = os.environ.get("PYRUCAST_FORMATION_IMG_DIR", tempfile.gettempdir())
    chemin = os.path.join(out, "plasticite.svg")
    plaque.plot(
        save=chemin, field=dernier["behaviour"], component="p", cmap="viridis", smooth=0
    )
    print(f"Plastic zone (p) written to {chemin}")


if __name__ == "__main__":
    main()
