# ANCHOR: script
"""Tutoriel débutant — 9.8.1 Plasticité. / Beginner tutorial — 9.8.1 Plasticity.

FR — La chape de la page « Dilatation, matériau hétérogène » (encastrée,
chargée par la force surfacique, chauffée, α en cloche) dépassait la limite
d'élasticité de son acier. Ce script lui donne une loi **plastique
parfaite** : au-delà de σy, la contrainte n'augmente plus, la matière se
déforme de façon permanente. La force et la température montent ensemble, en
20 pas, sur un temps fictif de 0 à 100. Deux calculs :

1. σy = 250 MPa, la même à toute température ;
2. σy qui baisse avec la température, lue dans une table.

La température est relue dans le fichier qu'écrit
`tutoriel/thermique_non_lineaire.py`. Le détail pas à pas est dans le livre,
page « Plasticité ».

EN — The lug of the "Dilatation, matériau hétérogène" page (clamped, loaded
by the surface force, heated, bell-shaped α) went past its steel's yield
stress. This script gives it a **perfectly plastic** law: beyond σy the
stress no longer rises, and the material deforms permanently. Force and
temperature rise together, in 20 steps, over a fictitious time from 0 to
100. Two analyses:

1. σy = 250 MPa, the same at every temperature;
2. σy dropping with temperature, read from a table.

The temperature is read back from the file `tutoriel/thermique_non_lineaire.py`
writes. The step-by-step walkthrough lives in the book's "Plasticité" page.

Lancement / Running ::

    pip install pyrucast
    python tutoriel/thermique_non_lineaire.py   # écrit / writes thermique.pyr
    python tutoriel/plasticite.py

    # Figures du livre / book figures (book/src/tutoriel/img/) :
    # PYRUCAST_TUTORIEL_IMG_DIR=book/src/tutoriel/img python tutoriel/plasticite.py
"""

import math
import os
import tempfile

import pyrucast as pc
from maillage import HEIGHT, HOLE_RADIUS, LENGTH, OUT, THICKNESS

# FR — La vue des figures de tutoriel/maillage.py : azimut, élévation, échelle.
# EN — The view of tutoriel/maillage.py's figures: azimuth, elevation, scale.
VUE = (-45, 25, 1.0)

# ── Données physiques / Physical data ──────────────────────────────────────
# ANCHOR: donnees
E, NU = 200e9, 0.3  # Pa, —
SIGMA_Y = 250e6  # Pa : limite d'élasticité à froid / yield stress when cold
ALPHA_0 = 1e-5  # 1/K
MU, SIGMA = LENGTH / 2.0, LENGTH / 5.0  # m : la cloche de α / α's bell
T_REF = 0.0  # °C
MASSE, G = 2500.0, 9.81  # kg, m/s²
POIDS = MASSE * G  # N
PRESSION = POIDS / (math.pi * HOLE_RADIUS * THICKNESS)  # Pa

# FR — Un temps fictif : il ne sert qu'à faire monter les chargements.
# EN — A fictitious time: it only drives the loads up.
T_FINAL, N_PAS = 100.0, 20

# FR — σy / σy à froid, selon la température (°C).
# EN — σy / cold σy, against temperature (°C).
TABLE_SIGMA_Y = [
    (0.0, 1.0),
    (100.0, 0.99),
    (150.0, 0.98),
    (200.0, 0.94),
    (215.0, 0.9),
    (230.0, 0.8),
    (235.0, 0.4),
    (240.0, 0.33),
    (250.0, 0.28),
    (275.0, 0.23),
    (300.0, 0.20),
]
AMPLIFICATION = 50.0
FICHIER = os.path.join(tempfile.gettempdir(), "thermique.pyr")
TOL = 1e-9  # m
# ANCHOR_END: donnees


def main() -> None:
    # ANCHOR: relecture
    # FR — Le volume et les températures du premier et du dernier instant.
    # EN — The volume and the temperatures of the first and the last instant.
    if not os.path.exists(FICHIER):
        raise SystemExit(
            f"{FICHIER} manque : lancer d'abord tutoriel/thermique_non_lineaire.py"
        )
    relu = pc.load(FICHIER)
    volume = relu["volume"]
    t_debut = relu["température (°C)"].frames()[0]
    t_fin = relu["température (°C)"].frames()[-1]
    peau = pc.mesh.consolidate(pc.mesh.skin(volume))
    # ANCHOR_END: relecture

    # ANCHOR: regions
    encastrement = pc.mesh.points_on_plane(volume, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    bas_axe = [LENGTH, -THICKNESS, HEIGHT / 2.0]
    haut_axe = [LENGTH, 2.0 * THICKNESS, HEIGHT / 2.0]
    noeuds_alesage = pc.mesh.points_on_cylinder(peau, bas_axe, haut_axe, HOLE_RADIUS)
    alesage = pc.mesh.elements_on(peau, noeuds_alesage, strict=True)
    z_alesage = pc.node_field.positions(alesage, ["Z"])
    noeuds_bas = pc.mesh.select(z_alesage, le=HEIGHT / 2.0 + TOL)
    demi_alesage = pc.mesh.consolidate(
        pc.mesh.elements_on(alesage, noeuds_bas, strict=True)
    )
    # ANCHOR_END: regions

    # ANCHOR: modele
    # FR — La plasticité parfaite en 3D : mêmes inconnues que l'élasticité,
    #      et une variable interne, p, la déformation plastique cumulée.
    # EN — 3-D perfect plasticity: the same unknowns as elasticity, and an
    #      internal variable, p, the cumulated plastic strain.
    fes = pc.FiniteElementSpace(volume)
    plastique = pc.model.plasticity_perfect(fes, "full_3d")
    modele = (
        plastique
        | pc.model.dirichlet(encastrement, plastique, "u_x")
        | pc.model.dirichlet(encastrement, plastique, "u_y")
        | pc.model.dirichlet(encastrement, plastique, "u_z")
    )

    # FR — α en cloche aux points de Gauss, comme à la page précédente ; le
    #      matériau porte en plus « sigma_y ».
    # EN — Bell-shaped α at the Gauss points, as on the previous page; the
    #      material also carries "sigma_y".
    x = pc.element_field.interp_to_gauss(pc.node_field.positions(volume, ["X"]), fes)
    cloche = pc.field.exp(-(((x - MU) / SIGMA) ** 2))
    alpha = (ALPHA_0 * (1.0 + 3.0 * cloche)).rename_component("X", "alpha")
    materiaux = (
        pc.element_field.material_field(
            modele, [("E", E), ("nu", NU), ("sigma_y", SIGMA_Y)]
        )
        | alpha
    )
    # ANCHOR_END: modele

    # ANCHOR: chargements
    # FR — La force surfacique de la page « Élasticité », puis deux
    #      évolutions sur le temps fictif : la force passe de 0 à 1 fois
    #      elle-même, la température du premier au dernier instant thermique.
    # EN — The "Élasticité" page's surface force, then two evolutions over
    #      the fictitious time: the force goes from 0 to 1 times itself, the
    #      temperature from the first to the last thermal instant.
    demi_fes = pc.FiniteElementSpace(demi_alesage)
    verticale = pc.model.flux(demi_fes, plastique, "f_z")
    densite_verticale = pc.element_field.material_field(
        verticale, [("phi_f_z", -PRESSION)]
    )
    force = pc.node_field.external_forces(verticale, densite_verticale)
    rampe_force = pc.Evolution([(0.0, force * 0.0), (T_FINAL, force)])
    rampe_temperature = pc.Evolution([(0.0, t_debut), (T_FINAL, t_fin)])

    instants = []
    for i in range(N_PAS + 1):
        instants.append(i * T_FINAL / N_PAS)
    # ANCHOR_END: chargements

    # ANCHOR: calcul_1
    # FR — 1) σy constante. `step_by_step` remplit `resultats["results"]` :
    #      un état par instant demandé.
    # EN — 1) Constant σy. `step_by_step` fills `resultats["results"]`: one
    #      state per requested instant.
    sigma_y_constante = {
        "model": modele,
        "times": instants,
        "materials": materiaux,
        "loads": rampe_force,
        "temperature": rampe_temperature,
        "t_ref": T_REF,
    }
    pc.stepping.step_by_step(sigma_y_constante)
    # ANCHOR_END: calcul_1

    # ANCHOR: table
    # FR — 2) σy selon la température : une évolution, T en abscisse,
    #      « sigma_y » en ordonnée ; hors de la table, la valeur du bord.
    # EN — 2) σy against temperature: an evolution, T as abscissa,
    #      "sigma_y" as ordinate; outside the table, the edge value.
    points = []
    for temperature, rapport in TABLE_SIGMA_Y:
        points.append((temperature, rapport * SIGMA_Y * 1e-6))
    pc.SubEvolution(points).plot(
        x_label="T (°C)",
        y_label="σy (MPa)",
        title="Limite d'élasticité selon la température",
        save=os.path.join(OUT, "plasticite-sigma-y.svg") if OUT else None,
    )
    sigma_y_de_t = []
    for temperature, rapport in TABLE_SIGMA_Y:
        sigma_y_de_t.append((temperature, rapport * SIGMA_Y))
    lois = pc.Evolution(
        sigma_y_de_t,
        abscissa_type="T",
        ordinate_type="sigma_y",
        out_of_range="clamp",
    )
    # ANCHOR_END: table

    # ANCHOR: calcul_2
    # FR — Le même calcul, avec la table : `material_laws` remplace
    #      « sigma_y » à chaque point de Gauss, selon sa température.
    # EN — The same analysis, with the table: `material_laws` replaces
    #      "sigma_y" at each Gauss point, from its temperature.
    sigma_y_variable = dict(sigma_y_constante, material_laws=lois)
    pc.stepping.step_by_step(sigma_y_variable)
    # ANCHOR_END: calcul_2

    # ANCHOR: evolutions
    # FR — p à chaque instant du second calcul, en millièmes, rangé dans une
    #      évolution et tracé d'un seul appel : dans la fenêtre, un curseur
    #      parcourt les pas. L'image enregistrée est celle du dernier pas.
    # EN — p at each instant of the second analysis, in thousandths, stored
    #      in an evolution and plotted in a single call: in the window, a
    #      slider runs through the steps. The saved image is the last step's.
    champs_p = []
    for resultat in sigma_y_variable["results"]:
        champs_p.append((resultat["time"], resultat["behaviour"] * 1000.0))
    evolution_p = pc.Evolution(champs_p)
    evolution_p.plot(
        mesh=volume,
        view=VUE,
        component="p",
        x_label="temps fictif",
        title="σy selon T — déformation plastique cumulée p (10⁻³)",
        frame=N_PAS,
        save=os.path.join(OUT, "plasticite-p.svg") if OUT else None,
    )

    # FR — von Mises au dernier pas du premier calcul, en MPa.
    # EN — von Mises at the first analysis's last step, in MPa.
    dernier_1 = sigma_y_constante["results"][-1]
    von_mises_1 = pc.element_field.von_mises(modele, dernier_1["behaviour"]) * 1e-6
    volume.plot(
        view=VUE,
        title="σy constante — von Mises au dernier pas (MPa)",
        field=von_mises_1,
        component="von_mises",
        cmap="viridis",
        save=os.path.join(OUT, "plasticite-von-mises.svg") if OUT else None,
    )
    # ANCHOR_END: evolutions

    # ANCHOR: deformee
    dernier_2 = sigma_y_variable["results"][-1]
    coords = volume.coords()
    deformee = coords.add_config("déformée ×50")
    pc.coords.displace_from(dernier_2["displacement"] * AMPLIFICATION, 0, deformee)
    with coords.in_config(deformee):
        volume.plot(
            view=VUE,
            title="σy selon T — déformée ×50 au dernier pas, p (10⁻³)",
            field=dernier_2["behaviour"] * 1000.0,
            component="p",
            cmap="viridis",
            save=os.path.join(OUT, "plasticite-deformee.svg") if OUT else None,
        )
    # ANCHOR_END: deformee

    for nom, calcul in (
        ("σy constante", sigma_y_constante),
        ("σy selon T", sigma_y_variable),
    ):
        iterations = []
        for resultat in calcul["results"]:
            iterations.append(resultat["iterations"])
        dernier = calcul["results"][-1]
        von_mises = pc.element_field.von_mises(modele, dernier["behaviour"]) * 1e-6
        print(f"{nom} : itérations par pas {iterations[1:]}, {sum(iterations)} en tout")
        print(
            f"{nom}, dernier pas : p max {dernier['behaviour'].max('p'):.4e}, "
            f"von Mises max {von_mises.max('von_mises'):.2f} MPa, "
            f"u_z min {dernier['displacement'].min('u_z') * 1000.0:.3f} mm, "
            f"réaction f_z {dernier['reactions'].sum('f_z'):.1f} N"
        )
    premier_pas_plastique = None
    for resultat in sigma_y_constante["results"]:
        if premier_pas_plastique is None and resultat["behaviour"].max("p") > 0.0:
            premier_pas_plastique = resultat["time"]
    print(f"σy constante : premier instant plastifié {premier_pas_plastique}")


if __name__ == "__main__":
    main()
# ANCHOR_END: script
