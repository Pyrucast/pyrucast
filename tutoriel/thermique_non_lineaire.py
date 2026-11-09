# ANCHOR: script
"""Tutoriel débutant — 4. Thermique non linéaire. / Beginner tutorial — 4. Nonlinear heat.

FR — Reprend le calcul de la page « Régime transitoire » — la chape percée à
0 °C, les mêmes sollicitations et les mêmes rampes — et ajoute un échange par
**rayonnement** : la grande face arrière de la pièce rayonne vers un
environnement lointain à −140 °C. Le rayonnement est en puissance quatrième de
la température : le problème n'est plus linéaire, et il se calcule en
**kelvins**. Le script mène deux calculs, l'un après l'autre :

1. **sans rayonnement**, en kelvins — il redonne le transitoire de la page
   précédente ;
2. **avec rayonnement** — même description, un terme de plus dans le modèle ;

puis compare les températures des quatre points suivis, parcourt le champ
dans le temps d'un seul tracé, et sauvegarde le calcul dans un fichier.

EN — Picks the transient page's analysis back up — the pierced lug at 0 °C,
the same loads and the same ramps — and adds a **radiation** exchange: the
part's large back face radiates towards a distant environment at −140 °C.
Radiation goes as the fourth power of the temperature: the problem is no
longer linear, and it is computed in **kelvins**. The script runs two
analyses, one after the other:

1. **without radiation**, in kelvins — it gives back the previous page's
   transient;
2. **with radiation** — the same description, one more term in the model;

then compares the temperatures of the four points followed, runs through
the field over time in a single plot, and saves the analysis to a file.

Lancement / Running ::

    pip install pyrucast
    python tutoriel/thermique_non_lineaire.py

    # Figures du livre / book figures (book/src/tutoriel/img/) :
    # PYRUCAST_TUTORIEL_IMG_DIR=book/src/tutoriel/img python tutoriel/thermique_non_lineaire.py
"""

import os
import tempfile

import pyrucast as pc
from maillage import HEIGHT, HOLE_RADIUS, LENGTH, OUT, THICKNESS, show, structured_mesh
from thermique import (
    FLUX_IMPOSE,
    H_CONV,
    K_COND,
    SOURCE_VOLUMIQUE,
    SOURCE_X_MAX,
    SOURCE_X_MIN,
    T_EXT,
    T_IMPOSEE,
    TOL,
    VUE,
    show_nodefield,
)
from thermique_transitoire import (
    CP,
    N_PAS,
    POINT_A,
    POINT_B,
    POINT_C,
    POINT_D,
    RHO,
    T_ALESAGE_CHAUD,
    T_DEBUT_CHARGES,
    T_FINAL,
    T_INITIALE,
    T_PLEINES_CHARGES,
)

# FR — La couleur de la face rayonnante sur les figures.
# EN — The radiating face's colour on the figures.
ROSE = (230, 80, 160)

# ── Données physiques / Physical data ──────────────────────────────────────
# ANCHOR: donnees
# FR — Le rayonnement : l'émissivité de la face, la température lointaine.
# EN — Radiation: the face's emissivity, the distant temperature.
EMISSIVITE = 0.9
T_INFINI = -140.0  # °C

# FR — Une puissance quatrième veut une température absolue : toutes les
#      températures du calcul sont décalées de 273,15 K.
# EN — A fourth power wants an absolute temperature: every temperature of the
#      analysis is shifted by 273.15 K.
ZERO_ABSOLU = 273.15  # K
# ANCHOR_END: donnees


def main() -> None:
    # ANCHOR: maillage
    # FR — Le maillage de tutoriel/maillage.py, tel quel, et sa peau.
    # EN — tutoriel/maillage.py's mesh, as is, and its skin.
    _, volume = structured_mesh(plot=False)
    peau = pc.mesh.consolidate(pc.mesh.skin(volume))
    # ANCHOR_END: maillage

    # ANCHOR: regions
    # FR — Les régions de la page « Régime stationnaire », redécoupées à
    #      l'identique : l'alésage, la face gauche, la face basse, la tranche.
    # EN — The regions of the steady-state page, cut out again the same way:
    #      the bore, the left face, the bottom face, the slice.
    bas_axe = [LENGTH, -THICKNESS, HEIGHT / 2.0]
    haut_axe = [LENGTH, 2.0 * THICKNESS, HEIGHT / 2.0]
    alesage = pc.mesh.consolidate(
        pc.mesh.points_on_cylinder(volume, bas_axe, haut_axe, HOLE_RADIUS)
    )

    noeuds_gauche = pc.mesh.points_on_plane(peau, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    face_gauche = pc.mesh.elements_on(peau, noeuds_gauche, strict=True)

    z_peau = pc.node_field.positions(peau, ["Z"])
    noeuds_bas = pc.mesh.select(z_peau, ge=-TOL, le=TOL)
    face_basse = pc.mesh.elements_on(peau, noeuds_bas, strict=True)

    x_volume = pc.node_field.positions(volume, ["X"])
    noeuds_source = pc.mesh.select(x_volume, ge=SOURCE_X_MIN, le=SOURCE_X_MAX)
    zone_source = pc.mesh.consolidate(
        pc.mesh.elements_on(volume, noeuds_source, strict=True)
    )
    # ANCHOR_END: regions

    # ANCHOR: face_rayonnante
    # FR — La face rayonnante : la grande face arrière, y = épaisseur, repérée
    #      par coordonnée comme la face basse.
    # EN — The radiating face: the large back face, y = thickness, located by
    #      coordinate like the bottom face.
    y_peau = pc.node_field.positions(peau, ["Y"])
    noeuds_arriere = pc.mesh.select(y_peau, ge=THICKNESS - TOL, le=THICKNESS + TOL)
    face_rayonnante = pc.mesh.elements_on(peau, noeuds_arriere, strict=True)

    face_rayonnante.unit().face_color = ROSE
    show(
        peau | face_rayonnante,
        "Face rayonnante",
        "thermique-cl-rayonnement.svg",
        wireframe=True,
    )
    # ANCHOR_END: face_rayonnante

    # ANCHOR: modele
    # FR — Le modèle du transitoire, plus le terme de rayonnement sur la face
    #      arrière, posé contre la conduction comme la convection.
    # EN — The transient's model, plus the radiation term on the back face,
    #      set against the conduction like the convection.
    fes = pc.FiniteElementSpace(volume)
    conduction = pc.model.heat_conduction(fes)
    basse_fes = pc.FiniteElementSpace(face_basse)
    modele = conduction | pc.model.boundary_transfer(basse_fes, conduction, ["T"])
    modele = modele | pc.model.dirichlet(alesage, conduction, "T")

    flux_gauche = pc.model.flux(pc.FiniteElementSpace(face_gauche), modele, "q")
    source = pc.model.flux(pc.FiniteElementSpace(zone_source), modele, "q")
    rayonnement = pc.model.radiation(pc.FiniteElementSpace(face_rayonnante), modele)
    # ANCHOR_END: modele

    # ANCHOR: materiaux
    # FR — Les deux champs matériau du transitoire, en kelvins, plus les
    #      coefficients du rayonnement : l'émissivité « emis » et la
    #      température lointaine « T_inf ».
    # EN — The transient's two material fields, in kelvins, plus the radiation
    #      coefficients: the emissivity "emis" and the distant temperature
    #      "T_inf".
    eteint = (
        pc.element_field.material_field(
            modele,
            [
                ("k", K_COND),
                ("h_T", H_CONV),
                ("a_ext_T", T_EXT + ZERO_ABSOLU),
                ("rho", RHO),
                ("cp", CP),
            ],
        )
        | pc.element_field.material_field(flux_gauche, [("phi_q", 0.0)])
        | pc.element_field.material_field(source, [("phi_q", 0.0)])
        | pc.element_field.material_field(
            rayonnement, [("emis", EMISSIVITE), ("T_inf", T_INFINI + ZERO_ABSOLU)]
        )
    )
    plein = (
        pc.element_field.material_field(
            modele,
            [
                ("k", K_COND),
                ("h_T", H_CONV),
                ("a_ext_T", T_EXT + ZERO_ABSOLU),
                ("rho", RHO),
                ("cp", CP),
            ],
        )
        | pc.element_field.material_field(flux_gauche, [("phi_q", FLUX_IMPOSE)])
        | pc.element_field.material_field(source, [("phi_q", SOURCE_VOLUMIQUE)])
        | pc.element_field.material_field(
            rayonnement, [("emis", EMISSIVITE), ("T_inf", T_INFINI + ZERO_ABSOLU)]
        )
    )
    materiaux = pc.Evolution(
        [
            (0.0, eteint),
            (T_DEBUT_CHARGES, eteint),
            (T_PLEINES_CHARGES, plein),
            (T_FINAL, plein),
        ]
    )
    # ANCHOR_END: materiaux

    # ANCHOR: temperature_imposee
    # FR — La température imposée, elle aussi en kelvins.
    # EN — The imposed temperature, in kelvins as well.
    froide = pc.NodeField(modele.multiplier_mesh(), ["imposed"]) + (
        T_INITIALE + ZERO_ABSOLU
    )
    chaude = pc.NodeField(modele.multiplier_mesh(), ["imposed"]) + (
        T_IMPOSEE + ZERO_ABSOLU
    )
    chargement = pc.Evolution(
        [(0.0, froide), (T_ALESAGE_CHAUD, chaude), (T_FINAL, chaude)]
    )
    # ANCHOR_END: temperature_imposee

    # ANCHOR: calculs
    instants = []
    for i in range(N_PAS + 1):
        instants.append(i * T_FINAL / N_PAS)
    depart = pc.NodeField(volume, ["T"]) + (T_INITIALE + ZERO_ABSOLU)

    # FR — Deux calculs décrits de la même façon : seul le modèle change. Le
    #      champ matériau porte les coefficients du rayonnement dans les deux ;
    #      sans le terme, personne ne les lit.
    # EN — Two analyses described the same way: only the model changes. The
    #      material field carries the radiation coefficients in both; without
    #      the term, nobody reads them.
    sans = {
        "model": modele | flux_gauche | source,
        "times": instants,
        "materials": materiaux,
        "loads": chargement,
        "initial": {"temperature": depart},
    }
    avec = {
        "model": modele | flux_gauche | source | rayonnement,
        "times": instants,
        "materials": materiaux,
        "loads": chargement,
        "initial": {"temperature": depart},
    }
    pc.stepping.step_by_step(sans)
    pc.stepping.step_by_step(avec)
    # ANCHOR_END: calculs

    # ANCHOR: courbes
    # FR — Les quatre points suivis, sans puis avec rayonnement, ramenés en °C.
    # EN — The four points followed, without then with radiation, back in °C.
    a = volume.nearest_node(POINT_A)
    b = volume.nearest_node(POINT_B)
    c = volume.nearest_node(POINT_C)
    d = volume.nearest_node(POINT_D)

    sans_a, sans_b, sans_c, sans_d = [], [], [], []
    avec_a, avec_b, avec_c, avec_d = [], [], [], []
    for r_sans, r_avec in zip(sans["results"], avec["results"]):
        t = r_sans["time"]
        sans_a.append((t, r_sans["temperature"].value(a, "T") - ZERO_ABSOLU))
        sans_b.append((t, r_sans["temperature"].value(b, "T") - ZERO_ABSOLU))
        sans_c.append((t, r_sans["temperature"].value(c, "T") - ZERO_ABSOLU))
        sans_d.append((t, r_sans["temperature"].value(d, "T") - ZERO_ABSOLU))
        avec_a.append((t, r_avec["temperature"].value(a, "T") - ZERO_ABSOLU))
        avec_b.append((t, r_avec["temperature"].value(b, "T") - ZERO_ABSOLU))
        avec_c.append((t, r_avec["temperature"].value(c, "T") - ZERO_ABSOLU))
        avec_d.append((t, r_avec["temperature"].value(d, "T") - ZERO_ABSOLU))

    # FR — Huit courbes sur une figure : A, B, C, D sans rayonnement, puis
    #      les mêmes avec.
    # EN — Eight curves on one figure: A, B, C, D without radiation, then the
    #      same with it.
    courbes = (
        pc.SubEvolution(sans_a)
        | pc.SubEvolution(sans_b)
        | pc.SubEvolution(sans_c)
        | pc.SubEvolution(sans_d)
        | pc.SubEvolution(avec_a)
        | pc.SubEvolution(avec_b)
        | pc.SubEvolution(avec_c)
        | pc.SubEvolution(avec_d)
    )
    courbes.plot(
        x_label="temps (s)",
        y_label="T (°C)",
        title="Température en A, B, C, D, sans puis avec rayonnement",
        save=os.path.join(OUT, "thermique-non-lineaire-courbes.svg") if OUT else None,
    )
    # ANCHOR_END: courbes

    # ANCHOR: champ_final
    # FR — Le champ du dernier instant avec rayonnement, ramené en °C.
    # EN — The last instant's field with radiation, back in °C.
    final = avec["results"][-1]["temperature"] - ZERO_ABSOLU
    show_nodefield(
        volume,
        final,
        "Avec rayonnement — température (°C) à 15 000 s",
        "thermique-non-lineaire-final.svg",
    )
    # ANCHOR_END: champ_final

    # ANCHOR: champ_temps
    # FR — Les 101 champs avec rayonnement, ramenés en °C, rangés dans une
    #      évolution et tracés d'un seul appel : dans la fenêtre, un curseur
    #      parcourt les instants. L'échelle reste celle du transitoire, du
    #      fluide (−80 °C) à l'alésage (250 °C). `frame` choisit l'instant de
    #      l'image enregistrée : le pas 50, à 7 500 s.
    # EN — The 101 fields with radiation, back in °C, stored in an evolution
    #      and plotted in a single call: in the window, a slider runs through
    #      the instants. The scale stays the transient's, from the fluid
    #      (−80 °C) to the bore (250 °C). `frame` picks the saved image's
    #      instant: step 50, at 7,500 s.
    champs = []
    for resultat in avec["results"]:
        champs.append((resultat["time"], resultat["temperature"] - ZERO_ABSOLU))
    evolution = pc.Evolution(champs)
    evolution.plot(
        mesh=volume,
        view=VUE,
        component="T",
        vmin=T_EXT,
        vmax=T_IMPOSEE,
        x_label="temps (s)",
        title="Avec rayonnement — température (°C)",
        frame=N_PAS // 2,
        save=os.path.join(OUT, "thermique-non-lineaire-t7500.svg") if OUT else None,
    )
    # ANCHOR_END: champ_temps

    # ANCHOR: sauvegarde
    # FR — Le volume et l'évolution des températures, écrits dans un fichier
    #      sous des clefs libres, puis relus : un dictionnaire à l'aller, un
    #      dictionnaire au retour. Un nœud du volume relu lit l'évolution
    #      relue, ici interpolée à 10 000 s, entre deux pas.
    # EN — The volume and the temperature evolution, written to a file under
    #      free keys, then read back: a dictionary on the way out, a
    #      dictionary on the way back. A node of the volume read back reads
    #      the evolution read back, interpolated here at 10,000 s, between
    #      two steps.
    chemin = os.path.join(tempfile.gettempdir(), "thermique.pyr")
    pc.save(chemin, {"volume": volume, "température (°C)": evolution})

    relu = pc.load(chemin)
    volume_relu = relu["volume"]
    evolution_relue = relu["température (°C)"]
    b_relu = volume_relu.nearest_node(POINT_B)
    t_b = evolution_relue.interpolate(10000.0).value(b_relu, "T")
    # ANCHOR_END: sauvegarde

    print(f"face rayonnante : {face_rayonnante.cell_count} QUA8")
    for nom, calcul in [("sans", sans), ("avec", avec)]:
        iterations = []
        for resultat in calcul["results"]:
            iterations.append(resultat["iterations"])
        dernier = calcul["results"][-1]["temperature"]
        print(
            f"{nom} rayonnement : {max(iterations)} itérations au plus par pas, "
            f"{sum(iterations)} en tout ; au dernier instant T min = "
            f"{dernier.min('T') - ZERO_ABSOLU:.2f} °C, T max = "
            f"{dernier.max('T') - ZERO_ABSOLU:.2f} °C"
        )
    print(
        f"points au dernier instant, sans : A {sans_a[-1][1]:.2f}, "
        f"B {sans_b[-1][1]:.2f}, C {sans_c[-1][1]:.2f}, D {sans_d[-1][1]:.2f} °C"
    )
    print(
        f"points au dernier instant, avec : A {avec_a[-1][1]:.2f}, "
        f"B {avec_b[-1][1]:.2f}, C {avec_c[-1][1]:.2f}, D {avec_d[-1][1]:.2f} °C"
    )
    milieu = evolution.frames()[N_PAS // 2]
    print(
        f"avec rayonnement à {instants[N_PAS // 2]:.0f} s : T min = "
        f"{milieu.min('T'):.2f} °C, T max = {milieu.max('T'):.2f} °C"
    )
    print(
        f"sauvegarde {chemin} : {os.path.getsize(chemin) / 1e6:.1f} Mo ; relu "
        f"{volume_relu.cell_count} HEX20, {len(evolution_relue.frames())} champs"
    )
    print(
        f"B relu à 10000 s : {t_b:.2f} °C, entre {avec_b[66][1]:.2f} °C à "
        f"{avec_b[66][0]:.0f} s et {avec_b[67][1]:.2f} °C à {avec_b[67][0]:.0f} s"
    )


if __name__ == "__main__":
    main()
# ANCHOR_END: script
