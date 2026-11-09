# ANCHOR: script
"""Tutoriel débutant — 3. Régime transitoire. / Beginner tutorial — 3. Transient.

FR — Reprend la **chape percée** et les sollicitations de la page « Régime
stationnaire », mais sur une pièce à 0 °C au départ, et avec des
sollicitations qui s'établissent au fil du temps : l'alésage monte à 250 °C
pendant les 1 500 premières secondes, le flux de la face gauche et la source
s'installent entre 4 500 s et 7 500 s, la convection agit dès le début. Le
calcul pas à pas (`pyrucast.stepping`) suit la pièce sur 15 000 s, en 100 pas :

1. **les chargements dans le temps** — deux champs matériau, sollicitations
   éteintes puis pleines, et une évolution qui passe de l'un à l'autre ; une
   évolution de la température imposée ;
2. **le calcul pas à pas** — un dictionnaire qui décrit le calcul, puis
   `pyrucast.stepping.step_by_step` ;
3. **les résultats** — la température en quatre points au fil du temps, le
   champ à trois instants, puis le flux de chaleur au dernier instant.

EN — Picks the **pierced lug** and the loads of the steady-state page back up,
but on a part at 0 °C to start with, and with loads that build up over time:
the bore rises to 250 °C over the first 1,500 seconds, the left face's flux
and the source set in between 4,500 s and 7,500 s, convection acts from the
start. The step-by-step analysis (`pyrucast.stepping`) follows the part over
15,000 s, in 100 steps:

1. **loads in time** — two material fields, loads off then on, and an
   evolution going from one to the other; an evolution of the imposed
   temperature;
2. **the step-by-step analysis** — a dictionary describing it, then
   `pyrucast.stepping.step_by_step`;
3. **the results** — the temperature at four points over time, the field at
   three instants, then the heat flux at the last instant.

Lancement / Running ::

    pip install pyrucast
    python tutoriel/thermique_transitoire.py

    # Figures du livre / book figures (book/src/tutoriel/img/) :
    # PYRUCAST_TUTORIEL_IMG_DIR=book/src/tutoriel/img python tutoriel/thermique_transitoire.py
"""

import os

import pyrucast as pc
from maillage import HEIGHT, HOLE_RADIUS, LENGTH, OUT, THICKNESS, structured_mesh
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
)

# ── Données physiques / Physical data ──────────────────────────────────────
# ANCHOR: donnees
# FR — Les données du stationnaire sont importées de tutoriel/thermique.py ;
#      le transitoire ajoute la capacité de l'acier et le temps.
# EN — The steady-state data come from tutoriel/thermique.py; the transient
#      adds the steel's capacity and the time.
RHO = 7800.0  # kg/m³
CP = 420.0  # J/kg/K
T_INITIALE = 0.0  # °C, toute la pièce au départ / the whole part at start

T_FINAL = 15000.0  # s
N_PAS = 100  # pas de temps / time steps

# FR — Les rampes : l'alésage atteint 250 °C à 0,1 t_f ; le flux de la face
#      gauche et la source s'installent de 0,3 t_f à 0,5 t_f.
# EN — The ramps: the bore reaches 250 °C at 0.1 t_f; the left face's flux
#      and the source set in from 0.3 t_f to 0.5 t_f.
T_ALESAGE_CHAUD = 0.1 * T_FINAL  # s
T_DEBUT_CHARGES = 0.3 * T_FINAL  # s
T_PLEINES_CHARGES = 0.5 * T_FINAL  # s

# FR — Les quatre points suivis, à mi-épaisseur.
# EN — The four points followed, at mid-thickness.
POINT_A = [0.0, THICKNESS / 2.0, 0.0]  # coin bas gauche / lower left corner
POINT_B = [0.45 * LENGTH, THICKNESS / 2.0, HEIGHT]  # bord du haut / top edge
POINT_C = [
    0.8 * LENGTH,
    THICKNESS / 2.0,
    HEIGHT / 2.0,
]  # avant le trou / before the hole
POINT_D = [LENGTH + HEIGHT / 2.0, THICKNESS / 2.0, HEIGHT / 2.0]  # la pointe / the tip

# FR — Les instants tracés, par leur rang : 1 500 s, 6 000 s, 15 000 s.
# EN — The instants plotted, by rank: 1,500 s, 6,000 s, 15,000 s.
PAS_TRACES = [10, 40, 100]
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

    # ANCHOR: modele
    # FR — Le modèle complet d'un tenant : conduction, convection, blocage de
    #      l'alésage, puis les deux termes de flux.
    # EN — The whole model at once: conduction, convection, the bore's
    #      constraint, then the two flux terms.
    fes = pc.FiniteElementSpace(volume)
    conduction = pc.model.heat_conduction(fes)
    basse_fes = pc.FiniteElementSpace(face_basse)
    modele = conduction | pc.model.boundary_transfer(basse_fes, conduction, ["T"])
    modele = modele | pc.model.dirichlet(alesage, conduction, "T")

    flux_gauche = pc.model.flux(pc.FiniteElementSpace(face_gauche), modele, "q")
    source = pc.model.flux(pc.FiniteElementSpace(zone_source), modele, "q")
    # ANCHOR_END: modele

    # ANCHOR: materiaux
    # FR — Deux champs matériau construits de la même façon : les
    #      sollicitations éteintes (phi_q = 0), puis pleines. La capacité
    #      « rho » × « cp » est ce que le transitoire ajoute.
    # EN — Two material fields built the same way: the loads off
    #      (phi_q = 0), then on. The capacity "rho" × "cp" is what the
    #      transient adds.
    eteint = (
        pc.element_field.material_field(
            modele,
            [
                ("k", K_COND),
                ("h_T", H_CONV),
                ("a_ext_T", T_EXT),
                ("rho", RHO),
                ("cp", CP),
            ],
        )
        | pc.element_field.material_field(flux_gauche, [("phi_q", 0.0)])
        | pc.element_field.material_field(source, [("phi_q", 0.0)])
    )
    plein = (
        pc.element_field.material_field(
            modele,
            [
                ("k", K_COND),
                ("h_T", H_CONV),
                ("a_ext_T", T_EXT),
                ("rho", RHO),
                ("cp", CP),
            ],
        )
        | pc.element_field.material_field(flux_gauche, [("phi_q", FLUX_IMPOSE)])
        | pc.element_field.material_field(source, [("phi_q", SOURCE_VOLUMIQUE)])
    )

    # FR — L'évolution reste éteinte jusqu'à 0,3 t_f, puis passe à pleine à
    #      0,5 t_f ; entre deux instants, elle interpole.
    # EN — The evolution stays off until 0.3 t_f, then turns on by 0.5 t_f;
    #      between two instants, it interpolates.
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
    # FR — La température imposée, posée sur les multiplicateurs comme au
    #      stationnaire : 0 °C au départ, 250 °C dès 0,1 t_f.
    # EN — The imposed temperature, set on the multipliers as in the steady
    #      state: 0 °C at the start, 250 °C from 0.1 t_f on.
    froide = pc.NodeField(modele.multiplier_mesh(), ["imposed"]) + T_INITIALE
    chaude = pc.NodeField(modele.multiplier_mesh(), ["imposed"]) + T_IMPOSEE
    chargement = pc.Evolution(
        [(0.0, froide), (T_ALESAGE_CHAUD, chaude), (T_FINAL, chaude)]
    )
    # ANCHOR_END: temperature_imposee

    # ANCHOR: calcul
    # FR — Les instants calculés : 0, 150, 300… 15 000 s.
    # EN — The instants computed: 0, 150, 300… 15,000 s.
    instants = []
    for i in range(N_PAS + 1):
        instants.append(i * T_FINAL / N_PAS)

    # FR — Le calcul se décrit par un dictionnaire, puis se lance.
    # EN — The analysis is described by a dictionary, then run.
    data = {
        "model": modele | flux_gauche | source,
        "times": instants,
        "materials": materiaux,
        "loads": chargement,
        "initial": {"temperature": pc.NodeField(volume, ["T"]) + T_INITIALE},
    }
    pc.stepping.step_by_step(data)
    resultats = data["results"]
    # ANCHOR_END: calcul

    # ANCHOR: courbes
    # FR — Les nœuds les plus proches des quatre points suivis.
    # EN — The nodes closest to the four points followed.
    a = volume.nearest_node(POINT_A)
    b = volume.nearest_node(POINT_B)
    c = volume.nearest_node(POINT_C)
    d = volume.nearest_node(POINT_D)

    # FR — Une liste de couples (temps, température) par point, remplie
    #      instant après instant.
    # EN — One list of (time, temperature) pairs per point, filled instant
    #      after instant.
    courbe_a, courbe_b, courbe_c, courbe_d = [], [], [], []
    for resultat in resultats:
        t, temperature = resultat["time"], resultat["temperature"]
        courbe_a.append((t, temperature.value(a, "T")))
        courbe_b.append((t, temperature.value(b, "T")))
        courbe_c.append((t, temperature.value(c, "T")))
        courbe_d.append((t, temperature.value(d, "T")))

    # FR — Quatre courbes réunies par `|` : une ligne par courbe sur la figure.
    # EN — Four curves gathered by `|`: one line per curve on the figure.
    courbes = (
        pc.SubEvolution(courbe_a)
        | pc.SubEvolution(courbe_b)
        | pc.SubEvolution(courbe_c)
        | pc.SubEvolution(courbe_d)
    )
    courbes.plot(
        x_label="temps (s)",
        y_label="T (°C)",
        title="Température en A, B, C, D",
        save=os.path.join(OUT, "thermique-transitoire-courbes.svg") if OUT else None,
    )
    # ANCHOR_END: courbes

    # ANCHOR: champ_temps
    # FR — Les champs de température, un par instant, rangés dans une
    #      évolution : dans la fenêtre, un curseur passe d'un instant à
    #      l'autre. L'échelle va du fluide (−80 °C) à l'alésage (250 °C).
    # EN — The temperature fields, one per instant, stored in an evolution:
    #      in the window, a slider moves from one instant to the next. The
    #      scale runs from the fluid (−80 °C) to the bore (250 °C).
    champs = []
    for resultat in resultats:
        champs.append((resultat["time"], resultat["temperature"]))
    evolution = pc.Evolution(champs)

    for pas in PAS_TRACES:
        fichier = f"thermique-transitoire-t{instants[pas]:.0f}.svg"
        evolution.plot(
            mesh=volume,
            view=VUE,
            component="T",
            vmin=T_EXT,
            vmax=T_IMPOSEE,
            frame=pas,
            title=f"Température (°C) à t = {instants[pas]:.0f} s",
            save=os.path.join(OUT, fichier) if OUT else None,
        )
    # ANCHOR_END: champ_temps

    # ANCHOR: flux
    # FR — Au dernier instant : le gradient aux points de Gauss, puis ce que la
    #      loi de conduction en tire, k·grad T. Le flux de Fourier, qui va du
    #      chaud vers le froid, est son opposé.
    # EN — At the last instant: the gradient at the Gauss points, then what
    #      the conduction law makes of it, k·grad T. Fourier's flux, from hot
    #      to cold, is its opposite.
    dernier = resultats[-1]["temperature"]
    conductivite = pc.element_field.material_field(conduction, [("k", K_COND)])
    gradient = pc.element_field.strain(conduction, dernier, conductivite)
    repos = pc.element_field.initial_state(conduction, conductivite)
    flux = -pc.element_field.integrate_behavior(
        conduction, gradient, conductivite, repos
    )

    # FR — Deux composantes dans le plan de la pièce : selon X, puis selon Z.
    # EN — Two components in the part's plane: along X, then along Z.
    for composante, fichier in [
        ("flux_x", "thermique-flux-x.svg"),
        ("flux_z", "thermique-flux-z.svg"),
    ]:
        volume.plot(
            view=VUE,
            field=flux,
            component=composante,
            title=f"Flux de chaleur {composante} (W/m²)",
            save=os.path.join(OUT, fichier) if OUT else None,
        )
    # ANCHOR_END: flux

    print(f"volume          : {volume.cell_count} HEX20")
    print(f"instants        : {len(resultats)}, de 0 à {T_FINAL:.0f} s")
    iterations = []
    for resultat in resultats:
        iterations.append(resultat["iterations"])
    print(f"itérations      : {max(iterations)} au plus par pas")
    for pas in PAS_TRACES:
        temperature = resultats[pas]["temperature"]
        print(
            f"t = {instants[pas]:>6.0f} s : T min = {temperature.min('T'):.2f} °C, "
            f"T max = {temperature.max('T'):.2f} °C"
        )
    t_froid, a_froid = courbe_a[0]
    for t, valeur in courbe_a:
        if valeur < a_froid:
            t_froid, a_froid = t, valeur
    print(f"A au plus froid : {a_froid:.2f} °C à {t_froid:.0f} s")
    print(
        f"points au dernier instant : A {dernier.value(a, 'T'):.2f}, "
        f"B {dernier.value(b, 'T'):.2f}, C {dernier.value(c, 'T'):.2f}, "
        f"D {dernier.value(d, 'T'):.2f} °C"
    )
    print(
        f"flux au dernier instant : flux_x de {flux.min('flux_x'):.0f} à "
        f"{flux.max('flux_x'):.0f} W/m², flux_z de {flux.min('flux_z'):.0f} à "
        f"{flux.max('flux_z'):.0f} W/m²"
    )


if __name__ == "__main__":
    main()
# ANCHOR_END: script
