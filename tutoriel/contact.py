# ANCHOR: script
"""Tutoriel débutant — 9.8.2 Contact. / Beginner tutorial — 9.8.2 Contact.

FR — La masse suspendue des pages précédentes devient un **appui** posé dans
l'alésage : un secteur de 90° du même rayon que le trou, mille fois plus
raide que la chape, qui ne peut que monter ou descendre. Le poids est appliqué
sur l'axe de l'appui, en 20 pas. Entre l'appui et la moitié basse de
l'alésage, un **contact** : les nœuds de l'appui ne traversent pas la paroi,
et le contact ne fait que pousser. Le détail pas à pas est dans le livre,
page « Contact ».

EN — The previous pages' hanging mass becomes a **support** laid in the
bore: a 90° sector of the hole's radius, a thousand times stiffer than the
lug, free only to move up or down. The weight is applied on the support's
axis, in 20 steps. Between the support and the lower half of the bore, a
**contact**: the support's nodes do not cross the wall, and the contact only
pushes. The step-by-step walkthrough lives in the book's "Contact" page.

Lancement / Running ::

    pip install pyrucast
    python tutoriel/contact.py

    # Figures du livre / book figures (book/src/tutoriel/img/) :
    # PYRUCAST_TUTORIEL_IMG_DIR=book/src/tutoriel/img python tutoriel/contact.py
"""

import math
import os

import pyrucast as pc
from maillage import HEIGHT, HOLE_RADIUS, LENGTH, OUT, THICKNESS, show, structured_mesh

# FR — La vue des figures de tutoriel/maillage.py : azimut, élévation, échelle.
# EN — The view of tutoriel/maillage.py's figures: azimuth, elevation, scale.
VUE = (-45, 25, 1.0)

ROUGE = (255, 0, 0)  # encastrement / clamp
ORANGE = (255, 140, 0)  # moitié basse de l'alésage / lower half of the bore
VERT = (0, 170, 0)  # l'appui / the support

# ── Données physiques / Physical data ──────────────────────────────────────
# ANCHOR: donnees
E, NU = 200e9, 0.3  # Pa, — : la chape / the lug
E_APPUI = 1000.0 * E  # Pa : l'appui, quasi indéformable / nearly rigid support
MASSE, G = 2500.0, 9.81  # kg, m/s²
POIDS = MASSE * G  # N
T_FINAL, N_PAS = 1.0, 20  # temps fictif / fictitious time
AMPLIFICATION = 50.0
TOL = 1e-9  # m
# ANCHOR_END: donnees


def main() -> None:
    # ANCHOR: chape
    # FR — La chape, ses régions : l'encastrement et la moitié basse de
    #      l'alésage, comme à la page « Élasticité ».
    # EN — The lug, its regions: the clamp and the lower half of the bore, as
    #      on the "Élasticité" page.
    _, volume = structured_mesh(plot=False)
    coords = volume.coords()
    peau = pc.mesh.consolidate(pc.mesh.skin(volume))
    encastrement = pc.mesh.points_on_plane(volume, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    face_gauche = pc.mesh.elements_on(peau, encastrement, strict=True)
    bas_axe = [LENGTH, -THICKNESS, HEIGHT / 2.0]
    haut_axe = [LENGTH, 2.0 * THICKNESS, HEIGHT / 2.0]
    noeuds_alesage = pc.mesh.points_on_cylinder(peau, bas_axe, haut_axe, HOLE_RADIUS)
    alesage = pc.mesh.elements_on(peau, noeuds_alesage, strict=True)
    z_alesage = pc.node_field.positions(alesage, ["Z"])
    noeuds_bas = pc.mesh.select(z_alesage, le=HEIGHT / 2.0 + TOL)
    demi_alesage = pc.mesh.consolidate(
        pc.mesh.elements_on(alesage, noeuds_bas, strict=True)
    )
    # ANCHOR_END: chape

    # ANCHOR: appui
    # FR — L'appui : un rayon, du centre du trou vers le bas à gauche, à 45°
    #      de la verticale, en 2 segments ; tourné de 90° autour de l'axe du
    #      trou en 7 secteurs, il balaie le quart bas du disque. Le nœud du
    #      centre est sur l'axe : les mailles qui le touchent sont des
    #      triangles. Puis l'épaisseur, en 2 couches.
    # EN — The support: a radius, from the hole's centre down to the left,
    #      45° off the vertical, in 2 segments; turned by 90° about the
    #      hole's axis in 7 sectors, it sweeps the disc's lower quarter. The
    #      centre node is on the axis: the cells touching it are triangles.
    #      Then the thickness, in 2 layers.
    centre = coords.add_node([LENGTH, 0.0, HEIGHT / 2.0])
    s = HOLE_RADIUS / math.sqrt(2.0)
    depart = coords.add_node([LENGTH - s, 0.0, HEIGHT / 2.0 - s])
    rayon = pc.mesh.line(centre, depart, 2)
    secteur = pc.mesh.revolve(
        rayon, math.pi / 2.0, 7, [LENGTH, 0.0, HEIGHT / 2.0], [0.0, -1.0, 0.0]
    )
    appui = pc.mesh.extrude(secteur, [0.0, THICKNESS, 0.0], 2)
    noeuds_appui = pc.mesh.consolidate(pc.mesh.to_poi1(appui))

    # FR — Sur l'appui : son axe, où s'applique le poids, et sa face
    #      cylindrique, qui touche l'alésage.
    # EN — On the support: its axis, where the weight is applied, and its
    #      cylindrical face, which touches the bore.
    axe = pc.mesh.consolidate(
        pc.mesh.points_on_line(
            noeuds_appui, [LENGTH, 0.0, HEIGHT / 2.0], [LENGTH, 1.0, HEIGHT / 2.0]
        )
    )
    face_appui = pc.mesh.consolidate(
        pc.mesh.points_on_cylinder(noeuds_appui, bas_axe, haut_axe, HOLE_RADIUS)
    )

    face_gauche.unit().face_color = ROUGE
    demi_alesage.unit().face_color = ORANGE
    for zone in appui:
        zone.face_color = VERT
    show(
        peau | face_gauche | demi_alesage | appui,
        "Encastrement, moitié basse de l'alésage, appui",
        "contact-regions.svg",
        wireframe=True,
    )
    # ANCHOR_END: appui

    # ANCHOR: modele
    # FR — Deux élasticités, une par pièce, réunies par `|`. La chape est
    #      encastrée ; l'appui ne peut ni avancer ni s'écarter (u_x, u_y
    #      bloqués), seulement monter ou descendre.
    # EN — Two elasticities, one per part, joined by `|`. The lug is clamped;
    #      the support can neither move forward nor sideways (u_x, u_y
    #      blocked), only up or down.
    chape = pc.model.elasticity(pc.FiniteElementSpace(volume), "full_3d")
    piece_appui = pc.model.elasticity(pc.FiniteElementSpace(appui), "full_3d")
    solides = chape | piece_appui
    modele = (
        solides
        | pc.model.dirichlet(encastrement, chape, "u_x")
        | pc.model.dirichlet(encastrement, chape, "u_y")
        | pc.model.dirichlet(encastrement, chape, "u_z")
        | pc.model.dirichlet(noeuds_appui, piece_appui, "u_x")
        | pc.model.dirichlet(noeuds_appui, piece_appui, "u_y")
    )

    # FR — Le contact : les nœuds de la face de l'appui (esclaves) ne
    #      traversent pas la moitié basse de l'alésage (maître), ses QUA8
    #      telles quelles, courbes.
    # EN — The contact: the nodes of the support's face (slaves) do not cross
    #      the lower half of the bore (master), its QUA8 as they are, curved.
    contact = pc.model.contact(face_appui, demi_alesage, solides, ["u_x", "u_y", "u_z"])
    modele = modele | contact

    # FR — Un matériau par pièce : chaque champ porte sur son modèle.
    # EN — One material per part: each field covers its own model.
    materiaux = pc.element_field.material_field(
        chape, [("E", E), ("nu", NU)]
    ) | pc.element_field.material_field(piece_appui, [("E", E_APPUI), ("nu", NU)])
    # ANCHOR_END: modele

    # ANCHOR: chargement
    # FR — Le poids, partagé entre les nœuds de l'axe de l'appui, et monté de
    #      0 à 1 sur le temps fictif, en 20 pas.
    # EN — The weight, shared between the nodes of the support's axis, and
    #      ramped from 0 to 1 over the fictitious time, in 20 steps.
    poids = pc.NodeField(axe, ["f_z"])
    for i in range(axe.cell_count):
        poids[0].set_value(axe.node(0, i, 0), "f_z", -POIDS / axe.cell_count)
    instants = []
    for i in range(N_PAS + 1):
        instants.append(i * T_FINAL / N_PAS)
    donnees = {
        "model": modele,
        "times": instants,
        "materials": materiaux,
        "loads": pc.Evolution([(0.0, poids * 0.0), (T_FINAL, poids)]),
    }
    pc.stepping.step_by_step(donnees)
    # ANCHOR_END: chargement

    # ANCHOR: enfoncement
    # FR — L'enfoncement de l'appui, pas après pas : u_z d'un nœud de son
    #      axe, en mm.
    # EN — The support's sinking, step after step: u_z of a node on its
    #      axis, in mm.
    noeud_axe = axe.node(0, 0, 0)
    enfoncement = []
    for resultat in donnees["results"]:
        enfoncement.append(
            (
                resultat["time"],
                resultat["displacement"].value(noeud_axe, "u_z") * 1000.0,
            )
        )
    pc.SubEvolution(enfoncement).plot(
        x_label="temps fictif",
        y_label="u_z de l'appui (mm)",
        title="Enfoncement de l'appui",
        save=os.path.join(OUT, "contact-enfoncement.svg") if OUT else None,
    )
    # ANCHOR_END: enfoncement

    # ANCHOR: efforts
    # FR — Les réactions du dernier pas. Sur les nœuds de l'alésage, que rien
    #      d'autre ne tient, ce sont les efforts du contact sur la chape ; on
    #      les sépare de part et d'autre du centre du trou, les nœuds juste
    #      dessous comptés du côté de l'encastrement.
    # EN — The last step's reactions. On the bore's nodes, which nothing else
    #      holds, they are the contact's forces on the lug; they are split on
    #      either side of the hole's centre, the nodes right below it counted
    #      on the clamp's side.
    dernier = donnees["results"][-1]
    reactions = dernier["reactions"]
    sur_l_encastrement = pc.node_field.restrict(reactions, encastrement)
    sur_l_alesage = pc.node_field.restrict(reactions, demi_alesage)
    x_alesage = pc.node_field.positions(demi_alesage, ["X"])
    cote_encastrement = pc.node_field.restrict(
        reactions, pc.mesh.select(x_alesage, le=LENGTH + TOL)
    )
    cote_bout_libre = pc.node_field.restrict(
        reactions, pc.mesh.select(x_alesage, gt=LENGTH + TOL)
    )
    # ANCHOR_END: efforts

    # ANCHOR: deformee
    deformee = coords.add_config("déformée ×50")
    pc.coords.displace_from(dernier["displacement"] * AMPLIFICATION, 0, deformee)
    with coords.in_config(deformee):
        (volume | appui).plot(
            view=VUE,
            title="Contact — déformée ×50 au dernier pas, u_z (mm)",
            field=dernier["displacement"] * 1000.0,
            component="u_z",
            cmap="viridis",
            save=os.path.join(OUT, "contact-deformee.svg") if OUT else None,
        )
    # ANCHOR_END: deformee

    iterations = []
    for resultat in donnees["results"]:
        iterations.append(resultat["iterations"])
    print(f"appui : {appui.element_types()} {[z.cell_count for z in appui]} mailles")
    print(
        f"appui : {noeuds_appui.cell_count} nœuds, {axe.cell_count} sur l'axe, "
        f"{face_appui.cell_count} sur la face"
    )
    print(f"itérations par pas {iterations[1:]}, {sum(iterations)} en tout")
    print(
        f"enfoncement de l'appui : {enfoncement[1][1]:.4f} mm au premier pas, "
        f"{enfoncement[-1][1]:.4f} mm au dernier"
    )
    print(
        f"contact sur la chape : f_x {sur_l_alesage.sum('f_x'):.1f} N, "
        f"f_z {sur_l_alesage.sum('f_z'):.1f} N"
    )
    print(
        f"  côté encastrement : f_z {cote_encastrement.sum('f_z'):.1f} N ; "
        f"côté bout libre : f_z {cote_bout_libre.sum('f_z'):.1f} N"
    )
    print(
        f"réactions de l'encastrement : f_x {sur_l_encastrement.sum('f_x'):.1f} N, "
        f"f_y {sur_l_encastrement.sum('f_y'):.1f} N, "
        f"f_z {sur_l_encastrement.sum('f_z'):.1f} N"
    )


if __name__ == "__main__":
    main()
# ANCHOR_END: script
