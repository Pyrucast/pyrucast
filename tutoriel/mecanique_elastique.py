# ANCHOR: script
"""Tutoriel débutant — 9.7.1 Élasticité. / Beginner tutorial — 9.7.1 Elasticity.

FR — Reprend la **chape percée** de la page « Maillage » (le volume HEX20
structuré de `tutoriel/maillage.py`), l'encastre par sa face gauche et y
suspend une masse de 2 500 kg par l'alésage. La masse est posée de trois
façons :

1. une **force ponctuelle** au bas de l'alésage ;
2. une **pression** sur la moitié basse de l'alésage ;
3. une **force surfacique verticale** sur la même moitié.

Les trois cas se résolvent sur la même matrice de rigidité. On lit ensuite les
réactions de l'encastrement, la déformée amplifiée, les déformations, les
contraintes et la contrainte de von Mises, puis von Mises le long de deux
lignes. Le détail pas à pas est dans le livre, page « Élasticité ».

EN — Picks the meshing page's **pierced lug** back up (the structured HEX20
volume of `tutoriel/maillage.py`), clamps it by its left face and hangs a
2,500 kg mass from its bore. The mass is laid on in three ways:

1. a **point force** at the bottom of the bore;
2. a **pressure** on the lower half of the bore;
3. a **vertical surface force** on the same half.

The three cases are solved on the same stiffness matrix. Then come the
clamp's reactions, the magnified deformed shape, the strains, the stresses and
the von Mises stress, and von Mises along two lines. The step-by-step
walkthrough lives in the book's "Élasticité" page.

Lancement / Running ::

    pip install pyrucast
    python tutoriel/mecanique_elastique.py

    # Figures du livre / book figures (book/src/tutoriel/img/) :
    # PYRUCAST_TUTORIEL_IMG_DIR=book/src/tutoriel/img python tutoriel/mecanique_elastique.py
"""

import math
import os

import pyrucast as pc
from maillage import HEIGHT, HOLE_RADIUS, LENGTH, OUT, THICKNESS, show, structured_mesh

# FR — La vue des figures de tutoriel/maillage.py : azimut, élévation, échelle.
# EN — The view of tutoriel/maillage.py's figures: azimuth, elevation, scale.
VUE = (-45, 25, 1.0)

# FR — Une couleur par région, tenue d'une figure à l'autre.
# EN — One colour per region, kept from one figure to the next.
ROUGE = (255, 0, 0)  # encastrement / clamp
ORANGE = (255, 140, 0)  # moitié basse de l'alésage / lower half of the bore
BLEU = (0, 0, 255)  # point de la masse / the mass's point

# ── Données physiques / Physical data ──────────────────────────────────────
# ANCHOR: donnees
E, NU = 200e9, 0.3  # Pa, — : un acier / a steel
MASSE, G = 2500.0, 9.81  # kg, m/s²
POIDS = MASSE * G  # N

# FR — Le poids réparti sur la moitié basse de l'alésage, d'aire π·r·e.
# EN — The weight spread over the lower half of the bore, of area π·r·e.
PRESSION = POIDS / (math.pi * HOLE_RADIUS * THICKNESS)  # Pa

# FR — La déformée est tracée 50 fois plus grande que nature.
# EN — The deformed shape is drawn 50 times larger than life.
AMPLIFICATION = 50.0

# FR — Une face plane vaut zéro à l'arrondi près : on sélectionne une bande.
# EN — A flat face is zero up to rounding: a band is selected, not a value.
TOL = 1e-9  # m
# ANCHOR_END: donnees


def main() -> None:
    # ANCHOR: maillage
    # FR — Le maillage de tutoriel/maillage.py, tel quel, et sa peau.
    # EN — tutoriel/maillage.py's mesh, as is, and its skin.
    _, volume = structured_mesh(plot=False)
    peau = pc.mesh.consolidate(pc.mesh.skin(volume))
    # ANCHOR_END: maillage

    # ── Régions / Regions ───────────────────────────────────────────────────
    # ANCHOR: regions
    # FR — L'encastrement : les nœuds du plan x = 0, et les faces qu'ils portent
    #      pour la figure.
    # EN — The clamp: the nodes of the plane x = 0, and the faces they carry,
    #      for the figure.
    encastrement = pc.mesh.points_on_plane(volume, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    face_gauche = pc.mesh.elements_on(peau, encastrement, strict=True)

    # FR — L'alésage : les faces de la peau sur le cylindre du trou ; puis sa
    #      moitié basse, z ≤ h/2.
    # EN — The bore: the skin's faces on the hole's cylinder; then its lower
    #      half, z ≤ h/2.
    bas_axe = [LENGTH, -THICKNESS, HEIGHT / 2.0]
    haut_axe = [LENGTH, 2.0 * THICKNESS, HEIGHT / 2.0]
    noeuds_alesage = pc.mesh.points_on_cylinder(peau, bas_axe, haut_axe, HOLE_RADIUS)
    alesage = pc.mesh.elements_on(peau, noeuds_alesage, strict=True)
    z_alesage = pc.node_field.positions(alesage, ["Z"])
    noeuds_bas = pc.mesh.select(z_alesage, le=HEIGHT / 2.0 + TOL)
    demi_alesage = pc.mesh.consolidate(
        pc.mesh.elements_on(alesage, noeuds_bas, strict=True)
    )

    # FR — Le point de la masse : le nœud de l'alésage le plus proche de son
    #      point le plus bas, au milieu de l'épaisseur.
    # EN — The mass's point: the bore node closest to its lowest point, in the
    #      middle of the thickness.
    point_masse = alesage.nearest_node(
        [LENGTH, THICKNESS / 2.0, HEIGHT / 2.0 - HOLE_RADIUS]
    )
    poi_masse = pc.mesh.poi1_from_nodes([point_masse])
    # ANCHOR_END: regions

    # ANCHOR: figure_regions
    face_gauche.unit().face_color = ROUGE
    demi_alesage.unit().face_color = ORANGE
    poi_masse.unit().face_color = BLEU
    show(
        peau | face_gauche | demi_alesage | poi_masse,
        "Encastrement, moitié basse de l'alésage, point de la masse",
        "mecanique-elastique-regions.svg",
        wireframe=True,
    )
    # ANCHOR_END: figure_regions

    # ── Modèle / Model ──────────────────────────────────────────────────────
    # ANCHOR: modele
    # FR — L'élasticité en 3D : primales u_x, u_y, u_z, duales f_x, f_y, f_z.
    # EN — 3-D elasticity: primals u_x, u_y, u_z, duals f_x, f_y, f_z.
    fes = pc.FiniteElementSpace(volume)
    elasticite = pc.model.elasticity(fes, "full_3d")

    # FR — L'encastrement : un blocage par composante, chacun avec ses
    #      multiplicateurs, qui porteront les réactions.
    # EN — The clamp: one constraint per component, each with its
    #      multipliers, which will carry the reactions.
    blocage_x = pc.model.dirichlet(encastrement, elasticite, "u_x")
    blocage_y = pc.model.dirichlet(encastrement, elasticite, "u_y")
    blocage_z = pc.model.dirichlet(encastrement, elasticite, "u_z")
    modele = elasticite | blocage_x | blocage_y | blocage_z

    materiaux = pc.element_field.material_field(modele, [("E", E), ("nu", NU)])
    K = pc.matrix.stiffness(modele, materiaux)
    # ANCHOR_END: modele

    # ── Trois façons de poser la masse / Three ways to lay the mass on ──────
    # ANCHOR: force_ponctuelle
    # FR — 1) Tout le poids sur un seul nœud, vers le bas.
    # EN — 1) The whole weight on a single node, downwards.
    force_ponctuelle = pc.NodeField(poi_masse, ["f_z"])
    force_ponctuelle[0].set_value(point_masse, "f_z", -POIDS)
    # ANCHOR_END: force_ponctuelle

    # ANCHOR: pression
    # FR — 2) Une pression sur la moitié basse : une densité « phi_n » le long
    #      de la normale sortante n. Négative, elle pousse dans la matière.
    # EN — 2) A pressure on the lower half: a density "phi_n" along the
    #      outward normal n. Negative, it pushes into the material.
    demi_fes = pc.FiniteElementSpace(demi_alesage)
    pression = pc.model.flux(demi_fes, elasticite, "n")
    densite_pression = pc.element_field.material_field(pression, [("phi_n", -PRESSION)])
    force_pression = pc.node_field.external_forces(pression, densite_pression)
    # ANCHOR_END: pression

    # ANCHOR: force_surfacique
    # FR — 3) La même densité, mais verticale : une composante « f_z » sur la
    #      même moitié.
    # EN — 3) The same density, but vertical: an "f_z" component on the same
    #      half.
    verticale = pc.model.flux(demi_fes, elasticite, "f_z")
    densite_verticale = pc.element_field.material_field(
        verticale, [("phi_f_z", -PRESSION)]
    )
    force_surfacique = pc.node_field.external_forces(verticale, densite_verticale)
    # ANCHOR_END: force_surfacique

    # ANCHOR: resolutions
    # FR — Trois seconds membres, une seule matrice : la factorisation de K est
    #      faite au premier appel et reprise aux deux suivants.
    # EN — Three right-hand sides, a single matrix: K is factorized on the
    #      first call and reused on the next two.
    u_ponctuelle = pc.solver.solve(K, force_ponctuelle)
    u_pression = pc.solver.solve(K, force_pression)
    u_surfacique = pc.solver.solve(K, force_surfacique)
    # ANCHOR_END: resolutions

    # ANCHOR: reactions
    # FR — Les réactions : les multiplicateurs λ de chaque blocage, lus sur
    #      ses nœuds multiplicateurs et sommés. λ est l'effort de la pièce sur
    #      le blocage ; l'encastrement, lui, pousse la pièce de −λ.
    # EN — The reactions: each constraint's multipliers λ, read on its
    #      multiplier nodes and summed. λ is the part's force on the
    #      constraint; the clamp pushes the part by −λ.
    reactions = []
    for u in (u_ponctuelle, u_pression, u_surfacique):
        r_x = -pc.node_field.restrict(u, blocage_x.multiplier_mesh()).sum("lambda")
        r_y = -pc.node_field.restrict(u, blocage_y.multiplier_mesh()).sum("lambda")
        r_z = -pc.node_field.restrict(u, blocage_z.multiplier_mesh()).sum("lambda")
        reactions.append((r_x, r_y, r_z))
    # ANCHOR_END: reactions

    # ANCHOR: deformee
    # FR — Une seconde configuration des mêmes nœuds, posée à X + 50·u ; le
    #      bloc `with` la rend active le temps de la figure.
    # EN — A second configuration of the same nodes, laid at X + 50·u; the
    #      `with` block makes it active for the time of the figure.
    coords = volume.coords()
    deformee = coords.add_config("déformée ×50")
    pc.coords.displace_from(u_surfacique * AMPLIFICATION, 0, deformee)
    with coords.in_config(deformee):
        volume.plot(
            view=VUE,
            title="Force surfacique — déformée ×50, u_z (mm)",
            field=u_surfacique * 1000.0,
            component="u_z",
            cmap="viridis",
            save=os.path.join(OUT, "mecanique-elastique-deformee.svg") if OUT else None,
        )
    # ANCHOR_END: deformee

    # ANCHOR: contraintes
    # FR — Aux points de Gauss : la déformation ε = [B]{U}, la contrainte
    #      σ = [C] ε, puis la contrainte équivalente de von Mises.
    # EN — At the Gauss points: the strain ε = [B]{U}, the stress σ = [C] ε,
    #      then the von Mises equivalent stress.
    deformation = pc.element_field.strain(modele, u_surfacique, materiaux)
    contrainte = pc.element_field.elastic_stress(modele, deformation, materiaux)
    von_mises = pc.element_field.von_mises(modele, contrainte)

    # FR — En MPa pour les figures : un champ se multiplie comme un nombre.
    # EN — In MPa for the figures: a field multiplies like a number.
    von_mises_mpa = von_mises * 1e-6
    volume.plot(
        view=VUE,
        title="Force surfacique — von Mises (MPa)",
        field=von_mises_mpa,
        component="von_mises",
        cmap="viridis",
        save=os.path.join(OUT, "mecanique-elastique-von-mises.svg") if OUT else None,
    )
    # ANCHOR_END: contraintes

    # ANCHOR: le_long_du_bord
    # FR — Le bord x = 0, y = 0, de bas en haut : 101 points sur une droite,
    #      un maillage POI1 dans l'ordre. Chaque point prend la valeur de
    #      l'élément qui le contient, tirée de ses seuls points de Gauss.
    # EN — The edge x = 0, y = 0, bottom to top: 101 points on a straight
    #      line, a POI1 mesh in order. Each point takes the value of the
    #      element holding it, drawn from its own Gauss points only.
    points_bord = []
    for i in range(101):
        points_bord.append(coords.add_node([0.0, 0.0, HEIGHT * i / 100.0]))
    bord = pc.mesh.poi1_from_nodes(points_bord)
    vm_bord = pc.node_field.project(von_mises_mpa, volume, bord)
    courbe_bord = vm_bord.along(bord)
    courbe_bord.plot(
        x_label="z (m)",
        y_label="von Mises (MPa)",
        title="von Mises le long du bord x = 0, y = 0",
        save=os.path.join(OUT, "mecanique-elastique-bord.svg") if OUT else None,
    )
    # ANCHOR_END: le_long_du_bord

    # ANCHOR: le_long_de_la_sinusoide
    # FR — Une ligne quelconque : une sinusoïde dans le plan médian y = e/2,
    #      de x = 0,01 L à 0,9 L, en 90 points. Elle traverse des éléments :
    #      la courbe saute à chaque frontière.
    # EN — Any line: a sinusoid in the mid-plane y = e/2, from x = 0.01 L to
    #      0.9 L, in 90 points. It crosses elements: the curve jumps at each
    #      boundary.
    points_sinusoide = []
    for i in range(1, 91):
        x = 0.01 * i * LENGTH
        z = HEIGHT / 2.0 + 0.4 * HEIGHT * math.sin(math.radians(3000.0 * x))
        points_sinusoide.append(coords.add_node([x, THICKNESS / 2.0, z]))
    sinusoide = pc.mesh.poi1_from_nodes(points_sinusoide)
    sinusoide.unit().face_color = ROUGE
    show(peau | sinusoide, "La sinusoïde", "mecanique-elastique-sinusoide.svg", True)
    vm_sinusoide = pc.node_field.project(von_mises_mpa, volume, sinusoide)
    courbe_sinusoide = vm_sinusoide.along(sinusoide)
    courbe_sinusoide.plot(
        x_label="abscisse curviligne (m)",
        y_label="von Mises (MPa)",
        title="von Mises le long de la sinusoïde",
        save=os.path.join(OUT, "mecanique-elastique-courbe.svg") if OUT else None,
    )
    # ANCHOR_END: le_long_de_la_sinusoide

    print(
        f"volume : {volume.cell_count} HEX20, encastrement : {encastrement.cell_count} nœuds"
    )
    print(f"moitié basse de l'alésage : {demi_alesage.cell_count} QUA8")
    print(f"point de la masse : {point_masse.position()}")
    print(f"poids : {POIDS:.1f} N, 2·poids/π : {2.0 * POIDS / math.pi:.1f} N")
    for nom, u, (r_x, r_y, r_z) in zip(
        ("force ponctuelle", "pression", "force surfacique"),
        (u_ponctuelle, u_pression, u_surfacique),
        reactions,
    ):
        print(
            f"{nom} : réactions {r_x:.1f} {r_y:.1f} {r_z:.1f} N ; "
            f"u_z le plus bas {u.min('u_z') * 1000.0:.3f} mm"
        )
    print(
        f"force surfacique, aux points de Gauss : von Mises de "
        f"{von_mises_mpa.min('von_mises'):.2f} à {von_mises_mpa.max('von_mises'):.2f} MPa"
    )
    print(
        f"le long du bord : von Mises de {vm_bord.min('von_mises'):.2f} "
        f"à {vm_bord.max('von_mises'):.2f} MPa"
    )
    print(
        f"le long de la sinusoïde : von Mises de {vm_sinusoide.min('von_mises'):.2f} "
        f"à {vm_sinusoide.max('von_mises'):.2f} MPa"
    )


if __name__ == "__main__":
    main()
# ANCHOR_END: script
