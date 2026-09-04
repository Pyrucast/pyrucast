# ANCHOR: script
"""Formation débutant — 2. Calcul thermique. / Beginner training — 2. Thermal.

FR — Reprend la **chape percée** du chapitre 1 — le volume HEX8 structuré est
importé de `formation/maillage.py`, aucune cote n'est redonnée — et y résout la
conduction **stationnaire** `div(-k·grad T) = q`, en deux temps :

1. **conduction seule** — température imposée sur l'alésage, flux imposé sur la
   face gauche ;
2. **conduction + convection + source** — film convectif sous la pièce et
   tranche chauffée, sans rien retoucher au reste.

Chaque étape est tracée avant d'être résolue, et les régions chargées sont
repérées **par leur géométrie** : par forme (`pyrucast.mesh.points_*`) ou par
coordonnée (`pyrucast.node_field.positions` + `pyrucast.mesh.select`). Ni
rayonnement ni terme transitoire. Le détail pas à pas est dans le livre, page
« Calcul thermique ».

EN — Picks chapter 1's **pierced lug** back up — the structured HEX8 volume is
imported from `formation/maillage.py`, not one dimension is restated — and
solves **steady** conduction `div(-k·grad T) = q` on it, in two steps:

1. **conduction alone** — imposed temperature on the bore, imposed flux on the
   left face;
2. **conduction + convection + source** — convective film under the part and
   heated slice, nothing else changes.

Each step is plotted before being solved, and the loaded regions are located
**by geometry**: by shape (`pyrucast.mesh.points_*`) or by coordinate
(`pyrucast.node_field.positions` + `pyrucast.mesh.select`). No radiation, no
transient term. The step-by-step walkthrough lives in the book's thermal page.

Lancement / Running ::

    maturin develop --release
    python formation/thermique.py

    # Figures du livre / book figures (book/src/formation/img/) :
    # PYRUCAST_FORMATION_IMG_DIR=book/src/formation/img python formation/thermique.py
"""

import os

import pyrucast as pc
from maillage import HEIGHT, HOLE_RADIUS, LENGTH, OUT, THICKNESS, show, structured_mesh

# FR — Vue commune aux figures du chapitre 1 : azimut, élévation, échelle.
# EN — The view shared by chapter 1's figures: azimuth, elevation, scale.
VUE = (-45, 25, 1.0)

# FR — Une couleur par région chargée, tenue d'une figure à l'autre.
# EN — One colour per loaded region, kept from one figure to the next.
BLEU = (0, 0, 255)  # alésage / bore
ROUGE = (255, 0, 0)  # face gauche / left face
TURQUOISE = (0, 190, 190)  # face convectée / convected face
VERT = (0, 170, 0)  # zone chauffée / heated zone

# ── Données physiques / Physical data ──────────────────────────────────────
# ANCHOR: donnees
K_COND = 50.0  # W/m/K
FLUX_IMPOSE = -40_000.0  # W/m², face gauche / left face
H_CONV, T_EXT = 240.0, -80.0  # W/m²/K, °C — convection, face z = 0
SOURCE_VOLUMIQUE = 2600e3  # W/m³, zone chauffée / heated zone (≈ 260 W)
T_IMPOSEE = 250.0  # °C, alésage / bore

# FR — La zone chauffée est une tranche de la pièce, entre deux abscisses.
# EN — The heated zone is a slice of the part, between two abscissae.
SOURCE_X_MIN, SOURCE_X_MAX = 0.33 * LENGTH, 0.51 * LENGTH

# FR — Une face plane vaut zéro à l'arrondi près : on sélectionne une bande.
# EN — A flat face is zero up to rounding: a band is selected, not a value.
TOL = 1e-9  # m
# ANCHOR_END: donnees


def show_nodefield(mesh: pc.Mesh, field: pc.NodeField, title: str, file: str) -> None:
    """FR — Trace `field` sur `mesh` : fenêtre interactive, ou SVG si `OUT`.

    EN — Plot `field` over `mesh`: interactive window, or SVG when `OUT` is set.
    """
    mesh.plot(
        view=VUE,
        title=title,
        field=field,
        component="T",
        cmap="viridis",
        smooth=1,
        save=os.path.join(OUT, file) if OUT else None,
    )


def main() -> None:
    # ANCHOR: maillage
    # FR — Le maillage du chapitre 1, tel quel ; `plot=False` : pas ses figures.
    # EN — Chapter 1's mesh, as is; `plot=False`: without its figures.
    _, volume = structured_mesh(plot=False)

    # FR — Les charges réparties s'intègrent sur des faces : il faut la peau.
    # EN — Distributed loads integrate over faces: the skin is needed.
    peau = pc.mesh.consolidate(pc.mesh.skin(volume))
    # ANCHOR_END: maillage

    # ── Étape 1 : régions / Step 1: regions ─────────────────────────────────
    # ANCHOR: alesage
    # FR — L'axe du trou : la normale du plan de la pièce (Y), par le centre.
    # EN — The hole's axis: the part plane's normal (Y), through the centre.
    bas_axe = [LENGTH, -THICKNESS, HEIGHT / 2.0]
    haut_axe = [LENGTH, 2.0 * THICKNESS, HEIGHT / 2.0]

    # FR — L'alésage : les nœuds sur le cylindre, lus à même le volume.
    # EN — The bore: the nodes on the cylinder, read straight off the volume.
    alesage = pc.mesh.consolidate(
        pc.mesh.points_on_cylinder(volume, bas_axe, haut_axe, HOLE_RADIUS)
    )
    # ANCHOR_END: alesage

    # ANCHOR: face_gauche
    # FR — La face gauche : les nœuds du plan x = 0, puis les QUA4 portés.
    # EN — The left face: the nodes of the plane x = 0, then the QUA4 they carry.
    noeuds_gauche = pc.mesh.points_on_plane(peau, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    face_gauche = pc.mesh.elements_on(peau, noeuds_gauche, strict=True)
    # ANCHOR_END: face_gauche

    # ANCHOR: figure_conduction
    # FR — Une couleur par région, la peau en fil de fer autour.
    # EN — One colour per region, the skin drawn as a wireframe around them.
    alesage.unit().face_color = BLEU
    face_gauche.unit().face_color = ROUGE
    show(
        peau | alesage | face_gauche,
        "Étape 1 — conditions aux limites",
        "thermique-cl-conduction.svg",
        wireframe=True,
    )
    # ANCHOR_END: figure_conduction

    # ── Étape 1 : calcul / Step 1: analysis ─────────────────────────────────
    # ANCHOR: modele_conduction
    # FR — Le modèle porte « T » (primal) et « q » (dual) sur tout le volume.
    # EN — The model carries "T" (primal) and "q" (dual) over the whole volume.
    fes = pc.FiniteElementSpace(volume)
    modele = pc.model.heat_conduction(fes)

    # FR — Dirichlet : le support bloqué, et un jumeau pour les multiplicateurs.
    # EN — Dirichlet: the constrained support, and a twin for the multipliers.
    multiplicateur_T = pc.mesh.translate(alesage, [0.0, 0.0, 0.0])
    modele = modele | pc.model.dirichlet(modele, "T", alesage, multiplicateur_T)

    # FR — Le flux imposé sur la face gauche est un terme du modèle.
    # EN — The imposed flux on the left face is a term of the model.
    gauche_fes = pc.FiniteElementSpace(face_gauche)
    modele = modele | pc.model.flux(gauche_fes, "q", "thermal")

    # FR — La conduction réclame « k », la charge sa densité.
    # EN — Conduction asks for "k", the load for its density.
    materiaux = pc.element_field.material_field(
        modele, [("k", K_COND), ("phi_q", FLUX_IMPOSE)]
    )
    # ANCHOR_END: modele_conduction

    # ANCHOR: charges_conduction
    flux_gauche = pc.node_field.external_forces(modele, materiaux)

    # FR — Température imposée, posée sur le maillage des multiplicateurs.
    # EN — Imposed temperature, set on the multipliers' mesh.
    temperature_imposee = pc.NodeField(multiplicateur_T, ["imposed_T"])
    temperature_imposee[0].add_to_component("imposed_T", T_IMPOSEE)
    # ANCHOR_END: charges_conduction

    # ANCHOR: resolution_conduction
    # FR — `[K]{T} = {P}` : matrice assemblée, second membre réuni par `|`.
    # EN — `[K]{T} = {P}`: assembled matrix, right-hand side gathered by `|`.
    K = pc.matrix.stiffness(modele, materiaux)
    t_conduction = pc.solver.solve(K, flux_gauche | temperature_imposee)
    show_nodefield(
        volume, t_conduction, "Étape 1 — température (°C)", "thermique-conduction.svg"
    )
    # ANCHOR_END: resolution_conduction

    # ── Étape 2 : régions / Step 2: regions ─────────────────────────────────
    # ANCHOR: face_basse
    # FR — La face convectée, z = 0 : repérée par coordonnée, pas par forme.
    # EN — The convected face, z = 0: located by coordinate, not by shape.
    z_peau = pc.node_field.positions(peau, ["Z"])
    noeuds_bas = pc.mesh.select(z_peau, ge=-TOL, le=TOL)
    face_basse = pc.mesh.elements_on(peau, noeuds_bas, strict=True)
    face_basse.unit().face_color = TURQUOISE
    show(
        peau | face_basse,
        "Étape 2 — surface convectée",
        "thermique-cl-convection.svg",
        wireframe=True,
    )
    # ANCHOR_END: face_basse

    # ANCHOR: zone_source
    # FR — La zone chauffée : même démarche sur X, en bande, et sur le volume.
    # EN — The heated zone: same approach on X, as a band, over the volume.
    x_volume = pc.node_field.positions(volume, ["X"])
    noeuds_source = pc.mesh.select(x_volume, ge=SOURCE_X_MIN, le=SOURCE_X_MAX)
    zone_source = pc.mesh.consolidate(
        pc.mesh.elements_on(volume, noeuds_source, strict=True)
    )
    zone_source.unit().face_color = VERT
    show(
        peau | zone_source,
        "Étape 2 — zone chauffée",
        "thermique-cl-source.svg",
        wireframe=True,
    )
    # ANCHOR_END: zone_source

    # ── Étape 2 : calcul / Step 2: analysis ─────────────────────────────────
    # ANCHOR: modele_complet
    # FR — La convection s'ajoute dans la matrice : `|` sur les mêmes ddl.
    # EN — Convection adds into the matrix: `|` on the very same dofs.
    basse_fes = pc.FiniteElementSpace(face_basse)
    modele = pc.model.heat_conduction(fes) | pc.model.boundary_transfer(
        basse_fes, [("T", "q")], "thermal"
    )
    modele = modele | pc.model.dirichlet(modele, "T", alesage, multiplicateur_T)

    # FR — Un seul champ matériau : « k » pour la conduction, « h » et son
    #      ambiant pour le film.
    # EN — A single material field: "k" for conduction, "h" and its ambient for
    #      the film.
    materiaux = pc.element_field.material_field(
        modele, [("k", K_COND), ("h_T", H_CONV), ("a_ext_T", T_EXT)]
    )
    # ANCHOR_END: modele_complet

    # ANCHOR: charges_complet
    # FR — Terme externe de la convection, h·T_ext : le modèle le porte.
    # EN — The convection's external term, h·T_ext: the model carries it.
    charge_convection = pc.node_field.external_forces(modele, materiaux)

    # FR — Source volumique sur des HEX8, donc une densité volumique. Une
    #      charge ne contribuant à aucune matrice, elle se tient très bien en
    #      modèle à elle seule — avec sa propre densité, distincte de celle du
    #      flux de bord bien qu'elles alimentent la même ligne « q ».
    # EN — A volume source over HEX8 cells. A load contributes to no matrix, so
    #      it stands perfectly well as a model of its own — with its own
    #      density, distinct from the boundary flux's though both feed "q".
    source_fes = pc.FiniteElementSpace(zone_source)
    source = pc.model.flux(source_fes, "q", "thermal")
    densite_source = pc.element_field.material_field(
        source, [("phi_q", SOURCE_VOLUMIQUE)]
    )
    charge_source = pc.node_field.external_forces(source, densite_source)
    # ANCHOR_END: charges_complet

    # ANCHOR: second_membre
    # FR — Les trois charges se touchent : support commun, puis `+` somme.
    # EN — The three loads touch: a common support first, then `+` really sums.
    noeuds_charges = pc.mesh.consolidate(
        pc.mesh.to_poi1(face_gauche | face_basse | zone_source)
    )
    second_membre = (
        pc.node_field.restrict(flux_gauche, noeuds_charges)
        + pc.node_field.restrict(charge_convection, noeuds_charges)
        + pc.node_field.restrict(charge_source, noeuds_charges)
    ) | temperature_imposee
    # ANCHOR_END: second_membre

    # ANCHOR: resolution_complet
    # FR — Même schéma qu'à l'étape 1, sur la matrice enrichie du terme convectif.
    # EN — Same pattern as step 1, on the matrix enriched with the film term.
    K = pc.matrix.stiffness(modele, materiaux)
    t_complet = pc.solver.solve(K, second_membre)
    show_nodefield(
        volume, t_complet, "Étape 2 — température (°C)", "thermique-complet.svg"
    )
    # ANCHOR_END: resolution_complet

    print(f"volume         : {volume.cell_count()} HEX8")
    print(f"face gauche    : {face_gauche.cell_count()} QUA4")
    print(f"face convectée : {face_basse.cell_count()} QUA4")
    print(f"zone chauffée  : {zone_source.cell_count()} HEX8")
    print(
        f"étape 1 : T min = {t_conduction.min('T'):.1f} °C, "
        f"T max = {t_conduction.max('T'):.1f} °C"
    )
    print(
        f"étape 2 : T min = {t_complet.min('T'):.1f} °C, "
        f"T max = {t_complet.max('T'):.1f} °C"
    )


if __name__ == "__main__":
    main()
# ANCHOR_END: script
