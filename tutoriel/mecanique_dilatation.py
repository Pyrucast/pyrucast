# ANCHOR: script
"""Tutoriel débutant — 9.7.2 Dilatation, matériau homogène. / Beginner
tutorial — 9.7.2 Thermal expansion, homogeneous material.

FR — Reprend la chape encastrée de la page « Élasticité », chargée par la
force surfacique verticale, et y ajoute la **dilatation thermique**. La
température est celle du dernier instant du calcul avec rayonnement, relue
dans le fichier qu'écrit `tutoriel/thermique_non_lineaire.py` : ce script-là
doit avoir tourné avant. La pièce se dilate de `α (T − T_ref)`, l'encastrement
l'en empêche en partie, et des contraintes naissent. Le détail pas à pas est
dans le livre, page « Dilatation, matériau homogène ».

EN — Takes the clamped lug of the "Élasticité" page back, loaded by the
vertical surface force, and adds **thermal expansion**. The temperature is
the last instant of the analysis with radiation, read back from the file
`tutoriel/thermique_non_lineaire.py` writes: that script must have run
first. The part expands by `α (T − T_ref)`, the clamp partly prevents it,
and stresses arise. The step-by-step walkthrough lives in the book's
"Dilatation, matériau homogène" page.

Lancement / Running ::

    pip install pyrucast
    python tutoriel/thermique_non_lineaire.py   # écrit / writes thermique.pyr
    python tutoriel/mecanique_dilatation.py

    # Figures du livre / book figures (book/src/tutoriel/img/) :
    # PYRUCAST_TUTORIEL_IMG_DIR=book/src/tutoriel/img python tutoriel/mecanique_dilatation.py
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
E, NU = 200e9, 0.3  # Pa, — : un acier / a steel
ALPHA = 1e-5  # 1/K : coefficient de dilatation / expansion coefficient
T_REF = 0.0  # °C : la pièce n'est pas dilatée à 0 °C / not expanded at 0 °C
MASSE, G = 2500.0, 9.81  # kg, m/s²
POIDS = MASSE * G  # N
PRESSION = POIDS / (math.pi * HOLE_RADIUS * THICKNESS)  # Pa
AMPLIFICATION = 50.0

# FR — Le fichier écrit par tutoriel/thermique_non_lineaire.py.
# EN — The file tutoriel/thermique_non_lineaire.py writes.
FICHIER = os.path.join(tempfile.gettempdir(), "thermique.pyr")

TOL = 1e-9  # m
# ANCHOR_END: donnees


def main() -> None:
    # ANCHOR: relecture
    # FR — Le volume et l'évolution des températures, relus sous les clefs
    #      où ils ont été écrits ; la température du dernier instant, en °C.
    # EN — The volume and the temperature evolution, read back under the
    #      keys they were written under; the last instant's temperature, in °C.
    if not os.path.exists(FICHIER):
        raise SystemExit(
            f"{FICHIER} manque : lancer d'abord tutoriel/thermique_non_lineaire.py"
        )
    relu = pc.load(FICHIER)
    volume = relu["volume"]
    evolution = relu["température (°C)"]
    temperature = evolution.frames()[-1]
    peau = pc.mesh.consolidate(pc.mesh.skin(volume))
    volume.plot(
        view=VUE,
        title="Température relue, dernier instant (°C)",
        field=temperature,
        component="T",
        cmap="viridis",
        save=os.path.join(OUT, "mecanique-dilatation-temperature.svg") if OUT else None,
    )
    # ANCHOR_END: relecture

    # ANCHOR: regions
    # FR — Les régions de la page « Élasticité », retrouvées sur le volume
    #      relu : l'encastrement et la moitié basse de l'alésage.
    # EN — The "Élasticité" page's regions, found again on the volume read
    #      back: the clamp and the lower half of the bore.
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
    # FR — Le modèle de la page « Élasticité » ; le matériau porte en plus
    #      « alpha », que la dilatation lira.
    # EN — The "Élasticité" page's model; the material also carries "alpha",
    #      which the expansion will read.
    fes = pc.FiniteElementSpace(volume)
    elasticite = pc.model.elasticity(fes, "full_3d")
    blocage_x = pc.model.dirichlet(encastrement, elasticite, "u_x")
    blocage_y = pc.model.dirichlet(encastrement, elasticite, "u_y")
    blocage_z = pc.model.dirichlet(encastrement, elasticite, "u_z")
    modele = elasticite | blocage_x | blocage_y | blocage_z
    materiaux = pc.element_field.material_field(
        modele, [("E", E), ("nu", NU), ("alpha", ALPHA)]
    )
    K = pc.matrix.stiffness(modele, materiaux)

    demi_fes = pc.FiniteElementSpace(demi_alesage)
    verticale = pc.model.flux(demi_fes, elasticite, "f_z")
    densite_verticale = pc.element_field.material_field(
        verticale, [("phi_f_z", -PRESSION)]
    )
    force_surfacique = pc.node_field.external_forces(verticale, densite_verticale)
    # ANCHOR_END: modele

    # ANCHOR: dilatation
    # FR — La température aux points de Gauss, puis la dilatation
    #      ε_th = α (T − T_ref), la contrainte qu'elle donnerait si rien ne la
    #      gênait, [C] ε_th, et ses forces nodales ∫ [B]ᵀ [C] ε_th dV.
    # EN — The temperature at the Gauss points, then the expansion
    #      ε_th = α (T − T_ref), the stress it would give were nothing in its
    #      way, [C] ε_th, and its nodal forces ∫ [B]ᵀ [C] ε_th dV.
    t_gauss = pc.element_field.interp_to_gauss(temperature, fes)
    eps_th = pc.element_field.thermal_strain(t_gauss, materiaux, fes, T_REF)
    sigma_th = pc.element_field.elastic_stress(modele, eps_th, materiaux)
    force_th = (
        pc.node_field.divergence(sigma_th, "sigma")
        .rename_component("div_sigma_x", "f_x")
        .rename_component("div_sigma_y", "f_y")
        .rename_component("div_sigma_z", "f_z")
    )
    # ANCHOR_END: dilatation

    # ANCHOR: second_membre
    # FR — Les deux chargements partagent les nœuds de l'alésage : la force
    #      surfacique est ramenée sur le support de la dilatation, puis `+`
    #      somme nœud à nœud.
    # EN — Both loads share the bore's nodes: the surface force is brought
    #      onto the expansion's support, then `+` sums node by node.
    second_membre = force_th + pc.node_field.restrict_like(force_surfacique, force_th)
    # ANCHOR_END: second_membre

    # ANCHOR: resolution
    u = pc.solver.solve(K, second_membre)
    r_x = -pc.node_field.restrict(u, blocage_x.multiplier_mesh()).sum("lambda")
    r_y = -pc.node_field.restrict(u, blocage_y.multiplier_mesh()).sum("lambda")
    r_z = -pc.node_field.restrict(u, blocage_z.multiplier_mesh()).sum("lambda")
    # ANCHOR_END: resolution

    # ANCHOR: deformee
    coords = volume.coords()
    deformee = coords.add_config("déformée ×50")
    pc.coords.displace_from(u * AMPLIFICATION, 0, deformee)
    with coords.in_config(deformee):
        volume.plot(
            view=VUE,
            title="Dilatation et force surfacique — déformée ×50, u_x (mm)",
            field=u * 1000.0,
            component="u_x",
            cmap="viridis",
            save=os.path.join(OUT, "mecanique-dilatation-deformee.svg")
            if OUT
            else None,
        )
    # ANCHOR_END: deformee

    # ANCHOR: contraintes
    # FR — Seule la part de la déformation que la dilatation n'explique pas
    #      crée une contrainte : σ = [C] (ε − ε_th).
    # EN — Only the part of the strain the expansion does not account for
    #      creates a stress: σ = [C] (ε − ε_th).
    deformation = pc.element_field.strain(modele, u, materiaux)
    contrainte = pc.element_field.elastic_stress(
        modele, deformation - eps_th, materiaux
    )
    von_mises_mpa = pc.element_field.von_mises(modele, contrainte) * 1e-6
    volume.plot(
        view=VUE,
        title="Dilatation et force surfacique — von Mises (MPa)",
        field=von_mises_mpa,
        component="von_mises",
        cmap="viridis",
        save=os.path.join(OUT, "mecanique-dilatation-von-mises.svg") if OUT else None,
    )
    # ANCHOR_END: contraintes

    print(f"relu : {volume.cell_count} HEX20, {len(evolution.frames())} champs")
    print(
        f"température relue : de {temperature.min('T'):.2f} à "
        f"{temperature.max('T'):.2f} °C"
    )
    print(
        f"forces de la dilatation, sommées : {force_th.sum('f_x'):.2e} "
        f"{force_th.sum('f_y'):.2e} {force_th.sum('f_z'):.2e} N"
    )
    print(f"réactions : {r_x:.1f} {r_y:.1f} {r_z:.1f} N")
    print(
        f"u_x de {u.min('u_x') * 1000.0:.3f} à {u.max('u_x') * 1000.0:.3f} mm, "
        f"u_z de {u.min('u_z') * 1000.0:.3f} à {u.max('u_z') * 1000.0:.3f} mm"
    )
    print(
        f"aux points de Gauss : von Mises de {von_mises_mpa.min('von_mises'):.2f} "
        f"à {von_mises_mpa.max('von_mises'):.2f} MPa"
    )


if __name__ == "__main__":
    main()
# ANCHOR_END: script
