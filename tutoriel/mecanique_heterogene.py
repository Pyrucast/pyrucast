# ANCHOR: script
"""Tutoriel débutant — 9.7.3 Dilatation, matériau hétérogène. / Beginner
tutorial — 9.7.3 Thermal expansion, heterogeneous material.

FR — Le calcul de la page « Dilatation, matériau homogène », avec un
coefficient de dilatation qui **varie le long de la pièce** : quatre fois plus
fort au milieu de la longueur qu'aux bouts, en cloche,

    α(x) = α₀ (1 + 3 exp(−((x − µ) / σ)²)),   µ = L/2,   σ = L/5.

α est calculé aux points de Gauss, là où la dilatation le lit. La
température est relue, comme à la page précédente, dans le fichier qu'écrit
`tutoriel/thermique_non_lineaire.py`. Le détail pas à pas est dans le livre,
page « Dilatation, matériau hétérogène ».

EN — The "Dilatation, matériau homogène" page's analysis, with an expansion
coefficient that **varies along the part**: four times as strong in the middle
of the length as at the ends, bell-shaped,

    α(x) = α₀ (1 + 3 exp(−((x − µ) / σ)²)),   µ = L/2,   σ = L/5.

α is computed at the Gauss points, where the expansion reads it. The
temperature is read back, as on the previous page, from the file
`tutoriel/thermique_non_lineaire.py` writes. The step-by-step walkthrough
lives in the book's "Dilatation, matériau hétérogène" page.

Lancement / Running ::

    pip install pyrucast
    python tutoriel/thermique_non_lineaire.py   # écrit / writes thermique.pyr
    python tutoriel/mecanique_heterogene.py

    # Figures du livre / book figures (book/src/tutoriel/img/) :
    # PYRUCAST_TUTORIEL_IMG_DIR=book/src/tutoriel/img python tutoriel/mecanique_heterogene.py
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
ALPHA_0 = 1e-5  # 1/K : α aux bouts de la pièce / α at the part's ends
MU, SIGMA = LENGTH / 2.0, LENGTH / 5.0  # m : centre et largeur de la cloche
T_REF = 0.0  # °C
MASSE, G = 2500.0, 9.81  # kg, m/s²
POIDS = MASSE * G  # N
PRESSION = POIDS / (math.pi * HOLE_RADIUS * THICKNESS)  # Pa
AMPLIFICATION = 50.0
FICHIER = os.path.join(tempfile.gettempdir(), "thermique.pyr")
TOL = 1e-9  # m
# ANCHOR_END: donnees


def main() -> None:
    # ANCHOR: relecture
    if not os.path.exists(FICHIER):
        raise SystemExit(
            f"{FICHIER} manque : lancer d'abord tutoriel/thermique_non_lineaire.py"
        )
    relu = pc.load(FICHIER)
    volume = relu["volume"]
    temperature = relu["température (°C)"].frames()[-1]
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

    # ANCHOR: alpha
    # FR — x aux points de Gauss : les positions des nœuds, interpolées comme
    #      tout champ aux nœuds. La formule s'écrit ensuite sur le champ comme
    #      sur un nombre ; `pc.field.exp` est l'exponentielle d'un champ.
    #      Le résultat s'appelle encore « X » : on le renomme « alpha ».
    # EN — x at the Gauss points: the nodes' positions, interpolated like any
    #      nodal field. The formula is then written on the field as on a
    #      number; `pc.field.exp` is a field's exponential. The result is
    #      still called "X": it is renamed "alpha".
    fes = pc.FiniteElementSpace(volume)
    x = pc.element_field.interp_to_gauss(pc.node_field.positions(volume, ["X"]), fes)
    cloche = pc.field.exp(-(((x - MU) / SIGMA) ** 2))
    alpha = (ALPHA_0 * (1.0 + 3.0 * cloche)).rename_component("X", "alpha")
    volume.plot(
        view=VUE,
        title="Coefficient de dilatation α (10⁻⁶/K)",
        field=alpha * 1e6,
        component="alpha",
        cmap="viridis",
        save=os.path.join(OUT, "mecanique-heterogene-alpha.svg") if OUT else None,
    )
    # ANCHOR_END: alpha

    # ANCHOR: modele
    # FR — Le champ matériau réunit E et ν, uniformes, et α, champ par
    #      éléments : `|` les juxtapose, chaque composante d'un seul côté.
    # EN — The material field gathers E and ν, uniform, and α, an element
    #      field: `|` lays them side by side, each component on one side only.
    elasticite = pc.model.elasticity(fes, "full_3d")
    blocage_x = pc.model.dirichlet(encastrement, elasticite, "u_x")
    blocage_y = pc.model.dirichlet(encastrement, elasticite, "u_y")
    blocage_z = pc.model.dirichlet(encastrement, elasticite, "u_z")
    modele = elasticite | blocage_x | blocage_y | blocage_z
    materiaux = pc.element_field.material_field(modele, [("E", E), ("nu", NU)]) | alpha
    K = pc.matrix.stiffness(modele, materiaux)

    demi_fes = pc.FiniteElementSpace(demi_alesage)
    verticale = pc.model.flux(demi_fes, elasticite, "f_z")
    densite_verticale = pc.element_field.material_field(
        verticale, [("phi_f_z", -PRESSION)]
    )
    force_surfacique = pc.node_field.external_forces(verticale, densite_verticale)
    # ANCHOR_END: modele

    # ANCHOR: dilatation
    # FR — Les mêmes lignes qu'à la page précédente : seul α a changé.
    # EN — The same lines as on the previous page: only α has changed.
    t_gauss = pc.element_field.interp_to_gauss(temperature, fes)
    eps_th = pc.element_field.thermal_strain(t_gauss, materiaux, fes, T_REF)
    sigma_th = pc.element_field.elastic_stress(modele, eps_th, materiaux)
    force_th = (
        pc.node_field.divergence(sigma_th, "sigma")
        .rename_component("div_sigma_x", "f_x")
        .rename_component("div_sigma_y", "f_y")
        .rename_component("div_sigma_z", "f_z")
    )
    second_membre = force_th + pc.node_field.restrict_like(force_surfacique, force_th)
    u = pc.solver.solve(K, second_membre)
    r_x = -pc.node_field.restrict(u, blocage_x.multiplier_mesh()).sum("lambda")
    r_y = -pc.node_field.restrict(u, blocage_y.multiplier_mesh()).sum("lambda")
    r_z = -pc.node_field.restrict(u, blocage_z.multiplier_mesh()).sum("lambda")
    # ANCHOR_END: dilatation

    # ANCHOR: resultats
    coords = volume.coords()
    deformee = coords.add_config("déformée ×50")
    pc.coords.displace_from(u * AMPLIFICATION, 0, deformee)
    with coords.in_config(deformee):
        volume.plot(
            view=VUE,
            title="α variable — déformée ×50, u_x (mm)",
            field=u * 1000.0,
            component="u_x",
            cmap="viridis",
            save=os.path.join(OUT, "mecanique-heterogene-deformee.svg")
            if OUT
            else None,
        )
    deformation = pc.element_field.strain(modele, u, materiaux)
    contrainte = pc.element_field.elastic_stress(
        modele, deformation - eps_th, materiaux
    )
    von_mises_mpa = pc.element_field.von_mises(modele, contrainte) * 1e-6
    volume.plot(
        view=VUE,
        title="α variable — von Mises (MPa)",
        field=von_mises_mpa,
        component="von_mises",
        cmap="viridis",
        save=os.path.join(OUT, "mecanique-heterogene-von-mises.svg") if OUT else None,
    )
    # ANCHOR_END: resultats

    print(
        f"α aux points de Gauss : de {alpha.min('alpha'):.4e} "
        f"à {alpha.max('alpha'):.4e} 1/K"
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
