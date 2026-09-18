"""Formation débutant — 3. Calcul mécanique (élasticité linéaire).

Reprend la plaque trouée : bord gauche encastré, effort ponctuel (masse
mass) spread over the lower half of the hole. Three load cases follow one
another, like sections 6/7/8 of the Cast3M training:

1. **élasticité pure** — effort seul ;
2. **+ dilatation thermique** — on réutilise le champ de température de
   `formation/thermique.py` (`ε_th = α·(T − T_ref)`, opérateur
   `field.thermal_strain`, l'équivalent Cast3M `EPTH`) ;
3. a paragraph (no code tested here) on the **heterogeneous material**
   (Cast3M varies `alpha(x)` through a formula on a field at the Gauss
   points) — see the book page for the detail.

Lancement ::

    maturin develop --release
    python formation/mecanique.py

    # To regenerate the book figure (book/src/formation/img/):
    # PYRUCAST_FORMATION_IMG_DIR=book/src/formation/img python formation/mecanique.py
"""

import os
import tempfile

import pyrucast as pc

LONGUEUR, HAUTEUR = 0.30, 0.10  # m
RAYON_TROU = 0.025  # m
CENTRE_TROU = (0.75 * LONGUEUR, HAUTEUR / 2.0)

E, NU, ALPHA = 200e9, 0.3, 1e-5  # acier
MASSE, G = 2500.0, 9.81  # kg, m/s^2 — mass hung from the hole
T_REF, T_IMPOSEE = 20.0, 250.0  # °C — dilatation thermique
K_COND = 50.0  # W/m/K


# ANCHOR: construction
def construire_plaque_trouee():
    """A holed rectangular plate, built edge by edge with the
    mailleurs dédiés (`line`, `circle`), fusionnés en un seul
    submeshes by `pyrucast.mesh.consolidate` before `triangulate_surface` — as
    in `formation/maillage.py`. Also returns the submeshes the mechanics and
    the thermics need: left edge (clamped end), lower half of the hole
    (loading) and the whole hole (imposed temperature, reused as is to stay on
    the same nodes as `plaque`)."""
    coords = pc.Coords(2)
    p1 = coords.add_node([0.0, 0.0])
    p2 = coords.add_node([LONGUEUR, 0.0])
    p3 = coords.add_node([LONGUEUR, HAUTEUR])
    p4 = coords.add_node([0.0, HAUTEUR])

    bas = pc.mesh.line(p1, p2, 10)
    droit = pc.mesh.line(p2, p3, 4)
    haut = pc.mesh.line(p3, p4, 10)
    gauche = pc.mesh.line(p4, p1, 4)
    boucle_ext = pc.mesh.consolidate(bas | droit | haut | gauche)

    centre = coords.add_node(list(CENTRE_TROU))
    trou = pc.mesh.circle(centre, [0.0, 0.0, 1.0], RAYON_TROU, 16)

    # Outer loop CCW, hole clockwise (CW): the orientation
    # `triangulate_surface` expects (the hole is inverted, `trou` stays usable below).
    contour = boucle_ext | pc.mesh.invert(trou)
    plaque = pc.mesh.triangulate_surface(contour, "TRI3", size=0.02)

    # Lower half of the hole (y < centre): support of the hung mass's force,
    # like Cast3M's `PRES 'MASS'` over half the circle.
    y = pc.node_field.positions(trou, ["Y"])
    noeuds_bas_trou = pc.mesh.select(y, lt=CENTRE_TROU[1])
    arc_bas = pc.mesh.elements_on(trou, noeuds_bas_trou, strict=True)

    return coords, plaque, gauche, arc_bas, trou


# ANCHOR_END: construction


def resoudre_thermique(plaque, trou):
    """Ré-sout la thermique de `formation/thermique.py` (version simplifiée,
    without convection or source, just T imposed on the hole) so as to reuse
    a non-uniform field in the second load case below.

    Important: we reuse the `trou` returned by `construire_plaque_trouee` —
    hence the same nodes as the hole's edge in `plaque` — rather than
    rebuilding a separate circle, which would give nodes disjoint from the
    real mesh and a Dirichlet with no effect on the solution."""
    fes = pc.FiniteElementSpace(plaque)
    modele_th = pc.model.heat_conduction(fes)

    trou_poi1 = pc.mesh.to_poi1(trou)
    multiplicateur = pc.mesh.translate(trou_poi1, [0.0, 0.0])
    modele_th = modele_th | pc.model.dirichlet(
        modele_th, "T", trou_poi1, multiplicateur
    )

    materiaux_th = pc.element_field.material_field(modele_th, [("k", K_COND)])
    temperature_imposee = pc.NodeField(multiplicateur, ["imposed_T"])
    temperature_imposee[0].add_to_component("imposed_T", T_IMPOSEE)

    K_th = pc.matrix.stiffness(modele_th, materiaux_th)
    return pc.solver.solve(K_th, temperature_imposee)


def main() -> None:
    _coords, plaque, gauche, arc_bas, trou = construire_plaque_trouee()
    fes = pc.FiniteElementSpace(plaque)
    arc_fes = pc.FiniteElementSpace(arc_bas)

    # ANCHOR: modele_elastique
    encastrement = pc.mesh.to_poi1(gauche)
    multiplicateur = pc.mesh.translate(encastrement, [0.0, 0.0])

    modele = pc.model.elasticity(fes, "plane_stress")
    modele = modele | pc.model.dirichlet(modele, "u_x", encastrement, multiplicateur)
    modele = modele | pc.model.dirichlet(modele, "u_y", encastrement, multiplicateur)

    # The hung mass's force, spread over the lower half of the hole —
    # analogue de FSUR 'MASS' / PRES 'MASS' (Cast3M section 6).
    pression = -MASSE * G / (2.0 * 3.14159265 * RAYON_TROU)
    modele = modele | pc.model.flux(arc_fes, modele, "f_y")
    materiaux = pc.element_field.material_field(
        modele, [("E", E), ("nu", NU), ("alpha", ALPHA), ("phi_f_y", pression)]
    )
    effort = pc.node_field.external_forces(modele, materiaux)

    K = pc.matrix.stiffness(modele, materiaux)
    # ANCHOR_END: modele_elastique

    # ANCHOR: cas1_elastique
    u1 = pc.solver.solve(K, effort)
    print(f"1) élasticité seule       : u_y(trou) ≈ {u1.min('u_y'):.6e} m")
    # ANCHOR_END: cas1_elastique

    # ANCHOR: cas2_thermique
    temperature = resoudre_thermique(plaque, trou)
    t_gauss = pc.element_field.interp_to_gauss(
        pc.node_field.restrict(temperature, plaque), fes
    )
    eps_th = pc.element_field.thermal_strain(t_gauss, materiaux, fes, T_REF)
    sig_th = pc.element_field.integrate_behavior(modele, eps_th, materiaux)
    f_th = (
        pc.node_field.divergence(sig_th, "sigma")
        .rename_component("div_sigma_x", "f_x")
        .rename_component("div_sigma_y", "f_y")
    )

    second_membre = f_th + pc.node_field.restrict_like(effort, f_th)
    u2 = pc.solver.solve(K, second_membre)
    print(f"2) + dilatation thermique : u_y(trou) ≈ {u2.min('u_y'):.6e} m")
    # ANCHOR_END: cas2_thermique

    # u2 also carries the Dirichlet's Lagrange multipliers: only (u_x, u_y)
    # are kept before computing a strain.
    u2_propre = pc.node_field.restrict_like(u2, pc.NodeField(plaque, ["u_x", "u_y"]))
    contraintes = pc.element_field.integrate_behavior(
        modele, pc.element_field.deformation(u2_propre, fes) - eps_th, materiaux
    )
    print(f"   σ_xx max ≈ {contraintes.max('sigma_xx'):.3e} Pa")

    out = os.environ.get("PYRUCAST_FORMATION_IMG_DIR", tempfile.gettempdir())
    chemin = os.path.join(out, "mecanique-deplacement.svg")
    plaque.plot(save=chemin, field=u2, component="u_y", cmap="coolwarm", smooth=1)
    print(f"Displacement u_y written to {chemin}")


if __name__ == "__main__":
    main()
