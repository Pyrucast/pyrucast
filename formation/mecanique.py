"""Formation débutant — 3. Calcul mécanique (élasticité linéaire).

Reprend la plaque trouée : bord gauche encastré, effort ponctuel (masse
suspendue) réparti sur la moitié basse du trou. Trois cas de charge sont
enchaînés, comme les sections 6/7/8 de la formation Cast3M :

1. **élasticité pure** — effort seul ;
2. **+ dilatation thermique** — on réutilise le champ de température de
   `formation/thermique.py` (`ε_th = α·(T − T_ref)`, opérateur
   `field.thermal_strain`, l'équivalent Cast3M `EPTH`) ;
3. un paragraphe (pas de code testé ici) sur le **matériau hétérogène**
   (Cast3M fait varier `alpha(x)` par une formule sur un champ aux points de
   Gauss) — voir la page du livre pour le détail.

Lancement ::

    maturin develop --release
    python formation/mecanique.py

    # Pour régénérer la figure du livre (book/src/formation/img/) :
    # PYRUCAST_FORMATION_IMG_DIR=book/src/formation/img python formation/mecanique.py
"""

import os
import tempfile

import pyrucast as pc

LONGUEUR, HAUTEUR = 0.30, 0.10  # m
RAYON_TROU = 0.025  # m
CENTRE_TROU = (0.75 * LONGUEUR, HAUTEUR / 2.0)

E, NU, ALPHA = 200e9, 0.3, 1e-5  # acier
MASSE, G = 2500.0, 9.81  # kg, m/s^2 — masse suspendue au trou
T_REF, T_IMPOSEE = 20.0, 250.0  # °C — dilatation thermique
K_COND = 50.0  # W/m/K


# ANCHOR: construction
def construire_plaque_trouee():
    """Plaque rectangulaire trouée, construite bord par bord avec les
    mailleurs dédiés (`line`, `circle`), fusionnés en un seul
    sous-maillage par `pyrucast.mesh.consolidate` avant `triangulate_surface` — comme
    dans `formation/maillage.py`. Renvoie aussi les sous-maillages utiles à
    la mécanique et à la thermique : bord gauche (encastrement), moitié
    basse du trou (chargement) et le trou complet (température imposée,
    réutilisé tel quel pour rester sur les mêmes nœuds que `plaque`)."""
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

    # Boucle extérieure CCW, trou horaire (CW) : orientation attendue par
    # `triangulate_surface` (le trou est inversé, `trou` reste utilisable ci-dessous).
    contour = boucle_ext | pc.mesh.invert(trou)
    plaque = pc.mesh.triangulate_surface(contour, "TRI3", size=0.02)

    # Moitié basse du trou (y < centre) : support de l'effort de la masse
    # suspendue, comme le `PRES 'MASS'` de Cast3M sur une moitié du cercle.
    y = pc.node_field.positions(trou, ["Y"])
    noeuds_bas_trou = pc.mesh.select(y, lt=CENTRE_TROU[1])
    arc_bas = pc.mesh.elements_on(trou, noeuds_bas_trou, strict=True)

    return coords, plaque, gauche, arc_bas, trou


# ANCHOR_END: construction


def resoudre_thermique(plaque, trou):
    """Ré-sout la thermique de `formation/thermique.py` (version simplifiée,
    sans convection ni source, juste T imposée sur le trou) pour réutiliser
    un champ non uniforme dans le second cas de charge ci-dessous.

    Important : on réutilise le `trou` renvoyé par `construire_plaque_trouee`
    — donc les mêmes nœuds que le bord du trou de `plaque` — plutôt que de
    reconstruire un cercle séparé, qui donnerait des nœuds disjoints du
    maillage réel et un Dirichlet sans effet sur la solution."""
    fes = pc.FiniteElementSpace(plaque)
    modele_th = pc.model.heat_conduction(fes)

    trou_poi1 = pc.mesh.to_poi1(trou)
    multiplicateur = pc.mesh.translate(trou_poi1, [0.0, 0.0])
    modele_th = modele_th | pc.model.dirichlet("T", "q", trou_poi1, multiplicateur)

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
    modele = modele | pc.model.dirichlet("u_x", "f_x", encastrement, multiplicateur)
    modele = modele | pc.model.dirichlet("u_y", "f_y", encastrement, multiplicateur)
    materiaux = pc.element_field.material_field(
        modele, [("E", E), ("nu", NU), ("alpha", ALPHA)]
    )

    # Effort de la masse suspendue, réparti sur la moitié basse du trou —
    # analogue de FSUR 'MASS' / PRES 'MASS' (Cast3M section 6).
    pression = -MASSE * G / (2.0 * 3.14159265 * RAYON_TROU)
    effort = pc.node_field.flux(arc_fes, pression, "f_y")

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
    f_th = pc.node_field.internal_forces(sig_th, modele)

    second_membre = f_th + pc.node_field.restrict_like(effort, f_th)
    u2 = pc.solver.solve(K, second_membre)
    print(f"2) + dilatation thermique : u_y(trou) ≈ {u2.min('u_y'):.6e} m")
    # ANCHOR_END: cas2_thermique

    # u2 porte aussi les multiplicateurs de Lagrange du Dirichlet : on ne
    # garde que (u_x, u_y) avant de calculer une déformation.
    u2_propre = pc.node_field.restrict_like(u2, pc.NodeField(plaque, ["u_x", "u_y"]))
    contraintes = pc.element_field.integrate_behavior(
        modele, pc.element_field.deformation(u2_propre, fes) - eps_th, materiaux
    )
    print(f"   σ_xx max ≈ {contraintes.max('sigma_xx'):.3e} Pa")

    out = os.environ.get("PYRUCAST_FORMATION_IMG_DIR", tempfile.gettempdir())
    chemin = os.path.join(out, "mecanique-deplacement.svg")
    plaque.plot(save=chemin, field=u2, component="u_y", cmap="coolwarm", smooth=1)
    print(f"Déplacement u_y écrit dans {chemin}")


if __name__ == "__main__":
    main()
