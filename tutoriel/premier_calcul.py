# ANCHOR: script
"""Tutoriel débutant — Premier calcul. / Beginner tutorial — First analysis.

FR — Le plus petit calcul complet : une plaque carrée d'acier de 10 cm de
côté, chauffée uniformément dans toute sa masse, dont le bord est tenu à
0 °C. On maille, on déclare la physique, on donne les coefficients, on résout
et on trace — les cinq gestes de tout calcul, que les pages suivantes
détaillent un par un.

EN — The smallest complete analysis: a 10 cm square steel plate, heated
uniformly throughout, whose boundary is held at 0 °C. Mesh, declare the
physics, give the coefficients, solve and plot — the five moves of every
analysis, which the following pages go through one by one.

Lancement / Running ::

    pip install pyrucast
    python tutoriel/premier_calcul.py

    # Figure du livre / book figure (book/src/tutoriel/img/) :
    # PYRUCAST_TUTORIEL_IMG_DIR=book/src/tutoriel/img python tutoriel/premier_calcul.py
"""

import os

import pyrucast as pc

# FR — Répertoire des figures : défini → export SVG, absent → fenêtre interactive.
# EN — Figure directory: set → SVG export, unset → interactive window.
OUT = os.environ.get("PYRUCAST_TUTORIEL_IMG_DIR")

# ── Données / Data ─────────────────────────────────────────────────────────
# ANCHOR: donnees
COTE = 0.10  # m, côté de la plaque / plate side
N = 20  # éléments par côté / elements per side
K_COND = 50.0  # W/m/K, acier / steel
SOURCE = 1.0e6  # W/m³, chauffage uniforme / uniform heating
# ANCHOR_END: donnees


def main() -> None:
    # ANCHOR: maillage
    # FR — Une ligne de N segments, balayée sur la hauteur : N × N QUA4.
    # EN — A line of N segments, swept over the height: N × N QUA4.
    coords = pc.Coords(2)
    a = coords.add_node([0.0, 0.0])
    b = coords.add_node([COTE, 0.0])
    plaque = pc.mesh.extrude(pc.mesh.line(a, b, N), [0.0, COTE], N)

    # FR — Les nœuds du bord, qui seront tenus à 0 °C.
    # EN — The boundary nodes, which will be held at 0 °C.
    bord = pc.mesh.to_poi1(pc.mesh.border(plaque))
    # ANCHOR_END: maillage

    # ANCHOR: modele
    # FR — La conduction sur la plaque, le bord bloqué, la source partout.
    # EN — Conduction over the plate, the boundary held, the source everywhere.
    fes = pc.FiniteElementSpace(plaque)
    modele = pc.model.heat_conduction(fes)
    modele = modele | pc.model.dirichlet(bord, modele, "T")
    modele = modele | pc.model.flux(fes, modele, "q")

    # FR — Les coefficients que le modèle réclame : « k » et la densité « phi_q ».
    # EN — The coefficients the model asks for: "k" and the density "phi_q".
    materiaux = pc.element_field.material_field(
        modele, [("k", K_COND), ("phi_q", SOURCE)]
    )
    # ANCHOR_END: modele

    # ANCHOR: resolution
    # FR — Second membre, matrice, résolution. Aucune température imposée à
    #      écrire : une valeur absente vaut 0.
    # EN — Right-hand side, matrix, solution. No imposed temperature to write:
    #      a missing value is 0.
    second_membre = pc.node_field.external_forces(modele, materiaux)
    K = pc.matrix.stiffness(modele, materiaux)
    temperature = pc.solver.solve(K, second_membre)
    print(f"T max = {temperature.max('T'):.2f} °C")
    # ANCHOR_END: resolution

    # ANCHOR: trace
    # FR — Une fenêtre s'ouvre ; le script se termine quand on la ferme.
    # EN — A window opens; the script ends when it is closed.
    plaque.plot(
        field=temperature,
        component="T",
        view=(0, 90, 1.0),  # FR — vue de dessus / EN — top view
        title="Température (°C)",
        save=os.path.join(OUT, "premier-calcul.svg") if OUT else None,
    )
    # ANCHOR_END: trace


# ANCHOR: lancement
if __name__ == "__main__":
    main()
# ANCHOR_END: lancement
# ANCHOR_END: script
