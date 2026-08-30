"""Formation débutant — 5. Contact (unilatéral, nœud-surface).

Patch-test classique : deux blocs élastiques empilés selon `y`, séparés par
un jeu initial `G0`. Une pression sur le bloc du haut ferme le contact et
transmet une contrainte uniforme à travers l'interface — l'équivalent
pyrucast du contact nœud-surface de Cast3M (section 10 de la formation),
piloté ici directement par le solveur actif-set `solve_unilateral` plutôt
que par `step_by_step` (qui ne sait pas encore composer thermique,
plasticité et contact dans une même table).

Lancement ::

    maturin develop --release
    python formation/contact.py

    # Pour régénérer la figure du livre (book/src/formation/img/) :
    # PYRUCAST_FORMATION_IMG_DIR=book/src/formation/img python formation/contact.py
"""

import os
import tempfile

import pyrucast as pc

E = 100.0
S = 5.0  # pression appliquée
G0 = 0.01  # jeu initial entre les deux blocs
N = 2  # grille N×N de QUA4 par bloc


def idx(i, j):
    return j * (N + 1) + i


def bloc(coords: pc.Coords, y0: float):
    """Bloc `[0,1] × [y0, y0+1]`, grille N×N de QUA4 — mailleurs dédiés
    (`line` pour les bords bas/haut, `sweep` entre les deux, comme
    dans `formation/maillage.py`). Renvoie `(mesh, grille)`, `grille[idx(i,j)]`
    étant le nœud `(i,j)` (`i` : abscisse, `j` : ordonnée)."""
    bas = pc.mesh.line(coords.add_node([0.0, y0]), coords.add_node([1.0, y0]), N)
    haut = pc.mesh.line(
        coords.add_node([0.0, y0 + 1.0]), coords.add_node([1.0, y0 + 1.0]), N
    )
    mesh = pc.mesh.sweep(bas, haut, N)

    grille = [None] * ((N + 1) * (N + 1))
    for cy in range(N):
        for cx in range(N):
            cell = cy * N + cx
            grille[idx(cx, cy)] = mesh.node(0, cell, 0)
            grille[idx(cx + 1, cy)] = mesh.node(0, cell, 1)
            grille[idx(cx + 1, cy + 1)] = mesh.node(0, cell, 2)
            grille[idx(cx, cy + 1)] = mesh.node(0, cell, 3)
    return mesh, grille


def clamp(nodes, var, dual):
    imposed = pc.mesh.poi1_from_nodes(nodes)
    multiplier = pc.mesh.barycenter(imposed)
    return pc.model.dirichlet(var, dual, imposed, multiplier)


def bord_horizontal(mesh: pc.Mesh, y: float) -> pc.Mesh:
    """Extrait, parmi les segments de bord de `mesh` (`pyrucast.mesh.border`,
    l'équivalent Cast3M `CONTOUR`), ceux d'ordonnée `y` — un bord existant du
    maillage, pas une ligne recréée à côté (`line` fabriquerait de
    nouveaux nœuds, disjoints de `mesh`)."""
    frontiere = pc.mesh.border(mesh)
    ordonnee = pc.node_field.positions(frontiere, ["Y"])
    noeuds = pc.mesh.select(ordonnee, ge=y - 1e-9, le=y + 1e-9)
    return pc.mesh.elements_on(frontiere, noeuds, strict=True)


def main() -> None:
    # ANCHOR: geometrie_contact
    coords = pc.Coords(2)
    mesh_bas, bas = bloc(coords, 0.0)
    mesh_haut, haut = bloc(coords, 1.0 + G0)
    mesh = mesh_bas | mesh_haut
    fes = pc.FiniteElementSpace(mesh)

    # Maître : bord haut du bloc bas (`contour` oriente déjà la frontière en
    # sens trigonométrique, donc ce bord court naturellement de droite à
    # gauche — la normale associée pointe vers +y). Esclave : nœuds du bord
    # bas du bloc haut.
    maitre = bord_horizontal(mesh_bas, 1.0)
    esclave = pc.mesh.poi1_from_nodes([haut[idx(i, 0)] for i in range(N + 1)])

    contact = pc.model.contact(esclave, maitre, [("u_x", "f_x"), ("u_y", "f_y")])
    # ANCHOR_END: geometrie_contact

    # ANCHOR: modele_contact
    modele = pc.model.elasticity(fes, "plane_stress")
    modele = modele | clamp(bas + haut, "u_x", "f_x")
    modele = modele | clamp([bas[idx(i, 0)] for i in range(N + 1)], "u_y", "f_y")
    modele = modele | contact

    materiaux = pc.element_field.material_field(modele, [("E", E), ("nu", 0.0)])
    # ANCHOR_END: modele_contact

    # ANCHOR: chargement_contact
    bord_haut = bord_horizontal(mesh_haut, 2.0 + G0)
    bord_haut_fes = pc.FiniteElementSpace(bord_haut)
    traction = pc.node_field.flux(bord_haut_fes, -S, "f_y")

    # `contact_gaps()` fournit le second membre du contact — l'équivalent
    # Cast3M de la préparation du problème unilatéral avant RESO.
    second_membre = traction | modele.contact_gaps()
    # ANCHOR_END: chargement_contact

    # ANCHOR: resolution_contact
    K = pc.matrix.stiffness(modele, materiaux)
    solution = pc.solver.solve_unilateral(K, modele, second_membre)
    # ANCHOR_END: resolution_contact

    print(f"Pression appliquée : {S}")
    for j in range(N + 1):
        uy_bas = solution.value(bas[idx(0, j)], "u_y")
        uy_haut = solution.value(haut[idx(0, j)], "u_y")
        print(f"  y={j / N:.2f} : u_y(bas)={uy_bas:.6e}  u_y(haut)={uy_haut:.6e}")

    # Réactions de contact : Σ(−λᵢ) doit reconstituer l'effort appliqué S.
    maillage_mult = contact.multiplier_mesh()
    lambdas = [
        solution.value(maillage_mult.node(0, r, 0), "lambda_contact")
        for r in range(N + 1)
    ]
    print(f"\nΣ(−λ) = {sum(-lam for lam in lambdas):.6f}  (attendu {S})")

    out = os.environ.get("PYRUCAST_FORMATION_IMG_DIR", tempfile.gettempdir())
    chemin = os.path.join(out, "contact.svg")
    maillage_mult.plot(
        save=chemin, field=solution, component="lambda_contact", cmap="viridis"
    )
    print(f"Réaction de contact (λ) écrite dans {chemin}")


if __name__ == "__main__":
    main()
