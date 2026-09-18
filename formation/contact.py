"""Formation débutant — 5. Contact (unilatéral, nœud-surface).

A classic patch test: two elastic blocks stacked along `y`, separated by an
initial gap `G0`. A pressure on the upper block closes the contact and
transmits a uniform stress across the interface — the pyrucast equivalent of
Cast3M's node-to-surface contact (section 10 of the training), driven here
straight by the active-set solver `solve_unilateral` rather than by
`step_by_step` (which cannot yet compose thermics, plasticity and contact in
a single table).

Lancement ::

    maturin develop --release
    python formation/contact.py

    # To regenerate the book figure (book/src/formation/img/):
    # PYRUCAST_FORMATION_IMG_DIR=book/src/formation/img python formation/contact.py
"""

import os
import tempfile

import pyrucast as pc

E = 100.0
S = 5.0  # pression appliquée
G0 = 0.01  # initial gap between the two blocks
N = 2  # N×N grid of QUA4 per block


def idx(i, j):
    return j * (N + 1) + i


def bloc(coords: pc.Coords, y0: float):
    """Bloc `[0,1] × [y0, y0+1]`, grille N×N de QUA4 — mailleurs dédiés
    (`line` for the bottom/top edges, `sweep` between them, as in
    `formation/maillage.py`). Returns `(mesh, grille)`, `grille[idx(i,j)]`
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


def clamp(target, nodes, var):
    imposed = pc.mesh.poi1_from_nodes(nodes)
    multiplier = pc.mesh.barycenter(imposed)
    return pc.model.dirichlet(target, var, imposed, multiplier)


def bord_horizontal(mesh: pc.Mesh, y: float) -> pc.Mesh:
    """Extracts, among `mesh`'s border segments (`pyrucast.mesh.border`, the
    Cast3M `CONTOUR` equivalent), those at ordinate `y` — an existing edge of
    the mesh, not a line rebuilt beside it (`line` would make
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

    # Master: top edge of the lower block (`contour` already orients the
    # boundary counter-clockwise, so this edge naturally runs right to left —
    # the associated normal points towards +y). Slave: nodes of the upper
    # block's bottom edge.
    maitre = bord_horizontal(mesh_bas, 1.0)
    esclave = pc.mesh.poi1_from_nodes([haut[idx(i, 0)] for i in range(N + 1)])

    elasticite = pc.model.elasticity(fes, "plane_stress")
    contact = pc.model.contact(elasticite, esclave, maitre, ["u_x", "u_y"])
    # ANCHOR_END: geometrie_contact

    # ANCHOR: modele_contact
    modele = pc.model.elasticity(fes, "plane_stress")
    modele = modele | clamp(modele, bas + haut, "u_x")
    modele = modele | clamp(modele, [bas[idx(i, 0)] for i in range(N + 1)], "u_y")
    modele = modele | contact

    # ANCHOR_END: modele_contact

    # ANCHOR: chargement_contact
    bord_haut = bord_horizontal(mesh_haut, 2.0 + G0)
    bord_haut_fes = pc.FiniteElementSpace(bord_haut)
    modele = modele | pc.model.flux(bord_haut_fes, modele, "f_y")
    materiaux = pc.element_field.material_field(
        modele, [("E", E), ("nu", 0.0), ("phi_f_y", -S)]
    )
    traction = pc.node_field.external_forces(modele, materiaux)

    # `contact_gaps()` supplies the contact's right-hand side — the Cast3M
    # equivalent of preparing the unilateral problem before RESO.
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
    print(f"Contact reaction (λ) written to {chemin}")


if __name__ == "__main__":
    main()
