"""Formation débutant — 6. Compléments : éléments structuraux + export.

Barre en traction (élément `SEG2`, `model.truss`) — l'équivalent pyrucast
des éléments `BARR`/`POUT` de Cast3M évoqués dans les compléments de la
formation. Le résultat est comparé à la solution analytique puis exporté
au format VTK (lisible par ParaView), l'équivalent du `SORT 'VTK'` de
Cast3M.

Lancement ::

    maturin develop --release
    python formation/complements.py
"""

import os
import tempfile

import pyrucast as pc

E, A, L, F = 210.0e9, 1.0e-4, 2.0, 1000.0


def encastrement(node, var, dual):
    imposed = pc.mesh.poi1_from_nodes([node])
    multiplier = pc.mesh.barycenter(imposed)
    return pc.model.dirichlet(var, dual, imposed, multiplier)


def main() -> None:
    # ANCHOR: barre
    coords = pc.Coords(2)
    n0 = coords.add_node([0.0, 0.0])
    n1 = coords.add_node([L, 0.0])
    mesh = pc.mesh.line(n0, n1, 1)
    fes = pc.FiniteElementSpace(mesh)

    modele = pc.model.truss(fes)
    modele = modele | encastrement(n0, "u_x", "f_x")
    modele = modele | encastrement(n0, "u_y", "f_y")
    modele = modele | encastrement(n1, "u_y", "f_y")  # pas de raideur transversale

    materiaux = pc.element_field.material_field(modele, [("E", E), ("A", A)])

    charge = pc.mesh.poi1_from_nodes([n1])
    second_membre = pc.NodeField(charge, ["f_x"])
    second_membre[0].set_value(n1, "f_x", F)

    K = pc.matrix.stiffness(modele, materiaux)
    solution = pc.solver.solve(K, second_membre)
    # ANCHOR_END: barre

    ux = solution.value(n1, "u_x")
    attendu = F * L / (E * A)
    print(f"u_x (bout) = {ux:.6e}   (analytique F·L/(E·A) = {attendu:.6e})")
    assert abs(ux - attendu) < 1e-10 * attendu

    # ANCHOR: export
    u_propre = pc.node_field.restrict_like(solution, pc.NodeField(mesh, ["u_x", "u_y"]))
    chemin = os.path.join(tempfile.gettempdir(), "barre.vtk")
    pc.export.export_vtk(mesh, chemin, u_propre)
    print(f"Champ de déplacement exporté (VTK, lisible par ParaView) : {chemin}")
    # ANCHOR_END: export


if __name__ == "__main__":
    main()
