"""Barre / treillis — barre en traction, comparée à l'analytique.

Physique
--------
Élément `SEG2` à 2 nœuds ne transmettant que l'effort axial. Loi : `N = E·A·ε`
avec `ε = du/ds` (déformation axiale le long de la barre). Rigidité globale
`K_e = (E·A/L)·[[c⊗c, -c⊗c], [-c⊗c, c⊗c]]`, où `c` est le cosinus directeur
(déduit des coordonnées des nœuds) — marche en 1-D/2-D/3-D.

Problème
--------
Barre horizontale de longueur `L`, encastrée à gauche (`u_x = u_y = 0`),
appuyée transversalement à droite (`u_y = 0`), force axiale `F` à droite.
Une barre n'ayant aucune raideur transversale, on bloque `u_y` aux deux nœuds.
Solution analytique : `u_x = F·L / (E·A)`.

Lancement ::

    maturin develop --features extension-module
    python examples/truss.py
"""

import pyrucast

E, A, L, F = 210.0e9, 1.0e-4, 2.0, 1000.0


def _clamp(node, var, dual):
    """Dirichlet homogène (u = 0) sur `var` au nœud `node`."""
    imposed = pyrucast.mesher.poi1_from_nodes([node])
    multiplier = pyrucast.mesher.barycenter(imposed)
    return pyrucast.Model.dirichlet(var, dual, imposed, multiplier)


def main() -> None:
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([L, 0.0])
    mesh = pyrucast.mesher.line_seg2(n0, n1, 1)  # un seul SEG2 (mailleur `line_seg2`)
    fes = pyrucast.FiniteElementSpace(mesh)

    model = pyrucast.Model.truss(fes)
    model = model | _clamp(n0, "u_x", "f_x")
    model = model | _clamp(n0, "u_y", "f_y")
    model = model | _clamp(n1, "u_y", "f_y")  # pas de raideur transversale

    materials = pyrucast.build.material_field(model, [("E", E), ("A", A)])

    load = pyrucast.mesher.poi1_from_nodes([n1])
    rhs = pyrucast.NodeField(load, ["f_x"])
    rhs[0].set_value(n1, "f_x", F)

    solution = pyrucast.solver.solve(pyrucast.assemble.stiffness(model, materials), rhs)

    ux = solution.value(n1, "u_x")
    expected = F * L / (E * A)
    print(f"u_x (bout) = {ux:.6e}   (analytique F·L/E·A = {expected:.6e})")
    assert abs(ux - expected) < 1e-10 * expected
    print("OK : élongation axiale conforme à F·L/(E·A).")


if __name__ == "__main__":
    main()
