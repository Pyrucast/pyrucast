"""Bar / truss — a bar in tension, compared with the analytical solution.

Physique
--------
A 2-node `SEG2` element transmitting the axial force only. Law: `N = E·A·ε`
with `ε = du/ds` (axial strain along the bar). Global stiffness
`K_e = (E·A/L)·[[c⊗c, -c⊗c], [-c⊗c, c⊗c]]`, where `c` is the direction cosine
(derived from the nodes' coordinates) — works in 1-D/2-D/3-D.

Problème
--------
A horizontal bar of length `L`, clamped on the left (`u_x = u_y = 0`),
transversally supported on the right (`u_y = 0`), axial force `F` on the right.
A bar having no transverse stiffness, `u_y` is blocked at both nodes.
Solution analytique : `u_x = F·L / (E·A)`.

Lancement ::

    maturin develop --features extension-module
    python examples/truss.py
"""

import pyrucast

E, A, L, F = 210.0e9, 1.0e-4, 2.0, 1000.0


def _clamp(target, node, var):
    """Homogeneous Dirichlet (u = 0) on `var` at node `node`."""
    imposed = pyrucast.mesh.poi1_from_nodes([node])
    multiplier = pyrucast.mesh.barycenter(imposed)
    return pyrucast.model.dirichlet(target, var, imposed, multiplier)


def main() -> None:
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([L, 0.0])
    mesh = pyrucast.mesh.line(n0, n1, 1)  # un seul SEG2 (mailleur `line`)
    fes = pyrucast.FiniteElementSpace(mesh)

    model = pyrucast.model.truss(fes)
    model = model | _clamp(model, n0, "u_x")
    model = model | _clamp(model, n0, "u_y")
    model = model | _clamp(model, n1, "u_y")  # no transverse stiffness

    materials = pyrucast.element_field.material_field(model, [("E", E), ("A", A)])

    load = pyrucast.mesh.poi1_from_nodes([n1])
    rhs = pyrucast.NodeField(load, ["f_x"])
    rhs[0].set_value(n1, "f_x", F)

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)

    ux = solution.value(n1, "u_x")
    expected = F * L / (E * A)
    print(f"u_x (bout) = {ux:.6e}   (analytique F·L/E·A = {expected:.6e})")
    assert abs(ux - expected) < 1e-10 * expected
    print("OK : élongation axiale conforme à F·L/(E·A).")


if __name__ == "__main__":
    main()
