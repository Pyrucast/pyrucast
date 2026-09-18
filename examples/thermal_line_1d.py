"""1-D thermal conduction — a heated line, compared with the analytical solution.

Problème
--------
On the segment [0, 1]:

  * at x = 0: a **heat source** (Neumann flux ``Q``);
  * at x = 1: an **imposed temperature** ``T = 20`` (Dirichlet).

In the steady regime without volume generation, ``T'' = 0``: the profile is
linear. The analytical solution is ::

    u(x) = 20 + (Q / k) * (1 - x)

and the Lagrange multiplier at the imposed node (the *reaction* holding
``T = 20``) is exactly ``Q``: all the flux injected at x = 0 leaves at
x = 1 (bilan d'énergie).

This is the Python equivalent of the Rust integration test ``tests/thermal_line.rs``
(and of the book's "Conduction thermique" chapter).

Lancement
---------
Once the extension is built in the venv ::

    maturin develop --features extension-module
    python examples/thermal_line_1d.py
"""

import pyrucast

# ── Problem data ────────────────────────────────────────────────────────────
K = 1.0  # conductivité
Q = 10.0  # source de chaleur (flux de Neumann) en x = 0
T_IMPOSED = 20.0  # température imposée en x = 1
N_ELEMS = 4


def main() -> None:
    h = 1.0 / N_ELEMS

    # ── Mesh: a line of N_ELEMS SEG2 on [0, 1] (the `line` mesher) ───────
    c = pyrucast.Coords(1)
    x0 = c.add_node([0.0])
    x1 = c.add_node([1.0])
    mesh = pyrucast.mesh.line(x0, x1, N_ELEMS)
    # Nœuds ordonnés le long de la ligne : x0, intérieurs…, x1.
    nodes = [mesh.node(0, i, 0) for i in range(N_ELEMS)] + [x1]
    fes = pyrucast.FiniteElementSpace(mesh)

    # ── Modèle : conduction + Dirichlet T = 20 en x = 1 ──────────────────────
    # The multipliers' support is built from the imposed node by the `barycenter`
    # mesher (a fresh co-located node). The model creates nothing.
    imposed = pyrucast.mesh.poi1_from_nodes([nodes[-1]])
    multiplier = pyrucast.mesh.barycenter(imposed)
    mult = multiplier.node(0, 0, 0)
    cible = pyrucast.model.heat_conduction(fes)

    model = cible | pyrucast.model.dirichlet(cible, "T", imposed, multiplier)

    # ── Matériau : k uniforme (Dirichlet ignoré automatiquement) ─────────────
    materials = pyrucast.element_field.material_field(model, [("k", K)])

    # ── Chargement : source Q en x = 0 (composante duale "q"), valeur imposée
    #    T = 20 at the multiplier node ("imposed_T" slot) ─────────────────────
    load_mesh = pyrucast.mesh.poi1_from_nodes([nodes[0], mult])
    rhs = pyrucast.NodeField(load_mesh, ["imposed_T", "q"])
    rhs[0].set_value(nodes[0], "q", Q)
    rhs[0].set_value(mult, "imposed_T", T_IMPOSED)

    # ── Assemblage + résolution ──────────────────────────────────────────────
    K_mat = pyrucast.matrix.stiffness(model, materials)
    solution = pyrucast.solver.solve(K_mat, rhs)

    # ── Compared with the analytical u(x) = 20 + (Q/k)(1 - x) ────────────────
    print(f"{'x':>6} {'T_calc':>12} {'T_exact':>12}")
    tol = 1e-10
    for i, node in enumerate(nodes):
        x = i * h
        expected = T_IMPOSED + (Q / K) * (1.0 - x)
        got = solution.value(node, "T")
        print(f"{x:6.3f} {got:12.6f} {expected:12.6f}")
        assert abs(got - expected) < tol, f"x={x}: {got} != {expected}"

    # La réaction (multiplicateur) équilibre le flux injecté : λ = Q.
    reaction = solution.value(mult, "lambda_T")
    print(f"\nréaction λ = {reaction:.6f}  (attendu {Q})")
    assert abs(reaction - Q) < tol

    print("\nOK: profile and reaction matching the analytical solution.")


if __name__ == "__main__":
    main()
