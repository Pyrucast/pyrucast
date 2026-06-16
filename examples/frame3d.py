"""Cadre 3-D (space frame) — console, flexions + torsion.

Physique
--------
Poutre 3-D à 6 DOFs par nœud (`u_x, u_y, u_z, r_x, r_y, r_z`). La rigidité
locale 12×12 combine l'effort axial (`E·A`), la torsion (`G·J`), et la flexion
de Timoshenko dans les deux plans principaux (`E·I_z`/`G·A_sy` et
`E·I_y`/`G·A_sz`). Elle est tournée dans le repère global `K = Tᵀ K_loc T` ;
les axes de section sont orientés automatiquement (référence Z globale).
L'élément (forme fermée, paramètres Φ) est nodalement exact pour des charges
en bout.

Problème
--------
Console le long de X, encastrée à la base (6 DOFs), charges au bout libre :
`f_y`, `f_z` et un moment de torsion `m_x`. Réponses découplées et exactes ::

    u_y = P_y·L³/(3·E·I_z) + P_y·L/(G·A_sy)
    u_z = P_z·L³/(3·E·I_y) + P_z·L/(G·A_sz)
    r_x = M_x·L/(G·J)

Lancement ::

    maturin develop --features extension-module
    python examples/frame3d.py
"""

import pyrucast

E, A, IY, IZ, J, G, ASY, ASZ = 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 10.0, 10.0
L, PY, PZ, MX, N = 1.0, 1.0, 1.0, 1.0, 2


def _clamp(node, var, dual):
    imposed = pyrucast.poi1_from_nodes([node])
    multiplier = pyrucast.barycenter(imposed)
    return pyrucast.Model.dirichlet(var, dual, imposed, multiplier)


def main() -> None:
    h = L / N
    c = pyrucast.Configuration(3)
    nodes = [c.add_node([i * h, 0.0, 0.0]) for i in range(N + 1)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(N):
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    fes = pyrucast.FiniteElementSpace(mesh)

    model = pyrucast.Model.frame3d(fes)
    for var, dual in (
        ("u_x", "f_x"), ("u_y", "f_y"), ("u_z", "f_z"),
        ("r_x", "m_x"), ("r_y", "m_y"), ("r_z", "m_z"),
    ):
        model = model | _clamp(nodes[0], var, dual)
    materials = pyrucast.material_field(
        model,
        [("E", E), ("A", A), ("I_y", IY), ("I_z", IZ),
         ("J", J), ("G", G), ("A_sy", ASY), ("A_sz", ASZ)],
    )

    load = pyrucast.Mesh(c, "POI1")
    load.unit().add_cell([nodes[-1]])
    rhs = pyrucast.NodeField(load, ["f_y", "f_z", "m_x"])
    rhs[0].set_value(nodes[-1], "f_y", PY)
    rhs[0].set_value(nodes[-1], "f_z", PZ)
    rhs[0].set_value(nodes[-1], "m_x", MX)
    solution = pyrucast.solve(pyrucast.stiffness(model, materials), rhs)

    tip = nodes[-1]
    uy = PY * L**3 / (3 * E * IZ) + PY * L / (G * ASY)
    uz = PZ * L**3 / (3 * E * IY) + PZ * L / (G * ASZ)
    rx = MX * L / (G * J)
    print(f"{'DOF':>5} {'calc':>12} {'exact':>12}")
    for name, got, exact in (
        ("u_y", solution.value(tip, "u_y"), uy),
        ("u_z", solution.value(tip, "u_z"), uz),
        ("r_x", solution.value(tip, "r_x"), rx),
    ):
        print(f"{name:>5} {got:12.6f} {exact:12.6f}")
        assert abs(got - exact) < 1e-9
    print("OK : flexions (2 plans) + torsion conformes à l'analytique.")


if __name__ == "__main__":
    main()
