"""Cadre 3-D (space frame) — console, flexions + torsion.

Physique
--------
A 3-D beam with 6 DOFs per node (`u_x, u_y, u_z, r_x, r_y, r_z`). The
locale 12×12 combine l'effort axial (`E·A`), la torsion (`G·J`), et la flexion
Timoshenko stiffness in both principal planes (`E·I_z`/`G·A_sy` and
`E·I_y`/`G·A_sz`). It is rotated into the global frame `K = Tᵀ K_loc T`; the
section axes are oriented automatically (global Z reference).
The element (closed form, Φ parameters) is nodally exact for loads
en bout.

Problème
--------
A cantilever along X, clamped at the base (6 DOFs), loads at the free end:
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


def _clamp(target, node, var):
    imposed = pyrucast.mesh.poi1_from_nodes([node])
    multiplier = pyrucast.mesh.barycenter(imposed)
    return pyrucast.model.dirichlet(target, var, imposed, multiplier)


def main() -> None:
    c = pyrucast.Coords(3)
    base = c.add_node([0.0, 0.0, 0.0])
    tip = c.add_node([L, 0.0, 0.0])
    mesh = pyrucast.mesh.line(base, tip, N)  # console le long de X (`line`)
    fes = pyrucast.FiniteElementSpace(mesh, interpolation="MODEL_EMBEDDED")

    model = pyrucast.model.timoshenko(fes)
    for var, dual in (
        ("u_x", "f_x"),
        ("u_y", "f_y"),
        ("u_z", "f_z"),
        ("r_x", "m_x"),
        ("r_y", "m_y"),
        ("r_z", "m_z"),
    ):
        model = model | _clamp(model, base, var)
    materials = pyrucast.element_field.material_field(
        model,
        [
            ("E", E),
            ("A", A),
            ("I_y", IY),
            ("I_z", IZ),
            ("J", J),
            ("G", G),
            ("A_sy", ASY),
            ("A_sz", ASZ),
        ],
    )

    load = pyrucast.mesh.poi1_from_nodes([tip])
    rhs = pyrucast.NodeField(load, ["f_y", "f_z", "m_x"])
    rhs[0].set_value(tip, "f_y", PY)
    rhs[0].set_value(tip, "f_z", PZ)
    rhs[0].set_value(tip, "m_x", MX)
    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)

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
    print("OK: bending (2 planes) + torsion matching the analytical solution.")


if __name__ == "__main__":
    main()
