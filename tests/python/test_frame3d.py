"""Python tests for the 3-D Timoshenko frame (space frame) physics."""

import pyrucast


def _clamp(target, node, var):
    imposed = pyrucast.mesh.poi1_from_nodes([node])
    multiplier = pyrucast.mesh.barycenter(imposed)
    return pyrucast.model.dirichlet(target, var, imposed, multiplier)


def test_cantilever_bending_and_torsion():
    """X-cantilever, tip loads f_y, f_z, m_x ⇒ exact deflections + twist."""
    E, A, IY, IZ, J, G, ASY, ASZ = 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 10.0, 10.0
    L, PY, PZ, MX, N = 1.0, 1.0, 1.0, 1.0, 2
    h = L / N

    c = pyrucast.Coords(3)
    nodes = [c.add_node([i * h, 0.0, 0.0]) for i in range(N + 1)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(N):
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
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
        model = model | _clamp(model, nodes[0], var)
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

    load = pyrucast.Mesh(c, "POI1")
    load.unit().add_cell([nodes[-1]])
    rhs = pyrucast.NodeField(load, ["f_y", "f_z", "m_x"])
    rhs[0].set_value(nodes[-1], "f_y", PY)
    rhs[0].set_value(nodes[-1], "f_z", PZ)
    rhs[0].set_value(nodes[-1], "m_x", MX)
    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)

    tip = nodes[-1]
    tol = 1e-9
    assert (
        abs(
            solution.value(tip, "u_y") - (PY * L**3 / (3 * E * IZ) + PY * L / (G * ASY))
        )
        < tol
    )
    assert (
        abs(
            solution.value(tip, "u_z") - (PZ * L**3 / (3 * E * IY) + PZ * L / (G * ASZ))
        )
        < tol
    )
    assert abs(solution.value(tip, "r_x") - MX * L / (G * J)) < tol
    assert abs(solution.value(tip, "u_x")) < tol


def test_frame3d_vars():
    c = pyrucast.Coords(3)
    a = c.add_node([0.0, 0.0, 0.0])
    b = c.add_node([1.0, 0.0, 0.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh, interpolation="MODEL_EMBEDDED")
    model = pyrucast.model.timoshenko(fes)
    assert model.primal_vars() == ["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"]
    assert model.dual_vars() == ["f_x", "f_y", "f_z", "m_x", "m_y", "m_z"]
