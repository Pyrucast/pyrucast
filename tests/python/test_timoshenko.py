"""Python tests for the Timoshenko beam physics."""

import pyrucast


def _clamp(node, var, dual):
    imposed = pyrucast.poi1_from_nodes([node])
    multiplier = pyrucast.barycenter(imposed)
    return pyrucast.Model.dirichlet(var, dual, imposed, multiplier)


def test_cantilever_converges_without_locking():
    """Slender cantilever, tip load P ⇒ w_tip = P·L³/(3EI) + P·L/(G·A_s)."""
    E, I, G, A_S, L, P, N = 1.0, 1.0, 30.0, 1.0, 1.0, 1.0, 40
    h = L / N
    c = pyrucast.Configuration(1)
    nodes = [c.add_node([i * h]) for i in range(N + 1)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(N):
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    fes = pyrucast.FiniteElementSpace(mesh)

    model = pyrucast.Model.timoshenko(fes)
    model = model + _clamp(nodes[0], "w", "f_w")
    model = model + _clamp(nodes[0], "theta", "m_theta")

    materials = pyrucast.material_field(model, [("E", E), ("I", I), ("G", G), ("A_s", A_S)])

    load_mesh = pyrucast.Mesh(c, "POI1")
    load_mesh.unit().add_cell([nodes[N]])
    rhs = pyrucast.NodeField(load_mesh, ["f_w"])
    rhs[0].set_value(nodes[N], "f_w", P)

    K = pyrucast.stiffness(model, materials)
    solution = pyrucast.solve(K, rhs)

    w_tip = solution.value(nodes[N], "w")
    analytical = P * L**3 / (3.0 * E * I) + P * L / (G * A_S)
    assert abs(w_tip - analytical) < 1e-2 * analytical


def test_timoshenko_vars():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    model = pyrucast.Model.timoshenko(fes)
    assert model.primal_vars() == ["w", "theta"]
    assert model.dual_vars() == ["f_w", "m_theta"]
