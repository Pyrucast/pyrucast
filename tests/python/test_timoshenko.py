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
    model = model | _clamp(nodes[0], "w", "f_w")
    model = model | _clamp(nodes[0], "theta", "m_theta")

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


def test_section_forces_cantilever():
    """COMP : M = E·I·θ', V = G·A_s·(w'−θ). V ≈ −P constant, M linéaire."""
    E, I, G, A_S, L, P, N = 1.0, 1.0, 30.0, 1.0, 1.0, 1.0, 40
    h = L / N
    c = pyrucast.Configuration(1)
    nodes = [c.add_node([i * h]) for i in range(N + 1)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(N):
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    fes = pyrucast.FiniteElementSpace(mesh)

    model = pyrucast.Model.timoshenko(fes)
    model = model | _clamp(nodes[0], "w", "f_w")
    model = model | _clamp(nodes[0], "theta", "m_theta")
    materials = pyrucast.material_field(model, [("E", E), ("I", I), ("G", G), ("A_s", A_S)])

    load = pyrucast.Mesh(c, "POI1")
    load.unit().add_cell([nodes[-1]])
    rhs = pyrucast.NodeField(load, ["f_w"])
    rhs[0].set_value(nodes[-1], "f_w", P)
    solution = pyrucast.solve(pyrucast.stiffness(model, materials), rhs)

    # (κ, γ) puis efforts de section M, V.
    deformation = pyrucast.beam_deformation(solution, fes)
    forces = pyrucast.integrate_behavior(model, deformation, materials)
    sub = forces[0]
    for cell in range(sub.cell_count()):
        assert abs(abs(sub.value(cell, 0, "V")) - P) < 2e-2 * P  # V ≈ ±P (constant)
    assert abs(abs(sub.value(0, 0, "M")) - P * L) < 5e-2 * P * L  # |M(0)| ≈ P·L
    assert abs(sub.value(sub.cell_count() - 1, 0, "M")) < 5e-2 * P * L  # |M(L)| ≈ 0


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
