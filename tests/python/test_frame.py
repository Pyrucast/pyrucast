"""Python tests for the planar frame (portique) physics."""

import math

import pyrucast


def _clamp(target, node, var):
    imposed = pyrucast.mesh.poi1_from_nodes([node])
    multiplier = pyrucast.mesh.barycenter(imposed)
    return pyrucast.model.dirichlet(target, var, imposed, multiplier)


def test_inclined_cantilever_perpendicular_load():
    """45° cantilever, perpendicular tip load ⇒ tip moves by the Timoshenko
    deflection along the perpendicular, with ~no axial displacement."""
    E, A, I, G, A_S, L, P, N = 1.0, 1.0, 1.0, 30.0, 1.0, 1.0, 1.0, 40
    c = s = 1.0 / math.sqrt(2.0)
    px, py = -s, c  # unit perpendicular
    h = L / N

    coords = pyrucast.Coords(2)
    nodes = [coords.add_node([i * h * c, i * h * s]) for i in range(N + 1)]
    mesh = pyrucast.Mesh(coords, "SEG2")
    for i in range(N):
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    fes = pyrucast.FiniteElementSpace(mesh, interpolation="MODEL_EMBEDDED")

    model = pyrucast.model.timoshenko(fes)
    for var in ("u_x", "u_y", "r_z"):
        model = model | _clamp(model, nodes[0], var)
    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("A", A), ("I", I), ("G", G), ("A_s", A_S)]
    )

    load = pyrucast.Mesh(coords, "POI1")
    load.unit().add_cell([nodes[-1]])
    rhs = pyrucast.NodeField(load, ["f_x", "f_y"])
    rhs[0].set_value(nodes[-1], "f_x", P * px)
    rhs[0].set_value(nodes[-1], "f_y", P * py)
    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)

    delta = P * L**3 / (3.0 * E * I) + P * L / (G * A_S)
    ux = solution.value(nodes[-1], "u_x")
    uy = solution.value(nodes[-1], "u_y")
    assert abs((ux * px + uy * py) - delta) < 1e-2 * delta  # transverse = δ
    assert abs(ux * c + uy * s) < 1e-6  # axial ≈ 0


def test_frame_vars():
    coords = pyrucast.Coords(2)
    a = coords.add_node([0.0, 0.0])
    b = coords.add_node([1.0, 0.0])
    mesh = pyrucast.Mesh(coords, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh, interpolation="MODEL_EMBEDDED")
    model = pyrucast.model.timoshenko(fes)
    assert model.primal_vars() == ["u_x", "u_y", "r_z"]
    assert model.dual_vars() == ["f_x", "f_y", "m_z"]
