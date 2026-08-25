"""Python tests for COMP — the geometric producers (`gradient`,
`deformation`) feeding `integrate_behavior`.

`gradient` / `deformation` depend only on the FE space (no model); the user
calls one of them to get a per-element field and hands it to
`integrate_behavior`, which runs the constitutive law. For a linear law the
weak-form flux `k·∇T` is exactly what the assembled stiffness encodes.
"""

import pyrucast


def _heat_setup(n_elems=4, length=1.0, k=2.0):
    """Heat-conduction model on [0, length] (n SEG2), conductivity `k`, and a
    nodal temperature field `T(x) = x`. Returns (model, fes, materials, t)."""
    h = length / n_elems
    c = pyrucast.Coords(1)
    nodes = [c.add_node([i * h]) for i in range(n_elems + 1)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(n_elems):
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    fes = pyrucast.FiniteElementSpace(mesh)
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", k)
    model = pyrucast.model.heat_conduction(fes)

    # Nodal temperature T(x) = x over a POI1 support of all nodes.
    t_mesh = pyrucast.Mesh(c, "POI1")
    for n in nodes:
        t_mesh.unit().add_cell([n])
    t = pyrucast.NodeField(t_mesh, ["T"])
    for i, n in enumerate(nodes):
        t[0].set_value(n, "T", i * h)  # T = x
    return model, fes, materials, t


def test_gradient_of_linear_temperature():
    _model, fes, _materials, t = _heat_setup(length=1.0)
    grad = pyrucast.element_field.gradient(t, fes)
    assert len(grad) == 1
    sub = grad[0]
    assert sub.components() == ["grad_T_x"]
    for cell in range(sub.cell_count()):
        for g in range(sub.gauss_count()):
            assert abs(sub.value(cell, g, "grad_T_x") - 1.0) < 1e-10  # ∇(x) = 1


def test_integrate_behavior_returns_weak_form_flux():
    k = 2.0
    model, fes, materials, t = _heat_setup(k=k)
    grad = pyrucast.element_field.gradient(t, fes)
    state = pyrucast.element_field.integrate_behavior(model, grad, materials)
    assert len(state) == 1
    sub = state[0]
    assert sub.components() == ["flux_x"]
    for cell in range(sub.cell_count()):
        for g in range(sub.gauss_count()):
            assert abs(sub.value(cell, g, "flux_x") - k) < 1e-10  # k·∇T = k·1


def test_has_behavior_true_for_hc_false_for_dirichlet():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    imposed = pyrucast.mesh.poi1_from_nodes([a])
    multiplier = pyrucast.mesh.barycenter(imposed)
    model = pyrucast.model.heat_conduction(fes) | pyrucast.model.dirichlet(
        "T", "q", imposed, multiplier
    )
    assert model[0].has_behavior() is True
    assert model[1].has_behavior() is False


def test_integrate_behavior_missing_material_errors():
    model, fes, _materials, t = _heat_setup()
    grad = pyrucast.element_field.gradient(t, fes)
    # Material field on the right subspace but lacking the "k" component.
    bad = pyrucast.ElementField(fes, ["unused"])
    try:
        pyrucast.element_field.integrate_behavior(model, grad, bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when material lacks 'k'")


def test_deformation_linearized_strain_2d():
    """u_x = 2x + 0.5y, u_y = 0.1x + 3y on a TRI3 ⇒ ε_xx=2, ε_yy=3, ε_xy=0.3."""
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([a, b, cc])
    fes = pyrucast.FiniteElementSpace(mesh)

    u_mesh = pyrucast.Mesh(c, "POI1")
    for n in (a, b, cc):
        u_mesh.unit().add_cell([n])
    u = pyrucast.NodeField(u_mesh, ["u_x", "u_y"])
    for n, x, y in [(a, 0.0, 0.0), (b, 1.0, 0.0), (cc, 0.0, 1.0)]:
        u[0].set_value(n, "u_x", 2.0 * x + 0.5 * y)
        u[0].set_value(n, "u_y", 0.1 * x + 3.0 * y)

    strain = pyrucast.element_field.deformation(u, fes)
    sub = strain[0]
    assert sub.components() == ["eps_xx", "eps_xy", "eps_yy"]
    for g in range(sub.gauss_count()):
        assert abs(sub.value(0, g, "eps_xx") - 2.0) < 1e-10
        assert abs(sub.value(0, g, "eps_yy") - 3.0) < 1e-10
        assert abs(sub.value(0, g, "eps_xy") - 0.3) < 1e-10
