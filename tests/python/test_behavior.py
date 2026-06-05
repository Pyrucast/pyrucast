"""Python tests for behaviour integration (Cast3m `COMP`).

`deformation` applies each physics's differential operator to a nodal
solution (∇T for heat conduction); `integrate_behavior` runs the
constitutive law (weak-form flux = k·∇T). For a linear law the result is
exactly what the assembled stiffness encodes.
"""

import pyrucast


def _solved_ramp(n_elems=4, k=2.0):
    """Solve heat conduction on [0, 1] with T(0)=0, T(1)=1 (so T(x)=x),
    conductivity `k`. Returns (model, fes, materials, solution)."""
    h = 1.0 / n_elems
    c = pyrucast.Configuration(1)
    nodes = [c.add_node([i * h]) for i in range(n_elems + 1)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(n_elems):
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    fes = pyrucast.FiniteElementSpace(mesh)
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", k)

    left = pyrucast.Model.dirichlet("T", "q", [nodes[0]])
    right = pyrucast.Model.dirichlet("T", "q", [nodes[-1]])
    ml = left[0].multiplier_mesh().node(0, 0, 0)
    mr = right[0].multiplier_mesh().node(0, 0, 0)
    model = pyrucast.Model.heat_conduction(fes) + left + right

    rhs_mesh = pyrucast.Mesh(c, "POI1")
    rhs_mesh.unit().add_cell([ml])
    rhs_mesh.unit().add_cell([mr])
    rhs = pyrucast.NodeField(rhs_mesh, ["T"])
    rhs.set_value(ml, "T", 0.0)
    rhs.set_value(mr, "T", 1.0)

    solution = pyrucast.solve(pyrucast.stiffness(model, materials), rhs)
    return model, fes, materials, solution


def test_has_behavior_true_for_hc_false_for_dirichlet():
    model, *_ = _solved_ramp()
    # Model.heat_conduction(fes) → one HC sub-model, then two Dirichlet.
    assert model[0].has_behavior() is True
    assert model[1].has_behavior() is False
    assert model[2].has_behavior() is False


def test_deformation_is_constant_gradient():
    model, _fes, _materials, solution = _solved_ramp()
    defo = pyrucast.deformation(model, solution)
    assert len(defo) == 1, "only the HC sub-model carries a behaviour"
    sub = defo[0]
    assert sub.components() == ["grad_T_x"]
    for g in range(sub.gauss_count()):
        # T(x) = x ⇒ ∇T = 1 everywhere.
        assert abs(sub.value(0, g, "grad_T_x") - 1.0) < 1e-10


def test_integrate_behavior_returns_weak_form_flux():
    k = 2.0
    model, _fes, materials, solution = _solved_ramp(k=k)
    defo = pyrucast.deformation(model, solution)
    state = pyrucast.integrate_behavior(model, defo, materials)
    assert len(state) == 1
    sub = state[0]
    assert sub.components() == ["flux_x"]
    for cell in range(sub.cell_count()):
        for g in range(sub.gauss_count()):
            # weak-form flux = k·∇T = 2·1 = 2.
            assert abs(sub.value(cell, g, "flux_x") - k) < 1e-10


def test_integrate_behavior_missing_material_errors():
    model, fes, _materials, solution = _solved_ramp()
    defo = pyrucast.deformation(model, solution)
    # Material field on the right subspace but lacking the "k" component.
    bad = pyrucast.ElementField(fes, ["unused"])
    try:
        pyrucast.integrate_behavior(model, defo, bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when material lacks 'k'")
