"""Python tests for Fickian diffusion — the `c` / `j` physics."""

import pyrucast


def _line(n_elems=4, length=1.0):
    """A line of `n_elems` SEG2 on `[0, length]`."""
    c = pyrucast.Coords(1)
    h = length / n_elems
    nodes = [c.add_node([i * h]) for i in range(n_elems + 1)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(n_elems):
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    return c, nodes, pyrucast.FiniteElementSpace(mesh), h


def test_fick_variables_and_material():
    _c, _nodes, fes, _h = _line()
    model = pyrucast.Model.fick(fes)
    assert model[0].primal_vars() == ["c"]
    assert model[0].dual_vars() == ["j"]
    assert model[0].material_components() == ["D"]
    assert model[0].physics() == ["diffusion"]


def test_fick_recovers_the_linear_profile():
    """Injected flux at one end, imposed concentration at the other."""
    d, flux, c_imposed = 2.0, 10.0, 1.0
    c, nodes, fes, h = _line()

    imposed = pyrucast.Mesh(c, "POI1")
    imposed.unit().add_cell([nodes[-1]])
    multiplier = pyrucast.mesh.barycenter(imposed)
    mult = multiplier.node(0, 0, 0)

    model = pyrucast.Model.fick(fes) | pyrucast.Model.dirichlet(
        "c", "j", imposed, multiplier
    )
    materials = pyrucast.element_field.material_field(model, [("D", d)])

    load = pyrucast.Mesh(c, "POI1")
    load.unit().add_cell([nodes[0]])
    load.unit().add_cell([mult])
    rhs = pyrucast.NodeField(load, ["imposed_c", "j"])
    rhs[0].set_value(nodes[0], "j", flux)
    rhs[0].set_value(mult, "imposed_c", c_imposed)

    stiffness = pyrucast.matrix.stiffness(model, materials)
    solution = pyrucast.solver.solve(stiffness, rhs)

    for i, node in enumerate(nodes):
        x = i * h
        expected = c_imposed + (flux / d) * (1.0 - x)
        assert abs(solution.value(node, "c") - expected) < 1e-10
    # Mass balance: the reaction at the imposed end equals the injected flux.
    assert abs(solution.value(mult, "lambda_c") - flux) < 1e-10


def test_orthotropic_fick_declares_its_axes():
    _c, _nodes, fes, _h = _line()
    model = pyrucast.Model.fick(fes, symmetry="orthotropic")
    required = model[0].material_components()
    for name in ("D_1", "D_2", "D_3", "V1X", "V1Y"):
        assert name in required, f"{name} missing from {required}"
    assert "D" not in required


def test_anisotropic_fick_takes_the_full_tensor():
    _c, _nodes, fes, _h = _line()
    required = pyrucast.Model.fick(fes, symmetry="anisotropic")[0].material_components()
    for name in ("D_11", "D_12", "D_22", "D_33"):
        assert name in required, f"{name} missing from {required}"


def test_diffusion_filters_apart_from_thermal():
    """The two share an operator, not a nature — `filter` must separate them."""
    _c, _nodes, fes, _h = _line()
    model = pyrucast.Model.fick(fes) | pyrucast.Model.heat_conduction(fes)
    assert len(model) == 2
    assert len(model.filter("diffusion")) == 1
    assert len(model.filter("thermal")) == 1
    assert len(model.filter("mechanical")) == 0


def test_filter_rejects_an_unknown_nature_and_lists_the_new_ones():
    _c, _nodes, fes, _h = _line()
    model = pyrucast.Model.fick(fes)
    try:
        model.filter("magnetic")
    except ValueError as exc:
        message = str(exc)
        assert "diffusion" in message
        assert "radiation" in message
    else:  # pragma: no cover - the filter must refuse
        raise AssertionError("an unknown nature must raise")
