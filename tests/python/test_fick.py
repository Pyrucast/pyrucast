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
    model = pyrucast.model.fick(fes, "H2")
    assert model[0].primal_vars() == ["c_H2"]
    assert model[0].dual_vars() == ["j_H2"]
    assert model[0].material_components() == ["D_H2"]
    assert model[0].physics() == ["diffusion"]


def test_fick_recovers_the_linear_profile():
    """Injected flux at one end, imposed concentration at the other."""
    d, flux, c_imposed = 2.0, 10.0, 1.0
    c, nodes, fes, h = _line()

    imposed = pyrucast.Mesh(c, "POI1")
    imposed.unit().add_cell([nodes[-1]])
    multiplier = pyrucast.mesh.barycenter(imposed)
    mult = multiplier.node(0, 0, 0)

    cible = pyrucast.model.fick(fes, "H2")

    model = cible | pyrucast.model.dirichlet(cible, "c_H2", imposed, multiplier)
    materials = pyrucast.element_field.material_field(model, [("D_H2", d)])

    load = pyrucast.Mesh(c, "POI1")
    load.unit().add_cell([nodes[0]])
    load.unit().add_cell([mult])
    rhs = pyrucast.NodeField(load, ["imposed_c_H2", "j_H2"])
    rhs[0].set_value(nodes[0], "j_H2", flux)
    rhs[0].set_value(mult, "imposed_c_H2", c_imposed)

    stiffness = pyrucast.matrix.stiffness(model, materials)
    solution = pyrucast.solver.solve(stiffness, rhs)

    for i, node in enumerate(nodes):
        x = i * h
        expected = c_imposed + (flux / d) * (1.0 - x)
        assert abs(solution.value(node, "c_H2") - expected) < 1e-10
    # Mass balance: the reaction at the imposed end equals the injected flux.
    assert abs(solution.value(mult, "lambda_c_H2") - flux) < 1e-10


def test_orthotropic_fick_declares_its_axes():
    _c, _nodes, fes, _h = _line()
    model = pyrucast.model.fick(fes, "H2", symmetry="orthotropic")
    required = model[0].material_components()
    # The diffusivities carry the species, the medium's own axes do not.
    for name in ("D_1_H2", "D_2_H2", "D_3_H2", "V1X", "V1Y"):
        assert name in required, f"{name} missing from {required}"
    assert "D_H2" not in required


def test_anisotropic_fick_takes_the_full_tensor():
    _c, _nodes, fes, _h = _line()
    required = pyrucast.model.fick(fes, "H2", symmetry="anisotropic")[
        0
    ].material_components()
    for name in ("D_11_H2", "D_12_H2", "D_22_H2", "D_33_H2"):
        assert name in required, f"{name} missing from {required}"


def test_two_species_share_a_mesh_without_colliding():
    """What the suffix is for: one mesh, two diffusing species, no collision."""
    _c, _nodes, fes, _h = _line()
    model = pyrucast.model.fick(fes, "H2") | pyrucast.model.fick(fes, "O2")
    assert len(model) == 2
    assert model.primal_vars() == ["c_H2", "c_O2"]
    assert model.dual_vars() == ["j_H2", "j_O2"]
    assert model[0].material_components() == ["D_H2"]
    assert model[1].material_components() == ["D_O2"]


def test_an_unnamed_species_is_rejected():
    _c, _nodes, fes, _h = _line()
    try:
        pyrucast.model.fick(fes, "")
    except RuntimeError as exc:
        assert "must be named" in str(exc)
    else:  # pragma: no cover - the constructor must refuse
        raise AssertionError("an empty species must raise")


def test_diffusion_filters_apart_from_thermal():
    """The two share an operator, not a nature — `filter` must separate them."""
    _c, _nodes, fes, _h = _line()
    model = pyrucast.model.fick(fes, "H2") | pyrucast.model.heat_conduction(fes)
    assert len(model) == 2
    assert len(model.filter("diffusion")) == 1
    assert len(model.filter("thermal")) == 1
    assert len(model.filter("mechanical")) == 0


def test_filter_rejects_an_unknown_nature_and_lists_the_new_ones():
    _c, _nodes, fes, _h = _line()
    model = pyrucast.model.fick(fes, "H2")
    try:
        model.filter("magnetic")
    except ValueError as exc:
        message = str(exc)
        assert "diffusion" in message
        assert "radiation" in message
    else:  # pragma: no cover - the filter must refuse
        raise AssertionError("an unknown nature must raise")
