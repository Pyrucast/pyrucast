"""Python tests for the interface exchange law `j·n = h(c1 - c2)`."""

import pyrucast


def _two_squares():
    """Two QUA4 squares side by side, with duplicated nodes at x = 1."""
    c = pyrucast.Coords(2)
    left = [
        c.add_node([0.0, 0.0]),
        c.add_node([1.0, 0.0]),
        c.add_node([1.0, 1.0]),
        c.add_node([0.0, 1.0]),
    ]
    right = [
        c.add_node([1.0, 0.0]),
        c.add_node([2.0, 0.0]),
        c.add_node([2.0, 1.0]),
        c.add_node([1.0, 1.0]),
    ]

    def square(nodes):
        m = pyrucast.Mesh(c, "QUA4")
        m.unit().add_cell(nodes)
        return pyrucast.FiniteElementSpace(m)

    def edge(a, b):
        m = pyrucast.Mesh(c, "SEG2")
        m.unit().add_cell([a, b])
        return pyrucast.FiniteElementSpace(m)

    return c, left, right, square, edge


def test_interface_declares_its_variables_and_material():
    _c, left, right, square, edge = _two_squares()
    bodies = pyrucast.model.fick(square(left), "H2") | pyrucast.model.fick(
        square(right), "H2"
    )
    model = pyrucast.model.interface_transfer(
        edge(left[1], left[2]), edge(right[0], right[3]), bodies, [("c_H2", "j_H2")]
    )
    assert model[0].primal_vars() == ["c_H2"]
    assert model[0].dual_vars() == ["j_H2"]
    assert model[0].material_components() == ["h_c_H2"]
    # The nature is the Fick models', not an argument.
    assert model[0].physics() == ["diffusion"]


def test_thermal_variant_is_a_contact_resistance():
    _c, left, right, square, edge = _two_squares()
    bodies = pyrucast.model.heat_conduction(
        square(left)
    ) | pyrucast.model.heat_conduction(square(right))
    model = pyrucast.model.interface_transfer(
        edge(left[1], left[2]), edge(right[0], right[3]), bodies, [("T", "q")]
    )
    assert model[0].primal_vars() == ["T"]
    assert model[0].dual_vars() == ["q"]
    assert model[0].material_components() == ["h_T"]
    # Same DOFs as heat conduction, so it couples straight into it.
    assert model[0].physics() == ["thermal"]


def test_a_mechanical_joint_transfers_several_quantities_at_once():
    """The generalisation: nothing about the law is thermal or diffusive.

    Given the displacement pairs, the very same interface becomes a bonded joint
    of finite stiffness — one coefficient per direction, and no new physics.
    """
    _c, left, right, square, edge = _two_squares()
    bodies = pyrucast.model.elasticity(
        square(left), "plane_stress"
    ) | pyrucast.model.elasticity(square(right), "plane_stress")
    model = pyrucast.model.interface_transfer(
        edge(left[1], left[2]),
        edge(right[0], right[3]),
        bodies,
        [("u_x", "f_x"), ("u_y", "f_y")],
    )
    assert model[0].primal_vars() == ["u_x", "u_y"]
    assert model[0].dual_vars() == ["f_x", "f_y"]
    assert model[0].material_components() == ["h_u_x", "h_u_y"]
    assert model[0].physics() == ["mechanical"]


def test_a_non_conforming_interface_is_rejected():
    c = pyrucast.Coords(2)
    a = [c.add_node([1.0, 0.0]), c.add_node([1.0, 1.0])]
    b = [c.add_node([1.5, 0.0]), c.add_node([1.5, 1.0])]

    def edge(nodes):
        m = pyrucast.Mesh(c, "SEG2")
        m.unit().add_cell(nodes)
        return pyrucast.FiniteElementSpace(m)

    # A construction-time modelling error surfaces as `RuntimeError`. The
    # target is valid, so that it is the geometry being refused.
    target = pyrucast.model.fick(edge(a), "H2")
    try:
        pyrucast.model.interface_transfer(edge(a), edge(b), target, [("c_H2", "j_H2")])
    except RuntimeError as exc:
        assert "not node-conforming" in str(exc)
    else:  # pragma: no cover - the constructor must refuse
        raise AssertionError("a non-conforming interface must raise")


def test_a_pair_the_target_does_not_assemble_is_rejected():
    """Coupled into a model that has no such rows, the law would tie nothing."""
    _c, left, right, square, edge = _two_squares()
    diffusion = pyrucast.model.fick(square(left), "H2")
    try:
        pyrucast.model.interface_transfer(
            edge(left[1], left[2]), edge(right[0], right[3]), diffusion, [("T", "q")]
        )
    except RuntimeError as exc:
        assert "assembles no `T` paired with `q`" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a pair the target does not assemble must raise")


def test_pairs_of_two_natures_are_rejected():
    """One interface carries one nature: build one per physics."""
    _c, left, right, square, edge = _two_squares()
    both = pyrucast.model.fick(square(left), "H2") | pyrucast.model.heat_conduction(
        square(left)
    )
    try:
        pyrucast.model.interface_transfer(
            edge(left[1], left[2]),
            edge(right[0], right[3]),
            both,
            [("c_H2", "j_H2"), ("T", "q")],
        )
    except RuntimeError as exc:
        assert "single nature" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("pairs of two natures must raise")


def test_transferring_nothing_is_rejected():
    """A law that transfers nothing has no matrix and no coefficient."""
    _c, left, right, square, edge = _two_squares()
    target = pyrucast.model.fick(square(left), "H2")
    try:
        pyrucast.model.interface_transfer(
            edge(left[1], left[2]), edge(right[0], right[3]), target, []
        )
    except RuntimeError as exc:
        assert "nothing to transfer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an empty component list must raise")


def test_the_field_jumps_by_q_over_h():
    """The physics: a flux q crossing the interface makes `c` jump by q/h."""
    d, q, h, c_right = 2.0, 10.0, 5.0, 1.0
    c, left, right, square, edge = _two_squares()

    imposed = pyrucast.Mesh(c, "POI1")
    imposed.unit().add_cell([right[1]])
    imposed.unit().add_cell([right[2]])
    multiplier = pyrucast.mesh.barycenter(imposed)

    modele_fick = pyrucast.model.fick(square(left), "H2") | pyrucast.model.fick(
        square(right), "H2"
    )
    model = (
        modele_fick
        | pyrucast.model.interface_transfer(
            edge(left[1], left[2]),
            edge(right[0], right[3]),
            modele_fick,
            [("c_H2", "j_H2")],
        )
        | pyrucast.model.dirichlet(modele_fick, "c_H2", imposed, multiplier)
    )
    # Uniform flux density on the far-left edge, as consistent nodal loads.
    inlet = pyrucast.Mesh(c, "SEG2")
    inlet.unit().add_cell([left[0], left[3]])
    model = model | pyrucast.model.flux(
        pyrucast.FiniteElementSpace(inlet), model, "j_H2"
    )
    materials = pyrucast.element_field.material_field(
        model, [("D_H2", d), ("h_c_H2", h), ("phi_j_H2", q)]
    )
    rhs = pyrucast.node_field.external_forces(model, materials)

    # The imposed concentration, on the two multiplier nodes.
    mult_mesh = model.multiplier_mesh()
    mults = [mult_mesh.node(0, i, 0) for i in range(len(mult_mesh[0]))]
    load = pyrucast.Mesh(c, "POI1")
    for m in mults:
        load.unit().add_cell([m])
    imposed_field = pyrucast.NodeField(load, ["imposed_c_H2"])
    for m in mults:
        imposed_field[0].set_value(m, "imposed_c_H2", c_right)

    solution = pyrucast.solver.solve(
        pyrucast.matrix.stiffness(model, materials), rhs | imposed_field
    )

    left_face = solution.value(left[1], "c_H2")
    right_face = solution.value(right[0], "c_H2")
    assert abs(solution.value(right[1], "c_H2") - c_right) < 1e-10
    assert abs((left_face - right_face) - q / h) < 1e-10
    # Each square drops q/D across its unit width.
    assert abs((solution.value(left[0], "c_H2") - left_face) - q / d) < 1e-10
