"""Python tests for the consistent mass (MASS) and heat-capacity (CAPA) matrices.

The consistent element mass of a unit QUA4 is (1/36)·[[4,2,1,2],…]; the
whole-matrix sum is n·ρ·V (mechanics) and ρ·cp·V (thermal). Rows are labelled by
the dual variable (f_x/f_y, q), columns by the primal one (u_x/u_y, T).
"""

import pyrucast


def _unit_quad():
    c = pyrucast.Coords(2)
    n = [
        c.add_node([0.0, 0.0]),
        c.add_node([1.0, 0.0]),
        c.add_node([1.0, 1.0]),
        c.add_node([0.0, 1.0]),
    ]
    mesh = pyrucast.Mesh(c, "QUA4")
    mesh.unit().add_cell(n)
    return pyrucast.FiniteElementSpace(mesh), n


def test_consistent_mass_of_unit_quad():
    rho = 2.0
    fes, n = _unit_quad()
    model = pyrucast.model.elasticity(fes, "plane_stress")
    materials = pyrucast.element_field.material_field(
        model, [("E", 1.0), ("nu", 0.3), ("rho", rho)]
    )

    m = pyrucast.matrix.mass(model, materials)
    tol = 1e-12
    assert abs(m.get(n[0], "f_x", n[0], "u_x") - rho * 4.0 / 36.0) < tol
    assert abs(m.get(n[0], "f_x", n[1], "u_x") - rho * 2.0 / 36.0) < tol
    assert abs(m.get(n[0], "f_x", n[2], "u_x") - rho * 1.0 / 36.0) < tol
    # Block-diagonal in the components (no u_y ↔ f_x coupling).
    assert abs(m.get(n[0], "f_x", n[0], "u_y")) < tol


def test_heat_capacity_of_unit_quad():
    rho, cp = 3.0, 5.0
    fes, n = _unit_quad()
    model = pyrucast.model.heat_conduction(fes)
    materials = pyrucast.element_field.material_field(
        model, [("k", 1.0), ("rho", rho), ("cp", cp)]
    )

    c = pyrucast.matrix.mass(model, materials)
    rc = rho * cp
    tol = 1e-12
    assert abs(c.get(n[0], "q", n[0], "T") - rc * 4.0 / 36.0) < tol
    assert abs(c.get(n[0], "q", n[1], "T") - rc * 2.0 / 36.0) < tol


def test_lumped_mass_is_diagonal():
    rho = 2.0
    fes, n = _unit_quad()
    model = pyrucast.model.elasticity(fes, "plane_stress")
    materials = pyrucast.element_field.material_field(
        model, [("E", 1.0), ("nu", 0.3), ("rho", rho)]
    )

    m = pyrucast.matrix.mass(model, materials)
    lumped = pyrucast.matrix.lump(m)
    tol = 1e-12
    # Diagonal = consistent-mass row sum = ρ/4; off-diagonals vanish.
    assert abs(lumped.get(n[0], "f_x", n[0], "u_x") - rho / 4.0) < tol
    assert abs(lumped.get(n[0], "f_x", n[1], "u_x")) < tol


def test_mass_requires_density():
    fes, _ = _unit_quad()
    model = pyrucast.model.elasticity(fes, "plane_stress")
    materials = pyrucast.element_field.material_field(model, [("E", 1.0), ("nu", 0.3)])
    try:
        pyrucast.matrix.mass(model, materials)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("expected mass() to require `rho`")
