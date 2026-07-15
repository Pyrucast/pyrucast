"""Python test for the geometric (initial-stress) stiffness (KSIG).

Under a uniform uniaxial stress σ_xx = σ on the unit QUA4,
K_g[(i,a),(j,a)] = σ · ∫ ∂N_i/∂x ∂N_j/∂x, with ∫(∂N_0/∂x)² = 1/3.
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


def test_geometric_stiffness_uniaxial():
    sig = 3.0
    fes, n = _unit_quad()
    model = pyrucast.Model.elasticity(fes, "plane_stress")
    materials = pyrucast.build.material_field(model, [("E", 1.0), ("nu", 0.3)])

    stress = pyrucast.ElementField(fes, ["sigma_xx", "sigma_yy", "sigma_xy"])
    stress[0].set_uniform("sigma_xx", sig)
    stress[0].set_uniform("sigma_yy", 0.0)
    stress[0].set_uniform("sigma_xy", 0.0)

    kg = pyrucast.assemble.geometric(model, materials, stress)
    tol = 1e-12
    assert abs(kg.get(n[0], "f_x", n[0], "u_x") - sig / 3.0) < tol
    # Same scalar on the u_y diagonal block (δ_ab).
    assert abs(kg.get(n[0], "f_y", n[0], "u_y") - sig / 3.0) < tol
    assert abs(kg.get(n[0], "f_x", n[1], "u_x") + sig / 3.0) < tol
    # No cross-component coupling.
    assert abs(kg.get(n[0], "f_x", n[0], "u_y")) < tol
