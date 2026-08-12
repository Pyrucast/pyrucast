"""Python test for the consistent (algorithmic) tangent (KTAN).

Checks that in the elastic regime the tangent equals the elastic stiffness, and
that in a plastic regime it is symmetric and softened (the plastic diagonal is
below the elastic one).
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
    return c, n, pyrucast.FiniteElementSpace(mesh)


def _uniform_strain(c, nodes, exx):
    u_mesh = pyrucast.Mesh(c, "POI1")
    for node in nodes:
        u_mesh.unit().add_cell([node])
    u = pyrucast.NodeField(u_mesh, ["u_x", "u_y"])
    xs = [0.0, 1.0, 1.0, 0.0]
    for node, x in zip(nodes, xs):
        u[0].set_value(node, "u_x", exx * x)
        u[0].set_value(node, "u_y", 0.0)
    return u


def _tangent_for(model, fes, materials, u):
    strain = pyrucast.element_field.deformation(u, fes)
    state = pyrucast.element_field.integrate_behavior(model, strain, materials)
    return pyrucast.matrix.tangent(model, materials, state)


def test_elastic_tangent_equals_stiffness():
    E, NU, SY = 70_000.0, 0.3, 200.0
    c, n, fes = _unit_quad()
    model = pyrucast.Model.plasticity_perfect(fes, "plane_strain")
    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("nu", NU), ("sigma_y", SY)]
    )

    kt = _tangent_for(model, fes, materials, _uniform_strain(c, n, 1e-5))
    k = pyrucast.matrix.stiffness(model, materials)
    tol = 1e-6
    for i in range(4):
        for j in range(4):
            assert (
                abs(kt.get(n[i], "f_x", n[j], "u_x") - k.get(n[i], "f_x", n[j], "u_x"))
                < tol
            )


def test_plastic_tangent_symmetric_and_softened():
    E, NU, SY = 70_000.0, 0.3, 200.0
    c, n, fes = _unit_quad()
    model = pyrucast.Model.plasticity_perfect(fes, "plane_strain")
    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("nu", NU), ("sigma_y", SY)]
    )

    kt = _tangent_for(model, fes, materials, _uniform_strain(c, n, 2e-2))
    k = pyrucast.matrix.stiffness(model, materials)

    # Symmetric.
    for i in range(4):
        for j in range(4):
            a = kt.get(n[i], "f_x", n[j], "u_x")
            b = kt.get(n[j], "f_x", n[i], "u_x")
            assert abs(a - b) < 1e-9

    # Plastic softening: the tangent diagonal is below the elastic one.
    kt_d = kt.get(n[1], "f_x", n[1], "u_x")
    k_d = k.get(n[1], "f_x", n[1], "u_x")
    assert kt_d < k_d
