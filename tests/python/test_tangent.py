"""Python test for the consistent (algorithmic) tangent (KTAN).

Checks that in the elastic regime the tangent equals the elastic stiffness, and
that in a plastic regime it is symmetric and softened (the plastic diagonal is
below the elastic one).
"""

import pyrucast
import pytest


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
    return pyrucast.matrix.tangent(model, materials, strain)


def test_elastic_tangent_equals_stiffness():
    E, NU, SY = 70_000.0, 0.3, 200.0
    c, n, fes = _unit_quad()
    model = pyrucast.model.plasticity_perfect(fes, "plane_strain")
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
    model = pyrucast.model.plasticity_perfect(fes, "plane_strain")
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


def test_model_tangent_matrix_method_in_elastic_regime():
    # Below the yield stress the tangent is the plane-stress stiffness of the
    # unit QUA4, whose closed form is k₁₁ = E(3 − ν)/(6(1 − ν²)) and
    # k₁₂ = E/(8(1 − ν)).
    E, NU, SY = 70_000.0, 0.3, 200.0
    c, n, fes = _unit_quad()
    model = pyrucast.model.plasticity_perfect(fes, "plane_stress")
    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("nu", NU), ("sigma_y", SY)]
    )
    strain = pyrucast.element_field.deformation(_uniform_strain(c, n, 1e-5), fes)

    kt = model.tangent_matrix(materials, strain)
    assert kt.get(n[0], "f_x", n[0], "u_x") == pytest.approx(
        E * (3.0 - NU) / (6.0 * (1.0 - NU**2))
    )
    assert kt.get(n[0], "f_x", n[0], "u_y") == pytest.approx(E / (8.0 * (1.0 - NU)))


def test_linear_conduction_tangent_is_its_conductivity():
    # Conduction is linear: its tangent is the conductivity, closed form
    # k/6·[4, −1, −2, −1] on the unit QUA4 — and the deformation passed is not
    # read, since there is no law to differentiate.
    _, n, fes = _unit_quad()
    model = pyrucast.model.heat_conduction(fes)
    k = 3.0
    materials = pyrucast.element_field.material_field(model, [("k", k)])

    kt = model.tangent_matrix(materials, pyrucast.ElementField(fes, ["grad_T_x"]))
    for j, coef in enumerate([4.0, -1.0, -2.0, -1.0]):
        assert kt.get(n[0], "q", n[j], "T") == pytest.approx(k * coef / 6.0)
