"""Python tests for perfect von Mises plasticity (the COMP brick).

The Newton loop lives in Python (not tested here); these tests exercise the
Rust brick: build the model, the material, a strain field via `deformation`,
then `integrate_behavior` and check the point-wise response (yield plateau,
internal state).
"""

import math

import pyrucast


def _unit_quad():
    """One QUA4 unit square; returns (coords, nodes, fes)."""
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


def _uniform_strain(c, nodes, eps_xx):
    """Displacement u_x = eps_xx · x (others 0) over a POI1 of `nodes`."""
    u_mesh = pyrucast.Mesh(c, "POI1")
    for node in nodes:
        u_mesh.unit().add_cell([node])
    u = pyrucast.NodeField(u_mesh, ["u_x", "u_y"])
    coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    for node, (x, _y) in zip(nodes, coords):
        u[0].set_value(node, "u_x", eps_xx * x)
        u[0].set_value(node, "u_y", 0.0)
    return u


def _vm_plane_stress(sxx, syy, sxy):
    return math.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)


def test_plasticity_caps_at_yield_plane_stress():
    E, NU, SY = 210_000.0, 0.3, 250.0
    c, nodes, fes = _unit_quad()
    model = pyrucast.Model.plasticity(fes, "plane_stress")
    materials = pyrucast.material_field(model, [("E", E), ("nu", NU), ("sigma_y", SY)])

    # Strain well past yield.
    u = _uniform_strain(c, nodes, 1e-2)
    strain = pyrucast.deformation(u, fes)
    state = pyrucast.integrate_behavior(model, strain, materials)
    sub = state[0]
    # Output carries stress + plastic-strain state + cumulated p.
    assert "sigma_xx" in sub.components()
    assert "eps_p_xx" in sub.components()
    assert "p" in sub.components()
    for g in range(sub.gauss_count()):
        vm = _vm_plane_stress(
            sub.value(0, g, "sigma_xx"),
            sub.value(0, g, "sigma_yy"),
            sub.value(0, g, "sigma_xy"),
        )
        assert abs(vm - SY) < 1e-2, f"von Mises {vm} != {SY}"
        assert sub.value(0, g, "p") > 0.0


def test_plasticity_elastic_below_yield():
    E, NU, SY = 210_000.0, 0.3, 250.0
    c, nodes, fes = _unit_quad()
    model = pyrucast.Model.plasticity(fes, "plane_stress")
    materials = pyrucast.material_field(model, [("E", E), ("nu", NU), ("sigma_y", SY)])

    # ε small ⇒ σ ≈ E·ε (plane-stress uniaxial-strain) well under yield.
    u = _uniform_strain(c, nodes, 1e-4)
    strain = pyrucast.deformation(u, fes)
    state = pyrucast.integrate_behavior(model, strain, materials)
    sub = state[0]
    for g in range(sub.gauss_count()):
        assert abs(sub.value(0, g, "p")) < 1e-14


def test_plasticity_rejects_inconsistent_model():
    _c, _n, fes = _unit_quad()
    for bad in ("solid", "nonsense"):
        try:
            pyrucast.Model.plasticity(fes, bad)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f"expected an error for model={bad!r}")
