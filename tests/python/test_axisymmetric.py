"""Python tests for axisymmetric (body-of-revolution) computations.

The geometry is the meridian plane of `Coords.axisymmetric()` — `x = r`,
`y = z`. It alone carries the `2πr` integration measure; the hoop strain
`ε_θθ = u_r / r` comes from the `"axisymmetric"` elasticity model.
"""

import math

import pyrucast


def _annulus(r0, r1, h, nr, nz):
    """QUA4 grid over the meridian rectangle r ∈ [r0, r1], z ∈ [0, h]."""
    c = pyrucast.Coords.axisymmetric()

    def idx(i, j):
        return j * (nr + 1) + i

    grid = [
        c.add_node([r0 + (r1 - r0) * i / nr, h * j / nz])
        for j in range(nz + 1)
        for i in range(nr + 1)
    ]
    mesh = pyrucast.Mesh(c, "QUA4")
    for j in range(nz):
        for i in range(nr):
            mesh.unit().add_cell(
                [
                    grid[idx(i, j)],
                    grid[idx(i + 1, j)],
                    grid[idx(i + 1, j + 1)],
                    grid[idx(i, j + 1)],
                ]
            )
    return c, grid, mesh, pyrucast.FiniteElementSpace(mesh), idx


def _clamp(target, nodes, var):
    imposed = pyrucast.mesh.poi1_from_nodes(nodes)
    multiplier = pyrucast.mesh.barycenter(imposed)
    return pyrucast.model.dirichlet(target, var, imposed, multiplier)


def test_coords_declare_the_revolution_frame():
    c = pyrucast.Coords.axisymmetric()
    assert c.dim == 2
    assert c.is_axisymmetric
    assert not pyrucast.Coords(2).is_axisymmetric
    # x is a radius: negative values are refused.
    try:
        c.add_node([-1.0, 0.0])
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("expected a negative radius to be refused")


def test_integral_measures_the_revolved_volume():
    """∫ 1 dΩ over the meridian rectangle is π(r1² − r0²)·h, not the plane area."""
    c, grid, mesh, fes, _ = _annulus(1.0, 3.0, 2.0, 4, 2)
    ones = pyrucast.NodeField(mesh, ["one"])
    ones.add_to_component("one", 1.0)

    volume = pyrucast.measure.integral(ones, "one", fes)
    assert abs(volume - math.pi * (3.0**2 - 1.0) * 2.0) < 1e-9


def test_lame_thick_cylinder_under_internal_pressure():
    """Thick cylinder under internal pressure, against the Lamé solution."""
    E, NU, P = 210_000.0, 0.3, 100.0
    A, B, H, NR, NZ = 1.0, 2.0, 0.5, 40, 1

    c, grid, mesh, fes, idx = _annulus(A, B, H, NR, NZ)

    # Plane strain: u_z = 0 on both z faces.
    ends = [grid[idx(i, j)] for i in range(NR + 1) for j in (0, NZ)]
    model = pyrucast.model.elasticity(fes, "axisymmetric")
    model = model | _clamp(model, ends, "u_y")
    # Internal pressure on r = a. The geometry being axisymmetric, the load
    # integrates ∫ 2πr N p — the true ring force, with no manual factor.
    inner = pyrucast.Mesh(c, "SEG2")
    for j in range(NZ):
        inner.unit().add_cell([grid[idx(0, j)], grid[idx(0, j + 1)]])
    inner_fes = pyrucast.FiniteElementSpace(inner)
    model = model | pyrucast.model.flux(inner_fes, model, "f_x")
    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("nu", NU), ("phi_f_x", P)]
    )
    rhs = pyrucast.node_field.external_forces(model, materials)

    K = pyrucast.matrix.stiffness(model, materials)
    solution = pyrucast.solver.solve(K, rhs)

    a2, b2 = A * A, B * B
    ca = P * a2 / (b2 - a2)
    cb = P * a2 * b2 / (b2 - a2)

    for i in range(NR + 1):
        n = grid[idx(i, 0)]
        r = A + (B - A) * i / NR
        exact = (1.0 + NU) / E * ((1.0 - 2.0 * NU) * ca * r + cb / r)
        assert abs(solution.value(n, "u_x") - exact) / exact < 5e-3
        assert abs(solution.value(n, "u_y")) < 1e-12


def test_model_and_geometry_must_agree():
    """A plane model on a body of revolution (or the reverse) is refused."""
    _c, _grid, _mesh, axi, _idx = _annulus(1.0, 2.0, 1.0, 1, 1)
    for bad in ("plane_strain", "plane_stress"):
        try:
            pyrucast.model.elasticity(axi, bad)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f"expected {bad!r} to be refused on a revolved body")

    c = pyrucast.Coords(2)
    a, b, d = c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0]), c.add_node([0.0, 1.0])
    plane = pyrucast.Mesh(c, "TRI3")
    plane.unit().add_cell([a, b, d])
    try:
        pyrucast.model.elasticity(pyrucast.FiniteElementSpace(plane), "axisymmetric")
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("expected 'axisymmetric' to need an axisymmetric Coords")
