"""Python tests for node-to-surface contact (`model.contact`) solved by the
active-set operator `solve_unilateral`.

Patch test: two elastic blocks stacked in `y` with an initial gap, every `u_x`
blocked (uniaxial column, plane stress, ν = 0). Pressing the top block closes
the contact and transmits a uniform stress through the interface; lifting it
releases every pair (λ = 0).
"""

import pytest

import pyrucast

E = 100.0
S = 5.0  # applied pressure
G0 = 0.01  # initial gap between the blocks
N = 2  # N×N QUA4 grid per block
H = 1.0 / N
TOL = 1e-9


def _block(c, mesh, y0):
    """Add an N×N QUA4 block `[0,1] × [y0, y0+1]` as a new unit of `mesh`."""
    grid = [c.add_node([i * H, y0 + j * H]) for j in range(N + 1) for i in range(N + 1)]
    idx = lambda i, j: j * (N + 1) + i
    unit = mesh.unit()
    for j in range(N):
        for i in range(N):
            unit.add_cell(
                [
                    grid[idx(i, j)],
                    grid[idx(i + 1, j)],
                    grid[idx(i + 1, j + 1)],
                    grid[idx(i, j + 1)],
                ]
            )
    return grid


def _clamp(target, nodes, var):
    imposed = pyrucast.mesh.poi1_from_nodes(nodes)
    mult = pyrucast.mesh.barycenter(imposed)
    return pyrucast.model.dirichlet(target, var, imposed, mult)


def _two_blocks():
    """Returns (coords, bottom, top, model, contact, materials)."""
    c = pyrucast.Coords(2)
    mesh = pyrucast.Mesh(c, "QUA4")
    bottom = _block(c, mesh, 0.0)
    top = _block(c, mesh, 1.0 + G0)
    fes = pyrucast.FiniteElementSpace(mesh)
    idx = lambda i, j: j * (N + 1) + i

    # Master: top edge of the bottom block, run in −x so the normal points +y.
    master = pyrucast.Mesh(c, "SEG2")
    for i in reversed(range(N)):
        master.unit().add_cell([bottom[idx(i + 1, N)], bottom[idx(i, N)]])
    # Slave: bottom edge nodes of the top block.
    slave = pyrucast.mesh.poi1_from_nodes([top[idx(i, 0)] for i in range(N + 1)])

    model = pyrucast.model.elasticity(fes, "plane_stress")
    contact = pyrucast.model.contact(model, slave, master, ["u_x", "u_y"])
    model = model | _clamp(model, bottom + top, "u_x")
    model = model | _clamp(model, [bottom[idx(i, 0)] for i in range(N + 1)], "u_y")
    model = model | contact

    materials = pyrucast.element_field.material_field(model, [("E", E), ("nu", 0.0)])
    return c, bottom, top, model, contact, materials


def test_patch_test_uniform_pressure_through_contact():
    """Pressure S on top transmits σ_yy = −S exactly through the contact."""
    c, bottom, top, model, contact, materials = _two_blocks()
    idx = lambda i, j: j * (N + 1) + i

    top_edge = pyrucast.Mesh(c, "SEG2")
    for i in range(N):
        top_edge.unit().add_cell([top[idx(i, N)], top[idx(i + 1, N)]])
    edge_fes = pyrucast.FiniteElementSpace(top_edge)
    model = model | pyrucast.model.flux(edge_fes, model, "f_y")
    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("nu", 0.0), ("phi_f_y", -S)]
    )
    traction = pyrucast.node_field.external_forces(model, materials)
    rhs = traction | model.contact_gaps()

    k = pyrucast.matrix.stiffness(model, materials)
    solution = pyrucast.solver.solve_unilateral(k, model, rhs)

    for j in range(N + 1):
        for i in range(N + 1):
            y = j * H
            assert abs(solution.value(bottom[idx(i, j)], "u_y") + S / E * y) < TOL
            assert (
                abs(solution.value(top[idx(i, j)], "u_y") + S / E * (1.0 + y) + G0)
                < TOL
            )

    # Contact reactions −λᵢ ≥ 0 sum to the applied resultant S.
    mult_mesh = contact.multiplier_mesh()
    mults = [mult_mesh.node(0, r, 0) for r in range(N + 1)]
    lambdas = [solution.value(m, "lambda_contact") for m in mults]
    assert all(lam <= TOL for lam in lambdas)
    assert abs(sum(-lam for lam in lambdas) - S) < TOL


def test_separation_releases_every_pair():
    """Lifting the top block opens every pair: λ = 0, bottom block untouched."""
    c, bottom, top, model, contact, materials = _two_blocks()
    idx = lambda i, j: j * (N + 1) + i
    lift = 0.1

    top_edge_nodes = [top[idx(i, N)] for i in range(N + 1)]
    lift_model = _clamp(model, top_edge_nodes, "u_y")
    model = model | lift_model
    rhs = lift_model.constraint_rhs([(n, lift) for n in top_edge_nodes])
    rhs = rhs | model.contact_gaps()

    k = pyrucast.matrix.stiffness(model, materials)
    solution = pyrucast.solver.solve_unilateral(k, model, rhs)

    for j in range(N + 1):
        for i in range(N + 1):
            assert abs(solution.value(top[idx(i, j)], "u_y") - lift) < TOL
            assert abs(solution.value(bottom[idx(i, j)], "u_y")) < TOL
    mult_mesh = contact.multiplier_mesh()
    for r in range(N + 1):
        m = mult_mesh.node(0, r, 0)
        assert abs(solution.value(m, "lambda_contact")) < TOL


def test_contact_gaps_requires_a_contact():
    """`contact_gaps` on a contact-free model raises a clear error."""
    c = pyrucast.Coords(2)
    mesh = pyrucast.Mesh(c, "QUA4")
    _block(c, mesh, 0.0)
    fes = pyrucast.FiniteElementSpace(mesh)
    model = pyrucast.model.elasticity(fes, "plane_stress")
    with pytest.raises(Exception, match="contact"):
        model.contact_gaps()


def test_components_must_match_dimension():
    """One (variable, dual) pair per space dimension, in ambient order."""
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    s = c.add_node([0.5, 0.5])
    master = pyrucast.Mesh(c, "SEG2")
    master.unit().add_cell([a, b])
    slave = pyrucast.mesh.poi1_from_nodes([s])
    barre = pyrucast.model.truss(pyrucast.FiniteElementSpace(master))
    with pytest.raises(Exception, match="component"):
        pyrucast.model.contact(barre, slave, master, ["u_y"])
