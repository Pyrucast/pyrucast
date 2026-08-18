"""Python tests for the embedded (immersed) constraint (`Model.embedded`).

An embedded constraint ties each node of an immersed mesh to the interpolation
of a host mesh at that node, via Lagrange multipliers — a bar « baignée » in a
volume. On a single HEX8 host with linear heat conduction whose eight corners
are pinned to a linear field, the trilinear interpolation reproduces the field
in the interior, so the immersed node's solved value equals the field there.
"""

import pyrucast

TOL = 1e-9

CORNERS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
]


def _field(c):
    return 1.0 + 2.0 * c[0] + 3.0 * c[1] + 4.0 * c[2]


def test_immersed_node_follows_host_interpolation():
    c = pyrucast.Coords(3)
    corners = [c.add_node(x) for x in CORNERS]

    # HEX8 host with linear heat conduction (k = 1).
    host = pyrucast.Mesh(c, "HEX8")
    host.unit().add_cell(corners)
    fes = pyrucast.FiniteElementSpace(host)
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", 1.0)
    base = pyrucast.Model.heat_conduction(fes)

    # Dirichlet pinning all eight corners to the linear field.
    corner_mesh = pyrucast.mesh.poi1_from_nodes(corners)
    corner_mult = pyrucast.mesh.barycenter(corner_mesh)
    dirichlet = pyrucast.Model.dirichlet("T", "q", corner_mesh, corner_mult)

    # Immersed node inside the cube, tied to the host.
    p = c.add_node([0.3, 0.6, 0.2])
    bar = pyrucast.mesh.poi1_from_nodes([p])
    embedded = pyrucast.Model.embedded(bar, host, [("T", "q")])
    emb_mult = embedded.multiplier_mesh().node(0, 0, 0)

    model = base | dirichlet | embedded

    # RHS: field value at each Dirichlet multiplier, 0 (tie) at the embedded one.
    rhs_mesh = pyrucast.Mesh(c, "POI1")
    for i in range(len(corners)):
        rhs_mesh.unit().add_cell([corner_mult.node(0, i, 0)])
    rhs_mesh.unit().add_cell([emb_mult])
    rhs = pyrucast.NodeField(rhs_mesh, ["imposed_T"])
    for i, x in enumerate(CORNERS):
        rhs[0].set_value(corner_mult.node(0, i, 0), "imposed_T", _field(x))
    rhs[0].set_value(emb_mult, "imposed_T", 0.0)

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)

    for node, x in zip(corners, CORNERS):
        assert abs(solution.value(node, "T") - _field(x)) < TOL
    # 1 + 2*0.3 + 3*0.6 + 4*0.2 = 4.2
    assert abs(solution.value(p, "T") - _field([0.3, 0.6, 0.2])) < TOL


def test_node_outside_host_is_rejected():
    """An immersed node outside the host mesh is an error at build time."""
    c = pyrucast.Coords(3)
    corners = [c.add_node(x) for x in CORNERS]
    host = pyrucast.Mesh(c, "HEX8")
    host.unit().add_cell(corners)

    outside = c.add_node([5.0, 5.0, 5.0])
    bar = pyrucast.mesh.poi1_from_nodes([outside])
    try:
        pyrucast.Model.embedded(bar, host, [("T", "q")])
        assert False, "expected an error for a node outside the host"
    except Exception:
        pass
