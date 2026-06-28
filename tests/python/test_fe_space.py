"""Python tests for FiniteElementSpace + SubFiniteElementSpace (Phase 2 step 5)."""

import math

import pyrucast


# ─── Structural acceptance / rejection ──────────────────────────────────────


def test_lagrange1_constructor_one_to_one_with_submeshes():
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([0.0, 1.0])
    n3 = c.add_node([1.0, 1.0])

    tri = pyrucast.Mesh(c, "TRI3")
    tri.unit().add_cell([n0, n1, n2])
    qua = pyrucast.Mesh(c, "QUA4")
    qua.unit().add_cell([n0, n1, n3, n2])
    mesh = tri | qua

    fes = pyrucast.FiniteElementSpace(mesh)
    assert len(fes) == 2
    assert fes[0].element_type == "TRI3"
    assert fes[0].gauss_count() == 3
    assert fes[1].element_type == "QUA4"
    assert fes[1].gauss_count() == 4
    for sub in fes:
        assert sub.interpolation == "LAGRANGE1"
        assert sub.quadrature == "GAUSS"
        assert sub.space_dim == 2


def test_lagrange1_classmethod_equivalent():
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([n0, n1, n2])

    fes_a = pyrucast.FiniteElementSpace(mesh)
    fes_b = pyrucast.FiniteElementSpace.lagrange1(mesh)
    assert len(fes_a) == len(fes_b) == 1
    assert fes_a[0].interpolation == fes_b[0].interpolation
    assert fes_a[0].quadrature == fes_b[0].quadrature


def test_with_choices_explicit_per_submesh():
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([n0, n1, n2])

    fes = pyrucast.FiniteElementSpace.with_choices(mesh, [("LAGRANGE1", "GAUSS")])
    assert len(fes) == 1
    assert fes[0].interpolation == "LAGRANGE1"


def test_with_choices_rejects_bad_length():
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([n0, n1, n2])

    try:
        pyrucast.FiniteElementSpace.with_choices(mesh, [])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for empty choices")


def test_unknown_interpolation_raises():
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([n0, n1, n2])

    try:
        pyrucast.FiniteElementSpace(mesh, interpolation="BOGUS")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown interpolation")


def test_unknown_quadrature_raises():
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([n0, n1, n2])

    try:
        pyrucast.FiniteElementSpace(mesh, quadrature="BOGUS")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown quadrature")


def test_rejects_mesh_with_poi1_submesh():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    mesh = pyrucast.Mesh(c, "POI1")
    mesh.unit().add_cell([a])
    try:
        pyrucast.FiniteElementSpace(mesh)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for POI1 mesh")


def test_rejects_empty_mesh():
    c = pyrucast.Coords(2)
    mesh = pyrucast.Mesh(c)
    try:
        pyrucast.FiniteElementSpace(mesh)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for empty mesh")


# ─── Reference-space tables ─────────────────────────────────────────────────


def test_reference_tables_partition_of_unity():
    """At every Gauss point, Σ_i N_i = 1 (partition of unity)."""
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([n0, n1, n2])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    for g in range(sub.gauss_count()):
        ns = sub.n_at_g(g)
        assert abs(sum(ns) - 1.0) < 1e-12


def test_reference_tables_dn_dxi_sums_to_zero():
    """For each ref direction k, Σ_i ∂N_i/∂ξ_k = 0 (derivative of partition)."""
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([1.0, 1.0])
    n3 = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "QUA4")
    mesh.unit().add_cell([n0, n1, n2, n3])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    n_nodes = sub.nodes_per_cell
    rd = sub.ref_dim
    for g in range(sub.gauss_count()):
        dn = sub.dn_at_g(g)
        for k in range(rd):
            s = sum(dn[i * rd + k] for i in range(n_nodes))
            assert abs(s) < 1e-12


def test_gauss_weights_sum_to_reference_volume():
    """Σ_g w_g = volume of the reference element."""
    expected = {
        "SEG2": 2.0,
        "TRI3": 0.5,
        "QUA4": 4.0,
        "TET4": 1.0 / 6.0,
        "HEX8": 8.0,
    }
    for et, vol in expected.items():
        c = pyrucast.Coords(3 if et in ("TET4", "HEX8") else 2)
        # Build a minimal valid cell of this element type so the FE space
        # constructor has something to wrap.
        if et == "SEG2":
            nodes = [c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0])]
        elif et == "TRI3":
            nodes = [
                c.add_node([0.0, 0.0]),
                c.add_node([1.0, 0.0]),
                c.add_node([0.0, 1.0]),
            ]
        elif et == "QUA4":
            nodes = [
                c.add_node(p) for p in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            ]
        elif et == "TET4":
            nodes = [
                c.add_node(p)
                for p in [
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ]
            ]
        else:  # HEX8
            nodes = [
                c.add_node(list(p))
                for p in [
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (1.0, 0.0, 1.0),
                    (1.0, 1.0, 1.0),
                    (0.0, 1.0, 1.0),
                ]
            ]
        mesh = pyrucast.Mesh(c, et)
        mesh.unit().add_cell([n for n in nodes])
        fes = pyrucast.FiniteElementSpace(mesh)
        sub = fes[0]
        s = sum(sub.gauss_weight(g) for g in range(sub.gauss_count()))
        assert abs(s - vol) < 1e-12, f"{et}: weights sum {s} ≠ {vol}"


# ─── Jacobian + |J| + dN/dx on closed-form cases ────────────────────────────


def test_seg2_jacobian_1d():
    """SEG2 of length L on the x-axis: |J| = L/2 at every Gauss point."""
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([5.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    for g in range(sub.gauss_count()):
        assert abs(sub.det_jacobian(0, g) - 2.5) < 1e-12


def test_seg2_jacobian_in_plane():
    """SEG2 of length 3 in the xy-plane: J = [3/2, 0]^T, |J| = 3/2."""
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 1.0])
    b = c.add_node([3.0, 1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    for g in range(sub.gauss_count()):
        jac = sub.jacobian(0, g)
        assert len(jac) == 2
        assert abs(jac[0] - 1.5) < 1e-12
        assert abs(jac[1]) < 1e-12
        assert abs(sub.det_jacobian(0, g) - 1.5) < 1e-12


def test_tri3_jacobian_planar():
    """TRI3 (0,0), (a,0), (0,b): |J| = a·b (twice the area)."""
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([3.0, 0.0])
    n2 = c.add_node([0.0, 4.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([n0, n1, n2])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    for g in range(sub.gauss_count()):
        assert abs(sub.det_jacobian(0, g) - 12.0) < 1e-12


def test_tri3_manifold_in_3d():
    """Same triangle but in a 3-D Coords: |J| stays at 12 via
    sqrt(det(JᵀJ))."""
    c = pyrucast.Coords(3)
    n0 = c.add_node([0.0, 0.0, 7.0])
    n1 = c.add_node([3.0, 0.0, 7.0])
    n2 = c.add_node([0.0, 4.0, 7.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([n0, n1, n2])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    assert sub.space_dim == 3
    assert sub.ref_dim == 2
    for g in range(sub.gauss_count()):
        assert abs(sub.det_jacobian(0, g) - 12.0) < 1e-12


def test_qua4_unit_square_integrates_to_area():
    """Σ_g w_g · |J(g)| over a unit QUA4 must equal the physical area = 1."""
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([1.0, 1.0])
    n3 = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "QUA4")
    mesh.unit().add_cell([n0, n1, n2, n3])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    area = sum(
        sub.gauss_weight(g) * sub.det_jacobian(0, g) for g in range(sub.gauss_count())
    )
    assert abs(area - 1.0) < 1e-12


def test_hex8_unit_cube_integrates_to_volume():
    c = pyrucast.Coords(3)
    pts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    nodes = [c.add_node(list(p)) for p in pts]
    mesh = pyrucast.Mesh(c, "HEX8")
    mesh.unit().add_cell([n for n in nodes])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    vol = sum(
        sub.gauss_weight(g) * sub.det_jacobian(0, g) for g in range(sub.gauss_count())
    )
    assert abs(vol - 1.0) < 1e-12


def test_tri3_dn_dx_constant_known_values():
    """For the triangle (0,0), (3,0), (0,4) and Lagrange-1, dN/dx is
    constant: ∂N₁/∂x = -1/3, ∂N₁/∂y = -1/4, etc."""
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([3.0, 0.0])
    n2 = c.add_node([0.0, 4.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([n0, n1, n2])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    for g in range(sub.gauss_count()):
        dn = sub.dn_dx(0, g)
        assert abs(dn[0] - (-1.0 / 3.0)) < 1e-12  # ∂N₁/∂x
        assert abs(dn[1] - (-1.0 / 4.0)) < 1e-12  # ∂N₁/∂y
        assert abs(dn[2] - (1.0 / 3.0)) < 1e-12  # ∂N₂/∂x
        assert abs(dn[3]) < 1e-12  # ∂N₂/∂y
        assert abs(dn[4]) < 1e-12  # ∂N₃/∂x
        assert abs(dn[5] - (1.0 / 4.0)) < 1e-12  # ∂N₃/∂y


# ─── On-the-fly: mesh displacement ──────────────────────────────────────────


def test_jacobian_reflects_mesh_displacement():
    """After set_coord on a node, the on-the-fly Jacobian must update."""
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]

    # Initial length = 1 ⇒ |J| = 1/2 over [-1, +1].
    assert abs(sub.det_jacobian(0, 0) - 0.5) < 1e-12

    # Move b from x=1 to x=4: length now 4, |J| = 2.
    b.set_coord([4.0, 0.0])
    assert abs(sub.det_jacobian(0, 0) - 2.0) < 1e-12


# ─── Iteration & error handling ─────────────────────────────────────────────


def test_index_out_of_range():
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([n0, n1, n2])
    fes = pyrucast.FiniteElementSpace(mesh)
    try:
        _ = fes[5]
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError for out-of-range subspace")


def test_negative_indexing_works():
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([n0, n1, n2])
    fes = pyrucast.FiniteElementSpace(mesh)
    assert fes[-1].element_type == "TRI3"


def test_gauss_index_out_of_range_raises():
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([n0, n1, n2])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    try:
        sub.det_jacobian(0, 999)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for out-of-range Gauss index")


def test_cell_index_out_of_range_raises():
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([n0, n1, n2])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    try:
        sub.jacobian(99, 0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for out-of-range cell")


# ─── repr / str ─────────────────────────────────────────────────────────────


def test_repr_and_str():
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([n0, n1, n2])
    fes = pyrucast.FiniteElementSpace(mesh)
    assert "FiniteElementSpace" in repr(fes)
    assert "1 subspace" in str(fes)
    sub = fes[0]
    assert "SubFiniteElementSpace" in repr(sub)
    s = str(sub)
    assert "TRI3" in s
    assert "LAGRANGE1" in s
    assert "GAUSS" in s


# ─── Aggregate addition (merge) ─────────────────────────────────────────────


def test_union_merges_subspaces():
    """`fes_a | fes_b` concatenates subspaces into a fresh space, like the
    other aggregates. No DOF check is performed."""
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([0.0, 1.0])
    n3 = c.add_node([1.0, 1.0])

    tri = pyrucast.Mesh(c, "TRI3")
    tri.unit().add_cell([n0, n1, n2])
    qua = pyrucast.Mesh(c, "QUA4")
    qua.unit().add_cell([n0, n1, n3, n2])

    fes_a = pyrucast.FiniteElementSpace(tri)
    fes_b = pyrucast.FiniteElementSpace(qua)
    merged = fes_a | fes_b

    assert len(merged) == 2
    assert merged[0].element_type == "TRI3"
    assert merged[1].element_type == "QUA4"
    # Operands are left untouched (fresh aggregate, first-seen order).
    assert len(fes_a) == 1
    assert len(fes_b) == 1
