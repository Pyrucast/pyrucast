"""Python tests for the Matrix / SubMatrix containers."""

import pyrucast


def _poi1(c, ids):
    """Build a unit POI1 Mesh in `c` over the given node ids (used as a
    block support — Matrix.block accepts a unitary Mesh)."""
    mesh = pyrucast.Mesh(c, "POI1")
    for nid in ids:
        mesh.unit().add_cell([nid])
    return mesh


def _make_block(c, row_ids, col_ids, dual_vars, primal_vars, symmetric=False):
    """A single COO block, returned as the `SubMatrix` view of a unit
    `Matrix` (SubMatrix is no longer constructed directly — see
    CONVENTIONS.md). The view supports `add_entry` / `get` / `n_rows`."""
    return pyrucast.Matrix.block(
        _poi1(c, row_ids),
        _poi1(c, col_ids),
        dual_vars,
        primal_vars,
        ordering="nodes_then_vars",
        symmetric=symmetric,
    )[0]


# ─── SubMatrix — construction ───────────────────────────────────────────────


def test_empty_sub_matrix():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    m = _make_block(c, [a, b], [a, b], ["q"], ["T"])
    assert m.n_rows() == 2
    assert m.n_cols() == 2
    assert m.entry_count() == 0
    assert m.symmetric is False


def test_sub_matrix_symmetric_flag():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    m = _make_block(c, [a], [a], ["q"], ["T"], symmetric=True)
    assert m.symmetric is True


def test_sub_matrix_factor_defaults_to_one():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    m = _make_block(c, [a], [a], ["q"], ["T"])
    assert m.factor == 1.0


# ─── SubMatrix — add_entry / get ────────────────────────────────────────────


def test_sub_matrix_add_entry_and_get():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    m = _make_block(c, [a, b], [a, b], ["q"], ["T"], symmetric=True)
    m.add_entry(a, "q", a, "T", 2.0)
    m.add_entry(a, "q", b, "T", -1.0)
    m.add_entry(b, "q", a, "T", -1.0)
    m.add_entry(b, "q", b, "T", 2.0)
    assert m.n_rows() == 2
    assert m.n_cols() == 2
    assert m.entry_count() == 4
    assert m.get(a, "q", a, "T") == 2.0
    assert m.get(a, "q", b, "T") == -1.0
    assert m.get(b, "q", a, "T") == -1.0
    assert m.get(b, "q", b, "T") == 2.0


def test_sub_matrix_get_unknown_returns_zero():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    m = _make_block(c, [a], [a], ["q"], ["T"])
    assert m.get(a, "x", a, "y") == 0.0


def test_sub_matrix_repeated_entries_sum():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    m = _make_block(c, [a], [a], ["q"], ["T"])
    m.add_entry(a, "q", a, "T", 2.0)
    m.add_entry(a, "q", a, "T", 1.5)
    m.add_entry(a, "q", a, "T", -0.5)
    assert m.get(a, "q", a, "T") == 3.0
    assert m.entry_count() == 3


def test_sub_matrix_dense_layout_is_row_major():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    m = _make_block(c, [a, b], [a, b], ["q"], ["T"])
    m.add_entry(a, "q", a, "T", 2.0)
    m.add_entry(a, "q", b, "T", -1.0)
    m.add_entry(b, "q", a, "T", -1.0)
    m.add_entry(b, "q", b, "T", 2.0)
    assert m.dense() == [2.0, -1.0, -1.0, 2.0]


def test_sub_matrix_mul_dense_against_known_block():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    m = _make_block(c, [a, b], [a, b], ["q"], ["T"], symmetric=True)
    m.add_entry(a, "q", a, "T", 2.0)
    m.add_entry(a, "q", b, "T", -1.0)
    m.add_entry(b, "q", a, "T", -1.0)
    m.add_entry(b, "q", b, "T", 2.0)
    assert m.mul_dense([1.0, 1.0]) == [1.0, 1.0]
    assert m.mul_dense([1.0, 2.0]) == [0.0, 3.0]


def test_sub_matrix_mul_dense_rejects_wrong_size():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    m = _make_block(c, [a], [a], ["q"], ["T"])
    m.add_entry(a, "q", a, "T", 1.0)
    try:
        m.mul_dense([1.0, 2.0])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for wrong x size")


def test_sub_matrix_row_and_col_dofs():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    m = _make_block(c, [a, b], [a, b], ["q"], ["T"])
    m.add_entry(a, "q", a, "T", 1.0)
    m.add_entry(b, "q", b, "T", 2.0)
    assert m.row_dofs() == [(a.id, "q"), (b.id, "q")]
    assert m.col_dofs() == [(a.id, "T"), (b.id, "T")]
    assert m.field_names() == ["q", "T"]


def test_sub_matrix_entries_preserves_insertion_order():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    m = _make_block(c, [a, b], [a, b], ["a"], ["b"])
    m.add_entry(a, "a", a, "b", 1.0)
    m.add_entry(b, "a", b, "b", 2.0)
    m.add_entry(a, "a", a, "b", 3.0)
    entries = m.entries()
    assert len(entries) == 3
    assert entries[0] == (a.id, "a", a.id, "b", 1.0)
    assert entries[1] == (b.id, "a", b.id, "b", 2.0)
    assert entries[2] == (a.id, "a", a.id, "b", 3.0)
    assert len(m) == 3


def test_sub_matrix_repr_and_str():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    m = _make_block(c, [a], [a], ["q"], ["T"], symmetric=True)
    m.add_entry(a, "q", a, "T", 2.0)
    assert "SubMatrix" in repr(m)
    assert "symmetric" in repr(m)
    s = str(m)
    assert "SubMatrix" in s
    assert "1 row" in s
    assert "symmetric" in s


# ─── Matrix aggregate ───────────────────────────────────────────────────────


def test_empty_matrix_aggregate():
    m = pyrucast.Matrix()
    assert m.n_rows() == 0
    assert m.n_cols() == 0
    assert m.entry_count() == 0
    assert m.symmetric is True  # vacuously
    assert len(m) == 0


def test_matrix_aggregates_two_blocks():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    # Block A: row a, cols (a, b).
    block_a = _make_block(c, [a], [a, b], ["q"], ["T"], symmetric=True)
    block_a.add_entry(a, "q", a, "T", 2.0)
    block_a.add_entry(a, "q", b, "T", -1.0)
    # Block B: row b, cols (a, b).
    block_b = _make_block(c, [b], [a, b], ["q"], ["T"], symmetric=True)
    block_b.add_entry(b, "q", a, "T", -1.0)
    block_b.add_entry(b, "q", b, "T", 2.0)

    k = pyrucast.Matrix()
    k.add_sub(block_a)
    k.add_sub(block_b)
    k.finalize()

    assert len(k) == 2
    assert len(k) == 2
    assert k.n_rows() == 2
    assert k.n_cols() == 2
    assert k.symmetric is True
    assert k.get(a, "q", a, "T") == 2.0
    assert k.get(b, "q", b, "T") == 2.0
    assert k.dense() == [2.0, -1.0, -1.0, 2.0]
    assert k.mul_dense([1.0, 1.0]) == [1.0, 1.0]


def test_matrix_compose_blocks_with_union():
    """`Matrix.block(...) + Matrix.block(...)` composes blocks the
    parent-level way; entries are filled through the block view `[0]`."""
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    ba = pyrucast.Matrix.block(
        _poi1(c, [a]), _poi1(c, [a, b]), ["q"], ["T"], symmetric=True
    )
    ba[0].add_entry(a, "q", a, "T", 2.0)
    ba[0].add_entry(a, "q", b, "T", -1.0)
    bb = pyrucast.Matrix.block(
        _poi1(c, [b]), _poi1(c, [a, b]), ["q"], ["T"], symmetric=True
    )
    bb[0].add_entry(b, "q", a, "T", -1.0)
    bb[0].add_entry(b, "q", b, "T", 2.0)

    k = ba | bb
    k.finalize()
    assert len(k) == 2
    assert k.n_rows() == 2
    assert k.n_cols() == 2
    assert k.symmetric is True
    assert k.dense() == [2.0, -1.0, -1.0, 2.0]


def test_matrix_get_sums_across_blocks():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    block_a = _make_block(c, [a], [a], ["q"], ["T"])
    block_a.add_entry(a, "q", a, "T", 2.0)
    block_b = _make_block(c, [a], [a], ["q"], ["T"])
    block_b.add_entry(a, "q", a, "T", 0.5)
    k = pyrucast.Matrix()
    k.add_sub(block_a)
    k.add_sub(block_b)
    assert k.get(a, "q", a, "T") == 2.5


def test_matrix_symmetric_is_and_of_blocks():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    block_a = _make_block(c, [a], [a], ["q"], ["T"], symmetric=True)
    block_b = _make_block(c, [a], [a], ["q"], ["T"], symmetric=False)
    k = pyrucast.Matrix()
    k.add_sub(block_a)
    k.add_sub(block_b)
    assert k.symmetric is False


def test_matrix_entries_concatenates_blocks():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    block_a = _make_block(c, [a], [a], ["q"], ["T"])
    block_a.add_entry(a, "q", a, "T", 1.0)
    block_b = _make_block(c, [b], [b], ["q"], ["T"])
    block_b.add_entry(b, "q", b, "T", 2.0)
    k = pyrucast.Matrix()
    k.add_sub(block_a)
    k.add_sub(block_b)
    entries = k.entries()
    assert len(entries) == 2
    assert entries[0] == (a.id, "q", a.id, "T", 1.0)
    assert entries[1] == (b.id, "q", b.id, "T", 2.0)


def test_matrix_mul_field_returns_matrix_vector_product():
    """`matrix * NodeField` (unchanged despite `__mul__` now also accepting a
    `float` — the two are dispatched by `extract`)."""
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    block = _make_block(c, [a, b], [a, b], ["q"], ["T"], symmetric=True)
    block.add_entry(a, "q", a, "T", 2.0)
    block.add_entry(a, "q", b, "T", -1.0)
    block.add_entry(b, "q", a, "T", -1.0)
    block.add_entry(b, "q", b, "T", 2.0)
    k = pyrucast.Matrix()
    k.add_sub(block)
    k.finalize()

    x_mesh = pyrucast.Mesh(c, "POI1")
    x_mesh.unit().add_cell([a])
    x_mesh.unit().add_cell([b])
    x = pyrucast.NodeField(x_mesh, ["T"])
    x[0].set_value(a, "T", 1.0)
    x[0].set_value(b, "T", 1.0)

    y = k * x
    assert y.value(a, "q") == 1.0
    assert y.value(b, "q") == 1.0


def test_matrix_mul_and_truediv_scale_by_factor():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    block = _make_block(c, [a, b], [a, b], ["q"], ["T"], symmetric=True)
    block.add_entry(a, "q", a, "T", 2.0)
    block.add_entry(b, "q", b, "T", 4.0)
    k = pyrucast.Matrix()
    k.add_sub(block)

    scaled = k * 2.5
    assert scaled[0].factor == 2.5
    assert k[0].factor == 1.0, "k must be untouched by scaling"
    scaled.finalize()
    assert scaled.get(a, "q", a, "T") == 5.0
    assert scaled.get(b, "q", b, "T") == 10.0

    halved = scaled / 2.0
    assert halved[0].factor == 1.25
    assert scaled[0].factor == 2.5, "/ must not mutate its source either"
    halved.finalize()
    assert halved.get(a, "q", a, "T") == 2.5
    assert halved.get(b, "q", b, "T") == 5.0


def test_assemble_reassembles_scaled_mass_union_stiffness():
    """`M/dt + K` via `(mass / dt) | stiffness` then `sys.assemble()`
    — the dynamics idiom documented in book/src/matrix.md. `K` carries the
    Dirichlet multiplier DOF that `M` doesn't (a constraint only ever enters
    the stiffness matrix); the union/reassembly must still work and leave
    that DOF's entries untouched by `M`."""
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)

    imposed = pyrucast.mesh.poi1_from_nodes([a])
    mult_mesh = pyrucast.mesh.barycenter(imposed)
    mult = mult_mesh.node(0, 0, 0)
    conduction = pyrucast.model.heat_conduction(fes)
    dirichlet = pyrucast.model.dirichlet(conduction, "T", imposed, mult_mesh)
    model = pyrucast.model.heat_conduction(fes) | dirichlet
    materials = pyrucast.element_field.material_field(
        model, [("k", 1.0), ("rho", 2.0), ("cp", 3.0)]
    )

    k = pyrucast.matrix.stiffness(model, materials)
    m = pyrucast.matrix.mass(model, materials)

    dt = 0.5
    m_dt = m / dt
    sys = m_dt | k
    sys.assemble()

    tol = 1e-12
    # K_e = k/h·[[1,-1],[-1,1]] = [[1,-1],[-1,1]] (h=1); C_e = ρcp·h/6·[[2,1],[1,2]]
    # = [[2,1],[1,2]] (ρ=2, cp=3); M/dt = C/0.5 = [[4,2],[2,4]].
    assert abs(sys.get(a, "q", a, "T") - (1.0 + 4.0)) < tol
    assert abs(sys.get(a, "q", b, "T") - (-1.0 + 2.0)) < tol
    assert abs(sys.get(b, "q", a, "T") - (-1.0 + 2.0)) < tol
    assert abs(sys.get(b, "q", b, "T") - (1.0 + 4.0)) < tol
    # The multiplier row/col is K's alone — M never carries it.
    assert sys.get(mult, "T", a, "T") == k.get(mult, "T", a, "T")
    assert sys.get(a, "q", mult, "T") == k.get(a, "q", mult, "T")


def test_matrix_repr_and_str():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    block_a = _make_block(c, [a], [a], ["q"], ["T"], symmetric=True)
    block_a.add_entry(a, "q", a, "T", 2.0)
    k = pyrucast.Matrix()
    k.add_sub(block_a)
    assert "Matrix" in repr(k)
    s = str(k)
    assert "Matrix" in s
    assert "1 row" in s
    assert "symmetric" in s


# ─── dump() — third display level ───────────────────────────────────────────


def test_sub_matrix_dump_prints_labeled_grid(capsys):
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    block = _make_block(c, [a, b], [a, b], ["q"], ["T"])
    block.add_entry(a, "q", a, "T", 2.0)
    block.add_entry(a, "q", b, "T", -1.0)
    block.add_entry(b, "q", b, "T", 2.0)

    # dump() prints to stdout and returns nothing.
    assert block.dump() is None
    s = capsys.readouterr().out
    assert s.splitlines()[0].startswith("SubMatrix")
    # In-line DOF labels on both axes + values at default precision (3).
    assert f"({a.id},q)" in s
    assert f"({a.id},T)" in s
    assert "2.000" in s and "-1.000" in s

    # precision is honoured.
    block.dump(precision=1)
    assert "2.0" in capsys.readouterr().out


def test_matrix_dump_prints_global_grid_and_elides(capsys):
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    block_a = _make_block(c, [a], [a, b], ["q"], ["T"], symmetric=True)
    block_a.add_entry(a, "q", a, "T", 2.0)
    block_b = _make_block(c, [b], [a, b], ["q"], ["T"], symmetric=True)
    block_b.add_entry(b, "q", b, "T", 2.0)
    k = pyrucast.Matrix()
    k.add_sub(block_a)
    k.add_sub(block_b)

    # Global labelled grid, built without finalize(), printed to stdout.
    k.dump()
    s = capsys.readouterr().out
    assert s.splitlines()[0].startswith("Matrix")
    assert f"({a.id},T)" in s
    # max_rows elides the grid and notes the overflow.
    k.dump(max_rows=1)
    assert "de plus" in capsys.readouterr().out
