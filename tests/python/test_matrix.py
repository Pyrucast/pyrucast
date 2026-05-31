"""Python tests for the Matrix / SubMatrix containers."""

import pyrucast


def _poi1(c, ids):
    """Build a POI1 SubMesh in `c` over the given node ids."""
    sm = pyrucast.SubMesh(c, "POI1")
    for nid in ids:
        sm.add_cell([nid])
    return sm


def _make_block(c, row_ids, col_ids, dual_vars, primal_vars, symmetric=False):
    return pyrucast.SubMatrix(
        _poi1(c, row_ids), _poi1(c, col_ids),
        dual_vars, primal_vars,
        ordering="nodes_then_vars",
        symmetric=symmetric,
    )


# ─── SubMatrix — construction ───────────────────────────────────────────────


def test_empty_sub_matrix():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    m = _make_block(c, [a, b], [a, b], ["q"], ["T"])
    assert m.n_rows() == 2
    assert m.n_cols() == 2
    assert m.entry_count() == 0
    assert m.symmetric is False


def test_sub_matrix_symmetric_flag():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    m = _make_block(c, [a], [a], ["q"], ["T"], symmetric=True)
    assert m.symmetric is True


# ─── SubMatrix — add_entry / get ────────────────────────────────────────────


def test_sub_matrix_add_entry_and_get():
    c = pyrucast.Configuration(1)
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
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    m = _make_block(c, [a], [a], ["q"], ["T"])
    assert m.get(a, "x", a, "y") == 0.0


def test_sub_matrix_repeated_entries_sum():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    m = _make_block(c, [a], [a], ["q"], ["T"])
    m.add_entry(a, "q", a, "T", 2.0)
    m.add_entry(a, "q", a, "T", 1.5)
    m.add_entry(a, "q", a, "T", -0.5)
    assert m.get(a, "q", a, "T") == 3.0
    assert m.entry_count() == 3


def test_sub_matrix_dense_layout_is_row_major():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    m = _make_block(c, [a, b], [a, b], ["q"], ["T"])
    m.add_entry(a, "q", a, "T", 2.0)
    m.add_entry(a, "q", b, "T", -1.0)
    m.add_entry(b, "q", a, "T", -1.0)
    m.add_entry(b, "q", b, "T", 2.0)
    assert m.dense() == [2.0, -1.0, -1.0, 2.0]


def test_sub_matrix_mul_dense_against_known_block():
    c = pyrucast.Configuration(1)
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
    c = pyrucast.Configuration(1)
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
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    m = _make_block(c, [a, b], [a, b], ["q"], ["T"])
    m.add_entry(a, "q", a, "T", 1.0)
    m.add_entry(b, "q", b, "T", 2.0)
    assert m.row_dofs() == [(a.id, "q"), (b.id, "q")]
    assert m.col_dofs() == [(a.id, "T"), (b.id, "T")]
    assert m.field_names() == ["q", "T"]


def test_sub_matrix_entries_preserves_insertion_order():
    c = pyrucast.Configuration(1)
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
    c = pyrucast.Configuration(1)
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
    c = pyrucast.Configuration(1)
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
    k.add_sub_matrix(block_a)
    k.add_sub_matrix(block_b)
    k.finalize()

    assert len(k) == 2
    assert k.sub_matrix_count() == 2
    assert k.n_rows() == 2
    assert k.n_cols() == 2
    assert k.symmetric is True
    assert k.get(a, "q", a, "T") == 2.0
    assert k.get(b, "q", b, "T") == 2.0
    assert k.dense() == [2.0, -1.0, -1.0, 2.0]
    assert k.mul_dense([1.0, 1.0]) == [1.0, 1.0]


def test_matrix_get_sums_across_blocks():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    block_a = _make_block(c, [a], [a], ["q"], ["T"])
    block_a.add_entry(a, "q", a, "T", 2.0)
    block_b = _make_block(c, [a], [a], ["q"], ["T"])
    block_b.add_entry(a, "q", a, "T", 0.5)
    k = pyrucast.Matrix()
    k.add_sub_matrix(block_a)
    k.add_sub_matrix(block_b)
    assert k.get(a, "q", a, "T") == 2.5


def test_matrix_symmetric_is_and_of_blocks():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    block_a = _make_block(c, [a], [a], ["q"], ["T"], symmetric=True)
    block_b = _make_block(c, [a], [a], ["q"], ["T"], symmetric=False)
    k = pyrucast.Matrix()
    k.add_sub_matrix(block_a)
    k.add_sub_matrix(block_b)
    assert k.symmetric is False


def test_matrix_entries_concatenates_blocks():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    block_a = _make_block(c, [a], [a], ["q"], ["T"])
    block_a.add_entry(a, "q", a, "T", 1.0)
    block_b = _make_block(c, [b], [b], ["q"], ["T"])
    block_b.add_entry(b, "q", b, "T", 2.0)
    k = pyrucast.Matrix()
    k.add_sub_matrix(block_a)
    k.add_sub_matrix(block_b)
    entries = k.entries()
    assert len(entries) == 2
    assert entries[0] == (a.id, "q", a.id, "T", 1.0)
    assert entries[1] == (b.id, "q", b.id, "T", 2.0)


def test_matrix_repr_and_str():
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    block_a = _make_block(c, [a], [a], ["q"], ["T"], symmetric=True)
    block_a.add_entry(a, "q", a, "T", 2.0)
    k = pyrucast.Matrix()
    k.add_sub_matrix(block_a)
    assert "Matrix" in repr(k)
    s = str(k)
    assert "Matrix" in s
    assert "1 row" in s
    assert "symmetric" in s
