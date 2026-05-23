"""Python tests for the Matrix container (Phase 2 step 8 — data layer)."""

import pyrucast


# ─── Construction ───────────────────────────────────────────────────────────


def test_empty_matrix():
    m = pyrucast.Matrix()
    assert m.n_rows() == 0
    assert m.n_cols() == 0
    assert m.entry_count() == 0
    assert m.symmetric is False


def test_symmetric_flag():
    m = pyrucast.Matrix(symmetric=True)
    assert m.symmetric is True


# ─── add_entry / get ────────────────────────────────────────────────────────


def test_add_entry_and_get():
    m = pyrucast.Matrix(symmetric=True)
    # Heat conduction on 2 nodes: rows = "q", cols = "T".
    m.add_entry(0, "q", 0, "T", 2.0)
    m.add_entry(0, "q", 1, "T", -1.0)
    m.add_entry(1, "q", 0, "T", -1.0)
    m.add_entry(1, "q", 1, "T", 2.0)
    assert m.n_rows() == 2
    assert m.n_cols() == 2
    assert m.entry_count() == 4
    assert m.get(0, "q", 0, "T") == 2.0
    assert m.get(0, "q", 1, "T") == -1.0
    assert m.get(1, "q", 0, "T") == -1.0
    assert m.get(1, "q", 1, "T") == 2.0


def test_get_unknown_returns_zero():
    m = pyrucast.Matrix()
    assert m.get(0, "x", 0, "y") == 0.0


def test_repeated_entries_sum():
    m = pyrucast.Matrix()
    m.add_entry(0, "q", 0, "T", 2.0)
    m.add_entry(0, "q", 0, "T", 1.5)
    m.add_entry(0, "q", 0, "T", -0.5)
    assert m.get(0, "q", 0, "T") == 3.0
    assert m.entry_count() == 3  # stored as-is in COO


# ─── Dense view ─────────────────────────────────────────────────────────────


def test_dense_layout_is_row_major():
    m = pyrucast.Matrix()
    m.add_entry(0, "q", 0, "T", 2.0)
    m.add_entry(0, "q", 1, "T", -1.0)
    m.add_entry(1, "q", 0, "T", -1.0)
    m.add_entry(1, "q", 1, "T", 2.0)
    assert m.dense() == [2.0, -1.0, -1.0, 2.0]


# ─── Matrix-vector ──────────────────────────────────────────────────────────


def test_mul_dense_against_known_matrix():
    m = pyrucast.Matrix(symmetric=True)
    m.add_entry(0, "q", 0, "T", 2.0)
    m.add_entry(0, "q", 1, "T", -1.0)
    m.add_entry(1, "q", 0, "T", -1.0)
    m.add_entry(1, "q", 1, "T", 2.0)
    assert m.mul_dense([1.0, 1.0]) == [1.0, 1.0]
    assert m.mul_dense([1.0, 2.0]) == [0.0, 3.0]


def test_mul_dense_rejects_wrong_size():
    m = pyrucast.Matrix()
    m.add_entry(0, "q", 0, "T", 1.0)
    try:
        m.mul_dense([1.0, 2.0])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for wrong x size")


# ─── DOF introspection ──────────────────────────────────────────────────────


def test_row_and_col_dofs_have_distinct_field_names():
    m = pyrucast.Matrix()
    m.add_entry(0, "q", 0, "T", 1.0)
    m.add_entry(1, "q", 1, "T", 2.0)
    rows = m.row_dofs()
    cols = m.col_dofs()
    assert rows == [(0, "q"), (1, "q")]
    assert cols == [(0, "T"), (1, "T")]
    assert m.field_names() == ["q", "T"]


def test_rectangular_matrix_with_lagrange_pattern():
    # Lagrange-multiplier block: row at multiplier node, col at constrained
    # real node. Rectangular structure (2 multiplier rows × 2 real cols).
    m = pyrucast.Matrix()
    m.add_entry(100, "T", 3, "T", 1.0)  # multiplier 100 constrains real 3
    m.add_entry(101, "T", 7, "T", 1.0)
    assert m.n_rows() == 2
    assert m.n_cols() == 2
    rows = m.row_dofs()
    cols = m.col_dofs()
    assert rows[0] == (100, "T")
    assert cols[0] == (3, "T")
    # "T" is interned once even though it appears on both rows and cols.
    assert m.field_names() == ["T"]


def test_entries_introspection_preserves_insertion_order():
    m = pyrucast.Matrix()
    m.add_entry(0, "a", 0, "b", 1.0)
    m.add_entry(1, "a", 1, "b", 2.0)
    m.add_entry(0, "a", 0, "b", 3.0)
    entries = m.entries()
    assert len(entries) == 3
    assert entries[0] == (0, "a", 0, "b", 1.0)
    assert entries[1] == (1, "a", 1, "b", 2.0)
    assert entries[2] == (0, "a", 0, "b", 3.0)
    assert len(m) == 3  # __len__ alias


# ─── repr / str ─────────────────────────────────────────────────────────────


def test_repr_and_str():
    m = pyrucast.Matrix(symmetric=True)
    m.add_entry(0, "q", 0, "T", 2.0)
    assert "Matrix" in repr(m)
    assert "symmetric" in repr(m)
    s = str(m)
    assert "Matrix" in s
    assert "1 row" in s
    assert "symmetric" in s
