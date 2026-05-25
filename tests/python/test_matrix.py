"""Python tests for the Matrix / SubMatrix containers."""

import pyrucast


# ─── SubMatrix — construction ───────────────────────────────────────────────


def test_empty_sub_matrix():
    m = pyrucast.SubMatrix()
    assert m.n_rows() == 0
    assert m.n_cols() == 0
    assert m.entry_count() == 0
    assert m.symmetric is False


def test_sub_matrix_symmetric_flag():
    m = pyrucast.SubMatrix(symmetric=True)
    assert m.symmetric is True


# ─── SubMatrix — add_entry / get ────────────────────────────────────────────


def test_sub_matrix_add_entry_and_get():
    m = pyrucast.SubMatrix(symmetric=True)
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


def test_sub_matrix_get_unknown_returns_zero():
    m = pyrucast.SubMatrix()
    assert m.get(0, "x", 0, "y") == 0.0


def test_sub_matrix_repeated_entries_sum():
    m = pyrucast.SubMatrix()
    m.add_entry(0, "q", 0, "T", 2.0)
    m.add_entry(0, "q", 0, "T", 1.5)
    m.add_entry(0, "q", 0, "T", -0.5)
    assert m.get(0, "q", 0, "T") == 3.0
    assert m.entry_count() == 3


def test_sub_matrix_dense_layout_is_row_major():
    m = pyrucast.SubMatrix()
    m.add_entry(0, "q", 0, "T", 2.0)
    m.add_entry(0, "q", 1, "T", -1.0)
    m.add_entry(1, "q", 0, "T", -1.0)
    m.add_entry(1, "q", 1, "T", 2.0)
    assert m.dense() == [2.0, -1.0, -1.0, 2.0]


def test_sub_matrix_mul_dense_against_known_block():
    m = pyrucast.SubMatrix(symmetric=True)
    m.add_entry(0, "q", 0, "T", 2.0)
    m.add_entry(0, "q", 1, "T", -1.0)
    m.add_entry(1, "q", 0, "T", -1.0)
    m.add_entry(1, "q", 1, "T", 2.0)
    assert m.mul_dense([1.0, 1.0]) == [1.0, 1.0]
    assert m.mul_dense([1.0, 2.0]) == [0.0, 3.0]


def test_sub_matrix_mul_dense_rejects_wrong_size():
    m = pyrucast.SubMatrix()
    m.add_entry(0, "q", 0, "T", 1.0)
    try:
        m.mul_dense([1.0, 2.0])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for wrong x size")


def test_sub_matrix_row_and_col_dofs():
    m = pyrucast.SubMatrix()
    m.add_entry(0, "q", 0, "T", 1.0)
    m.add_entry(1, "q", 1, "T", 2.0)
    assert m.row_dofs() == [(0, "q"), (1, "q")]
    assert m.col_dofs() == [(0, "T"), (1, "T")]
    assert m.field_names() == ["q", "T"]


def test_sub_matrix_entries_preserves_insertion_order():
    m = pyrucast.SubMatrix()
    m.add_entry(0, "a", 0, "b", 1.0)
    m.add_entry(1, "a", 1, "b", 2.0)
    m.add_entry(0, "a", 0, "b", 3.0)
    entries = m.entries()
    assert len(entries) == 3
    assert entries[0] == (0, "a", 0, "b", 1.0)
    assert entries[1] == (1, "a", 1, "b", 2.0)
    assert entries[2] == (0, "a", 0, "b", 3.0)
    assert len(m) == 3


def test_sub_matrix_repr_and_str():
    m = pyrucast.SubMatrix(symmetric=True)
    m.add_entry(0, "q", 0, "T", 2.0)
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
    a = pyrucast.SubMatrix(symmetric=True)
    a.add_entry(0, "q", 0, "T", 2.0)
    a.add_entry(0, "q", 1, "T", -1.0)
    b = pyrucast.SubMatrix(symmetric=True)
    b.add_entry(1, "q", 0, "T", -1.0)
    b.add_entry(1, "q", 1, "T", 2.0)

    k = pyrucast.Matrix()
    k.add_sub_matrix(a)
    k.add_sub_matrix(b)

    assert len(k) == 2
    assert k.sub_matrix_count() == 2
    assert k.n_rows() == 2
    assert k.n_cols() == 2
    assert k.symmetric is True
    assert k.get(0, "q", 0, "T") == 2.0
    assert k.get(1, "q", 1, "T") == 2.0
    assert k.dense() == [2.0, -1.0, -1.0, 2.0]
    assert k.mul_dense([1.0, 1.0]) == [1.0, 1.0]


def test_matrix_get_sums_across_blocks():
    a = pyrucast.SubMatrix()
    a.add_entry(0, "q", 0, "T", 2.0)
    b = pyrucast.SubMatrix()
    b.add_entry(0, "q", 0, "T", 0.5)
    k = pyrucast.Matrix()
    k.add_sub_matrix(a)
    k.add_sub_matrix(b)
    assert k.get(0, "q", 0, "T") == 2.5


def test_matrix_symmetric_is_and_of_blocks():
    a = pyrucast.SubMatrix(symmetric=True)
    b = pyrucast.SubMatrix(symmetric=False)
    k = pyrucast.Matrix()
    k.add_sub_matrix(a)
    k.add_sub_matrix(b)
    assert k.symmetric is False


def test_matrix_entries_concatenates_blocks():
    a = pyrucast.SubMatrix()
    a.add_entry(0, "q", 0, "T", 1.0)
    b = pyrucast.SubMatrix()
    b.add_entry(1, "q", 1, "T", 2.0)
    k = pyrucast.Matrix()
    k.add_sub_matrix(a)
    k.add_sub_matrix(b)
    entries = k.entries()
    assert len(entries) == 2
    assert entries[0] == (0, "q", 0, "T", 1.0)
    assert entries[1] == (1, "q", 1, "T", 2.0)


def test_matrix_repr_and_str():
    a = pyrucast.SubMatrix(symmetric=True)
    a.add_entry(0, "q", 0, "T", 2.0)
    k = pyrucast.Matrix()
    k.add_sub_matrix(a)
    assert "Matrix" in repr(k)
    s = str(k)
    assert "Matrix" in s
    assert "1 row" in s
    assert "symmetric" in s
