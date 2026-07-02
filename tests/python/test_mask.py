"""Python tests for `mask` (per-component 0/1 field) and the comparison
sugar (`field >= x`, `> x`, `<= x`, `< x`) that builds masks."""

import pytest

import pyrucast


def _poi1_field(values, components=("T",)):
    """Single-zone POI1 NodeField over `len(values)` 1-D nodes, one row of
    component values per node (`values[i]` is the row for node i)."""
    c = pyrucast.Coords(1)
    nodes = [c.add_node([float(i)]) for i in range(len(values))]
    mesh = pyrucast.Mesh(c, "POI1")
    for n in nodes:
        mesh.unit().add_cell([n])
    f = pyrucast.NodeField(mesh, list(components))
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            f[0].set(i, j, v)
    return f, nodes


def _values(field):
    """Flat 0/1 values of the (single) zone, node → component order."""
    sub = field[0]
    nc = sub.component_count()
    return [sub.get(i, j) for i in range(sub.node_count()) for j in range(nc)]


# ─── mask() ──────────────────────────────────────────────────────────────────


def test_mask_keeps_structure_and_flags_band():
    f, _ = _poi1_field([[0.0], [10.0], [20.0], [30.0], [40.0]])
    m = pyrucast.mask(f, ge=10.0, le=30.0)
    assert isinstance(m, pyrucast.NodeField)
    assert len(m) == len(f)
    assert m.components() == ["T"]
    assert _values(m) == [0.0, 1.0, 1.0, 1.0, 0.0]


def test_mask_strict_vs_inclusive():
    f, _ = _poi1_field([[0.0], [5.0], [9.0]])
    assert _values(pyrucast.mask(f, gt=5.0)) == [0.0, 0.0, 1.0]  # 5 excluded
    assert _values(pyrucast.mask(f, ge=5.0)) == [0.0, 1.0, 1.0]  # 5 included
    assert _values(pyrucast.mask(f, lt=5.0)) == [1.0, 0.0, 0.0]  # 5 excluded
    assert _values(pyrucast.mask(f, le=5.0)) == [1.0, 1.0, 0.0]  # 5 included


def test_mask_is_per_component_no_and():
    # node0: U=1 V=9 ; node1: U=9 V=1 — each component independent.
    f, _ = _poi1_field([[1.0, 9.0], [9.0, 1.0]], components=("U", "V"))
    m = pyrucast.mask(f, ge=0.0, le=5.0)
    assert _values(m) == [1.0, 0.0, 0.0, 1.0]


def test_mask_component_filter_leaves_others_identity():
    f, _ = _poi1_field([[9.0, 9.0], [1.0, 1.0]], components=("U", "V"))
    m = pyrucast.mask(f, ge=0.0, le=5.0, components=["U"])
    # V stays 1.0 (identity), only U gets a real 0/1.
    assert _values(m) == [0.0, 1.0, 1.0, 1.0]


def test_mask_multiplies_to_zero_out_of_band():
    f, _ = _poi1_field([[-2.0], [3.0], [7.0]])
    kept = f * pyrucast.mask(f, ge=0.0)  # zero the negatives
    assert _values(kept) == [0.0, 3.0, 7.0]


def test_mask_requires_a_bound():
    f, _ = _poi1_field([[1.0]])
    with pytest.raises(RuntimeError):
        pyrucast.mask(f)


def test_mask_rejects_two_lower_bounds():
    f, _ = _poi1_field([[1.0]])
    with pytest.raises(RuntimeError):
        pyrucast.mask(f, ge=0.0, gt=0.0)


# ─── comparison sugar ────────────────────────────────────────────────────────


def test_comparison_operators_build_masks():
    f, _ = _poi1_field([[0.0], [5.0], [9.0]])
    assert _values(f >= 5.0) == [0.0, 1.0, 1.0]
    assert _values(f > 5.0) == [0.0, 0.0, 1.0]
    assert _values(f <= 5.0) == [1.0, 1.0, 0.0]
    assert _values(f < 5.0) == [1.0, 0.0, 0.0]
    assert isinstance(f >= 5.0, pyrucast.NodeField)


def test_comparison_equivalent_to_mask():
    f, _ = _poi1_field([[0.0], [5.0], [9.0]])
    assert _values(f > 5.0) == _values(pyrucast.mask(f, gt=5.0))


def test_equality_not_overridden():
    f, _ = _poi1_field([[1.0]])
    # `==` must keep Python identity semantics, not return a mask field.
    assert (f == f) is True
    assert (f == 5.0) is False
