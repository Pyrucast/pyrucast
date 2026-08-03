"""Python tests for `filter_components` (component extraction, Cast3M `EXCO`)
and `rename_component` (metadata-only rename)."""

import pytest

import pyrucast


def _poi1_field(values, components):
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


# ─── filter_components() ─────────────────────────────────────────────────────


def test_filter_single_name_keeps_only_it():
    f, _ = _poi1_field([[1.0, 2.0, 3.0]], components=("U", "V", "W"))
    g = f.filter_components("V")
    assert isinstance(g, pyrucast.NodeField)
    assert g.components() == ["V"]
    assert g[0].get(0, 0) == 2.0


def test_filter_list_keeps_subset_in_field_order():
    f, _ = _poi1_field([[1.0, 2.0, 3.0]], components=("U", "V", "W"))
    # Request order does not matter — the field's own order is kept.
    g = f.filter_components(["W", "U"])
    assert g.components() == ["U", "W"]
    assert g[0].get(0, 0) == 1.0
    assert g[0].get(0, 1) == 3.0


def test_filter_accepts_superset_like_primal_vars():
    # A solve-result-like field: primal [u_x, u_y] plus a dual [lambda].
    f, _ = _poi1_field([[1.0, 2.0, 9.0]], components=("u_x", "u_y", "lambda"))
    # Passing a list that names components the field lacks (as primal_vars may)
    # is fine — the extras are ignored, the dual is stripped.
    g = f.filter_components(["u_x", "u_y", "T"])
    assert g.components() == ["u_x", "u_y"]


def test_filter_no_op_returns_equivalent_field():
    # Zone already carries only requested components (extras in the request are
    # ignored): the filter is a no-op and returns the same components/values.
    # The handle-sharing optimization is checked at the Rust level.
    f, _ = _poi1_field([[1.0, 2.0]], components=("u_x", "u_y"))
    g = f.filter_components(["u_x", "u_y", "lambda"])
    assert g.components() == ["u_x", "u_y"]
    assert g[0].get(0, 0) == 1.0
    assert g[0].get(0, 1) == 2.0


def test_filter_none_present_errors():
    f, _ = _poi1_field([[1.0]], components=("T",))
    with pytest.raises(RuntimeError):
        f.filter_components(["nope"])


# ─── rename_component() ──────────────────────────────────────────────────────


def test_rename_preserves_values():
    f, _ = _poi1_field([[5.0, 6.0]], components=("U", "V"))
    g = f.rename_component("U", "DX")
    assert g.components() == ["DX", "V"]
    assert g[0].get(0, 0) == 5.0
    assert g[0].get(0, 1) == 6.0


def test_rename_absent_source_errors():
    f, _ = _poi1_field([[1.0]], components=("T",))
    with pytest.raises(RuntimeError):
        f.rename_component("nope", "X")


def test_rename_collision_errors():
    f, _ = _poi1_field([[1.0, 2.0]], components=("U", "V"))
    with pytest.raises(RuntimeError):
        f.rename_component("U", "V")


# ─── __getitem__ sugar (numpy/pandas-style) ─────────────────────────────────


def test_getitem_string_selects_one_component():
    f, _ = _poi1_field([[1.0, 2.0, 3.0]], components=("U", "V", "W"))
    g = f["V"]
    assert isinstance(g, pyrucast.NodeField)
    assert g.components() == ["V"]
    assert g[0].get(0, 0) == 2.0


def test_getitem_list_selects_subset():
    f, _ = _poi1_field([[1.0, 2.0, 3.0]], components=("U", "V", "W"))
    assert f[["U", "W"]].components() == ["U", "W"]
    # A tuple key works too.
    assert f["U", "W"].components() == ["U", "W"]


def test_getitem_int_and_slice_unchanged():
    f, _ = _poi1_field([[1.0], [2.0]], components=("T",))
    # int → the zone (a SubNodeField), not a filtered field.
    assert isinstance(f[0], pyrucast.SubNodeField)
    # slice → a fresh aggregate.
    assert isinstance(f[0:1], pyrucast.NodeField)
    assert len(f[0:1]) == 1


def test_getitem_unknown_component_raises():
    f, _ = _poi1_field([[1.0]], components=("T",))
    with pytest.raises(RuntimeError):
        _ = f["nope"]


# ─── sub-field flavour ───────────────────────────────────────────────────────


def test_filter_and_rename_on_subnodefield():
    f, _ = _poi1_field([[1.0, 2.0, 3.0]], components=("U", "V", "W"))
    sub = f[0]
    g = sub.filter_components(["U", "W"])
    assert isinstance(g, pyrucast.SubNodeField)
    assert g.components() == ["U", "W"]
    r = sub.rename_component("V", "VV")
    assert r.components() == ["U", "VV", "W"]


# ─── __getitem__ sugar on sub-fields ─────────────────────────────────────────


def test_subnodefield_getitem_value_still_works():
    f, nodes = _poi1_field([[5.0, 6.0]], components=("U", "V"))
    sub = f[0]
    # (node, component) → the scalar value (unchanged behaviour).
    assert sub[nodes[0], "U"] == 5.0
    assert sub[nodes[0], "V"] == 6.0


def test_subnodefield_getitem_string_and_list_filter():
    f, _ = _poi1_field([[1.0, 2.0, 3.0]], components=("U", "V", "W"))
    sub = f[0]
    assert isinstance(sub["V"], pyrucast.SubNodeField)
    assert sub["V"].components() == ["V"]
    assert sub[["U", "W"]].components() == ["U", "W"]


def test_u1_indexed_by_u2_components():
    # The target idiom: reproject u1 onto the components of u2 — both flavours.
    f, nodes = _poi1_field([[1.0, 2.0, 9.0]], components=("u_x", "u_y", "lambda"))
    u2, _ = _poi1_field([[0.0, 0.0]], components=("u_x", "u_y"))
    # aggregate[list]
    agg = f[u2.components()]
    assert agg.components() == ["u_x", "u_y"]
    # sub[list]
    sub = f[0][u2.components()]
    assert sub.components() == ["u_x", "u_y"]
    assert sub[nodes[0], "u_x"] == 1.0
