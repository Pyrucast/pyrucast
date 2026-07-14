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
    g = pyrucast.field.filter_components(f, "V")
    assert isinstance(g, pyrucast.NodeField)
    assert g.components() == ["V"]
    assert g[0].get(0, 0) == 2.0


def test_filter_list_keeps_subset_in_field_order():
    f, _ = _poi1_field([[1.0, 2.0, 3.0]], components=("U", "V", "W"))
    # Request order does not matter — the field's own order is kept.
    g = pyrucast.field.filter_components(f, ["W", "U"])
    assert g.components() == ["U", "W"]
    assert g[0].get(0, 0) == 1.0
    assert g[0].get(0, 1) == 3.0


def test_filter_accepts_superset_like_primal_vars():
    # A solve-result-like field: primal [u_x, u_y] plus a dual [lambda].
    f, _ = _poi1_field([[1.0, 2.0, 9.0]], components=("u_x", "u_y", "lambda"))
    # Passing a list that names components the field lacks (as primal_vars may)
    # is fine — the extras are ignored, the dual is stripped.
    g = pyrucast.field.filter_components(f, ["u_x", "u_y", "T"])
    assert g.components() == ["u_x", "u_y"]


def test_filter_no_op_returns_equivalent_field():
    # Zone already carries only requested components (extras in the request are
    # ignored): the filter is a no-op and returns the same components/values.
    # The handle-sharing optimization is checked at the Rust level.
    f, _ = _poi1_field([[1.0, 2.0]], components=("u_x", "u_y"))
    g = pyrucast.field.filter_components(f, ["u_x", "u_y", "lambda"])
    assert g.components() == ["u_x", "u_y"]
    assert g[0].get(0, 0) == 1.0
    assert g[0].get(0, 1) == 2.0


def test_filter_none_present_errors():
    f, _ = _poi1_field([[1.0]], components=("T",))
    with pytest.raises(RuntimeError):
        pyrucast.field.filter_components(f, ["nope"])


# ─── rename_component() ──────────────────────────────────────────────────────


def test_rename_preserves_values():
    f, _ = _poi1_field([[5.0, 6.0]], components=("U", "V"))
    g = pyrucast.field.rename_component(f, "U", "DX")
    assert g.components() == ["DX", "V"]
    assert g[0].get(0, 0) == 5.0
    assert g[0].get(0, 1) == 6.0


def test_rename_absent_source_errors():
    f, _ = _poi1_field([[1.0]], components=("T",))
    with pytest.raises(RuntimeError):
        pyrucast.field.rename_component(f, "nope", "X")


def test_rename_collision_errors():
    f, _ = _poi1_field([[1.0, 2.0]], components=("U", "V"))
    with pytest.raises(RuntimeError):
        pyrucast.field.rename_component(f, "U", "V")


# ─── sub-field flavour ───────────────────────────────────────────────────────


def test_filter_and_rename_on_subnodefield():
    f, _ = _poi1_field([[1.0, 2.0, 3.0]], components=("U", "V", "W"))
    sub = f[0]
    g = pyrucast.field.filter_components(sub, ["U", "W"])
    assert isinstance(g, pyrucast.SubNodeField)
    assert g.components() == ["U", "W"]
    r = pyrucast.field.rename_component(sub, "V", "VV")
    assert r.components() == ["U", "VV", "W"]
