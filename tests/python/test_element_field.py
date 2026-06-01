"""Python tests for ElementField (Phase 2 step 6)."""

import pyrucast


def _tri3_subspace(n_cells=1):
    """Helper: returns (config, mesh, fes) — fes is a FE space with a
    single subspace."""
    c = pyrucast.Configuration(2)
    if n_cells == 1:
        nodes = [
            c.add_node([0.0, 0.0]),
            c.add_node([1.0, 0.0]),
            c.add_node([0.0, 1.0]),
        ]
        mesh = pyrucast.Mesh(c, "TRI3")
        mesh.add_cell([nodes[0], nodes[1], nodes[2]])
    else:
        # Fan of n_cells triangles sharing the origin.
        apex = c.add_node([0.0, 0.0])
        perim = []
        for i in range(n_cells + 1):
            t = i / n_cells
            perim.append(c.add_node([1.0, t]))
        mesh = pyrucast.Mesh(c, "TRI3")
        for i in range(n_cells):
            mesh.add_cell([apex, perim[i], perim[i + 1]])
    fes = pyrucast.FiniteElementSpace(mesh)
    return c, mesh, fes


def _subfield(fes, components):
    """Parent-level construction: build the ElementField and return its
    single sub-field view. SubElementField is no longer constructed
    directly — see CONVENTIONS.md (« Agrégats : un ou plusieurs »)."""
    return pyrucast.ElementField(fes, components)[0]


# ─── Construction ───────────────────────────────────────────────────────────


def test_new_zero_initialized():
    _, _, fes = _tri3_subspace()
    f = _subfield(fes, ["E", "nu"])
    assert f.cell_count() == 1
    assert f.gauss_count() == 3  # TRI3 Hammer
    assert f.component_count() == 2
    assert f.components() == ["E", "nu"]
    for g in range(3):
        assert f.get(0, g, 0) == 0.0
        assert f.get(0, g, 1) == 0.0


def test_new_rejects_empty_components():
    _, _, fes = _tri3_subspace()
    try:
        _subfield(fes, [])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for empty components")


def test_new_rejects_duplicate_components():
    _, _, fes = _tri3_subspace()
    try:
        _subfield(fes, ["E", "E"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for duplicates")


def test_uniform_per_component_via_set_uniform():
    _, _, fes = _tri3_subspace()
    f = _subfield(fes, ["E", "nu", "rho"])
    f.set_uniform("E", 210e9)
    f.set_uniform("nu", 0.3)
    f.set_uniform("rho", 7800.0)
    for g in range(3):
        assert f.get(0, g, 0) == 210e9
        assert f.get(0, g, 1) == 0.3
        assert f.get(0, g, 2) == 7800.0


# ─── Get / set ──────────────────────────────────────────────────────────────


def test_get_set_roundtrip_multi_cell():
    _, _, fes = _tri3_subspace(n_cells=3)
    f = _subfield(fes, ["sigma_xx", "sigma_yy"])
    assert f.cell_count() == 3
    f.set(0, 0, 0, 1.0)
    f.set(1, 2, 1, -3.5)
    assert f.get(0, 0, 0) == 1.0
    assert f.get(1, 2, 1) == -3.5
    assert f.get(0, 0, 1) == 0.0


def test_value_set_value_by_name():
    _, _, fes = _tri3_subspace()
    f = _subfield(fes, ["T", "P"])
    f.set_value(0, 1, "P", 42.0)
    assert f.value(0, 1, "P") == 42.0
    try:
        f.value(0, 0, "unknown")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for unknown component")


def test_point_values_returns_all_components():
    _, _, fes = _tri3_subspace()
    f = _subfield(fes, ["a", "b", "c"])
    f.set(0, 1, 0, 1.0)
    f.set(0, 1, 1, 2.0)
    f.set(0, 1, 2, 3.0)
    assert f.point_values(0, 1) == [1.0, 2.0, 3.0]


def test_out_of_bounds_errors():
    _, _, fes = _tri3_subspace()
    f = _subfield(fes, ["x"])
    for bad in [(99, 0, 0), (0, 99, 0), (0, 0, 99)]:
        try:
            f.get(*bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"expected RuntimeError for {bad}")


# ─── Bulk fillers ───────────────────────────────────────────────────────────


def test_set_uniform_fills_one_component():
    _, _, fes = _tri3_subspace(n_cells=2)
    f = _subfield(fes, ["E", "nu"])
    f.set_uniform("E", 210e9)
    for cell in range(2):
        for g in range(3):
            assert f.get(cell, g, 0) == 210e9
            assert f.get(cell, g, 1) == 0.0


def test_set_cell_uniform_only_touches_one_cell():
    _, _, fes = _tri3_subspace(n_cells=2)
    f = _subfield(fes, ["rho"])
    f.set_cell_uniform(1, "rho", 7800.0)
    for g in range(3):
        assert f.get(0, g, 0) == 0.0
        assert f.get(1, g, 0) == 7800.0


# ─── Scalar ops on a component ──────────────────────────────────────────────


def test_component_scalar_ops_isolate_components():
    _, _, fes = _tri3_subspace()
    f = _subfield(fes, ["a", "b"])
    f.set_uniform("a", 10.0)
    f.set_uniform("b", 1.0)
    f.add_to_component("a", 5.0)
    f.sub_to_component("a", 2.0)
    f.mul_to_component("a", 3.0)
    f.div_to_component("a", 13.0)
    # a: 10 → 15 → 13 → 39 → 3.0
    for g in range(3):
        assert abs(f.get(0, g, 0) - 3.0) < 1e-12
        assert f.get(0, g, 1) == 1.0  # b unchanged


def test_div_by_zero_errors():
    _, _, fes = _tri3_subspace()
    f = _subfield(fes, ["x"])
    try:
        f.div_to_component("x", 0.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on /0")


# ─── Operators with f64 ─────────────────────────────────────────────────────


def test_operator_add_f64_returns_new_field():
    _, _, fes = _tri3_subspace()
    f = _subfield(fes, ["x"])
    f.set(0, 1, 0, 4.0)
    g = f + 10.0
    assert g.get(0, 1, 0) == 14.0
    assert f.get(0, 1, 0) == 4.0  # f untouched


def test_operator_chained_sub_mul_div():
    _, _, fes = _tri3_subspace()
    f = _subfield(fes, ["x"])
    f.set_uniform("x", 12.0)
    g = (f - 2.0) * 3.0 / 2.0
    for gp in range(3):
        assert abs(g.get(0, gp, 0) - 15.0) < 1e-12


# ─── __getitem__ / __setitem__ ──────────────────────────────────────────────


def test_dunder_get_set_item():
    _, _, fes = _tri3_subspace()
    f = _subfield(fes, ["sigma"])
    f[0, 1, "sigma"] = 5.0
    assert f[0, 1, "sigma"] == 5.0


# ─── repr / str ─────────────────────────────────────────────────────────────


def test_repr_and_str():
    _, _, fes = _tri3_subspace(n_cells=2)
    f = _subfield(fes, ["E", "nu"])
    assert "ElementField" in repr(f)
    s = str(f)
    assert "ElementField" in s
    assert "2 cell(s)" in s
    assert "3 gauss" in s
    assert "2 component(s)" in s
    assert "E, nu" in s
