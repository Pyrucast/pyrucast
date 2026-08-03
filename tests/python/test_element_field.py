"""Python tests for ElementField (Phase 2 step 6)."""

import pyrucast


def _tri3_subspace(n_cells=1):
    """Helper: returns (config, mesh, fes) — fes is a FE space with a
    single subspace."""
    c = pyrucast.Coords(2)
    if n_cells == 1:
        nodes = [
            c.add_node([0.0, 0.0]),
            c.add_node([1.0, 0.0]),
            c.add_node([0.0, 1.0]),
        ]
        mesh = pyrucast.Mesh(c, "TRI3")
        mesh.unit().add_cell([nodes[0], nodes[1], nodes[2]])
    else:
        # Fan of n_cells triangles sharing the origin.
        apex = c.add_node([0.0, 0.0])
        perim = []
        for i in range(n_cells + 1):
            t = i / n_cells
            perim.append(c.add_node([1.0, t]))
        mesh = pyrucast.Mesh(c, "TRI3")
        for i in range(n_cells):
            mesh.unit().add_cell([apex, perim[i], perim[i + 1]])
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


def test_operator_pow_scalar():
    _, _, fes = _tri3_subspace()
    f = _subfield(fes, ["x"])
    f.set_uniform("x", 3.0)
    g = f**2.0
    for gp in range(3):
        assert abs(g.get(0, gp, 0) - 9.0) < 1e-12
    assert f.get(0, 0, 0) == 3.0  # f untouched


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


def test_min_max_per_component():
    c, mesh, fes = _tri3_subspace(n_cells=2)
    ef = pyrucast.ElementField(fes, ["k"])
    sub = ef[0]
    sub.set_cell_uniform(0, "k", 2.0)
    sub.set_cell_uniform(1, "k", -5.0)
    # Sub-field level.
    assert sub.min("k") == -5.0
    assert sub.max("k") == 2.0
    # Aggregate level: folds across the sub-fields.
    assert ef.min("k") == -5.0
    assert ef.max("k") == 2.0
    assert ef.components() == ["k"]
    try:
        ef.min("missing")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for unknown component")


# ─── Union `|` (composition, uniform with the other aggregates) ──────────────


def test_union_distinct_supports_stay_separate():
    # Two element fields on distinct FE spaces (distinct supports) → the
    # union keeps both zones.
    _, _, fes_a = _tri3_subspace()
    _, _, fes_b = _tri3_subspace()
    a = pyrucast.ElementField(fes_a, ["E"])
    b = pyrucast.ElementField(fes_b, ["nu"])
    f = a | b
    assert len(f) == 2


def test_union_same_support_keeps_zones_separate():
    # Unlike NodeField (which fuses), ElementField union verifies rather than
    # merges: two fields on the *same* subspace with different components
    # aggregate into two zones, each keeping its own component.
    _, _, fes = _tri3_subspace()
    a = pyrucast.ElementField(fes, ["E"])
    b = pyrucast.ElementField(fes, ["nu"])
    a[0].set_uniform("E", 210.0)
    b[0].set_uniform("nu", 0.3)
    f = a | b
    assert len(f) == 2
    assert f.components() == ["E", "nu"]
    assert f[0].value(0, 0, "E") == 210.0
    assert f[1].value(0, 0, "nu") == 0.3


def test_union_same_support_conflict_raises():
    _, _, fes = _tri3_subspace()
    a = pyrucast.ElementField(fes, ["E"])
    b = pyrucast.ElementField(fes, ["E"])
    a[0].set_uniform("E", 1.0)
    b[0].set_uniform("E", 2.0)
    try:
        a | b
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on conflicting shared component")


def test_subfield_union_subfield_builds_aggregate():
    # `sub | sub` → a fresh ElementField holding both (distinct supports here).
    _, _, fes_a = _tri3_subspace()
    _, _, fes_b = _tri3_subspace()
    sa = pyrucast.ElementField(fes_a, ["E"])[0]
    sb = pyrucast.ElementField(fes_b, ["nu"])[0]
    f = sa | sb
    assert len(f) == 2


# ─── Arithmetic operators (binary, strict) ───────────────────────────────────


def test_subfield_plus_subfield_same_support():
    _, _, fes = _tri3_subspace()
    a = pyrucast.ElementField(fes, ["E"])[0]
    b = pyrucast.ElementField(fes, ["E"])[0]
    a.set_uniform("E", 3.0)
    b.set_uniform("E", 4.0)
    s = a + b
    for g in range(3):
        assert s.value(0, g, "E") == 7.0
    # operands untouched
    assert a.value(0, 0, "E") == 3.0


def test_subfield_plus_subfield_disjoint_components_passes_through():
    # Union/passthrough arithmetic: disjoint components each pass through raw,
    # no error (mirrors the Rust `subfield_operator_uses_union_passthrough`).
    _, _, fes = _tri3_subspace()
    a = pyrucast.ElementField(fes, ["E"])[0]
    a.set_uniform("E", 210.0)
    b = pyrucast.ElementField(fes, ["nu"])[0]
    b.set_uniform("nu", 0.3)
    s = a + b
    assert s.components() == ["E", "nu"]
    assert s.value(0, 0, "E") == 210.0
    assert s.value(0, 0, "nu") == 0.3


def test_subfield_plus_subfield_distinct_support_raises():
    _, _, fa = _tri3_subspace()
    _, _, fb = _tri3_subspace()
    a = pyrucast.ElementField(fa, ["E"])[0]
    b = pyrucast.ElementField(fb, ["E"])[0]
    try:
        a + b
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on distinct supports")


def test_subfield_div_by_zero_is_inf():
    import math

    _, _, fes = _tri3_subspace()
    a = pyrucast.ElementField(fes, ["E"])[0]
    b = pyrucast.ElementField(fes, ["E"])[0]
    a.set_uniform("E", 1.0)  # b stays zero
    s = a / b
    assert math.isinf(s.value(0, 0, "E"))


def test_field_scalar_op_hits_every_zone():
    _, _, fes_a = _tri3_subspace()
    _, _, fes_b = _tri3_subspace()
    a = pyrucast.ElementField(fes_a, ["E"])
    b = pyrucast.ElementField(fes_b, ["E"])
    a[0].set_uniform("E", 1.0)
    b[0].set_uniform("E", 2.0)
    f = a | b  # two zones
    g = f + 10.0
    assert g[0].value(0, 0, "E") == 11.0
    assert g[1].value(0, 0, "E") == 12.0


def test_field_plus_field_same_decomposition():
    _, _, fes = _tri3_subspace()
    a = pyrucast.ElementField(fes, ["E"])
    b = pyrucast.ElementField(fes, ["E"])
    a[0].set_uniform("E", 3.0)
    b[0].set_uniform("E", 4.0)
    s = a + b
    assert s[0].value(0, 0, "E") == 7.0


def test_field_plus_field_distinct_decomposition_aggregates():
    # Field-level union/passthrough: two fields over distinct supports do not
    # overlap, so they aggregate into a two-zone field (no error).
    _, _, fa = _tri3_subspace()
    _, _, fb = _tri3_subspace()
    a = pyrucast.ElementField(fa, ["E"])
    a[0].set_uniform("E", 3.0)
    b = pyrucast.ElementField(fb, ["E"])
    b[0].set_uniform("E", 4.0)
    s = a + b
    assert len(s) == 2


def test_field_plus_subfield_targets_matching_zone():
    _, _, fa = _tri3_subspace()
    _, _, fb = _tri3_subspace()
    a = pyrucast.ElementField(fa, ["E"])
    b = pyrucast.ElementField(fb, ["E"])
    a[0].set_uniform("E", 1.0)
    b[0].set_uniform("E", 2.0)
    f = a | b  # zone 0 on fa-support, zone 1 on fb-support
    sub = pyrucast.ElementField(fa, ["E"])[0]  # same support as zone 0
    sub.set_uniform("E", 10.0)
    g = f + sub
    assert len(g) == 2
    assert g[0].value(0, 0, "E") == 11.0  # matching zone updated
    assert g[1].value(0, 0, "E") == 2.0  # other zone unchanged


def test_field_per_component_in_place():
    _, _, fes = _tri3_subspace()
    f = pyrucast.ElementField(fes, ["E", "nu"])
    f[0].set_uniform("E", 100.0)
    f.add_to_component("E", 1.0)
    assert f[0].value(0, 0, "E") == 101.0
    assert f[0].value(0, 0, "nu") == 0.0
    try:
        f.add_to_component("missing", 1.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for unknown component")


def test_op_bad_operand_raises_type_error():
    _, _, fes = _tri3_subspace()
    f = pyrucast.ElementField(fes, ["E"])
    for bad_lhs in (f, f[0]):
        try:
            bad_lhs + "nope"
        except TypeError:
            pass
        else:
            raise AssertionError("expected TypeError for str operand")


def test_consolidate_element_fuses_component_disjoint_zones():
    """`consolidate_element` fuses same-support, component-disjoint zones
    (e.g. per-physics material zones left side by side by a union) into one zone
    carrying the union of their components."""
    _, _, fes = _tri3_subspace()
    a = pyrucast.ElementField(fes, ["k"])
    a[0].set_uniform("k", 2.0)
    b = pyrucast.ElementField(fes, ["E"])
    b[0].set_uniform("E", 5.0)

    union = a | b
    assert len(union) == 2  # two zones side by side (ElementField union does not fuse)

    fused = pyrucast.element_field.consolidate(union)
    assert len(fused) == 1
    sub = fused[0]
    assert set(sub.components()) == {"k", "E"}
    assert sub.value(0, 0, "k") == 2.0
    assert sub.value(0, 0, "E") == 5.0
