"""Python tests for ElementField (Phase 2 step 6)."""

import math

import pyrucast
import pytest


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


def _two_spaces():
    """Two distinct FE spaces on one mesh: two supports, one `Coords`."""
    _, mesh, fes = _tri3_subspace()
    return fes, pyrucast.FiniteElementSpace(mesh)


def _subfield(fes, components):
    """Parent-level construction: build the ElementField and return its
    single sub-field view. SubElementField is no longer constructed
    directly — see CONVENTIONS.md (« Agrégats : un ou plusieurs »)."""
    return pyrucast.ElementField(fes, components)[0]


# ─── Construction ───────────────────────────────────────────────────────────


def test_a_component_given_a_value_holds_it_at_every_gauss_point():
    """A `(name, value)` entry starts at `value`, a bare name at zero."""
    _, _, fes = _tri3_subspace(n_cells=2)
    ef = pyrucast.ElementField(fes, [("E", 210e9), "sigma_y"])
    assert ef.components() == ["E", "sigma_y"]
    assert ef.min("E") == ef.max("E") == 210e9
    assert ef.min("sigma_y") == ef.max("sigma_y") == 0.0


def test_an_empty_field_takes_zones_with_the_union():
    _, _, fes = _tri3_subspace()
    empty = pyrucast.ElementField.empty()
    assert len(empty) == 0
    assert empty.components() == []
    filled = empty | pyrucast.ElementField(fes, [("E", 1.0)])
    assert len(filled) == 1
    assert filled.max("E") == 1.0


def test_new_zero_initialized():
    _, _, fes = _tri3_subspace()
    f = _subfield(fes, ["E", "nu"])
    assert f.cell_count == 1
    assert f.gauss_count == 3  # TRI3 Hammer
    assert f.component_count == 2
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


def test_a_subspace_gives_a_field_of_that_one_zone():
    fes, other = _two_spaces()
    both = fes | other
    ef = pyrucast.ElementField(both[1], [("k", 2.0)])
    assert len(ef) == 1
    assert ef.max("k") == 2.0
    # Zones add up on one support only: the zone stands on `both[1]`.
    on_second = pyrucast.ElementField(both, [("k", 1.0)])[1]
    assert (ef[0] + on_second).max("k") == 3.0
    with pytest.raises(RuntimeError, match="same support"):
        ef[0] + pyrucast.ElementField(both, ["k"])[0]


def test_values_fill_a_zone_and_read_back_through_numpy():
    np = pytest.importorskip("numpy")
    _, _, fes = _tri3_subspace(n_cells=2)
    z = pyrucast.ElementField(fes, [("E", 1.0), ("NU", 0.3)])[0]
    z.set_value(1, 2, "E", 7.0)
    again = pyrucast.ElementField(fes[0], z.components(), values=np.asarray(z))
    assert np.asarray(again[0]).tolist() == np.asarray(z).tolist()
    # One component: the last dimension may be left out.
    k = pyrucast.ElementField(
        fes, ["k"], values=np.arange(2.0 * z.gauss_count).reshape(2, -1)
    )
    assert k[0].value(1, 0, "k") == z.gauss_count


def test_values_refusals():
    np = pytest.importorskip("numpy")
    fes, other = _two_spaces()
    with pytest.raises(ValueError, match="2 subspaces"):
        pyrucast.ElementField(fes | other, ["E"], values=np.zeros((1, 3)))
    with pytest.raises(ValueError, match=r"shape \(3, 1\)"):
        pyrucast.ElementField(fes, ["E"], values=np.zeros((3, 1)))
    with pytest.raises(TypeError, match="names its columns"):
        pyrucast.ElementField(fes, [("E", 1.0)], values=np.zeros((1, 3)))


# ─── Get / set ──────────────────────────────────────────────────────────────


def test_get_set_roundtrip_multi_cell():
    _, _, fes = _tri3_subspace(n_cells=3)
    f = _subfield(fes, ["sigma_xx", "sigma_yy"])
    assert f.cell_count == 3
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


# ─── Lending the values (buffer protocol) ───────────────────────────────────


def test_numpy_reads_a_zone_in_place_cells_then_gauss_points_then_components():
    np = pytest.importorskip("numpy")
    _, _, fes = _tri3_subspace(n_cells=2)
    z = _subfield(fes, ["E", "NU"])
    for cell in range(z.cell_count):
        for gp in range(z.gauss_count):
            z.set_value(cell, gp, "E", 100.0 * cell + gp)
            z.set_value(cell, gp, "NU", -1.0 - gp)
    a = np.asarray(z)
    assert a.shape == (2, z.gauss_count, 2)
    assert a.tolist() == [
        [[z.value(cell, gp, c) for c in z.components()] for gp in range(z.gauss_count)]
        for cell in range(z.cell_count)
    ]
    assert np.shares_memory(a, np.asarray(z))
    assert not a.flags.writeable


_ZONE_WRITES = {
    "set": lambda z: z.set(0, 0, 0, 1.0),
    "set_value": lambda z: z.set_value(0, 0, "E", 1.0),
    "set_uniform": lambda z: z.set_uniform("E", 1.0),
    "set_cell_uniform": lambda z: z.set_cell_uniform(0, "E", 1.0),
    "setitem": lambda z: z.__setitem__((0, 0, "E"), 1.0),
    "add_to_component": lambda z: z.add_to_component("E", 1.0),
    "sub_to_component": lambda z: z.sub_to_component("E", 1.0),
    "mul_to_component": lambda z: z.mul_to_component("E", 3.0),
    "div_to_component": lambda z: z.div_to_component("E", 4.0),
}


@pytest.mark.parametrize("write", _ZONE_WRITES.values(), ids=_ZONE_WRITES.keys())
def test_a_zone_refuses_every_write_while_a_view_reads_it(write):
    _, _, fes = _tri3_subspace()
    z = pyrucast.ElementField(fes, [("E", 2.0)])[0]
    view = memoryview(z)
    assert view.shape == (1, z.gauss_count, 1)
    with pytest.raises(BufferError, match="1 buffer view"):
        write(z)
    assert view.tolist() == [[[2.0]] * z.gauss_count]
    view.release()
    write(z)
    assert z.get(0, 0, 0) != 2.0


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
    _c, _mesh, fes = _tri3_subspace(n_cells=2)
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
    # Without a component: the whole field, every component pooled.
    assert sub.min() == -5.0
    assert ef.max() == 2.0


def test_norm_of_an_element_field():
    """Every Gauss point counts: three per TRI3 cell, 2 in one cell, -5 in the other."""
    _c, _mesh, fes = _tri3_subspace(n_cells=2)
    ef = pyrucast.ElementField(fes, ["k"])
    ef[0].set_cell_uniform(0, "k", 2.0)
    ef[0].set_cell_uniform(1, "k", -5.0)
    gauss = ef[0].gauss_count  # per cell
    assert ef.norm(0) == 5.0
    assert ef.norm(1) == pytest.approx(gauss * 7.0)
    assert ef.norm() == pytest.approx(math.sqrt(gauss * 29.0))
    assert ef[0].norm() == ef.norm()
    assert ef.components() == ["k"]
    try:
        ef.min("missing")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for unknown component")


def test_norm_of_an_element_field_on_the_components_named():
    """Two TRI3 cells, every Gauss point holding k = 2 and E = 100."""
    _c, _mesh, fes = _tri3_subspace(n_cells=2)
    ef = pyrucast.ElementField(fes, [("k", 2.0), ("E", 100.0)])
    points = 2 * ef[0].gauss_count
    assert ef.norm(0, ["k"]) == 2.0
    assert ef.norm(1, ["k"]) == pytest.approx(points * 2.0)
    assert ef[0].norm(components=["k"]) == pytest.approx(math.sqrt(points * 4.0))
    assert ef.norm(0) == 100.0
    assert ef.norm(components=["nope"]) == 0.0


# ─── Union `|` (composition, uniform with the other aggregates) ──────────────


def test_union_distinct_supports_stay_separate():
    # Two element fields on distinct FE spaces (distinct supports) → the
    # union keeps both zones.
    fes_a, fes_b = _two_spaces()
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
    fes_a, fes_b = _two_spaces()
    sa = pyrucast.ElementField(fes_a, ["E"])[0]
    sb = pyrucast.ElementField(fes_b, ["nu"])[0]
    f = sa | sb
    assert len(f) == 2


# ─── Arithmetic operators (binary, union with passthrough) ──────────────────


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
    fa, fb = _two_spaces()
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
    fes_a, fes_b = _two_spaces()
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
    fa, fb = _two_spaces()
    a = pyrucast.ElementField(fa, ["E"])
    a[0].set_uniform("E", 3.0)
    b = pyrucast.ElementField(fb, ["E"])
    b[0].set_uniform("E", 4.0)
    s = a + b
    assert len(s) == 2


def test_field_plus_subfield_targets_matching_zone():
    fa, fb = _two_spaces()
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


# ─── Back to the space, scalar on the left, one error type ──────────────────


def test_support_gives_back_the_shared_space():
    _, mesh, fes = _tri3_subspace(n_cells=2)
    k = pyrucast.ElementField(fes, ["k"])
    k[0].set_uniform("k", 2.0)
    # A field built on the space read back combines point by point with k.
    e = pyrucast.ElementField(k.support(), ["k"]) + 1.0
    assert len(k + e) == 1
    zone = pyrucast.ElementField(fes, ["E"])[0]
    assert len(zone.support() | k.support()[0]) == 1  # one subspace, not two
    # Another space on the same mesh is another support.
    autre = pyrucast.FiniteElementSpace(mesh)
    assert len(pyrucast.ElementField(autre, ["k"]) + k) == 2


def test_support_lists_one_subspace_per_zone():
    # The union keeps k and E side by side on one subspace: the space lists
    # it twice, zone i ↔ subspace i.
    _, _mesh, fes = _tri3_subspace(n_cells=2)
    both = pyrucast.ElementField(fes, ["k"]) | pyrucast.ElementField(fes, ["E"])
    support = both.support()
    assert len(support) == 2
    for i in range(2):
        assert len(support[i] | both[i].support()) == 1  # the same subspace
    assert len(support | support) == 1  # the union keeps it once


def test_a_scalar_on_the_left_mirrors_the_right():
    _, _, fes = _tri3_subspace()
    f = pyrucast.ElementField(fes, ["k"])
    f[0].set_uniform("k", 1.5)
    assert (2.0 * f)[0].value(0, 0, "k") == (f * 2.0)[0].value(0, 0, "k") == 3.0
    assert (2.0 + f)[0].value(0, 0, "k") == (f + 2.0)[0].value(0, 0, "k") == 3.5
    # Not commutative: the scalar on the left is the left operand.
    assert (1.0 - f)[0].value(0, 0, "k") == -0.5
    assert (3.0 / f)[0].value(0, 0, "k") == 2.0
    assert (4.0**f)[0].value(0, 0, "k") == 8.0  # 4 ** 1.5
    assert (-f)[0].value(0, 0, "k") == -1.5
    zone = f[0]
    assert [(2 * zone).value(0, 1, "k"), (2 + zone).value(0, 1, "k")] == [3.0, 3.5]
    assert [(1 - zone).value(0, 1, "k"), (3 / zone).value(0, 1, "k")] == [-0.5, 2.0]
    assert [(4**zone).value(0, 1, "k"), (-zone).value(0, 1, "k")] == [8.0, -1.5]
    for call in (
        lambda: "x" * f,
        lambda: "x" + f,
        lambda: "x" ** f,
        lambda: "x" - zone,
        lambda: "x" / zone,
        lambda: "x" ** zone,
    ):
        with pytest.raises(TypeError):
            call()


def test_subscript_raises_like_value():
    _, _, fes = _tri3_subspace()
    zone = pyrucast.ElementField(fes, ["k"])[0]
    for call in (lambda: zone[0, 0, "rho"], lambda: zone.value(0, 0, "rho")):
        with pytest.raises(RuntimeError, match="unknown component"):
            call()
    for call in (lambda: zone[5, 0, "k"], lambda: zone.value(5, 0, "k")):
        with pytest.raises(RuntimeError, match="cell index 5"):
            call()


def test_side_by_side_zones_pair_per_component():
    _, _, fes = _tri3_subspace()
    left = pyrucast.ElementField(fes, ["E", "k"])
    left[0].set_uniform("E", 10.0)
    left[0].set_uniform("k", 20.0)
    a = pyrucast.ElementField(fes, ["E", "nu"])
    a[0].set_uniform("E", 1.0)
    a[0].set_uniform("nu", 0.3)
    b = pyrucast.ElementField(fes, ["k"])
    b[0].set_uniform("k", 2.0)
    right = a | b  # two zones on one subspace, each component in one of them
    s = left + right
    assert len(s) == 1  # the right's extras join the left's only zone
    assert [s[0].value(0, 0, c) for c in ("E", "k", "nu")] == [11.0, 22.0, 0.3]
    points = left[0].gauss_count
    assert (
        pyrucast.measure.xty(left, right)
        == pyrucast.measure.xty(right, left)
        == 50.0 * points
    )


def test_a_field_is_not_built_on_a_subspace_listed_twice():
    _, _, fes = _tri3_subspace()
    fes.add_sub(fes[0])  # concatenation, no deduplication: the subspace twice
    with pytest.raises(RuntimeError, match="subspaces 0 and 1"):
        pyrucast.ElementField(fes, ["E"])
    assert len(pyrucast.ElementField(fes | fes, ["E"])) == 1


def _grid_and_rank():
    """A 2×2 QUA4 grid (cells 0, 1 below, 2, 3 above), its left column `sel`
    (cells 0 and 2), its space `fes`, and `k`: the rank of the cell at each
    Gauss point."""
    c = pyrucast.Coords(2)
    n = [c.add_node([float(i % 3), float(i // 3)]) for i in range(9)]
    grid = pyrucast.Mesh(c, "QUA4")
    for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        m = 3 * j + i
        grid.unit().add_cell([n[m], n[m + 1], n[m + 4], n[m + 3]])
    sel = pyrucast.mesh.elements_on(
        grid, pyrucast.mesh.poi1_from_nodes([n[i] for i in (0, 1, 3, 4, 6, 7)])
    )
    fes = pyrucast.FiniteElementSpace(grid)
    k = pyrucast.ElementField(fes, ["k"])
    for cell in range(4):
        k[0].set_cell_uniform(cell, "k", float(cell))
    return grid, sel, fes, k


def _per_cell(field, component):
    zone = field[0]
    return [zone.value(cell, 0, component) for cell in range(zone.cell_count)]


def test_restrict_onto_a_second_space_of_the_same_mesh():
    """Two spaces on one mesh: the values pass, and the result adds to a field
    of the second space point by point, in one zone."""
    grid, _, _, k = _grid_and_rank()
    other = pyrucast.FiniteElementSpace(grid)
    assert len(k + pyrucast.ElementField(other, [("k", 1.0)])) == 2
    moved = pyrucast.element_field.restrict(k, other)
    total = moved + pyrucast.ElementField(other, [("k", 1.0)])
    assert len(total) == 1
    assert _per_cell(total, "k") == [1.0, 2.0, 3.0, 4.0]


def test_restrict_onto_a_selection_and_back():
    """On the left column: cells 0 and 2. Read back on the grid, the cells the
    selection lacks are 0."""
    _, sel, fes, k = _grid_and_rank()
    part = fes.restrict(sel)
    left = k.restrict(part)
    assert _per_cell(left, "k") == [0.0, 2.0]
    assert _per_cell(left.restrict(fes), "k") == [0.0, 0.0, 2.0, 0.0]


def test_restrict_does_not_read_another_definition():
    """A REDUCED space is another definition: nothing is read. `restrict` gives
    no zone, `restrict_like` the zone of its target, all at 0."""
    grid, _, _, k = _grid_and_rank()
    reduced = pyrucast.FiniteElementSpace(grid, quadrature="REDUCED")
    assert len(pyrucast.element_field.restrict(k, reduced)) == 0
    target = pyrucast.ElementField(reduced, ["k"])
    r = pyrucast.element_field.restrict_like(k, target)
    assert r[0].gauss_count == 1
    assert _per_cell(r, "k") == [0.0] * 4


def test_restrict_takes_the_components_of_the_zones_sharing_a_cell():
    """`k` on the left column, `m` on the right one: the left column takes `k`
    alone, the whole grid both."""
    grid, sel, fes, k = _grid_and_rank()
    left = fes.restrict(sel)
    right = fes.restrict(pyrucast.mesh.difference(grid, sel))
    both = k.restrict(left) | pyrucast.ElementField(right, [("m", 7.0)])
    assert pyrucast.element_field.restrict(both, left).components() == ["k"]
    r = pyrucast.element_field.restrict(both, fes)
    assert r.components() == ["k", "m"]
    assert _per_cell(r, "m") == [0.0, 7.0, 0.0, 7.0]


def test_restrict_like_takes_the_target_components():
    """The components of the target, those `k` lacks at 0."""
    grid, _, _, k = _grid_and_rank()
    target = pyrucast.ElementField(pyrucast.FiniteElementSpace(grid), ["q", "k"])
    r = pyrucast.element_field.restrict_like(k, target)
    assert r.components() == ["q", "k"]
    assert _per_cell(r, "k") == [0.0, 1.0, 2.0, 3.0]
    assert _per_cell(r, "q") == [0.0] * 4
    assert len(target + r) == 1
