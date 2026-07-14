"""Python tests for NodeField (aggregate) and SubNodeField (zone view)."""

import gc as pygc

import pyrucast


def _poi1_with(n_nodes, dim=2):
    c = pyrucast.Coords(dim)
    coords = [0.0] * dim
    nodes = [c.add_node([float(i)] + coords[1:]) for i in range(n_nodes)]
    mesh = pyrucast.Mesh(c, "POI1")
    for n in nodes:
        mesh.unit().add_cell([n])
    return c, nodes, mesh


def _two_zone_mesh():
    """Two TRI3 zones sharing an interface edge (nodes n1, n2)."""
    c = pyrucast.Coords(2)
    n0 = c.add_node([0.0, 0.0])
    n1 = c.add_node([1.0, 0.0])
    n2 = c.add_node([0.0, 1.0])
    n3 = c.add_node([1.0, 1.0])
    za = pyrucast.Mesh(c, "TRI3")
    za.unit().add_cell([n0, n1, n2])
    zb = pyrucast.Mesh(c, "TRI3")
    zb.unit().add_cell([n1, n3, n2])
    return c, [n0, n1, n2, n3], za | zb


# ─── Construction ────────────────────────────────────────────────────────────


def test_from_mesh_zero_initialized():
    c, _nodes, sm = _poi1_with(3)
    f = pyrucast.NodeField(sm, ["T"])
    assert len(f) == 1
    assert f.node_count() == 3
    assert f.components() == ["T"]
    sub = f.unit()
    assert sub.component_count() == 1
    for i in range(3):
        assert sub.get(i, 0) == 0.0


def test_non_poi1_support_uses_distinct_nodes():
    # A non-POI1 mesh is accepted: each zone's sub-field is supported on
    # the distinct nodes of its submesh.
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    seg = pyrucast.Mesh(c, "SEG2")
    seg.unit().add_cell([a, b])
    f = pyrucast.NodeField(seg, ["T"])
    assert f.node_count() == 2
    assert f.value(a, "T") == 0.0


def test_rejects_empty_components():
    c, _nodes, sm = _poi1_with(1)
    try:
        pyrucast.NodeField(sm, [])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for empty components")


def test_rejects_duplicate_components():
    c, _nodes, sm = _poi1_with(1)
    try:
        pyrucast.NodeField(sm, ["UX", "UX"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for duplicate components")


def test_one_sub_per_submesh():
    c, nodes, mesh = _two_zone_mesh()
    f = pyrucast.NodeField(mesh, ["T"])
    assert len(f) == 2
    # 4 distinct nodes; the interface nodes are stored once per zone.
    assert f.node_count() == 4
    assert f[0].node_count() == 3
    assert f[1].node_count() == 3


def test_components_per_submesh():
    c, nodes, mesh = _two_zone_mesh()
    f = pyrucast.NodeField.with_components_per_submesh(mesh, [["T"], ["UX", "UY"]])
    assert f.components() == ["T", "UX", "UY"]
    # T lives on zone 0 only: defined at n0, absent at n3.
    assert f.value(nodes[0], "T") == 0.0
    try:
        f.value(nodes[3], "T")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for (n3, T)")


# ─── Zone view (SubNodeField) ────────────────────────────────────────────────


def test_get_set_multi_component():
    c, _nodes, sm = _poi1_with(2)
    f = pyrucast.NodeField(sm, ["UX", "UY", "UZ"])
    sub = f.unit()
    sub.set(0, 0, 1.0)
    sub.set(0, 1, 2.0)
    sub.set(0, 2, 3.0)
    sub.set(1, 1, -7.0)
    assert sub.node_values(0) == [1.0, 2.0, 3.0]
    assert sub.node_values(1) == [0.0, -7.0, 0.0]


def test_by_node_id_access():
    c, nodes, sm = _poi1_with(2)
    f = pyrucast.NodeField(sm, ["T", "P"])
    sub = f.unit()
    ci_p = sub.component_index("P")
    assert ci_p == 1
    sub.set_by_node(nodes[1], ci_p, 42.0)
    assert sub.get_by_node(nodes[1], ci_p) == 42.0


def test_unknown_node_or_component():
    c, _nodes, sm = _poi1_with(1)
    f = pyrucast.NodeField(sm, ["T"])
    sub = f.unit()
    assert sub.component_index("missing") is None
    other = c.add_node(
        [99.0, 99.0]
    )  # alive in the config but not in the field's support
    try:
        sub.get_by_node(other, 0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for unknown NodeId")


def test_subfield_getitem_setitem():
    c, nodes, sm = _poi1_with(2)
    f = pyrucast.NodeField(sm, ["UX", "UY"])
    f[0][nodes[0], "UX"] = 7.0
    f[0][nodes[1], "UY"] = -3.0
    assert f[0][nodes[0], "UX"] == 7.0
    assert f[0][nodes[1], "UY"] == -3.0
    assert f[0][nodes[0], "UY"] == 0.0


# ─── Aggregate reads, coherence ──────────────────────────────────────────────


def test_value_first_zone_wins_and_check():
    c, nodes, mesh = _two_zone_mesh()
    f = pyrucast.NodeField(mesh, ["T"])
    interface = nodes[1]
    # Diverging values at the interface: reads pick zone 0, check() raises.
    f[0].set_value(interface, "T", 1.0)
    f[1].set_value(interface, "T", 2.0)
    assert f.value(interface, "T") == 1.0
    try:
        f.check()
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError from check() on conflict")
    # Re-aligned values: check() passes.
    f[1].set_value(interface, "T", 1.0)
    f.check()


def test_field_protects_nodes_from_gc():
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    nid = a.id
    sm = pyrucast.Mesh(c, "POI1")
    sm.unit().add_cell([a])
    field = pyrucast.NodeField(sm, ["T"])

    # The sub-field shares the SubMesh handle, so per-node refcounts are
    # only Node + SubMesh = 2 (the field adds no per-node incref).
    assert c.refcount(nid) == 2

    del a
    del sm
    pygc.collect()
    # The field still holds a clone of the SubMesh handle, so the
    # SubMesh stays alive and keeps the node alive (refcount = 1).
    assert c.refcount(nid) == 1
    assert c.is_alive(nid)
    assert c.gc() == 0

    del field
    pygc.collect()
    assert c.gc() == 1
    assert not c.is_alive(nid)


def test_values_batch_list_submesh_and_mesh():
    c, nodes, sm = _poi1_with(4)
    f = pyrucast.NodeField(sm, ["T"])
    for i, n in enumerate(nodes):
        f[0].set_value(n, "T", float(i * 10))  # 0, 10, 20, 30

    # 1) A plain list of nodes → values in the same order (duplicates kept).
    assert f.values([nodes[2], nodes[0], nodes[2]], "T") == [20.0, 0.0, 20.0]

    # 2) A POI1 Mesh → its points in connectivity order.
    probe = pyrucast.Mesh(c, "POI1")
    for n in (nodes[3], nodes[1]):
        probe.unit().add_cell([n])
    assert f.values(probe, "T") == [30.0, 10.0]

    # 3) A POI1 SubMesh (the single zone of that mesh).
    assert f.values(probe.unit(), "T") == [30.0, 10.0]


def test_values_batch_errors_on_absent_and_bad_arg():
    c, nodes, mesh = _two_zone_mesh()
    f = pyrucast.NodeField.with_components_per_submesh(mesh, [["T"], ["UX", "UY"]])
    # T lives on zone 0 only: absent at n3 → raises, like value().
    try:
        f.values([nodes[3]], "T")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for (n3, T)")
    # A wrong argument type is rejected.
    try:
        f.values("nope", "T")
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for str argument")


# ─── coordinates / set_coordinates / displace ────────────────────────────────


def test_coordinates_poi1_mesh_xyz():
    c = pyrucast.Coords(3)
    a = c.add_node([1.0, 2.0, 3.0])
    b = c.add_node([4.0, 5.0, 6.0])
    mesh = pyrucast.Mesh(c, "POI1")
    mesh.unit().add_cell([a])
    mesh.unit().add_cell([b])

    f = pyrucast.field.coordinates(mesh)
    assert f.components() == ["X", "Y", "Z"]
    assert f.node_count() == 2
    assert f.value(a, "X") == 1.0
    assert f.value(a, "Y") == 2.0
    assert f.value(a, "Z") == 3.0
    assert f.value(b, "Z") == 6.0


def test_coordinates_converts_non_poi1_and_deduplicates():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    cc = c.add_node([0.5, 1.0])
    d = c.add_node([1.5, 1.0])

    tri = pyrucast.Mesh(c, "TRI3")
    tri.unit().add_cell([a, b, cc])
    tri.unit().add_cell([b, d, cc])

    f = pyrucast.field.coordinates(tri)
    assert f.components() == ["X", "Y"]  # 2-D ⇒ X, Y only
    assert f.node_count() == 4  # shared nodes appear once
    assert f.value(cc, "X") == 0.5
    assert f.value(d, "X") == 1.5


def test_coordinates_component_subset():
    c = pyrucast.Coords(3)
    a = c.add_node([1.0, 2.0, 3.0])
    mesh = pyrucast.Mesh(c, "POI1")
    mesh.unit().add_cell([a])

    f = pyrucast.field.coordinates(mesh, ["X", "Z"])
    assert f.components() == ["X", "Z"]
    assert f.value(a, "X") == 1.0
    assert f.value(a, "Z") == 3.0


def test_coordinates_rejects_axis_beyond_dimension():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    mesh = pyrucast.Mesh(c, "POI1")
    mesh.unit().add_cell([a])
    try:
        pyrucast.field.coordinates(mesh, ["Z"])  # no Z in 2-D
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for Z on a 2-D mesh")


def test_set_coordinates_writes_positions():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 1.0])
    mesh = pyrucast.Mesh(c, "POI1")
    mesh.unit().add_cell([a])
    mesh.unit().add_cell([b])

    # Read current positions into a field, move node a, write it all back.
    f = pyrucast.field.coordinates(mesh)  # components X, Y
    f[0][a, "X"] = 10.0
    f[0][a, "Y"] = 20.0
    pyrucast.field.set_coordinates(f)  # default components ["X", "Y"]
    assert a.coord() == [10.0, 20.0]
    assert b.coord() == [1.0, 1.0]


def test_displace_adds_displacement():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 1.0])
    mesh = pyrucast.Mesh(c, "POI1")
    mesh.unit().add_cell([a])
    mesh.unit().add_cell([b])

    d = pyrucast.NodeField(mesh, ["ux", "uy"])
    d[0][a, "ux"] = 5.0
    d[0][a, "uy"] = -1.0
    d[0][b, "ux"] = 2.0
    pyrucast.field.displace(d)  # default components ["ux", "uy"]
    assert a.coord() == [5.0, -1.0]
    assert b.coord() == [3.0, 1.0]


def test_repr_str_node_field():
    c, _nodes, sm = _poi1_with(3)
    f = pyrucast.NodeField(sm, ["UX", "UY"])
    assert "NodeField" in repr(f)
    assert "1 subfield(s)" in str(f)
    s = str(f.unit())
    assert "3 node(s)" in s
    assert "2 component(s)" in s
    assert "UX, UY" in s


# ─── Operators ───────────────────────────────────────────────────────────────


def test_subfield_add_scalar():
    c, nodes, sm = _poi1_with(2)
    f = pyrucast.NodeField(sm, ["T"])
    f[0].set_value(nodes[0], "T", 1.0)
    f[0].set_value(nodes[1], "T", 2.0)
    g = f[0] + 10.0
    assert g.value(nodes[0], "T") == 11.0
    assert g.value(nodes[1], "T") == 12.0
    # The original is untouched.
    assert f.value(nodes[0], "T") == 1.0


def test_subfield_pow_scalar():
    c, nodes, sm = _poi1_with(2)
    f = pyrucast.NodeField(sm, ["T"])
    f[0].set_value(nodes[0], "T", 2.0)
    f[0].set_value(nodes[1], "T", 3.0)
    g = f[0] ** 2.0
    assert g.value(nodes[0], "T") == 4.0
    assert g.value(nodes[1], "T") == 9.0
    # Fractional exponent: square root.
    h = f[0] ** 0.5
    assert abs(h.value(nodes[1], "T") - 3.0**0.5) < 1e-12
    # Original untouched.
    assert f.value(nodes[0], "T") == 2.0


def test_field_pow_scalar_aggregate():
    c, nodes, sm = _poi1_with(2)
    f = pyrucast.NodeField(sm, ["T"])
    f[0].set_value(nodes[0], "T", 3.0)
    g = f**3.0
    assert g.value(nodes[0], "T") == 27.0


def test_subfield_pow_field_elementwise():
    # field ** field is strict and element-by-element.
    c, nodes, sm = _poi1_with(2)
    base = pyrucast.NodeField(sm, ["T"])
    expo = pyrucast.NodeField(sm, ["T"])
    base[0].set_value(nodes[0], "T", 2.0)
    base[0].set_value(nodes[1], "T", 5.0)
    expo[0].set_value(nodes[0], "T", 3.0)
    expo[0].set_value(nodes[1], "T", 2.0)
    g = base[0] ** expo[0]
    assert g.value(nodes[0], "T") == 8.0
    assert g.value(nodes[1], "T") == 25.0


def test_pow_rejects_modulo():
    c, nodes, sm = _poi1_with(1)
    f = pyrucast.NodeField(sm, ["T"])
    f[0].set_value(nodes[0], "T", 2.0)
    # Ternary pow(base, exp, mod) is meaningless on float fields.
    try:
        pow(f[0], 2.0, 3.0)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for pow with modulo")


# ─── element-wise unary maths (numpy-style top-level functions) ───────────────


def test_unary_math_on_aggregate_and_subfield():
    import math

    c, nodes, sm = _poi1_with(3)
    f = pyrucast.NodeField(sm, ["T"])
    f[0].set_value(nodes[0], "T", 0.0)
    f[0].set_value(nodes[1], "T", math.pi)
    f[0].set_value(nodes[2], "T", 4.0)

    cosf = pyrucast.field.cos(f)  # aggregate → aggregate
    assert abs(cosf.value(nodes[0], "T") - 1.0) < 1e-12
    assert abs(cosf.value(nodes[1], "T") + 1.0) < 1e-12
    # Original is untouched.
    assert f.value(nodes[1], "T") == math.pi

    assert abs(pyrucast.field.sqrt(f).value(nodes[2], "T") - 2.0) < 1e-12
    assert abs(pyrucast.field.exp(f).value(nodes[0], "T") - 1.0) < 1e-12

    # Works on a single zone (SubNodeField) too.
    g = pyrucast.field.sin(f[0])
    assert abs(g.value(nodes[1], "T") - math.sin(math.pi)) < 1e-12


def test_unary_math_log_unguarded_is_nan():
    import math

    c, nodes, sm = _poi1_with(1)
    f = pyrucast.NodeField(sm, ["T"])
    f[0].set_value(nodes[0], "T", -1.0)
    assert math.isnan(pyrucast.field.log(f).value(nodes[0], "T"))


def test_unary_math_rejects_bad_operand():
    try:
        pyrucast.field.cos("nope")
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for non-field operand")


def test_union_is_structural_merge():
    # `a | b` unites zones (shared handles) — it does NOT add values. The two
    # zones live on distinct support SubMeshes, so they stay separate.
    c, nodes, mesh = _two_zone_mesh()
    a = pyrucast.NodeField(mesh[0], ["T"])
    b = pyrucast.NodeField(mesh[1], ["P"])
    a[0].set_value(nodes[0], "T", 1.0)
    f = a | b
    assert len(f) == 2
    assert f.components() == ["T", "P"]
    assert f.value(nodes[0], "T") == 1.0
    assert f.value(nodes[3], "P") == 0.0


def test_union_aggregate_plus_subfield():
    c, nodes, mesh = _two_zone_mesh()
    a = pyrucast.NodeField(mesh[0], ["T"])
    b = pyrucast.NodeField(mesh[1], ["T"])
    f = a | b[0]
    assert len(f) == 2


def test_union_field_to_itself_dedups_by_handle():
    # Same handle on both sides: the union deduplicates by handle, so a single
    # zone remains (and the store mutex is never re-locked).
    c, nodes, sm = _poi1_with(1)
    a = pyrucast.NodeField(sm, ["T"])
    f = a | a
    assert len(f) == 1
    f.check()  # coherent


def test_union_rejects_bad_operand():
    c, _nodes, sm = _poi1_with(1)
    f = pyrucast.NodeField(sm, ["T"])
    try:
        f | "nope"
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for str operand")


# ─── merge / restrict ────────────────────────────────────────────────────────


def test_merge_compatible_and_conflict():
    # a on {n0, n1}, b on {n1, n2} — n1 shared. Union = {n0, n1, n2}.
    c = pyrucast.Coords(1)
    n0 = c.add_node([0.0])
    n1 = c.add_node([1.0])
    n2 = c.add_node([2.0])
    sm_a = pyrucast.Mesh(c, "POI1")
    sm_a.unit().add_cell([n0])
    sm_a.unit().add_cell([n1])
    sm_b = pyrucast.Mesh(c, "POI1")
    sm_b.unit().add_cell([n1])
    sm_b.unit().add_cell([n2])
    a = pyrucast.NodeField(sm_a, ["T"])
    b = pyrucast.NodeField(sm_b, ["T"])
    a[0].set_value(n0, "T", 5.0)
    a[0].set_value(n1, "T", 3.0)
    b[0].set_value(n1, "T", 3.0)  # same value at the shared node → compatible
    b[0].set_value(n2, "T", 9.0)

    m = pyrucast.field.merge(a, b)
    assert m.node_count() == 3
    assert m.value(n0, "T") == 5.0
    assert m.value(n1, "T") == 3.0
    assert m.value(n2, "T") == 9.0

    # Conflicting value at the shared node → error.
    b[0].set_value(n1, "T", 7.0)
    try:
        pyrucast.field.merge(a, b)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on conflicting merge")


def test_restrict_to_mesh_subset():
    c, nodes, sm = _poi1_with(3)
    f = pyrucast.NodeField(sm, ["T"])
    for i, n in enumerate(nodes):
        f[0].set_value(n, "T", float(i + 1))

    # Mesh covering only nodes[0] and nodes[2].
    mesh = pyrucast.Mesh(c, "POI1")
    mesh.unit().add_cell([nodes[0]])
    mesh.unit().add_cell([nodes[2]])

    r = pyrucast.field.restrict(f, mesh)
    assert r.node_count() == 2
    assert r.value(nodes[0], "T") == 1.0
    assert r.value(nodes[2], "T") == 3.0


def test_restrict_twice_to_element_mesh_is_subtractable():
    """Two restricts onto the *same* element (TRI3) mesh land on its cached
    POI1 companion, so they subtract node-by-node rather than passing through
    as two disjoint zones."""
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    d = c.add_node([0.0, 1.0])
    tri = pyrucast.Mesh(c, "TRI3")
    tri.unit().add_cell([a, b, d])

    def field(va, vb, vd):
        sm = pyrucast.Mesh(c, "POI1")
        for n in (a, b, d):
            sm.unit().add_cell([n])
        f = pyrucast.NodeField(sm, ["v"])
        f[0].set_value(a, "v", va)
        f[0].set_value(b, "v", vb)
        f[0].set_value(d, "v", vd)
        return f

    a2 = pyrucast.field.restrict(field(1.0, 2.0, 3.0), tri)
    b2 = pyrucast.field.restrict(field(0.5, 0.5, 0.5), tri)

    # Shared support ⇒ genuine difference (else a2's values would pass through).
    diff = a2 - b2
    assert diff.value(a, "v") == 0.5
    assert diff.value(b, "v") == 1.5
    assert diff.value(d, "v") == 2.5


def test_restrict_like_lands_on_target_support():
    c, nodes, sm = _poi1_with(3)

    # Target: nodes[0], nodes[1], components [u_x, u_y].
    tmesh = pyrucast.Mesh(c, "POI1")
    tmesh.unit().add_cell([nodes[0]])
    tmesh.unit().add_cell([nodes[1]])
    target = pyrucast.NodeField(tmesh, ["u_x", "u_y"])

    # Source (`du`-like): all 3 nodes, extra `lambda` component (multiplier).
    source = pyrucast.NodeField(sm, ["u_x", "u_y", "lambda"])
    source[0].set_value(nodes[0], "u_x", 1.0)
    source[0].set_value(nodes[1], "u_y", 2.0)
    source[0].set_value(nodes[2], "u_x", 9.0)  # node dropped
    source[0].set_value(nodes[0], "lambda", 7.0)  # component dropped

    r = pyrucast.field.restrict_like(source, target)
    assert r.node_count() == 2
    assert r.value(nodes[0], "u_x") == 1.0
    assert r.value(nodes[1], "u_y") == 2.0
    assert r.value(nodes[1], "u_x") == 0.0  # absent → 0

    # Lands on target's own support ⇒ the `+` operator applies directly.
    s = target + r
    assert s.value(nodes[0], "u_x") == 1.0
    assert s.value(nodes[1], "u_y") == 2.0


# ─── min / max ───────────────────────────────────────────────────────────────


def test_min_max_per_component():
    c, nodes, sm = _poi1_with(3)
    f = pyrucast.NodeField(sm, ["U", "V"])
    for i, n in enumerate(nodes):
        f[0].set_value(n, "U", float(i + 1))  # 1, 2, 3
        f[0].set_value(n, "V", -float(i + 1))  # -1, -2, -3
    # Zone view and aggregate agree on a single-zone field.
    assert f[0].min("U") == 1.0
    assert f.min("U") == 1.0
    assert f.max("U") == 3.0
    assert f.min("V") == -3.0
    assert f.max("V") == -1.0
    try:
        f.min("missing")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for unknown component")


def test_min_max_fold_across_zones():
    c, nodes, mesh = _two_zone_mesh()
    f = pyrucast.NodeField(mesh, ["T"])
    f[0].set_value(nodes[0], "T", -2.0)
    f[1].set_value(nodes[3], "T", 5.0)
    assert f.min("T") == -2.0
    assert f.max("T") == 5.0


# ─── consolidate ─────────────────────────────────────────────────────────────


def test_consolidate_keeps_distinct_supports_separate():
    # Fusion is by support handle: the two TRI3 zones live on distinct
    # SubMeshes, so consolidate keeps both — but the shared interface node
    # (same value) passes the cross-support check.
    c, nodes, mesh = _two_zone_mesh()
    f = pyrucast.NodeField(mesh, ["T"])
    f[0].set_value(nodes[0], "T", 1.0)
    f[0].set_value(nodes[1], "T", 2.0)
    f[1].set_value(nodes[1], "T", 2.0)  # interface: same value
    f[1].set_value(nodes[3], "T", 4.0)

    g = pyrucast.consolidate(f)
    assert len(g) == 2
    assert g.node_count() == 4
    assert g.value(nodes[1], "T") == 2.0
    assert g.value(nodes[3], "T") == 4.0


def test_consolidate_rejects_incoherent_field():
    c, nodes, mesh = _two_zone_mesh()
    f = pyrucast.NodeField(mesh, ["T"])
    f[0].set_value(nodes[1], "T", 1.0)
    f[1].set_value(nodes[1], "T", 2.0)
    try:
        pyrucast.consolidate(f)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError on incoherent field")


def test_consolidate_keeps_distinct_component_sets_separate():
    c, nodes, mesh = _two_zone_mesh()
    f = pyrucast.NodeField.with_components_per_submesh(mesh, [["T"], ["UX", "UY"]])
    g = pyrucast.consolidate(f)
    assert len(g) == 2
    assert g.components() == ["T", "UX", "UY"]


def test_consolidate_still_dispatches_on_mesh():
    # The top-level consolidate dispatches on type: Mesh → mesher op.
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    d = c.add_node([0.0, 1.0])
    za = pyrucast.Mesh(c, "TRI3")
    za.unit().add_cell([a, b, d])
    zb = pyrucast.Mesh(c, "TRI3")
    zb.unit().add_cell([a, b, d])  # duplicate cell in a second submesh
    m = pyrucast.consolidate(za | zb)
    assert len(m) == 1  # one submesh per element type, duplicates dropped


def test_subfield_union_subfield_builds_aggregate():
    # The original report: `SubNodeField | SubNodeField` must work, building a
    # NodeField (distinct supports here → two zones).
    c, nodes, mesh = _two_zone_mesh()
    sa = pyrucast.NodeField(mesh[0], ["T"])[0]
    sb = pyrucast.NodeField(mesh[1], ["P"])[0]
    f = sa | sb
    assert len(f) == 2
    assert f.components() == ["T", "P"]


# ─── Arithmetic operators (binary, strict) ───────────────────────────────────


def _two_poi1_zones():
    """Two single-node POI1 zones on distinct supports, same Coords."""
    c = pyrucast.Coords(1)
    n0 = c.add_node([0.0])
    n1 = c.add_node([1.0])
    za = pyrucast.Mesh(c, "POI1")
    za.unit().add_cell([n0])
    zb = pyrucast.Mesh(c, "POI1")
    zb.unit().add_cell([n1])
    return c, [n0, n1], za, zb


def test_subfield_plus_subfield_same_support():
    # Two node fields on the *same* POI1 mesh share the support handle.
    c, nodes, sm = _poi1_with(2)
    a = pyrucast.NodeField(sm, ["T"])[0]
    b = pyrucast.NodeField(sm, ["T"])[0]
    a.set_value(nodes[0], "T", 1.0)
    a.set_value(nodes[1], "T", 2.0)
    b.set_value(nodes[0], "T", 10.0)
    b.set_value(nodes[1], "T", 20.0)
    s = a + b
    assert s.value(nodes[0], "T") == 11.0
    assert s.value(nodes[1], "T") == 22.0


def test_subfield_plus_subfield_disjoint_components_passes_through():
    # Union/passthrough arithmetic: disjoint components each pass through raw,
    # no error (mirrors the Rust `subfield_operator_uses_union_passthrough`).
    c, nodes, sm = _poi1_with(1)
    a = pyrucast.NodeField(sm, ["T"])[0]
    a.set_value(nodes[0], "T", 5.0)
    b = pyrucast.NodeField(sm, ["P"])[0]
    b.set_value(nodes[0], "P", 9.0)
    s = a + b
    assert s.components() == ["T", "P"]
    assert s.value(nodes[0], "T") == 5.0
    assert s.value(nodes[0], "P") == 9.0


def test_field_plus_field_same_decomposition():
    c, nodes, sm = _poi1_with(2)
    a = pyrucast.NodeField(sm, ["T"])
    b = pyrucast.NodeField(sm, ["T"])
    a[0].set_value(nodes[0], "T", 1.0)
    b[0].set_value(nodes[0], "T", 5.0)
    s = a + b
    assert s.value(nodes[0], "T") == 6.0


def test_field_scalar_and_per_component():
    c, nodes, sm = _poi1_with(2)
    f = pyrucast.NodeField(sm, ["T"])
    f[0].set_value(nodes[0], "T", 1.0)
    g = f * 3.0
    assert g.value(nodes[0], "T") == 3.0
    assert f.value(nodes[0], "T") == 1.0  # original untouched
    f.add_to_component("T", 100.0)
    assert f.value(nodes[0], "T") == 101.0


def test_field_plus_subfield_targets_matching_zone():
    c, nodes, za, zb = _two_poi1_zones()
    f = pyrucast.NodeField(za, ["T"]) | pyrucast.NodeField(zb, ["T"])
    assert len(f) == 2
    f[0].set_value(nodes[0], "T", 1.0)
    f[1].set_value(nodes[1], "T", 7.0)
    sub = pyrucast.NodeField(za, ["T"])[0]  # same support as zone 0
    sub.set_value(nodes[0], "T", 10.0)
    g = f + sub
    assert len(g) == 2
    assert g.value(nodes[0], "T") == 11.0  # matching zone updated
    assert g.value(nodes[1], "T") == 7.0  # other zone unchanged


def test_op_bad_operand_raises_type_error():
    c, nodes, sm = _poi1_with(1)
    f = pyrucast.NodeField(sm, ["T"])
    for bad_lhs in (f, f[0]):
        try:
            bad_lhs + "nope"
        except TypeError:
            pass
        else:
            raise AssertionError("expected TypeError for str operand")
