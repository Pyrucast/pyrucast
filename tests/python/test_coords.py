"""Python tests for Coords + Node (Phase 2)."""

import gc as pygc

import pyrucast


def test_create_coords():
    c = pyrucast.Coords(dim=2)
    assert c.dim == 2
    assert c.node_count() == 0
    assert c.capacity() == 0
    assert c.active == 0
    assert c.names() == ["default"]


def test_add_node_returns_node():
    c = pyrucast.Coords(2)
    n = c.add_node([1.0, 2.0])
    assert n.id == 0
    assert n.coord() == [1.0, 2.0]
    assert c.node_count() == 1
    assert c.refcount(n.id) == 1


def test_gc_protects_referenced_nodes():
    c = pyrucast.Coords(1)
    n = c.add_node([42.0])
    nid = n.id
    # While Python holds `n`, refcount >= 1 → gc collects nothing.
    assert c.gc() == 0
    assert c.is_alive(nid)
    # del + collect forces the Drop of the Rust-side Node.
    del n
    pygc.collect()
    assert c.refcount(nid) == 0
    assert c.gc() == 1
    assert not c.is_alive(nid)


def test_ids_are_stable_after_gc():
    c = pyrucast.Coords(1)
    n = c.add_node([0.0])
    first_id = n.id
    del n
    pygc.collect()
    c.gc()
    m = c.add_node([1.0])
    assert m.id != first_id
    assert m.id == 1
    assert c.capacity() == 2


def test_set_coord_via_node():
    c = pyrucast.Coords(2)
    n = c.add_node([0.0, 0.0])
    n.set_coord([3.0, 4.0])
    assert n.coord() == [3.0, 4.0]


def test_configs_select():
    c = pyrucast.Coords(2)
    n = c.add_node([0.0, 0.0])
    s2 = c.add_config("deformed")
    assert s2 == 1
    c.select(1)
    n.set_coord([10.0, 20.0])
    c.select(0)
    assert n.coord() == [0.0, 0.0]
    c.select(1)
    assert n.coord() == [10.0, 20.0]


def test_acquire_shares_id():
    c = pyrucast.Coords(1)
    n = c.add_node([5.0])
    nid = n.id
    m = c.acquire(nid)
    assert m.id == nid
    assert c.refcount(nid) == 2
    del n
    del m
    pygc.collect()
    assert c.refcount(nid) == 0


def test_permutation_validation():
    c = pyrucast.Coords(1)
    _nodes = [c.add_node([float(k)]) for k in range(3)]
    c.set_permutation([2, 1, 0])
    assert c.permutation() == [2, 1, 0]
    c.clear_permutation()
    assert c.permutation() is None


def test_repr_and_str():
    c = pyrucast.Coords(2)
    n = c.add_node([0.0, 0.0])
    assert "Coords" in repr(c)
    s = str(c)
    assert "dim=2" in s
    assert "identity" in s
    assert "Node" in repr(n)
    assert str(n).startswith("<Node #")
