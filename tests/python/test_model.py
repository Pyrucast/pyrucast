"""Python tests for Model + Physics (Phase 2 step 7)."""

import pyrucast


def _seg2_heat_model(length=1.0, k=1.0, dirichlet_left=False):
    """Build a 1-D heat-conduction model on a single SEG2 element.
    Returns (config, mesh, fes, sub_fespace, materials, model, a, b).
    """
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([length])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]

    # ElementField aggregate: one SubElementField per FE subspace.
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", k)

    model = pyrucast.Model.heat_conduction(fes)
    if dirichlet_left:
        imposed = pyrucast.poi1_from_nodes([a])
        multiplier = pyrucast.barycenter(imposed)
        model = model | pyrucast.Model.dirichlet("T", "q", imposed, multiplier)
    return c, mesh, fes, sub, materials, model, a, b


# ─── Variable name introspection ────────────────────────────────────────────


def test_heat_conduction_alone_primal_dual():
    _, _, _, _, _, model, *_ = _seg2_heat_model()
    assert model.primal_vars() == ["T"]
    assert model.dual_vars() == ["q"]


def test_with_dirichlet_includes_lagrange_names():
    _, _, _, _, _, model, *_ = _seg2_heat_model(dirichlet_left=True)
    assert model.primal_vars() == ["T", "lambda_T"]
    # The Dirichlet dual is "imposed_T" — distinct from the primal "T".
    assert model.dual_vars() == ["q", "imposed_T"]


# ─── Stiffness assembly ─────────────────────────────────────────────────────


def test_heat_conduction_single_seg2_stiffness():
    length = 2.0
    k_val = 1.5
    _, _, _, _, materials, model, a, b = _seg2_heat_model(length=length, k=k_val)
    K = pyrucast.stiffness(model, materials)
    assert K.n_rows() == 2
    assert K.n_cols() == 2
    expected = k_val / length
    tol = 1e-12
    assert abs(K.get(a, "q", a, "T") - expected) < tol
    assert abs(K.get(a, "q", b, "T") + expected) < tol
    assert abs(K.get(b, "q", a, "T") + expected) < tol
    assert abs(K.get(b, "q", b, "T") - expected) < tol


def test_two_seg2_assembly_is_tridiagonal():
    c = pyrucast.Coords(1)
    n0 = c.add_node([0.0])
    n1 = c.add_node([1.0])
    n2 = c.add_node([2.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([n0, n1])
    mesh.unit().add_cell([n1, n2])
    fes = pyrucast.FiniteElementSpace(mesh)
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", 1.0)

    model = pyrucast.Model.heat_conduction(fes)
    K = pyrucast.stiffness(model, materials)
    assert K.n_rows() == 3
    assert K.n_cols() == 3

    def v(i, j):
        return K.get(i, "q", j, "T")

    tol = 1e-12
    assert abs(v(n0, n0) - 1.0) < tol
    assert abs(v(n0, n1) + 1.0) < tol
    assert abs(v(n0, n2)) < tol
    assert abs(v(n1, n0) + 1.0) < tol
    assert abs(v(n1, n1) - 2.0) < tol
    assert abs(v(n1, n2) + 1.0) < tol
    assert abs(v(n2, n0)) < tol
    assert abs(v(n2, n1) + 1.0) < tol
    assert abs(v(n2, n2) - 1.0) < tol


# ─── Dirichlet block ────────────────────────────────────────────────────────


def test_dirichlet_creates_multiplier_node_and_writes_both_blocks():
    c, _, _, _, materials, model, a, _b = _seg2_heat_model(dirichlet_left=True)
    # The Coords now has 3 live nodes (2 real + 1 multiplier).
    assert c.node_count() == 3

    K = pyrucast.stiffness(model, materials)
    assert K.n_rows() == 3
    assert K.n_cols() == 3

    # Locate the multiplier node: the row labelled "imposed_T" sits on the
    # one new node id.
    rows = K.row_dofs()
    mult_id = next(nid for (nid, name) in rows if name == "imposed_T")
    assert mult_id != a.id  # new node
    mult = c.acquire(mult_id)

    # C entry (mult, "imposed_T") × (a, "T")
    assert K.get(mult, "imposed_T", a, "T") == 1.0
    # Cᵀ entry (a, "q") × (mult, "lambda_T")
    assert K.get(a, "q", mult, "lambda_T") == 1.0


def test_dirichlet_empty_constraint_mesh_rejected():
    c = pyrucast.Coords(1)
    empty = pyrucast.Mesh(c, "POI1")  # one POI1 submesh, zero cells
    try:
        pyrucast.Model.dirichlet("T", "q", empty, empty)
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("expected error for empty constraint mesh")


# ─── Mass (v0: empty stub) ──────────────────────────────────────────────────


def test_mass_is_empty_in_v0():
    _, _, _, _, _, model, *_ = _seg2_heat_model()
    M = pyrucast.mass(model)
    assert M.n_rows() == 0
    assert M.n_cols() == 0


# ─── Physics nature + filtering ─────────────────────────────────────────────


def test_submodel_physics_nature():
    _, _, _, _, _, model, *_ = _seg2_heat_model(dirichlet_left=True)
    # physics() is a list of nature tags (one per plain physics).
    assert [model[i].physics() for i in range(len(model))] == [
        ["thermal"],
        ["constraint"],
    ]


def test_model_filter_by_physics():
    _, _, _, _, _, model, *_ = _seg2_heat_model(dirichlet_left=True)
    assert len(model) == 2

    thermal = model.filter("thermal")
    assert len(thermal) == 1 and thermal[0].physics() == ["thermal"]

    constraint = model.filter("constraint")
    assert len(constraint) == 1 and constraint[0].physics() == ["constraint"]

    # A nature no sub-model has yields an empty model.
    assert len(model.filter("mechanical")) == 0


def test_model_filter_unknown_tag_raises():
    _, _, _, _, _, model, *_ = _seg2_heat_model()
    try:
        model.filter("bogus")
    except ValueError as err:
        assert "bogus" in str(err)
    else:
        raise AssertionError("expected ValueError for unknown physics tag")


def test_assembled_blocks_carry_physics_and_matrix_filter():
    _, _, _, _, materials, model, *_ = _seg2_heat_model(dirichlet_left=True)
    K = pyrucast.stiffness(model, materials)

    # Every assembled block is tagged (computed heat block + literal C/Cᵀ).
    assert all(K[i].physics() for i in range(len(K)))
    # The matrix as a whole reports both natures present.
    present = set(K.physics())
    assert {"thermal", "constraint"} <= present

    kc = K.filter("constraint")
    assert len(kc) == 2  # the Dirichlet C / Cᵀ pair
    assert all(kc[i].physics() == ["constraint"] for i in range(len(kc)))

    kt = K.filter("thermal")
    assert len(kt) == 1


# ─── repr / str ─────────────────────────────────────────────────────────────


def test_repr_and_str():
    _, _, _, _, _, model, *_ = _seg2_heat_model(dirichlet_left=True)
    assert "Model" in repr(model)
    s = str(model)
    assert "Model" in s
    assert "2 sub-model" in s
