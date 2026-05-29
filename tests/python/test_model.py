"""Python tests for Model + Physics (Phase 2 step 7)."""

import pyrucast


def _seg2_heat_model(length=1.0, k=1.0, dirichlet_left=False):
    """Build a 1-D heat-conduction model on a single SEG2 element.
    Returns (config, mesh, fes, sub_fespace, materials, model, a, b).
    """
    c = pyrucast.Configuration(1)
    a = c.add_node([0.0])
    b = c.add_node([length])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.add_cell([a.id, b.id])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]

    # ElementField aggregate: one SubElementField per FE subspace.
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", k)

    model = pyrucast.Model()
    model.add_sub(pyrucast.SubModel.heat_conduction(sub))
    if dirichlet_left:
        model.add_sub(
            pyrucast.SubModel.dirichlet(c, "T", "q", [a.id])
        )
    return c, mesh, fes, sub, materials, model, a, b


# ─── Variable name introspection ────────────────────────────────────────────


def test_heat_conduction_alone_primal_dual():
    _, _, _, _, _, model, *_ = _seg2_heat_model()
    assert model.primal_vars() == ["T"]
    assert model.dual_vars() == ["q"]


def test_with_dirichlet_includes_lagrange_names():
    _, _, _, _, _, model, *_ = _seg2_heat_model(dirichlet_left=True)
    assert model.primal_vars() == ["T", "lambda_T"]
    assert model.dual_vars() == ["q", "T"]


# ─── Stiffness assembly ─────────────────────────────────────────────────────


def test_heat_conduction_single_seg2_stiffness():
    length = 2.0
    k_val = 1.5
    _, _, _, _, materials, model, a, b = _seg2_heat_model(length=length, k=k_val)
    K = model.stiffness(materials)
    assert K.n_rows() == 2
    assert K.n_cols() == 2
    expected = k_val / length
    tol = 1e-12
    assert abs(K.get(a.id, "q", a.id, "T") - expected) < tol
    assert abs(K.get(a.id, "q", b.id, "T") + expected) < tol
    assert abs(K.get(b.id, "q", a.id, "T") + expected) < tol
    assert abs(K.get(b.id, "q", b.id, "T") - expected) < tol


def test_two_seg2_assembly_is_tridiagonal():
    c = pyrucast.Configuration(1)
    n0 = c.add_node([0.0])
    n1 = c.add_node([1.0])
    n2 = c.add_node([2.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.add_cell([n0.id, n1.id])
    mesh.add_cell([n1.id, n2.id])
    fes = pyrucast.FiniteElementSpace(mesh)
    sub = fes[0]
    materials = pyrucast.ElementField(fes, ["k"])
    materials[0].set_uniform("k", 1.0)

    model = pyrucast.Model()
    model.add_sub(pyrucast.SubModel.heat_conduction(sub))
    K = model.stiffness(materials)
    assert K.n_rows() == 3
    assert K.n_cols() == 3

    def v(i, j):
        return K.get(i, "q", j, "T")

    tol = 1e-12
    assert abs(v(n0.id, n0.id) - 1.0) < tol
    assert abs(v(n0.id, n1.id) + 1.0) < tol
    assert abs(v(n0.id, n2.id)) < tol
    assert abs(v(n1.id, n0.id) + 1.0) < tol
    assert abs(v(n1.id, n1.id) - 2.0) < tol
    assert abs(v(n1.id, n2.id) + 1.0) < tol
    assert abs(v(n2.id, n0.id)) < tol
    assert abs(v(n2.id, n1.id) + 1.0) < tol
    assert abs(v(n2.id, n2.id) - 1.0) < tol


# ─── Dirichlet block ────────────────────────────────────────────────────────


def test_dirichlet_creates_multiplier_node_and_writes_both_blocks():
    c, _, _, _, materials, model, a, _b = _seg2_heat_model(dirichlet_left=True)
    # The Configuration now has 3 live nodes (2 real + 1 multiplier).
    assert c.node_count() == 3

    K = model.stiffness(materials)
    assert K.n_rows() == 3
    assert K.n_cols() == 3

    # Locate the multiplier node: the row labelled "T" sits on the
    # one new node id.
    rows = K.row_dofs()
    mult_id = next(nid for (nid, name) in rows if name == "T")
    assert mult_id != a.id  # new node

    # C entry (mult, "T") × (a, "T")
    assert K.get(mult_id, "T", a.id, "T") == 1.0
    # Cᵀ entry (a, "q") × (mult, "lambda_T")
    assert K.get(a.id, "q", mult_id, "lambda_T") == 1.0


def test_dirichlet_empty_constraint_list_rejected():
    c = pyrucast.Configuration(1)
    try:
        pyrucast.SubModel.dirichlet(c, "T", "q", [])
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for empty constraints")


# ─── Mass (v0: empty stub) ──────────────────────────────────────────────────


def test_mass_is_empty_in_v0():
    _, _, _, _, _, model, *_ = _seg2_heat_model()
    M = model.mass()
    assert M.n_rows() == 0
    assert M.n_cols() == 0


# ─── repr / str ─────────────────────────────────────────────────────────────


def test_repr_and_str():
    _, _, _, _, _, model, *_ = _seg2_heat_model(dirichlet_left=True)
    assert "Model" in repr(model)
    s = str(model)
    assert "Model" in s
    assert "2 sub-model" in s
