"""Python tests for the `flux` load sub-model (Cast3m FLUX/SOUR).

A distributed load is a **term of the model**, not a vector built beside it: it
joins the model, its density joins the material, and one asks the model for its
external forces. Its derivative with respect to the solution is zero, so it
contributes to no matrix.
"""

import pyrucast


def _line(n_cells):
    """A unit SEG2 line of `n_cells` equal cells, and its FE space."""
    c = pyrucast.Coords(1)
    nodes = [c.add_node([i / n_cells]) for i in range(n_cells + 1)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(n_cells):
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    return nodes, pyrucast.FiniteElementSpace(mesh)


def test_uniform_flux_consistent_loads_on_seg2_line():
    """Uniform flux φ on a SEG2 line: interior node gets φ·h, ends φ·h/2."""
    (a, b, d), fes = _line(2)

    phi, h = 3.0, 0.5
    model = pyrucast.model.flux(fes, pyrucast.model.heat_conduction(fes), "q")
    materials = pyrucast.element_field.material_field(model, [("phi_q", phi)])
    load = pyrucast.node_field.external_forces(model, materials)

    assert abs(load.value(a, "q") - phi * h / 2) < 1e-12
    assert abs(load.value(b, "q") - phi * h) < 1e-12
    assert abs(load.value(d, "q") - phi * h / 2) < 1e-12
    total = load.value(a, "q") + load.value(b, "q") + load.value(d, "q")
    assert abs(total - phi * 1.0) < 1e-12  # Σ = φ · L


def test_density_from_element_field_matches_the_uniform_one():
    """A hand-built material field gives the same loads as the scalar that
    `material_field` would have spread for us."""
    (a, _), fes = _line(1)
    phi = 7.5
    model = pyrucast.model.flux(fes, pyrucast.model.heat_conduction(fes), "q")

    by_hand = pyrucast.ElementField(fes, ["phi_q"])
    by_hand[0].set_uniform("phi_q", phi)
    from_field = pyrucast.node_field.external_forces(model, by_hand)

    spread = pyrucast.element_field.material_field(model, [("phi_q", phi)])
    from_uniform = pyrucast.node_field.external_forces(model, spread)
    assert abs(from_field.value(a, "q") - from_uniform.value(a, "q")) < 1e-12


def test_a_load_contributes_to_no_matrix():
    """`∂r/∂u = 0` : a given density does not move when the solution does, so a
    load assembles to nothing."""
    _, fes = _line(1)
    model = pyrucast.model.flux(fes, pyrucast.model.heat_conduction(fes), "q")
    materials = pyrucast.element_field.material_field(model, [("phi_q", 1.0)])

    k = pyrucast.matrix.stiffness(model, materials)
    assert k.dense() == []


def test_a_missing_density_is_named():
    """Forgetting the density is refused at assembly, by name — not read as a
    load of zero."""
    _, fes = _line(1)
    model = pyrucast.model.flux(fes, pyrucast.model.heat_conduction(fes), "q")
    try:
        pyrucast.element_field.material_field(model, [("k", 1.0)])
    except (ValueError, RuntimeError) as e:
        assert "phi_q" in str(e)
    else:
        raise AssertionError("expected a missing-component error")
