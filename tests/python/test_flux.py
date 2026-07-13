"""Python tests for the `flux` load operator (Cast3m FLUX/PRES)."""

import pyrucast


def test_uniform_flux_consistent_loads_on_seg2_line():
    """Uniform flux φ on a SEG2 line: interior node gets φ·h, ends φ·h/2."""
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([0.5])
    d = c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    mesh.unit().add_cell([b, d])
    fes = pyrucast.FiniteElementSpace(mesh)

    phi = 3.0
    h = 0.5
    load = pyrucast.assemble.flux(fes[0], phi, "q")
    assert abs(load.value(a, "q") - phi * h / 2) < 1e-12
    assert abs(load.value(b, "q") - phi * h) < 1e-12
    assert abs(load.value(d, "q") - phi * h / 2) < 1e-12
    total = load.value(a, "q") + load.value(b, "q") + load.value(d, "q")
    assert abs(total - phi * 1.0) < 1e-12  # Σ = φ · L


def test_flux_from_element_field_matches_uniform():
    """A single-component ElementField density gives the same loads as the
    uniform value it carries."""
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)

    phi = 7.5
    ef = pyrucast.ElementField(fes, ["phi"])
    ef[0].set_uniform("phi", phi)

    from_field = pyrucast.assemble.flux(fes[0], ef[0], "q")
    from_uniform = pyrucast.assemble.flux(fes[0], phi, "q")
    assert abs(from_field.value(a, "q") - from_uniform.value(a, "q")) < 1e-12


def test_flux_rejects_bad_density():
    """A density that is neither a float nor a SubElementField is rejected."""
    c = pyrucast.Coords(1)
    a = c.add_node([0.0])
    b = c.add_node([1.0])
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([a, b])
    fes = pyrucast.FiniteElementSpace(mesh)

    try:
        pyrucast.assemble.flux(fes[0], "not a density", "q")
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("expected an error for an invalid density")
