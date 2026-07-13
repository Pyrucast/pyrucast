"""Python test for the divergence operator (adjoint of gradient)."""

import pyrucast


def test_divergence_uniform_1d_telescopes():
    """Two SEG2 on [0,2], uniform F=(a) ⇒ weak divergence [−a, 0, +a]."""
    c = pyrucast.Coords(1)
    n = [c.add_node([float(i)]) for i in range(3)]
    mesh = pyrucast.Mesh(c, "SEG2")
    mesh.unit().add_cell([n[0], n[1]])
    mesh.unit().add_cell([n[1], n[2]])
    fes = pyrucast.FiniteElementSpace(mesh)

    a = 3.0
    field = pyrucast.ElementField(fes, ["Fx"])
    field[0].set_uniform("Fx", a)

    div = pyrucast.field.divergence(field)
    assert abs(div.value(n[0], "div") + a) < 1e-12  # −a
    assert abs(div.value(n[1], "div")) < 1e-12  # 0 (interior)
    assert abs(div.value(n[2], "div") - a) < 1e-12  # +a


def test_divergence_rejects_wrong_component_count():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    d = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([a, b, d])
    fes = pyrucast.FiniteElementSpace(mesh)
    field = pyrucast.ElementField(fes, ["Fx"])  # 1 comp on a 2-D space
    try:
        pyrucast.field.divergence(field)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("expected an error for wrong component count")
