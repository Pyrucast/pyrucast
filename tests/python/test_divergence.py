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
    field = pyrucast.ElementField(fes, ["F_x"])
    field[0].set_uniform("F_x", a)

    div = pyrucast.node_field.divergence(field, "F")
    assert abs(div.value(n[0], "div_F") + a) < 1e-12  # −a
    assert abs(div.value(n[1], "div_F")) < 1e-12  # 0 (interior)
    assert abs(div.value(n[2], "div_F") - a) < 1e-12  # +a


def test_divergence_rejects_a_name_of_neither_rank():
    c = pyrucast.Coords(2)
    a = c.add_node([0.0, 0.0])
    b = c.add_node([1.0, 0.0])
    d = c.add_node([0.0, 1.0])
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell([a, b, d])
    fes = pyrucast.FiniteElementSpace(mesh)
    field = pyrucast.ElementField(fes, ["F_x"])  # ni F_x,F_y ni F_xx,F_xy,F_yy
    try:
        pyrucast.node_field.divergence(field, "F")
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("expected an error: neither a vector nor a tensor")


def test_divergence_of_a_tensor_is_a_vector():
    """Le même opérateur, un rang au-dessus : le préfixe suffit à trancher.

    La divergence d'un tenseur des contraintes uniforme est de somme nulle —
    c'est l'équilibre global, et ce sont les forces internes d'un solide.
    """
    c = pyrucast.Coords(2)
    n = [c.add_node(p) for p in ([0.0, 0.0], [2.0, 0.0], [0.0, 2.0])]
    mesh = pyrucast.Mesh(c, "TRI3")
    mesh.unit().add_cell(n)
    fes = pyrucast.FiniteElementSpace(mesh)

    sig = pyrucast.ElementField(fes, ["sigma_xx", "sigma_xy", "sigma_yy"])
    sig[0].set_uniform("sigma_xx", 100.0)

    div = pyrucast.node_field.divergence(sig, "sigma")
    assert div[0].components() == ["div_sigma_x", "div_sigma_y"]
    total = sum(div.value(node, "div_sigma_x") for node in n)
    assert abs(total) < 1e-9
