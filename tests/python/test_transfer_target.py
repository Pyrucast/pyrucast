"""The exchange laws take their nature from the model they couple into.

`boundary_transfer` and `radiation` write into rows of another physics. They are
built against it — their `target` — which must assemble every `(primal, dual)`
pair they name and which gives them their nature: nothing in `("T", "q")` says
« thermal », the conduction assembling it does. The interface is covered by
`test_interface_transfer.py`.
"""

import pyrucast


def _plate_and_edge():
    """A QUA4 plate and the FE space of its left edge."""
    c = pyrucast.Coords(2)
    n = [c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0])]
    plate = pyrucast.Mesh(c, "QUA4")
    plate.unit().add_cell(n)
    edge = pyrucast.Mesh(c, "SEG2")
    edge.unit().add_cell([n[0], n[3]])
    return pyrucast.FiniteElementSpace(plate), pyrucast.FiniteElementSpace(edge)


def _raises(call, fragment):
    """A construction-time modelling error surfaces as `RuntimeError`."""
    try:
        call()
    except RuntimeError as exc:
        assert fragment in str(exc), str(exc)
    else:  # pragma: no cover - the constructor must refuse
        raise AssertionError(f"expected a RuntimeError mentioning {fragment!r}")


def test_a_film_is_thermal_because_its_conduction_is():
    plate, edge = _plate_and_edge()
    conduction = pyrucast.model.heat_conduction(plate)
    film = pyrucast.model.boundary_transfer(edge, conduction, [("T", "q")])
    assert film[0].physics() == ["thermal"]
    assert film[0].material_components() == ["h_T", "a_ext_T"]


def test_a_foundation_is_mechanical_because_its_elasticity_is():
    plate, edge = _plate_and_edge()
    elasticity = pyrucast.model.elasticity(plate, "plane_stress")
    foundation = pyrucast.model.boundary_transfer(
        edge, elasticity, [("u_x", "f_x"), ("u_y", "f_y")]
    )
    assert foundation[0].physics() == ["mechanical"]
    assert foundation[0].material_components() == [
        "h_u_x",
        "h_u_y",
        "a_ext_u_x",
        "a_ext_u_y",
    ]


def test_a_pair_the_target_does_not_assemble_is_rejected():
    plate, edge = _plate_and_edge()
    conduction = pyrucast.model.heat_conduction(plate)
    _raises(
        lambda: pyrucast.model.boundary_transfer(edge, conduction, [("u_x", "f_x")]),
        "assembles no `u_x` paired with `f_x`",
    )


def test_pairs_of_two_natures_are_rejected():
    plate, edge = _plate_and_edge()
    both = pyrucast.model.heat_conduction(plate) | pyrucast.model.elasticity(
        plate, "plane_stress"
    )
    _raises(
        lambda: pyrucast.model.boundary_transfer(
            edge, both, [("T", "q"), ("u_x", "f_x")]
        ),
        "single nature",
    )


def test_radiation_needs_a_conduction_beneath_it():
    plate, edge = _plate_and_edge()
    conduction = pyrucast.model.heat_conduction(plate)
    radiation = pyrucast.model.radiation(edge, conduction)
    # Its natures are its own; the target only proves `q` is assembled.
    assert radiation[0].physics() == ["thermal", "radiation"]
    _raises(
        lambda: pyrucast.model.radiation(edge, pyrucast.model.fick(plate, "H2")),
        "assembles no `T` paired with `q`",
    )
