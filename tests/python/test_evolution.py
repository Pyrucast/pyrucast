"""Python tests for Evolution / SubEvolution — interpolation, types, and the
field-through-curve transfer-function mode."""

import pytest

import pyrucast


def _temperature_field(values):
    """A single-zone POI1 NodeField of component 'T' holding `values`."""
    c = pyrucast.Coords(1)
    nodes = [c.add_node([float(i)]) for i in range(len(values))]
    mesh = pyrucast.Mesh(c, "POI1")
    for n in nodes:
        mesh.unit().add_cell([n])
    field = pyrucast.NodeField(mesh, ["T"])
    for n, v in zip(nodes, values):
        field[0].set_value(n, "T", v)
    return field, nodes


# ─── Scalar interpolation (unchanged behaviour) ──────────────────────────────


def test_scalar_interpolation_midpoint():
    se = pyrucast.SubEvolution([(0.0, 10.0), (1.0, 20.0)])
    assert se.interpolate(0.5) == 15.0


def test_aggregate_scalar_returns_list():
    e = pyrucast.Evolution([(0.0, 10.0), (1.0, 20.0)])
    assert e.interpolate(0.5) == [15.0]


def test_out_of_range_override():
    se = pyrucast.SubEvolution([(0.0, 10.0), (1.0, 20.0)])
    with pytest.raises(Exception):
        se.interpolate(2.0)
    assert se.interpolate(2.0, out_of_range="clamp") == 20.0


# ─── Types ───────────────────────────────────────────────────────────────────


def test_types_round_trip():
    e = pyrucast.Evolution(
        [(0.0, 0.0), (1.0, 1.0)], abscissa_type="T", ordinate_type="young"
    )
    assert e.abscissa_type() == "T"
    assert e.ordinate_type() == "young"


def test_ordinate_type_rejected_on_field_evolution():
    field, _ = _temperature_field([1.0, 2.0])
    with pytest.raises(Exception):
        pyrucast.Evolution([(0.0, field)], ordinate_type="oops")


# ─── Field-through-curve (transfer function) ─────────────────────────────────


def test_interpolate_field_maps_pointwise():
    # E(T) doubling curve, typed on both axes.
    law = pyrucast.Evolution(
        [(0.0, 0.0), (10.0, 20.0)], abscissa_type="T", ordinate_type="E"
    )
    temperature, nodes = _temperature_field([0.0, 5.0, 10.0])
    young = law.interpolate(temperature)
    assert young.components() == ["E"]
    assert young[0].value(nodes[0], "E") == 0.0
    assert young[0].value(nodes[1], "E") == 10.0
    assert young[0].value(nodes[2], "E") == 20.0


def test_interpolate_field_type_mismatch():
    # Abscissa type 'P' has no counterpart in a field whose component is 'T'.
    law = pyrucast.Evolution([(0.0, 0.0), (1.0, 1.0)], abscissa_type="P")
    temperature, _ = _temperature_field([0.5])
    with pytest.raises(Exception):
        law.interpolate(temperature)


def test_interpolate_field_requires_abscissa_type():
    law = pyrucast.Evolution([(0.0, 0.0), (1.0, 1.0)])
    temperature, _ = _temperature_field([0.5])
    with pytest.raises(Exception):
        law.interpolate(temperature)


def test_interpolate_field_requires_single_curve():
    # A two-curve aggregate is ambiguous for a field map → rejected.
    agg = pyrucast.SubEvolution(
        [(0.0, 0.0), (1.0, 1.0)], abscissa_type="T"
    ) | pyrucast.SubEvolution([(0.0, 0.0), (1.0, 1.0)], abscissa_type="T")
    temperature, _ = _temperature_field([0.5])
    with pytest.raises(Exception):
        agg.interpolate(temperature)


def test_sub_evolution_interpolate_field():
    curve = pyrucast.SubEvolution(
        [(0.0, 0.0), (10.0, 20.0)], abscissa_type="T", ordinate_type="E"
    )
    temperature, nodes = _temperature_field([2.5])
    out = curve.interpolate(temperature[0])
    assert out.components() == ["E"]
    assert out.value(nodes[0], "E") == 5.0
