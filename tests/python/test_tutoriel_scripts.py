"""What the beginner tutorial scripts (`tutoriel/*.py`) compute.

`script/run_examples.sh` only proves that the scripts run to the end. This
file checks what the book says they compute, measured on the solved fields
rather than read back from the script's own declarations: a script that
imposes a temperature on the wrong multiplier nodes still runs, but the bore
is no longer held at that temperature.

The scripts plot every step; the plotting helpers are replaced so that the
test needs neither a screen nor the `viz` feature.
"""

import pathlib

import pyrucast
import pytest

TUTORIEL = pathlib.Path(__file__).resolve().parents[2] / "tutoriel"


@pytest.fixture
def thermal_fields(monkeypatch):
    """Run `tutoriel/thermique.py`; return its solved volume and temperatures."""
    monkeypatch.syspath_prepend(str(TUTORIEL))
    import thermique

    solved = {}
    monkeypatch.setattr(thermique, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        thermique,
        "show_nodefield",
        lambda mesh, field, title, file: solved.__setitem__(file, (mesh, field)),
    )
    thermique.main()
    return thermique, solved


@pytest.mark.parametrize(
    "figure", ["thermique-conduction.svg", "thermique-complet.svg"]
)
def test_thermal_bore_is_held_at_the_imposed_temperature(thermal_fields, figure):
    """Both steps hold the bore at T_IMPOSEE, and nothing is hotter than it."""
    thermique, solved = thermal_fields
    volume, temperature = solved[figure]

    # The bore nodes, found again from the geometry alone.
    bore = pyrucast.mesh.consolidate(
        pyrucast.mesh.points_on_cylinder(
            volume,
            [thermique.LENGTH, -thermique.THICKNESS, thermique.HEIGHT / 2.0],
            [thermique.LENGTH, 2.0 * thermique.THICKNESS, thermique.HEIGHT / 2.0],
            thermique.HOLE_RADIUS,
        )
    )
    on_bore = pyrucast.node_field.restrict(temperature, bore)

    assert on_bore.node_count > 0
    assert on_bore.min("T") == pytest.approx(thermique.T_IMPOSEE, abs=1e-9)
    assert on_bore.max("T") == pytest.approx(thermique.T_IMPOSEE, abs=1e-9)
    assert temperature.max("T") == pytest.approx(thermique.T_IMPOSEE, abs=1e-9)
