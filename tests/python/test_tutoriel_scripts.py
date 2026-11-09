"""What the beginner tutorial scripts (`tutoriel/*.py`) compute.

`script/run_examples.sh` only proves that the scripts run to the end. This
file checks what the book says they compute, measured on the solved fields
rather than read back from the script's own declarations: a script that
imposes a temperature on the wrong multiplier nodes still runs, but the bore
is no longer held at that temperature.

`tutoriel/thermique.py` plots every step; its plotting helpers are replaced.
The stepping scripts call `plot` directly: their figures go to a temporary
directory, as `script/run_examples.sh` sends them.
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


def _bore(script, mesh):
    """The bore nodes of `mesh`, found again from the geometry alone."""
    return pyrucast.mesh.consolidate(
        pyrucast.mesh.points_on_cylinder(
            mesh,
            [script.LENGTH, -script.THICKNESS, script.HEIGHT / 2.0],
            [script.LENGTH, 2.0 * script.THICKNESS, script.HEIGHT / 2.0],
            script.HOLE_RADIUS,
        )
    )


@pytest.fixture(scope="module")
def stepping_runs(tmp_path_factory):
    """Run a stepping script of the tutorial once, figures in a temporary
    directory; return the module and the `data` of each `step_by_step` call.

    The calls are recorded by wrapping `pyrucast.stepping.step_by_step`: the
    script stays exactly as the book shows it.
    """
    runs = {}

    def run(name):
        if name in runs:
            return runs[name]
        with pytest.MonkeyPatch.context() as mp:
            mp.syspath_prepend(str(TUTORIEL))
            script = __import__(name)
            mp.setattr(script, "OUT", str(tmp_path_factory.mktemp(name)))
            recorded = []
            step_by_step = pyrucast.stepping.step_by_step

            def recording(data):
                step_by_step(data)
                recorded.append(data)

            mp.setattr(pyrucast.stepping, "step_by_step", recording)
            script.main()
        runs[name] = (script, recorded)
        return runs[name]

    return run


def test_transient_holds_the_bore_once_its_ramp_is_over(stepping_runs):
    """From 0.1 t_f on, the bore sits at T_IMPOSEE at every instant."""
    script, (data,) = stepping_runs("thermique_transitoire")
    support = pyrucast.mesh.consolidate(data["initial"]["temperature"].support())
    bore = _bore(script, support)
    held = [r for r in data["results"] if r["time"] >= script.T_ALESAGE_CHAUD - 1e-9]
    assert len(held) == 91
    for record in held:
        on_bore = pyrucast.node_field.restrict(record["temperature"], bore)
        assert on_bore.min("T") == pytest.approx(script.T_IMPOSEE, abs=1e-9)
        assert on_bore.max("T") == pytest.approx(script.T_IMPOSEE, abs=1e-9)


def test_transient_settles_onto_the_steady_state(stepping_runs, thermal_fields):
    """At the last instant, node by node, the transient is the steady field of
    `tutoriel/thermique.py` (step 2), solved directly on its own mesh."""
    _, (data,) = stepping_runs("thermique_transitoire")
    last = data["results"][-1]["temperature"]
    _, solved = thermal_fields
    steady_volume, steady = solved["thermique-complet.svg"]
    transient_nodes = pyrucast.mesh.consolidate(last.support())
    steady_nodes = pyrucast.mesh.consolidate(pyrucast.mesh.to_poi1(steady_volume))
    assert steady_nodes.cell_count == transient_nodes.cell_count == 4000
    worst = 0.0
    for i in range(steady_nodes.cell_count):
        node = steady_nodes.node(0, i, 0)
        twin = transient_nodes.nearest_node(node.position())
        assert twin.position() == pytest.approx(node.position(), abs=1e-12)
        worst = max(worst, abs(last.value(twin, "T") - steady.value(node, "T")))
    # Measured 0.013 °C (2026-11-09): the part has not quite settled at t_f.
    # A ramp gone wrong — a source never on, a flux left off — leaves tens of
    # degrees.
    assert worst < 0.05
