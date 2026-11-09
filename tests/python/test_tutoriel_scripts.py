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
            figures = str(tmp_path_factory.mktemp(name))
            # The helpers `show` and `show_nodefield`, reused from
            # tutoriel/maillage.py and tutoriel/thermique.py, read their own
            # module's OUT.
            for module in (script, __import__("maillage"), __import__("thermique")):
                mp.setattr(module, "OUT", figures)
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


def test_kelvin_run_without_radiation_is_the_transient(stepping_runs):
    """Shifted back by 273.15, the nonlinear script's run without radiation is
    the °C transient of `tutoriel/thermique_transitoire.py`, at every instant
    and every node: conduction and convection only see differences."""
    script, (sans, _) = stepping_runs("thermique_non_lineaire")
    _, (transient,) = stepping_runs("thermique_transitoire")
    in_celsius = pyrucast.mesh.consolidate(
        transient["results"][0]["temperature"].support()
    )
    in_kelvin = pyrucast.mesh.consolidate(sans["results"][0]["temperature"].support())
    assert in_celsius.cell_count == in_kelvin.cell_count == 4000
    nodes = [in_celsius.node(0, i, 0) for i in range(in_celsius.cell_count)]
    twins = [in_kelvin.nearest_node(node.position()) for node in nodes]
    assert len(sans["results"]) == len(transient["results"]) == 101
    worst = 0.0
    for celsius, kelvin in zip(transient["results"], sans["results"]):
        values_c = celsius["temperature"].values(nodes, "T")
        values_k = kelvin["temperature"].values(twins, "T")
        for c, k in zip(values_c, values_k):
            worst = max(worst, abs(k - script.ZERO_ABSOLU - c))
    # Measured 3.4e-13 K (2026-11-09): the shift is exact up to round-off.
    assert worst < 1e-9


def test_radiation_cools_every_point_followed(stepping_runs):
    """With radiation, each of the four points ends colder than without."""
    script, (sans, avec) = stepping_runs("thermique_non_lineaire")
    nodes = pyrucast.mesh.consolidate(sans["results"][0]["temperature"].support())
    for point in (script.POINT_A, script.POINT_B, script.POINT_C, script.POINT_D):
        node = nodes.nearest_node(point)
        without = sans["results"][-1]["temperature"].value(node, "T")
        with_radiation = avec["results"][-1]["temperature"].value(node, "T")
        assert with_radiation < without, point
