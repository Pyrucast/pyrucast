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

import itertools
import math
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
    directory; return the module, the `data` of each `step_by_step` call and
    the path of each file written by `pyrucast.save`.

    The calls are recorded by wrapping `pyrucast.stepping.step_by_step` and
    `pyrucast.save`: the script stays exactly as the book shows it. The
    constraints the script builds are recorded too, in `run.calls[name]`:
    `"dirichlet"` lists `(variable, constraint)`, `"contact"` the
    `(slave nodes, sub-model)` of each contact; with `solves=True`, `"solve"` lists `(matrix, rhs, solution)`
    — only for the scripts that do not step, whose solves are their own.
    """
    runs = {}
    calls = {}

    def run(name, solves=False):
        if name in runs:
            return runs[name]
        recorded_calls = {"dirichlet": [], "contact": [], "solve": []}
        calls[name] = recorded_calls
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

            saved = []
            save = pyrucast.save

            def saving(path, objects):
                save(path, objects)
                saved.append(path)

            dirichlet = pyrucast.model.dirichlet
            contact = pyrucast.model.contact
            solve = pyrucast.solver.solve

            def recording_dirichlet(nodes, target, variable, *args, **kwargs):
                constraint = dirichlet(nodes, target, variable, *args, **kwargs)
                recorded_calls["dirichlet"].append((variable, constraint))
                return constraint

            def recording_contact(slave, *args, **kwargs):
                sub_model = contact(slave, *args, **kwargs)
                recorded_calls["contact"].append((slave, sub_model))
                return sub_model

            def recording_solve(matrix, rhs, *args, **kwargs):
                solution = solve(matrix, rhs, *args, **kwargs)
                recorded_calls["solve"].append((matrix, rhs, solution))
                return solution

            mp.setattr(pyrucast.stepping, "step_by_step", recording)
            mp.setattr(pyrucast, "save", saving)
            mp.setattr(pyrucast.model, "dirichlet", recording_dirichlet)
            mp.setattr(pyrucast.model, "contact", recording_contact)
            if solves:
                mp.setattr(pyrucast.solver, "solve", recording_solve)
            script.main()
        runs[name] = (script, recorded, saved)
        return runs[name]

    run.calls = calls
    return run


def test_transient_holds_the_bore_once_its_ramp_is_over(stepping_runs):
    """From 0.1 t_f on, the bore sits at T_IMPOSEE at every instant."""
    script, (data,), _ = stepping_runs("thermique_transitoire")
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
    _, (data,), _ = stepping_runs("thermique_transitoire")
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
    script, (sans, _), _ = stepping_runs("thermique_non_lineaire")
    _, (transient,), _ = stepping_runs("thermique_transitoire")
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
    script, (sans, avec), _ = stepping_runs("thermique_non_lineaire")
    nodes = pyrucast.mesh.consolidate(sans["results"][0]["temperature"].support())
    for point in (script.POINT_A, script.POINT_B, script.POINT_C, script.POINT_D):
        node = nodes.nearest_node(point)
        without = sans["results"][-1]["temperature"].value(node, "T")
        with_radiation = avec["results"][-1]["temperature"].value(node, "T")
        assert with_radiation < without, point


def test_saved_file_holds_the_radiation_run_in_celsius(stepping_runs):
    """The file the nonlinear script writes, loaded again, holds its volume
    and, at every instant and every node, the radiation run's temperature in
    °C. The values are read through the nodes of the loaded volume: a field
    that came back on copies of them would read 0 there, a node it does not
    carry."""
    script, (_, avec), (path,) = stepping_runs("thermique_non_lineaire")
    loaded = pyrucast.load(path)
    volume = loaded["volume"]
    evolution = loaded["température (°C)"]
    assert volume.cell_count == 640
    assert evolution.shared_abscissas() == [r["time"] for r in avec["results"]]

    computed = pyrucast.mesh.consolidate(avec["results"][0]["temperature"].support())
    assert computed.cell_count == 4000
    nodes = [computed.node(0, i, 0) for i in range(computed.cell_count)]
    twins = [volume.nearest_node(node.position()) for node in nodes]
    for node, twin in zip(nodes, twins):
        assert twin.position() == pytest.approx(node.position(), abs=1e-12)
    worst = 0.0
    for frame, record in zip(evolution.frames(), avec["results"]):
        read = frame.values(twins, "T")
        kelvin = record["temperature"].values(nodes, "T")
        for celsius, k in zip(read, kelvin):
            worst = max(worst, abs(celsius - (k - script.ZERO_ABSOLU)))
    # The file holds the very values the script computed: 0 expected.
    assert worst < 1e-12


# ── Mécanique / Mechanics ─────────────────────────────────────────────────

WEIGHT = 2500.0 * 9.81  # N, the tutorial's 2,500 kg mass


def _reaction(solution, constraints, variable):
    """The force the clamp exerts on the part along `variable`: minus the
    sum of the multipliers of the constraints on that variable."""
    total = 0.0
    for name, constraint in constraints:
        if name == variable:
            on_multipliers = pyrucast.node_field.restrict(
                solution, constraint.multiplier_mesh()
            )
            total -= on_multipliers.sum("lambda")
    return total


def test_elasticity_clamp_gives_back_each_way_of_laying_the_mass(stepping_runs):
    """The clamp returns the load each case applies: the weight for the point
    force and for the vertical surface force, 2mg/π for the pressure — normal
    to the half-cylinder, it only carries its horizontal projection. Measured
    on the solved multipliers, against the weight computed here."""
    stepping_runs("mecanique_elastique", solves=True)
    calls = stepping_runs.calls["mecanique_elastique"]
    expected = [WEIGHT, 2.0 * WEIGHT / math.pi, WEIGHT]
    assert len(calls["solve"]) == 3
    for (_, rhs, u), carried in zip(calls["solve"], expected):
        applied = -rhs.sum("f_z")
        # The QUA8 half-bore's area is π r e to 5e-7 (measured 2026-11-10):
        # the surface force carries the weight to that much. The pressure's
        # vertical part is exact, the projection of straight edges.
        assert applied == pytest.approx(carried, rel=1e-6)
        assert _reaction(u, calls["dirichlet"], "u_z") == pytest.approx(
            applied, rel=1e-9
        )
        assert abs(_reaction(u, calls["dirichlet"], "u_x")) < 1e-6 * WEIGHT
        assert abs(_reaction(u, calls["dirichlet"], "u_y")) < 1e-6 * WEIGHT


@pytest.mark.parametrize("name", ["mecanique_dilatation", "mecanique_heterogene"])
def test_expansion_adds_no_resultant_the_clamp_still_carries_the_weight(
    stepping_runs, name
):
    """The right-hand side is the surface force plus the expansion's nodal
    forces: it sums to the weight alone, the expansion's forces summing to
    zero, and the clamp gives back the weight."""
    stepping_runs("thermique_non_lineaire")  # writes the file the script reads
    stepping_runs(name, solves=True)
    calls = stepping_runs.calls[name]
    ((_, rhs, u),) = calls["solve"]
    applied = -rhs.sum("f_z")
    # The surface force alone, the weight to the half-bore's area (5e-7).
    assert applied == pytest.approx(WEIGHT, rel=1e-6)
    assert abs(rhs.sum("f_x")) < 1e-9 * WEIGHT
    assert abs(rhs.sum("f_y")) < 1e-9 * WEIGHT
    assert _reaction(u, calls["dirichlet"], "u_z") == pytest.approx(applied, rel=1e-9)
    assert abs(_reaction(u, calls["dirichlet"], "u_x")) < 1e-6 * WEIGHT


def test_bell_shaped_expansion_lengthens_the_lug_further(stepping_runs):
    """α ≥ α₀ everywhere, up to 4 α₀ mid-length, under the same temperature:
    the free end moves further than with a uniform α₀. An α the expansion
    did not read would leave both runs equal."""
    stepping_runs("thermique_non_lineaire")
    stepping_runs("mecanique_dilatation", solves=True)
    stepping_runs("mecanique_heterogene", solves=True)
    ((_, _, uniform),) = stepping_runs.calls["mecanique_dilatation"]["solve"]
    ((_, _, bell),) = stepping_runs.calls["mecanique_heterogene"]["solve"]
    assert bell.max("u_x") > 1.2 * uniform.max("u_x")


def _yield_stress(temperature, table, sigma_y):
    """σy at each Gauss point of `temperature` (component "T"), interpolated
    here in `table` — `(T, σy / sigma_y)` pairs — and held at the last value
    beyond it."""
    result = temperature * 0.0
    for (t0, r0), (t1, r1) in itertools.pairwise(table):
        inside = temperature.mask(ge=t0, lt=t1)
        slope = (r1 - r0) * sigma_y / (t1 - t0)
        result = result + inside * (r0 * sigma_y + slope * (temperature - t0))
    beyond = temperature.mask(ge=table[-1][0])
    return result + beyond * (table[-1][1] * sigma_y)


def test_plasticity_never_exceeds_the_yield_stress(stepping_runs):
    """At the last step, von Mises stays under σy at every Gauss point: 250 MPa
    in the first run, σy of each point's temperature — interpolated here in
    the script's table — in the second. Both runs plastify, and the clamp
    still carries the weight."""
    stepping_runs("thermique_non_lineaire")
    script, (constant, varying), _ = stepping_runs("plasticite")
    model = constant["model"]
    for data in (constant, varying):
        last = data["results"][-1]
        assert last["behaviour"].max("p") > 0.0
        # The equilibrium holds to the stepping's precision (1e-4).
        assert last["reactions"].sum("f_z") == pytest.approx(WEIGHT, rel=1e-4)
    vm = pyrucast.element_field.von_mises(model, constant["results"][-1]["behaviour"])
    assert vm.max("von_mises") <= script.SIGMA_Y * (1.0 + 1e-9)

    fes = model.fespace()
    temperature = pyrucast.element_field.interp_to_gauss(
        varying["temperature"].interpolate(script.T_FINAL), fes
    )
    limit = _yield_stress(temperature, script.TABLE_SIGMA_Y, script.SIGMA_Y)
    vm = pyrucast.element_field.von_mises(model, varying["results"][-1]["behaviour"])
    excess = vm - limit.rename_component("T", "von_mises")
    assert excess.max("von_mises") <= 1e-9 * script.SIGMA_Y
    # The check above has teeth: somewhere hot, the limit is under half of
    # 250 MPa, where a run ignoring the table would sit at 250 MPa.
    assert limit.min("T") < 0.5 * script.SIGMA_Y


def test_contact_carries_the_weight_and_never_pulls(stepping_runs):
    """At every step, the contact pushes each node of the support up — the
    support is held in x and y only, so its vertical reactions are the
    contact's — never down, and in all it carries the applied weight. The
    support sinks further at each step."""
    script, (data,), _ = stepping_runs("contact")
    ((slaves, _),) = stepping_runs.calls["contact"]["contact"]
    assert slaves.cell_count == 24
    centre = [script.LENGTH, 0.0, script.HEIGHT / 2.0]
    support = pyrucast.mesh.consolidate(data["results"][0]["displacement"].support())
    axis_node = support.nearest_node(centre)
    assert axis_node.position() == pytest.approx(centre, abs=1e-12)
    previous = 0.0
    for record in data["results"][1:]:
        fraction = record["time"] / script.T_FINAL
        on_slaves = pyrucast.node_field.restrict(record["reactions"], slaves)
        # Measured 1.3e-4 N at most (5e-9 of the weight, 2026-11-10): the
        # round-off of the multipliers. A node pulled would carry newtons.
        assert on_slaves.min("f_z") >= -1e-7 * WEIGHT
        assert on_slaves.sum("f_z") == pytest.approx(fraction * WEIGHT, rel=1e-4)
        sinking = record["displacement"].value(axis_node, "u_z")
        assert sinking < previous
        previous = sinking
