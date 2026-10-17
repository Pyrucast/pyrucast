"""The loop over time (Cast3M ``PASAPAS``)."""

from .fields import infinity_norm, part
from .heat import heat_step, heat_step_dupont, heat_step_linear
from .mechanics import mechanical_step
from .options import resolve_options
from .problem import describe_problem
from .results import store_step
from .state import initial_state
from .time_step import adjust_time_step

__all__ = ["step_by_step"]

_HEAT = {"nonlinear": heat_step, "linear": heat_step_linear, "dupont": heat_step_dupont}


def _reached(t, target):
    return abs(t - target) <= 1e-12 * max(1.0, abs(target))


def _coherent(previous, current, tolerance):
    """Whether two passes gave the same temperature, to ``tolerance``
    relative to its size."""
    if previous is None:
        return False
    change = infinity_norm(part(current, ["T"]) - part(previous, ["T"]), ["T"])
    return change <= tolerance * max(infinity_norm(current, ["T"]), 1e-300)


def _one_step(data, problem, state, instant, options):
    """Heat then mechanics from ``state.time`` to ``instant``, repeated in
    passes until the temperature no longer changes when ``coherence`` is on.
    Returns the outcomes ``(heat, mechanics)``; the state is not touched."""
    hook = options.hook
    heat = mechanics = None
    temperature = state.temperature
    previous = None
    for k in range(options.coupling_passes):
        if problem.heat is not None:
            heat = _HEAT[options.heat_scheme](data, problem, state, instant, options)
            temperature = heat.temperature
            hook("after_heat")(data, state, heat)
        elif problem.prescribed_temperature_at(instant) is not None:
            temperature = problem.prescribed_temperature_at(instant)
        if problem.mechanics is not None:
            mechanics = mechanical_step(
                data, problem, state, instant, temperature, options
            )
            hook("after_mechanics")(data, state, mechanics)
        if options.coupling_passes == 1:
            break
        if k > 0 and _coherent(previous, temperature, options.coherence_tolerance):
            break
        if k == options.coupling_passes - 1:
            raise RuntimeError(
                f"step_by_step: the temperature still changes after {k + 1} coupling passes "
                f"at t = {instant}"
            )
        previous = temperature
    return heat, mechanics, temperature


def _commit(problem, state, instant, heat, mechanics, temperature):
    """Make the step's results the state the next one starts from. Under
    automatic control only the fraction ``α`` of the step was applied: the
    time advances by ``α (instant − t)``."""
    if mechanics is not None and mechanics.fraction < 1.0:
        instant = state.time + mechanics.fraction * (instant - state.time)
    dt = instant - state.time
    if heat is not None or problem.prescribed_temperature_at(instant) is not None:
        state.temperature_before = state.temperature
        state.temperature = temperature
    if heat is not None:
        state.heat_behaviour = heat.behaviour
    converged = True
    if mechanics is not None:
        converged = mechanics.converged
        state.solution = mechanics.solution
        state.behaviour = mechanics.behaviour
        state.materials = mechanics.materials
        state.reactions = mechanics.reactions
        state.velocity = (
            mechanics.velocity if mechanics.velocity is not None else state.velocity
        )
        state.acceleration = (
            mechanics.acceleration
            if mechanics.acceleration is not None
            else state.acceleration
        )
        state.iterations = mechanics.iterations
        state.accelerations = mechanics.accelerations
        state.substeps_before = mechanics.substeps
        state.load_fraction = mechanics.fraction
        state.load_multiplier = mechanics.multiplier
    elif heat is not None:
        converged = heat.converged
        state.materials = heat.materials
        state.iterations = heat.iterations
    state.failed_before = not state.converged
    state.converged = converged
    state.dt_before = dt
    state.time = instant


def step_by_step(data):
    """Run a step-by-step computation and fill ``data["results"]`` (Cast3M
    ``PASAPAS``).

    ``data`` is the input table: ``model`` and ``times`` are required,
    ``materials`` too; ``loads``, ``temperature`` (prescribed, without a
    thermal model), ``material_laws``, ``initial`` and every setting of
    ``resolve_options`` are optional. At each instant the heat problem is
    solved, then the mechanics with the temperature reached; the results are
    recorded at ``save_times``. With ``adaptive_step`` the instants between
    two targets are chosen from the iteration counts and a failed step is
    replayed shorter. Returns ``data``.
    """
    options = resolve_options(data)
    problem = describe_problem(data, options)
    state = initial_state(data, problem, options)
    data["results"] = []
    store_step(data, problem, state, options, initial=True)

    targets = [
        t for t in options.times if t > state.time and not _reached(t, state.time)
    ]
    k = 0
    dt = None
    while k < len(targets):
        target = targets[k]
        if options.adaptive_step:
            instant, dt = adjust_time_step(state, target, dt)
        else:
            instant = target
        heat, mechanics, temperature = _one_step(data, problem, state, instant, options)
        failed = (mechanics is not None and not mechanics.converged) or (
            heat is not None and not heat.converged
        )
        if failed and options.adaptive_step:
            # Not kept: the step is replayed, halved.
            state.failed_before = True
            state.converged = False
            if dt < 1e-10 * (target - options.times[0]):
                raise RuntimeError(
                    f"step_by_step: no convergence at t = {instant}, step exhausted"
                )
            continue
        if failed:
            raise RuntimeError(
                f"step_by_step: no convergence at t = {instant} "
                f"({options.max_substeps} substeps of forced convergence)"
            )
        _commit(problem, state, instant, heat, mechanics, temperature)
        store_step(data, problem, state, options)
        options.hook("after_step")(data, state)
        if _reached(state.time, target):
            k += 1
            state.replays = 0
        else:
            state.replays += 1
            if state.replays > options.automatic_replays:
                raise RuntimeError(
                    f"step_by_step: automatic control still short of t = {target} after "
                    f"{options.automatic_replays} steps"
                )
    return data
