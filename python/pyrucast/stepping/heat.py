"""The heat step, in its three schemes (Cast3M ``TRANSNON``, ``TRANSLIN``,
``DUPONT2``).

The same bookkeeping as the mechanics: the unknown carries the temperature and
the multipliers of the imposed temperatures, every solve returns a
temperature correction with the multipliers in total, and the residual comes
from integrating the laws — conduction, radiation — at the current
temperature. A model without heat capacity (no ``cp``) is solved steady.
"""

from dataclasses import dataclass

from .. import element_field, matrix, node_field, solver
from .anderson import Accelerator
from .fields import carried, infinity_norm, part, shifted
from .materials import materials_at

__all__ = ["HeatOutcome", "heat_step", "heat_step_dupont", "heat_step_linear"]

_TINY = 1e-300


@dataclass
class HeatOutcome:
    """A heat step: the temperature reached (with the multipliers), the laws'
    state there, the materials it was solved with, and how."""

    converged: bool
    temperature: object
    behaviour: object
    materials: object
    iterations: int


def _temperatures(problem, field):
    """The temperature components of ``field`` (multipliers left out)."""
    return part(field, ["T"])


def _multipliers(problem, field):
    names = [v for v in problem.heat.primal_vars() if v != "T"]
    return part(field, names) if names else None


def _advanced(problem, temperature, increment):
    """Temperature ``+ δT``, multipliers as returned (totals)."""
    moved = shifted(
        _temperatures(problem, temperature), _temperatures(problem, increment)
    )
    multipliers = _multipliers(problem, increment)
    return moved if multipliers is None else moved | multipliers


def _loads(data, problem, state, t, materials, options):
    """The heat loads at ``t``: the given ones, the hook's, and those the
    model carries (fluxes, the ambient of an exchange) — one field each."""
    loads = [problem.loads_at(t), options.hook("extra_heat_load")(data, state, t)]
    carried_loads = node_field.external_forces(problem.heat, materials)
    if len(carried_loads) > 0:
        loads.append(carried_loads)
    return [load for load in loads if load is not None]


def _materials(data, problem, state, temperature, t, options):
    """The materials at ``t`` and ``temperature``, through the
    ``heat_material_update`` hook when given."""
    materials = materials_at(problem, temperature, t)
    updated = options.hook("heat_material_update")(data, state)
    return materials if updated is None else updated


def _laws(problem, temperature, materials, dt):
    """The heat laws integrated at ``temperature`` (no history)."""
    strain = element_field.strain(problem.heat_domain, temperature, materials)
    return element_field.integrate_behavior(
        problem.heat_domain, strain, materials, dt=dt
    )


def _system(problem, loads, evaluated, behaviour, constrained, materials, extra=None):
    """``Σ loads − F(T_eval) [− extra]`` on the temperature rows and
    ``g − C T`` on the imposed rows, with ``T`` the ``constrained``
    temperature — the right-hand side whose solve returns the multipliers in
    total. Returns ``(system, out_of_balance)``, the second less ``Cᵀλ``."""
    temperature = _temperatures(problem, evaluated)
    internal = node_field.internal_forces(
        problem.heat, behaviour, temperature, materials
    )
    if problem.heat_constraints is not None and constrained is not evaluated:
        shift = _temperatures(problem, constrained) - node_field.restrict_like(
            temperature, _temperatures(problem, constrained)
        )
        internal = internal + node_field.restrict_like(
            node_field.internal_forces(
                problem.heat_constraints, behaviour, shift, materials
            ),
            internal,
        )
    system = internal * -1.0
    for load in loads:
        system = system + node_field.restrict_like(load, internal)
    if extra is not None:
        system = system - node_field.restrict_like(extra, internal)
    multipliers = _multipliers(problem, constrained)
    if multipliers is None or problem.heat_constraints is None:
        return system, system
    reactions = node_field.internal_forces(
        problem.heat_constraints, behaviour, multipliers, materials
    )
    return system, system - node_field.restrict_like(reactions, system)


def _operator(problem, materials, temperature, weight, capacity_over_dt):
    """``weight·K(T) + C/Δt + constraints``: the domain's tangent at
    ``temperature`` (the stiffness for a linear law, the radiative
    ``4σεT³`` film for radiation) scaled by the scheme's weight, the
    capacity, and the constraints left unscaled."""
    strain = element_field.strain(problem.heat_domain, temperature, materials)
    operator = matrix.tangent(problem.heat_domain, materials, strain) * weight
    if capacity_over_dt is not None:
        operator = operator + capacity_over_dt
    if problem.heat_constraints is not None:
        operator = operator + matrix.stiffness(problem.heat_constraints, materials)
    # A sum of operators is a list of blocks until folded into one system.
    operator.assemble()
    return operator


def _capacity(problem, materials, dt):
    """``C/Δt``, or ``None`` for a steady problem."""
    if not problem.transient_heat or dt <= 0.0:
        return None
    return matrix.mass(problem.heat_domain, materials) * (1.0 / dt)


def _solve(problem, operator, rhs):
    return solver.solve_unilateral(operator, problem.heat, rhs)


def heat_step(data, problem, state, t_end, options):
    """The heat problem from ``state.time`` to ``t_end`` by the θ-scheme,
    iterated (Cast3M ``TRANSNON``).

    The operator ``θK + C/Δt`` is built once at the start temperature; the
    residual ``F − F_int(T_θ) − C(T − Tₙ)/Δt`` is exact at
    ``T_θ = Tₙ + θ(T − Tₙ)``, with the materials re-evaluated there at every
    iteration. Converged when the correction is below ``precision`` relative
    to the temperature's size; Anderson's correction (unweighted) every other
    iteration from the fourth. A steady problem uses ``θ = 1``.
    """
    dt = t_end - state.time
    theta = options.theta if problem.transient_heat else 1.0
    start = state.temperature
    materials = _materials(data, problem, state, start, t_end, options)
    capacity = _capacity(problem, materials, dt)
    operator = _operator(
        problem, materials, _temperatures(problem, start), theta, capacity
    )
    loads = _loads(data, problem, state, t_end, materials, options)
    accelerator = None
    if options.acceleration:
        accelerator = Accelerator(None, False, None)
    size_start = infinity_norm(start, ["T"])
    temperature = start
    pause = 0
    for i in range(1, options.max_iterations + 1):
        pause -= 1
        evaluated = (
            shifted(
                _temperatures(problem, start) * (1.0 - theta),
                _temperatures(problem, temperature) * theta,
            )
            if theta != 1.0
            else _temperatures(problem, temperature)
        )
        behaviour = _laws(problem, evaluated, materials, dt)
        storage = None
        if capacity is not None:
            moved = _temperatures(problem, temperature) - node_field.restrict_like(
                _temperatures(problem, start), _temperatures(problem, temperature)
            )
            storage = capacity * moved
        system, out = _system(
            problem, loads, evaluated, behaviour, temperature, materials, storage
        )
        rhs = system
        if accelerator is not None:
            correction = accelerator(
                part(out, ["q"]), pause <= 0 and i > 3 and i % 2 == 0, None
            )
            if correction is not None:
                pause = 2
                rhs = rhs - node_field.restrict_like(correction, rhs)
        increment = _solve(problem, operator, rhs)
        temperature = _advanced(problem, temperature, increment)
        change = infinity_norm(increment, ["T"])
        size = 0.5 * (size_start + infinity_norm(temperature, ["T"]))
        if change <= options.precision * max(size, _TINY):
            behaviour = _laws(
                problem, _temperatures(problem, temperature), materials, dt
            )
            return HeatOutcome(True, temperature, behaviour, materials, i)
        if problem.laws is not None:
            materials = _materials(data, problem, state, temperature, t_end, options)
    return HeatOutcome(False, temperature, behaviour, materials, options.max_iterations)


def heat_step_linear(data, problem, state, t_end, options):
    """The heat problem in a single solve, materials frozen at the start
    (Cast3M ``TRANSLIN``): ``(θK + C/Δt) ΔT = F − K Tₙ`` linearised at
    ``Tₙ`` — radiation included through its tangent, where ``TRANSLIN``
    drops it."""
    dt = t_end - state.time
    theta = options.theta if problem.transient_heat else 1.0
    start = state.temperature
    materials = _materials(data, problem, state, start, state.time, options)
    capacity = _capacity(problem, materials, dt)
    operator = _operator(
        problem, materials, _temperatures(problem, start), theta, capacity
    )
    loads = _loads(data, problem, state, t_end, materials, options)
    behaviour = _laws(problem, _temperatures(problem, start), materials, dt)
    system, _ = _system(problem, loads, start, behaviour, start, materials)
    temperature = _advanced(problem, start, _solve(problem, operator, system))
    behaviour = _laws(problem, _temperatures(problem, temperature), materials, dt)
    return HeatOutcome(True, temperature, behaviour, materials, 1)


def heat_step_dupont(data, problem, state, t_end, options):
    """The heat problem by Dupont's three-level scheme (Cast3M ``DUPONT2``).

    ``(C/Δt + (½+λ)K) T₂ = (C/Δt + (2λ−½)K) T₁ − λK T₀ + F``, linearised at
    ``T* = T₁ + ½ Δt/Δt₀ (T₁ − T₀)`` — the temperature extrapolated to the
    middle of the step, where the materials are read and the sources taken;
    imposed temperatures are read at the end. Unlike ``DUPONT2``, ``T₀`` is the
    true previous temperature, so the scheme stays three-level from one step
    to the next. The first step, or a step whose length changed, goes through
    ``heat_step``.
    """
    dt = t_end - state.time
    before = state.temperature_before
    if (
        not problem.transient_heat
        or before is None
        or state.dt_before <= 0.0
        or abs(dt - state.dt_before) > 1e-5 * dt
    ):
        return heat_step(data, problem, state, t_end, options)
    lam = options.dupont_relaxation
    middle = state.time + 0.5 * dt
    t1 = _temperatures(problem, state.temperature)
    t0 = node_field.restrict_like(_temperatures(problem, before), t1)
    extrapolated = t1 + (t1 - t0) * (0.5 * dt / state.dt_before)
    materials = _materials(data, problem, state, extrapolated, middle, options)
    capacity = _capacity(problem, materials, dt)
    operator = _operator(problem, materials, extrapolated, 0.5 + lam, capacity)
    imposed = (
        list(problem.heat_constraints.dual_vars()) if problem.heat_constraints else []
    )
    sources = [
        part(load, [c for c in carried(load) if c not in imposed])
        for load in _loads(data, problem, state, middle, materials, options)
    ]
    targets = [
        part(load, imposed)
        for load in _loads(data, problem, state, t_end, materials, options)
    ]
    loads = [load for load in sources + targets if load is not None]
    behaviour = _laws(problem, extrapolated, materials, dt)
    # The flux at the weighted temperature, linearised at T*.
    weighted = t1 * (1.0 - lam) + t0 * lam
    tangent = matrix.tangent(
        problem.heat_domain,
        materials,
        element_field.strain(problem.heat_domain, extrapolated, materials),
    )
    system, _ = _system(
        problem,
        loads,
        extrapolated,
        behaviour,
        state.temperature,
        materials,
        tangent * (weighted - extrapolated),
    )
    temperature = _advanced(
        problem, state.temperature, _solve(problem, operator, system)
    )
    behaviour = _laws(problem, _temperatures(problem, temperature), materials, dt)
    return HeatOutcome(True, temperature, behaviour, materials, 1)
