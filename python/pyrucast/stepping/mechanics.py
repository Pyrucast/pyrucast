"""One mechanical step and the parts it is made of (Cast3M ``UNPAS``).

The unknown is the full solution — displacement and the constraints'
multipliers — and every linear solve returns a displacement **correction**
with the multipliers in **total**: the multipliers are then the current
reactions, which is what lets a unilateral constraint test their sign.

In large displacements — laws reading a large-transformation measure — the
body stays on the reference configuration: its internal forces are those of
the nominal stress ``P = F S``. The operator, the follower loads and the
geometric stiffness are evaluated on the current configuration
``x = X + u`` (``configuration.Configurations``), and the operator is rebuilt
there at the second iteration, then every ``max(delta_iterations, 20)``
iterations (Cast3M ``K_e(GEOM2) + K_σ``).

The iteration is a modified Newton: a constant operator (the elastic
stiffness, or the consistent tangent on request), residuals evaluated exactly
by integrating the laws, Anderson acceleration every other iteration, and
substeps of forced convergence when an iteration stalls.
"""

from dataclasses import dataclass, field
from typing import Optional

from .. import element_field, matrix, measure, node_field, solver
from .anderson import Accelerator
from .control import AutomaticControl, IndirectControl
from .fields import carried, infinity_norm, part, shifted
from .materials import materials_at

__all__ = [
    "Balance",
    "Equilibrium",
    "MechanicalOutcome",
    "StepData",
    "Tracker",
    "balance",
    "convergence_status",
    "equilibrium_iterations",
    "integrate_increment",
    "iteration_operator",
    "mechanical_step",
    "nonlinear_predictor",
    "reference_forces",
    "step_data",
]

_TINY = 1e-300
# Large displacements: the operator is rebuilt at least every so many
# iterations, and a substep that needs more rebuilds than this has failed
# (Cast3M UNPAS).
_REBUILD_EVERY = 20
_MAX_REBUILDS = 20
# What the convergence tests read as a strain: the measure each law reads.
_STRAIN_PREFIXES = ("eps", "E_", "H_")


# ─── What a (sub)step is solved for ─────────────────────────────────────────


@dataclass(frozen=True)
class StepData:
    """A (sub)step: its interval, and everything evaluated at its end — the
    materials, the nodal loads (one field per source, each reprojected on its
    own), the loads the model carries, and the strain imposed on the laws
    (thermal). Moduli that move between the two ends need nothing more: every
    law computes its stress from its elastic strain with the materials of the
    end (see ``Behavior`` in the Rust documentation)."""

    t_start: float
    t_end: float
    materials: object
    loads: tuple
    imposed_strain: Optional[object]


def _thermal_strain(problem, materials, temperature, options):
    """``α (T − T_ref)`` at the Gauss points, or ``None`` without a
    temperature or without ``alpha`` in the materials."""
    if temperature is None or "alpha" not in carried(materials):
        return None
    at_points = element_field.interp_to_gauss(
        temperature.filter_components(["T"]), problem.fespace
    )
    return element_field.thermal_strain(
        at_points, materials, problem.fespace, options.t_ref, problem.strain_measure
    )


def step_data(data, problem, state, start, t_end, temperature, options):
    """The ``StepData`` from the substep start ``start`` to ``t_end``.

    ``start`` carries the substep's starting time. The loads are read at
    ``t_end``, plus
    whatever the ``extra_mechanical_load`` hook adds; the loads the model
    itself carries (follower-free fluxes) are computed from the materials.
    """
    materials = materials_at(problem, temperature, t_end)
    loads = [
        problem.loads_at(t_end),
        options.hook("extra_mechanical_load")(data, state, t_end),
    ]
    carried_loads = node_field.external_forces(problem.step_loads, materials)
    if len(carried_loads) > 0:
        loads.append(carried_loads)
    return StepData(
        t_start=start.time,
        t_end=t_end,
        materials=materials,
        loads=tuple(load for load in loads if load is not None),
        imposed_strain=_thermal_strain(problem, materials, temperature, options),
    )


# ─── Algo 5: the laws ───────────────────────────────────────────────────────


def integrate_increment(problem, solution, step, behaviour_start):
    """The laws integrated from the substep start to ``solution`` (Algo 5).

    The strain each law reads comes from the model's own strain operator,
    minus the imposed strain; the laws step from ``behaviour_start`` (the
    converged state of the substep start, ``None`` for the rest state).
    Returns ``(strain, behaviour)``.
    """
    strain = element_field.strain(problem.mechanics_domain, solution, step.materials)
    if step.imposed_strain is not None:
        strain = strain - step.imposed_strain
    behaviour = element_field.integrate_behavior(
        problem.mechanics_domain,
        strain,
        step.materials,
        prev=behaviour_start,
        dt=step.t_end - step.t_start,
    )
    return strain, behaviour


# ─── The forces ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Balance:
    """The forces at an iterate.

    ``system`` is the right-hand side the operator is solved against —
    ``F_ext − F_int`` on the displacement rows, ``u_imp − C u`` on the
    constraint rows — so that the solve returns the multipliers in total.
    ``out_of_balance`` subtracts the current reactions ``Cᵀλ`` from it: the
    true residual, zero at equilibrium. ``reactions`` is ``Cᵀλ`` itself.
    """

    system: object
    out_of_balance: object
    reactions: Optional[object]


def balance(problem, solution, behaviour, step, inertia=None):
    """The ``Balance`` at ``solution`` with the laws' state ``behaviour``.

    ``inertia``, given in dynamics, is ``M·a`` at this iterate. In large
    displacements the internal forces are those of the nominal stress
    ``P = F S`` on the reference configuration, and the loads that follow the
    wall are evaluated on the configuration of this iterate.
    """
    displacement = part(solution, problem.displacement_names)
    stress = behaviour
    if problem.configurations is not None:
        gradient = element_field.deformation(displacement, problem.fespace, "gradient")
        stress = element_field.stress_measure(behaviour, gradient, "pk1")
    internal = node_field.internal_forces(
        problem.mechanics, stress, displacement, step.materials
    )
    system = internal * -1.0
    for load in step.loads:
        system = system + node_field.restrict_like(load, internal)
    if problem.follower_loads is not None:
        with problem.configurations.current_at(displacement):
            follower = node_field.external_forces(
                problem.follower_loads, step.materials
            )
        system = system + node_field.restrict_like(follower, internal)
    if inertia is not None:
        system = system - node_field.restrict_like(inertia, internal)
    multipliers = part(solution, problem.multiplier_names)
    if multipliers is None:
        return Balance(system, system, None)
    reactions = node_field.restrict_like(
        node_field.internal_forces(
            problem.mechanics_constraints, behaviour, multipliers, step.materials
        ),
        internal,
    )
    return Balance(system, system - reactions, reactions)


def _solve(problem, operator, rhs):
    """``operator⁻¹ rhs`` — the unilateral solver, which is the plain one when
    no constraint is an inequality; its factorization stays on ``operator``."""
    return solver.solve_unilateral(operator, problem.mechanics, rhs)


def _advanced(problem, solution, increment):
    """The solution after a solve: displacement ``+ δu``, multipliers as
    returned (they are totals)."""
    displacement = shifted(
        part(solution, problem.displacement_names),
        part(increment, problem.displacement_names),
    )
    multipliers = part(increment, problem.multiplier_names)
    return displacement if multipliers is None else displacement | multipliers


def _work(problem, primal, dual):
    """``Σ primal·dual`` over the model's (unknown, dual) pairs — ``δu·f`` and
    ``λ·g`` — the scalar product of an increment with a force."""
    names = [p for p in carried(primal) if p in problem.duality]
    if not names:
        return 0.0
    renamed = primal.filter_components(names)
    for name in names:
        renamed = renamed.rename_component(name, problem.duality[name])
    duals = [problem.duality[n] for n in names if problem.duality[n] in carried(dual)]
    if not duals:
        return 0.0
    against = dual.filter_components(duals)
    return measure.xty(node_field.restrict_like(renamed, against), against)


# ─── The operator ───────────────────────────────────────────────────────────


def iteration_operator(
    problem, state, step, options, strain=None, behaviour=None, at=None
):
    """The operator a substep iterates with.

    By default the elastic stiffness at the end-of-step materials, with the
    constraints; it is **kept** in the state and reused while the materials
    are the same object and the time step is unchanged (Cast3M ``RRRR``), so
    its factorization is reused too. With ``tangent`` and a ``strain``, the
    consistent tangent at that iterate instead, never kept. An explicit
    ``augmented_stiffness`` is added; in dynamics, ``4/Δt² M``.

    In large displacements ``at = (solution, behaviour)`` is the iterate: the
    operator is built on the configuration ``X + u`` there — the stiffness
    (or tangent) plus, unless ``geometric_stiffness`` is off, the geometric
    stiffness of its Cauchy stress — and never kept from one substep to the
    next.
    """
    dt = step.t_end - step.t_start
    if problem.configurations is not None:
        return _current_operator(problem, state, step, options, strain, behaviour, at)
    if options.tangent and strain is not None:
        operator = matrix.tangent(
            problem.mechanics, step.materials, strain, prev=behaviour, dt=dt
        )
    else:
        key = (step.materials, dt if options.dynamics else None)
        cached = state.operator_key
        if state.operator is not None and cached[0] is key[0] and cached[1] == key[1]:
            return state.operator
        operator = matrix.stiffness(problem.mechanics, step.materials)
    if options.augmented_stiffness is not None:
        operator = operator + options.augmented_stiffness
    if options.dynamics and dt > 0.0:
        operator = operator + _mass(problem, state, step.materials) * (4.0 / dt**2)
    if options.augmented_stiffness is not None or (options.dynamics and dt > 0.0):
        # A sum of operators is a list of blocks until folded into one system.
        operator.assemble()
    if not (options.tangent and strain is not None):
        state.operator = operator
        state.operator_key = (step.materials, dt if options.dynamics else None)
    return operator


def _current_operator(problem, state, step, options, strain, behaviour, at):
    """The operator of an iterate in large displacements: ``K_e + K_σ`` on its
    configuration (Cast3M ``K_e(GEOM2) + K_σ``)."""
    dt = step.t_end - step.t_start
    solution, reached = at
    displacement = part(solution, problem.displacement_names)
    sigma = None
    if options.geometric_stiffness is not False and reached is not None:
        # F and σ belong to the reference configuration: evaluated before the
        # current one becomes active.
        gradient = element_field.deformation(displacement, problem.fespace, "gradient")
        sigma = element_field.stress_measure(reached, gradient, "cauchy")
    mass = None
    if options.dynamics and dt > 0.0:
        # Kept on the reference configuration for the inertia of the residual,
        # ρ₀ dV₀: built there first.
        mass = _mass(problem, state, step.materials) * (4.0 / dt**2)
    with problem.configurations.current_at(displacement):
        if options.tangent and strain is not None:
            operator = matrix.tangent(
                problem.mechanics, step.materials, strain, prev=behaviour, dt=dt
            )
        else:
            operator = matrix.stiffness(problem.mechanics, step.materials)
        if sigma is not None:
            operator = operator + matrix.geometric(
                problem.mechanics_domain, step.materials, sigma
            )
        if options.augmented_stiffness is not None:
            operator = operator + options.augmented_stiffness
        if mass is not None:
            operator = operator + mass
        # Folded here: assembling evaluates the blocks on the configuration
        # active at that moment — the operator's mass therefore on the current
        # one, an approximation the operator may make.
        operator.assemble()
    return operator


def _mass(problem, state, materials):
    """The mass matrix of the mechanical domain, kept while the materials
    are the same object."""
    if state.mass is None or state.mass_materials is not materials:
        state.mass = matrix.mass(problem.mechanics_domain, materials)
        state.mass_materials = materials
    return state.mass


# ─── F_ref, and the predictor of the next step ──────────────────────────────


def reference_forces(
    problem, step, first_correction, balance_start, internal_scale, floor=0.0
):
    """The force and moment scales the residual is compared to (``F_ref``,
    ``M_ref``), from the first solve of the step.

    ``F_ref = |δx⁰·(F_ext ; −g)| / ‖δu⁰‖∞ + 2 F_s``, bounded below by the
    largest applied force — ``F_s`` being ``internal_scale``, the largest
    internal force at the start of the step with and without its imposed
    strain — and by ``floor`` (in large displacements ``‖K uₙ‖∞``, as Cast3M).
    ``M_ref`` divides the same work by the largest rotation. Each falls back to
    1 when vanishing (a step with nothing applied).
    """
    applied = None
    for load in step.loads:
        aligned = node_field.restrict_like(load, balance_start.system)
        applied = aligned if applied is None else applied + aligned
    rows = [n for n in carried(balance_start.system) if n not in problem.force_names]
    rows = [n for n in rows if n not in problem.moment_names]
    work = 0.0
    if applied is not None:
        work = _work(
            problem,
            first_correction,
            part(applied, problem.force_names + problem.moment_names),
        )
    constraint_rows = part(balance_start.system, rows) if rows else None
    if constraint_rows is not None:
        work -= _work(problem, first_correction, constraint_rows)
    largest_move = infinity_norm(first_correction, problem.translation_names)
    work = abs(work) + internal_scale * largest_move
    force = work / (largest_move + _TINY) + internal_scale
    force = max(force, infinity_norm(applied, problem.force_names), floor)
    if force < _TINY:
        force = 1.0
    moment = 1.0
    if problem.moment_names:
        moment = (
            work / (infinity_norm(first_correction, problem.rotation_names) + _TINY)
            + _TINY
        )
    return force, moment


def nonlinear_predictor(
    problem, state, options, first_correction, increment, reference, dt
):
    """The share of the previous step's non-linear forces to add to this
    step's first residual, or ``None`` (Cast3M ``INITIALISATION``).

    The previous step left ``F_NL = K Δx − ΔR⁰ + r`` — what the elastic
    operator mispredicted. If it converged without substeps and was
    non-linear, this step starts from ``a·F_NL``, with ``a`` the ratio of the
    two steps' load increments projected on their first corrections, bounded by
    ``Δt/Δt_prev``; ``a = Δt/Δt_prev`` when the load does not move (creep,
    thermal), zero on a significant unloading; kept only when ``0 < a < 20``.

    Never in large displacements: a finite rotation is not in the kernel of
    the operator, so ``K Δx`` reads forces where nothing deforms, and the
    operator itself changes from one step to the next — the previous step's
    misprediction says nothing about this one's.
    """
    previous = state.nonlinear_forces
    if (
        not options.predictor
        or problem.configurations is not None
        or options.dynamics
        or previous is None
        or not state.converged
        or state.substeps_before > 0
        or infinity_norm(previous, problem.force_names) <= options.precision * reference
    ):
        return None
    both = shifted(
        part(state.first_correction, problem.displacement_names),
        part(first_correction, problem.displacement_names),
    )
    on_previous = _work(problem, both, previous)
    now = _work(problem, both, increment)
    before = _work(problem, both, state.load_increment)
    ratio = now / (before + _TINY)
    by_time = dt / state.dt_before if state.dt_before > _TINY else 0.0
    moving = _work(
        problem, part(first_correction, problem.displacement_names), increment
    )
    largest = infinity_norm(first_correction, problem.translation_names)
    if abs(moving) < options.precision * reference * largest or abs(
        on_previous
    ) > 200.0 * abs(before):
        amplitude = 0.0 if ratio < -0.05 else by_time
    else:
        amplitude = ratio
    amplitude = min(amplitude, by_time)
    if not 0.0 < amplitude < 20.0:
        return None
    return previous * amplitude


def _nonlinear_forces(problem, operator, start, solution, increment, residual):
    """``K Δx − ΔR⁰ + r`` on the force rows: what the operator alone would not
    have predicted. Zero for a linear step."""
    step_increment = solution - node_field.restrict_like(start, solution)
    predicted = operator * step_increment
    forces = list(problem.force_names + problem.moment_names)
    out = part(predicted, forces) - node_field.restrict_like(
        part(increment, forces), part(predicted, forces)
    )
    return out + node_field.restrict_like(part(residual, forces), out)


# ─── Algo 6: is it over? ────────────────────────────────────────────────────


@dataclass
class Tracker:
    """What the convergence tests remember across iterations, and — for the
    iteration cap — across substeps."""

    max_iterations: int
    history: list = field(default_factory=list)
    stalled: bool = False
    stall_tolerance: float = 1e-5
    relaxation: float = 1.0

    def reset(self):
        """A new substep: the history and the stall flag start over, the cap
        does not (Cast3M ``ZMAXIT``)."""
        self.history = []
        self.stalled = False
        self.stall_tolerance = 1e-5
        self.relaxation = 1.0


def convergence_status(tracker, i, measures, options, skip, controlled):
    """``"converged"``, ``"relaxed"``, ``"failed"`` or ``"continue"`` after
    iteration ``i`` (Algo 6).

    ``measures`` holds ``force`` and ``moment`` (residual over its
    reference), ``strain_change`` (strain change over the iteration) and
    ``strain_increment`` (strain increment of the step). Converged when the
    residual and the strain change are below their precisions. A residual
    that has not dropped by 1 % in ``delta_iterations`` iterations, or the
    cap reached, flags a stall: the cap triples and, with a relaxation
    below 1, the target is brought closer. Failure is declared on a stall that
    no longer moves the strain.
    """
    tracker.history.append(measures["force"])
    if not skip:
        if (
            measures["force"] < options.precision
            and measures["strain_change"] < options.strain_precision
            and measures["moment"] < options.moment_precision
        ):
            return "relaxed" if tracker.relaxation < 1.0 else "converged"
    d = options.delta_iterations
    reference = 0.99 * tracker.history[i - 1 - d] if i > d else float("inf")
    if not skip and not tracker.stalled and i > 1 and not controlled:
        if i >= tracker.max_iterations or measures["force"] > reference:
            tracker.stalled = True
            tracker.max_iterations = 3 * i
            tracker.relaxation = options.relaxation
            skip = True
    if tracker.stalled and i > 15:
        tracker.stall_tolerance *= 2.0
    if not skip and measures["strain_increment"] < tracker.stall_tolerance:
        if i >= tracker.max_iterations or (
            tracker.stalled
            and measures["force"] > reference
            and measures["strain_increment"] >= 0.25
        ):
            return "failed"
    return "continue"


# ─── Algo 4: the equilibrium iterations ─────────────────────────────────────


@dataclass
class Equilibrium:
    """Where the iterations of a substep ended."""

    status: str
    solution: object
    behaviour: object
    strain: object
    balance: Balance
    iterations: int
    accelerations: int
    step: StepData
    controller: Optional[object] = None


def equilibrium_iterations(
    problem,
    state,
    operator,
    step,
    start,
    first,
    reference,
    tracker,
    options,
    skip_first,
    inertia=None,
    retarget=None,
    controller=None,
    control_step=None,
):
    """Iterate a substep to equilibrium (Algo 4).

    ``start`` is the substep's starting point (solution, behaviour, strain),
    ``first`` its ``Balance`` — the predictor already added. Each iteration:
    the consistent tangent if asked, Anderson's correction every other
    iteration from the fourth, a solve, the laws integrated at the new
    iterate, the forces, and the tests of ``convergence_status``. When a stall
    brings the target closer (relaxation below 1), ``retarget(fraction)``
    gives the step toward it and the forces are re-evaluated there. A
    ``controller`` (automatic or indirect control) adjusts each solve and the
    load level, ``control_step(level)`` giving the step at that level.
    """
    force_ref, moment_ref = reference
    accelerator = None
    if options.acceleration:
        accelerator = Accelerator(
            problem.weights,
            options.stability,
            lambda increment, forces: _work(problem, increment, forces),
        )
    solution, behaviour, strain, current = (
        start.solution,
        start.behaviour,
        start.strain,
        first,
    )
    start_strain = start.strain
    previous_increment = None
    pause = 0
    accelerations = 0
    controlled = options.automatic or options.indirect_control is not None
    large = problem.configurations is not None
    rebuild_every = max(options.delta_iterations, _REBUILD_EVERY)
    rebuilds = 0
    for i in range(1, 10**9):
        pause -= 1
        if options.tangent and i >= 2 and i % options.tangent_every == 0:
            operator = iteration_operator(
                problem,
                state,
                step,
                options,
                strain,
                start.behaviour,
                at=(solution, behaviour),
            )
        elif large and i > 1 and (i - 2) % rebuild_every == 0:
            # The operator of the configuration reached (Cast3M K_e(GEOM2) + K_σ),
            # at the second iteration — the first correction has moved the
            # geometry — then every `rebuild_every` iterations (UNPAS l. 1540-1795).
            rebuilds += 1
            if rebuilds > _MAX_REBUILDS:
                return Equilibrium(
                    "failed",
                    solution,
                    behaviour,
                    strain,
                    current,
                    i - 1,
                    accelerations,
                    step,
                    controller,
                )
            operator = iteration_operator(
                problem, state, step, options, at=(solution, behaviour)
            )
            pause = 4
        rhs = current.system
        if accelerator is not None:
            forces = part(current.out_of_balance, problem.force_names)
            correction = accelerator(forces, pause <= 0 and i > 3, previous_increment)
            if correction is not None:
                pause = 2
                accelerations += 1
                rhs = rhs - node_field.restrict_like(correction, rhs)
        increment = _solve(problem, operator, rhs)
        previous_increment = part(increment, problem.displacement_names)
        solution = _advanced(problem, solution, increment)
        if controller is not None:
            solution = controller.after_solve(
                i, start.solution, solution, step.materials
            )
            level = controller.pending()
            if level is not None:
                step = control_step(level)
        previous_strain = strain
        strain, behaviour = integrate_increment(
            problem, solution, step, start.behaviour
        )
        previous = current
        current = balance(
            problem, solution, behaviour, step, inertia(solution) if inertia else None
        )
        if controller is not None:
            level = controller.after_balance(start.solution, solution, current)
            if level is not None:
                step = control_step(level)
                strain, behaviour = integrate_increment(
                    problem, solution, step, start.behaviour
                )
                current = balance(
                    problem,
                    solution,
                    behaviour,
                    step,
                    inertia(solution) if inertia else None,
                )

        # How much the reactions moved: from none, all of them. Each evaluation
        # builds its own support, so align before subtracting.
        change = current.reactions
        if change is not None and previous.reactions is not None:
            change = change - node_field.restrict_like(previous.reactions, change)
        reaction_change = infinity_norm(change, problem.force_names)
        names = [n for n in carried(strain) if n.startswith(_STRAIN_PREFIXES)]
        measures = {
            "force": max(
                infinity_norm(current.out_of_balance, problem.force_names),
                reaction_change,
            )
            / force_ref,
            "moment": infinity_norm(current.out_of_balance, problem.moment_names)
            / moment_ref,
            "strain_change": infinity_norm(strain - previous_strain, names),
            "strain_increment": infinity_norm(strain - start_strain, names),
        }
        if options.trace:
            print(
                f"  it {i:4d}  residual {measures['force']:.3e}  strain change "
                f"{measures['strain_change']:.3e}  step strain {measures['strain_increment']:.3e}"
                f"{'  accelerated' if pause == 2 else ''}"
            )
        relaxation = tracker.relaxation
        status = convergence_status(
            tracker, i, measures, options, skip_first and i == 1, controlled
        )
        if status != "continue":
            return Equilibrium(
                status,
                solution,
                behaviour,
                strain,
                current,
                i,
                accelerations,
                step,
                controller,
            )
        if tracker.relaxation < relaxation and retarget is not None:
            step = retarget(tracker.relaxation)
            strain, behaviour = integrate_increment(
                problem, solution, step, start.behaviour
            )
            current = balance(
                problem,
                solution,
                behaviour,
                step,
                inertia(solution) if inertia else None,
            )


# ─── Algo 3: the step ───────────────────────────────────────────────────────


@dataclass
class MechanicalOutcome:
    """A mechanical step: converged or not, where it ended, how it got there."""

    converged: bool
    solution: object
    behaviour: object
    strain: object
    materials: object
    reactions: Optional[object]
    iterations: int
    accelerations: int
    substeps: int
    velocity: Optional[object] = None
    acceleration: Optional[object] = None
    # The share of the step's load applied (automatic control) and the pilot
    # load multiplier reached (indirect control).
    fraction: float = 1.0
    multiplier: float = 0.0


@dataclass
class _Start:
    """The starting point of a substep."""

    time: float
    solution: object
    behaviour: object
    strain: Optional[object] = None


def _newmark(problem, state, step, dt):
    """``x ↦ M a(x)`` with ``a = 4/Δt² (u − uₙ) − 4/Δt vₙ − aₙ`` (Newmark,
    γ = ½, β = ¼), and the update of velocity and acceleration."""
    mass = _mass(problem, state, step.materials)
    u_n = part(state.solution, problem.displacement_names)

    def accelerations(solution):
        u = part(solution, problem.displacement_names)
        moved = u - node_field.restrict_like(u_n, u)
        return (
            moved * (4.0 / dt**2)
            - node_field.restrict_like(state.velocity, u) * (4.0 / dt)
            - node_field.restrict_like(state.acceleration, u)
        )

    def inertia(solution):
        return mass * accelerations(solution)

    def motion(solution):
        u = part(solution, problem.displacement_names)
        moved = u - node_field.restrict_like(u_n, u)
        velocity = moved * (2.0 / dt) - node_field.restrict_like(state.velocity, u)
        return velocity, accelerations(solution)

    return inertia, motion


def mechanical_step(data, problem, state, t_end, temperature, options):
    """Solve the mechanics from ``state.time`` to ``t_end`` (Algo 3).

    Substeps of forced convergence: when the iterations stall, the iterate
    reached becomes the start of a new substep — toward a closer target when
    the relaxation is below 1 — until convergence or ``max_substeps``.
    Without forced convergence a stall is a failure. Returns a
    ``MechanicalOutcome``; the state is only updated by the caller, except
    for what the next step's predictor and operator read.
    """
    start = _Start(
        time=state.time,
        solution=state.solution,
        behaviour=state.behaviour,
    )
    tracker = Tracker(max_iterations=options.max_iterations)
    forces = problem.force_names + problem.moment_names
    reference = None
    nonlinear = None
    iterations = accelerations = 0
    result = None
    for substep in range(options.max_substeps):
        tracker.reset()
        step = step_data(data, problem, state, start, t_end, temperature, options)
        dt = step.t_end - step.t_start
        inertia = motion = None
        if options.dynamics and dt > 0.0:
            inertia, motion = _newmark(problem, state, step, dt)
        start.strain, behaviour0 = integrate_increment(
            problem, start.solution, step, start.behaviour
        )
        at = (start.solution, behaviour0)
        if options.tangent:
            operator = iteration_operator(
                problem, state, step, options, start.strain, start.behaviour, at=at
            )
        else:
            operator = iteration_operator(problem, state, step, options, at=at)
        first = balance(
            problem,
            start.solution,
            behaviour0,
            step,
            inertia(start.solution) if inertia else None,
        )
        first_correction = _solve(problem, operator, first.system)
        # This step's load increment: the first residual, less what the
        # previous step left unbalanced.
        increment = part(first.out_of_balance, forces)
        if state.final_residual is not None:
            increment = increment - node_field.restrict_like(
                part(state.final_residual, forces), increment
            )
        if reference is None:
            internal_scale = infinity_norm(
                first.system - _applied(step, first.system), problem.force_names
            )
            floor = 0.0
            if problem.configurations is not None:
                floor = infinity_norm(
                    operator * part(start.solution, problem.displacement_names),
                    problem.force_names,
                )
            reference = reference_forces(
                problem, step, first_correction, first, internal_scale, floor
            )
        controller, control_step = _controller(
            data,
            problem,
            state,
            start,
            t_end,
            temperature,
            options,
            operator,
            step,
            first,
            first_correction,
        )
        if controller is not None:
            # The iterations start from the step at the controller's level.
            step = control_step(controller.level)
            behaviour0 = integrate_increment(
                problem, start.solution, step, start.behaviour
            )[1]
            first = balance(problem, start.solution, behaviour0, step)
        predicted = None
        if substep == 0 and controller is None:
            predicted = nonlinear_predictor(
                problem, state, options, first_correction, increment, reference[0], dt
            )
        if predicted is not None:
            system = first.system + node_field.restrict_like(predicted, first.system)
            first = Balance(system, first.out_of_balance, first.reactions)
        here = start

        def retarget(fraction, here=here):
            target = here.time + fraction * (t_end - here.time)
            return step_data(data, problem, state, here, target, temperature, options)

        result = equilibrium_iterations(
            problem,
            state,
            operator,
            step,
            start,
            first,
            reference,
            tracker,
            options,
            skip_first=predicted is not None,
            inertia=inertia,
            retarget=retarget,
            controller=controller,
            control_step=control_step,
        )
        found = _nonlinear_forces(
            problem,
            operator,
            start.solution,
            result.solution,
            increment,
            result.balance.out_of_balance,
        )
        nonlinear = (
            found
            if nonlinear is None
            else nonlinear + node_field.restrict_like(found, nonlinear)
        )
        accelerations += result.accelerations
        if result.status == "converged":
            iterations += result.iterations
            state.nonlinear_forces = nonlinear
            state.load_increment = increment
            state.first_correction = first_correction
            state.final_residual = result.balance.out_of_balance
            velocity = acceleration = None
            if motion is not None:
                velocity, acceleration = motion(result.solution)
            outcome = MechanicalOutcome(
                True,
                result.solution,
                result.behaviour,
                result.strain,
                result.step.materials,
                result.balance.reactions,
                iterations,
                accelerations,
                substep,
                velocity,
                acceleration,
            )
            if isinstance(controller, AutomaticControl):
                outcome.fraction = controller.fraction
            elif isinstance(controller, IndirectControl):
                outcome.multiplier = controller.multiplier
            return outcome
        if result.status == "relaxed":
            iterations += result.iterations
        elif not options.forced_convergence:
            break
        # The iterate reached starts the next substep, at the target it was
        # solved for, toward the same end.
        start = _Start(
            time=result.step.t_end,
            solution=result.solution,
            behaviour=result.behaviour,
        )
    state.nonlinear_forces = nonlinear
    return MechanicalOutcome(
        False,
        result.solution,
        result.behaviour,
        result.strain,
        result.step.materials,
        result.balance.reactions,
        iterations,
        accelerations,
        options.max_substeps,
    )


def _controller(
    data,
    problem,
    state,
    start,
    t_end,
    temperature,
    options,
    operator,
    step,
    first,
    first_correction,
):
    """The load controller of a substep and the step at a given load level,
    or ``(None, None)`` without control."""
    if options.automatic:
        before = state.load_fraction
        fraction = 1.0 if before >= 1.0 else min(before / (1.0 - before), 1.0)

        def control_step(level):
            target = start.time + level * (t_end - start.time)
            return step_data(data, problem, state, start, target, temperature, options)

        controller = AutomaticControl(
            problem,
            fraction,
            options.automatic_target,
            part(first_correction, problem.displacement_names),
            part(first.out_of_balance, problem.force_names),
            lambda increment, forces: _work(problem, increment, forces),
        )
        controller.level = fraction
        return controller, control_step
    if options.indirect_control is not None:
        settings = options.indirect_control
        pilot = settings["load"]
        response = _solve(
            problem, operator, node_field.restrict_like(pilot, first.system)
        )

        def control_step(level):
            loads = step.loads + (pilot * level,)
            return StepData(
                step.t_start, step.t_end, step.materials, loads, step.imposed_strain
            )

        controller = IndirectControl(
            problem, response, float(settings["increment"]), state.load_multiplier
        )
        controller.level = state.load_multiplier
        return controller, control_step
    return None, None


def _applied(step, like):
    """The step's applied loads, summed on the support of ``like``."""
    total = None
    for load in step.loads:
        aligned = node_field.restrict_like(load, like)
        total = aligned if total is None else total + aligned
    return total if total is not None else like * 0.0
