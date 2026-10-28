"""The evolving state of a step-by-step computation (Cast3M ``PAS_INIT``).

What the loop carries from one step to the next — the converged solution,
the behaviour state, the iteration operator while it stays valid, what the
next step's predictor needs — gathered in one mutable ``State``.
"""

from dataclasses import dataclass, field

from .fields import uniform
from .materials import materials_at

__all__ = ["State", "initial_state"]


@dataclass
class State:
    """Everything the next step starts from.

    ``solution`` is the full mechanical unknown — displacement **and** the
    constraints' multipliers, the multipliers being the current reactions;
    ``behaviour`` the converged output of the laws (stresses, internal
    variables, strains); ``temperature`` likewise for the heat problem, with
    the previous one kept for a three-level scheme.
    """

    time: float
    materials: object
    solution: object | None = None
    behaviour: object | None = None
    temperature: object | None = None
    temperature_before: object | None = None
    velocity: object | None = None
    acceleration: object | None = None
    # Cᵀλ: the forces the constraints apply, at the last converged step.
    reactions: object | None = None
    heat_behaviour: object | None = None
    # The iteration operator, kept while it stays valid (Cast3M RRRR): the
    # materials it was built from and, in dynamics, the time step.
    operator: object | None = None
    operator_key: tuple = (None, None)
    mass: object | None = None
    mass_materials: object | None = None
    # What the next step's predictor reads (Cast3M FNONL, INCREMENT, ZDEP1, RESIDU).
    nonlinear_forces: object | None = None
    load_increment: object | None = None
    first_correction: object | None = None
    final_residual: object | None = None
    dt_before: float = 0.0
    substeps_before: int = 0
    converged: bool = True
    failed_before: bool = False
    iterations: int = 0
    accelerations: int = 0
    # Automatic control: the fraction of the step's load applied (AUTOCOEF).
    load_fraction: float = 1.0
    # Indirect control: the load multiplier reached.
    load_multiplier: float = 0.0
    replays: int = 0
    log: list = field(default_factory=list)


def initial_state(data, problem, options):
    """The state at the first instant (Cast3M ``PAS_INIT``).

    ``data["initial"]`` may give ``displacement``, ``behaviour`` (stresses and
    internal variables), ``temperature``, ``velocity`` and ``acceleration``;
    what it does not give starts at rest: zero displacement and motion,
    the laws' rest state, and a uniform temperature equal to ``t_ref`` (the
    stress-free temperature). With a prescribed temperature the initial one is
    the prescription at the first instant.
    """
    initial = data.get("initial", {})
    t0 = options.times[0]

    if problem.heat is not None:
        temperature = initial.get("temperature")
        if temperature is None:
            temperature = uniform(
                problem.heat_domain.fespace().mesh(), ["T"], options.t_ref
            )
    else:
        temperature = problem.prescribed_temperature_at(t0)

    state = State(time=t0, materials=materials_at(problem, temperature, t0))
    state.temperature = temperature

    if problem.mechanics is not None:
        mesh = problem.fespace.mesh()
        names = problem.displacement_names
        at_rest = lambda key: (
            initial[key] if initial.get(key) is not None else uniform(mesh, names, 0.0)
        )
        state.solution = at_rest("displacement")
        state.behaviour = initial.get("behaviour")
        if options.dynamics:
            state.velocity = at_rest("velocity")
            state.acceleration = at_rest("acceleration")
    return state
