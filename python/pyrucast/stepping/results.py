"""The record of a step."""

import math

__all__ = ["store_step"]


def store_step(data, problem, state, options, initial=False):
    """Append the state reached to ``data["results"]`` when its time is one of
    ``save_times`` and the step converged (the initial state is always kept).

    Each record holds ``time``, ``displacement`` (multipliers left out),
    ``reactions`` (``−Cᵀλ``, the forces the supports exert), ``behaviour``
    (stresses, internal variables, strains), ``temperature``, ``velocity`` and
    ``acceleration`` in dynamics, the pilot ``load_multiplier`` under
    indirect control, and the counts ``iterations``, ``accelerations``,
    ``converged``. In large displacements ``behaviour`` holds the Lagrangian
    stress the laws write (``S_*`` or ``T_*``); any other measure is the
    caller's to compute from ``displacement`` and ``behaviour``, through
    ``deformation(…, "gradient")`` and ``stress_measure``.
    """
    if not initial and (
        not state.converged
        or not any(
            math.isclose(state.time, s, rel_tol=1e-12, abs_tol=1e-12)
            for s in options.save_times
        )
    ):
        return
    solution, forces = state.solution, state.constraint_forces
    record = {
        "time": state.time,
        "displacement": (
            None if solution is None else solution[problem.displacement_names]
        ),
        "reactions": (
            None
            if forces is None
            else forces[problem.force_names + problem.moment_names] * -1.0
        ),
        "behaviour": state.behaviour,
        "temperature": None if state.temperature is None else state.temperature["T"],
        "velocity": state.velocity,
        "acceleration": state.acceleration,
        "load_multiplier": state.load_multiplier,
        "iterations": state.iterations,
        "accelerations": state.accelerations,
        "converged": state.converged,
    }
    data.setdefault("results", []).append(record)
