"""The record of a step (Cast3M ``PAS_RESU``)."""

from .fields import part

__all__ = ["store_step"]


def _close(t, times, tolerance=1e-12):
    return any(abs(t - s) <= tolerance * max(1.0, abs(s)) for s in times)


def store_step(data, problem, state, options, initial=False):
    """Append the state reached to ``data["results"]`` when its time is one of
    ``save_times`` and the step converged (the initial state is always kept).

    Each record holds ``time``, ``displacement`` (multipliers left out),
    ``reactions`` (``−Cᵀλ``, the forces the supports exert), ``behaviour``
    (stresses, internal variables, strains), ``temperature``, ``velocity`` and
    ``acceleration`` in dynamics, the pilot ``load_multiplier`` under
    indirect control, and the counts ``iterations``, ``accelerations``,
    ``converged``.
    """
    if not initial and (
        not state.converged or not _close(state.time, options.save_times)
    ):
        return
    reactions = state.reactions * -1.0 if state.reactions is not None else None
    data.setdefault("results", []).append(
        {
            "time": state.time,
            "displacement": part(state.solution, problem.displacement_names),
            "reactions": part(reactions, problem.force_names + problem.moment_names),
            "behaviour": state.behaviour,
            "temperature": part(state.temperature, ["T"]),
            "velocity": state.velocity,
            "acceleration": state.acceleration,
            "load_multiplier": state.load_multiplier,
            "iterations": state.iterations,
            "accelerations": state.accelerations,
            "converged": state.converged,
        }
    )
