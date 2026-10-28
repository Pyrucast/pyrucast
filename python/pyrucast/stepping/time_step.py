"""The time step chosen from the last one's difficulty (Cast3M ``PAS_AJUSTE``)."""

__all__ = ["adjust_time_step"]

_SHRINK = 2.0**-0.5
_GROW = 2.0**0.5


def adjust_time_step(state, target, dt):
    """The next instant on the way to ``target``, and the step to remember.

    After a converged step: more than 20 iterations shrink the step by √2,
    fewer than 7 grow it by √2 — unless the step before failed. After a
    failure the step is halved. The instant never passes ``target``.
    Returns ``(instant, dt)``; ``dt`` is ``None`` before the first step, which
    then goes straight to ``target``.
    """
    if dt is None:
        return target, target - state.time
    if state.converged:
        if state.iterations > 20:
            dt *= _SHRINK
        elif state.iterations < 7 and not state.failed_before:
            dt *= _GROW
    else:
        dt *= 0.5
    instant = min(state.time + dt, target)
    return instant, dt
