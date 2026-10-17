"""Materials at a given instant and state (Cast3M ``PAS_ETAT`` + ``PAS_MATE``)."""

from .. import element_field

__all__ = ["materials_at"]


def materials_at(problem, temperature, t):
    """The materials at time ``t`` and temperature ``temperature``.

    The base materials are read at ``t`` (a field, or an ``Evolution`` of
    fields). When the problem has ``material_laws`` — scalar curves of the
    state, such as ``E(T)`` or ``sigma_y(T, phi)`` — each is applied to the
    temperature at the Gauss points, and its output replaces the component of
    the same name: the laws are functions applied to a field, like ``cos``.
    Without laws, or without a temperature yet, the base materials are
    returned as they are.
    """
    base = problem.base_materials_at(t)
    if problem.laws is None or temperature is None:
        return base
    at_points = element_field.interp_to_gauss(
        temperature.filter_components(["T"]), problem.fespace
    )
    varied = problem.laws.interpolate(at_points)
    # One zone per space: the zones of several physics sharing a space are
    # fused (a constant both read, such as `rho`, appears once), the law
    # outputs leave them, and the laws' zone joins them.
    fused = element_field.consolidate(base)
    kept = [
        name
        for zone in fused
        for name in zone.components()
        if name not in problem.law_outputs
    ]
    return element_field.consolidate(fused.filter_components(kept) | varied)
