"""What is being solved, read once from the input (Cast3M ``PAS_MODL``).

The model is split by physics, each part with its own supports; the loads, the
materials and a prescribed temperature become functions of time; the names
the convergence tests read are listed. A ``Problem`` never changes during the
computation: everything that does lives in the ``State``.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from .. import element_field, node_field
from .._pyrucast import ElementField, Evolution, FiniteElementSpace
from .fields import carried

__all__ = ["Problem", "describe_problem"]


def _of_time(spec):
    """A function of time from a field or an ``Evolution`` of fields; ``None``
    stays ``None``. Decided here, once: the loop calls it without asking."""
    if spec is None:
        return lambda _t: None
    if isinstance(spec, Evolution):
        return spec.interpolate
    return lambda _t: spec


def _part(model, nature):
    """The sub-models of ``model`` of one nature, its domain (constraints
    excluded) and its constraints — ``None`` for a physics the model lacks."""
    part = model.filter(nature)
    domain = part.exclude("constraint")
    if len(domain) == 0:
        return None, None, None
    constraints = part.filter("constraint")
    return part, domain, (constraints if len(constraints) > 0 else None)


def _nodal_weights(domain, names):
    """``(Σ cell measures around each node + ε)^-½`` on every component of
    ``names`` — the weight that puts residuals at comparable scales before
    they are combined (Cast3M ``ZJAC``)."""
    fespace = domain.fespace()
    ones = ElementField(fespace, list(names))
    for zone in ones:
        for name in names:
            zone.set_uniform(name, 1.0)
    centres = FiniteElementSpace(fespace.mesh(), quadrature="REDUCED")
    volumes = node_field.sum_to_nodes(element_field.cell_integral(ones, centres))
    return (volumes + 1e-300) ** -0.5


@dataclass(frozen=True)
class Problem:
    """The model split by physics, and the data as functions of time.

    ``mechanics`` / ``heat`` are the parts with their constraints,
    ``*_domain`` the same without, ``*_constraints`` the constraints alone
    (``None`` when absent). The name tuples are what the convergence tests
    read: forces and moments of the mechanical duals, the multipliers of the
    constraints.
    """

    model: object
    mechanics: Optional[object]
    mechanics_domain: Optional[object]
    mechanics_constraints: Optional[object]
    heat: Optional[object]
    heat_domain: Optional[object]
    heat_constraints: Optional[object]
    fespace: Optional[object]
    displacement_names: tuple
    multiplier_names: tuple
    translation_names: tuple
    rotation_names: tuple
    force_names: tuple
    moment_names: tuple
    duality: dict
    loads_at: Callable
    base_materials_at: Callable
    prescribed_temperature_at: Callable
    laws: Optional[object]
    law_outputs: tuple
    transient_heat: bool
    radiating: bool
    weights: Optional[object]
    indirect_domain: Optional[object] = None


def describe_problem(data, options):
    """The ``Problem`` of ``data``: the model split by physics, its loads,
    materials and prescribed temperature as functions of time.

    Refuses a model with neither a mechanical nor a thermal domain, and a
    prescribed temperature given alongside a thermal model (the model then
    computes it).
    """
    model = data["model"]
    mechanics, mechanics_domain, mechanics_constraints = _part(model, "mechanical")
    heat, heat_domain, heat_constraints = _part(model, "thermal")
    if mechanics is None and heat is None:
        raise ValueError(
            "step_by_step: the model has neither a mechanical nor a thermal domain"
        )
    if heat is not None and data.get("temperature") is not None:
        raise ValueError(
            "step_by_step: `temperature` prescribes what the thermal model computes — "
            "give one or the other"
        )

    materials_at = _of_time(data["materials"])
    first_materials = materials_at(options.times[0])
    if mechanics is not None:
        fespace = mechanics_domain.fespace()
        displacement_names = tuple(mechanics_domain.primal_vars())
        # Unknowns and duals are listed in step: `u_x` with `f_x`, a multiplier
        # with its imposed value.
        duality = dict(zip(mechanics.primal_vars(), mechanics.dual_vars(), strict=True))
        multiplier_names = tuple(
            v for v in mechanics.primal_vars() if v not in displacement_names
        )
        duals = mechanics_domain.dual_vars()
        force_names = tuple(d for d in duals if not d.startswith("m_"))
        moment_names = tuple(d for d in duals if d.startswith("m_"))
        rotation_names = tuple(
            v for v in displacement_names if duality[v].startswith("m_")
        )
        translation_names = tuple(
            v for v in displacement_names if v not in rotation_names
        )
        weights = (
            _nodal_weights(mechanics_domain, force_names)
            if options.acceleration
            else None
        )
    else:
        fespace = heat_domain.fespace()
        displacement_names = multiplier_names = translation_names = rotation_names = ()
        force_names = moment_names = ()
        duality = {}
        weights = None

    laws = data.get("material_laws")
    # One output component per curve, named by its ordinate type.
    law_outputs = (
        tuple(laws[i].ordinate_type() for i in range(len(laws))) if laws else ()
    )

    return Problem(
        model=model,
        mechanics=mechanics,
        mechanics_domain=mechanics_domain,
        mechanics_constraints=mechanics_constraints,
        heat=heat,
        heat_domain=heat_domain,
        heat_constraints=heat_constraints,
        fespace=fespace,
        displacement_names=displacement_names,
        multiplier_names=multiplier_names,
        translation_names=translation_names,
        rotation_names=rotation_names,
        force_names=force_names,
        moment_names=moment_names,
        duality=duality,
        loads_at=_of_time(data.get("loads")),
        base_materials_at=materials_at,
        prescribed_temperature_at=_of_time(data.get("temperature")),
        laws=laws,
        law_outputs=law_outputs,
        transient_heat=heat is not None and "cp" in carried(first_materials),
        radiating=heat is not None and len(model.filter("radiation")) > 0,
        weights=weights,
        indirect_domain=(
            (options.indirect_control or {}).get("region") or mechanics_domain
            if options.indirect_control is not None
            else None
        ),
    )
