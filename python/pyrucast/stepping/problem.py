"""What is being solved, read once from the input (Cast3M ``PAS_MODL``).

The model is split by physics, each part with its own supports; the loads, the
materials and a prescribed temperature become functions of time; the names
the convergence tests read are listed. A ``Problem`` never changes during the
computation: everything that does lives in the ``State``.
"""

from collections.abc import Callable
from dataclasses import dataclass

from .. import element_field, node_field
from .._pyrucast import ElementField, Evolution, FiniteElementSpace
from .configuration import configurations
from .fields import carried, uniform

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


# The strain measure a law reads, from the prefix of what it reads.
_MEASURES = {"E": "green_lagrange", "H": "hencky"}


def _strain_measure(domain, fespace, displacement_names, materials):
    """The strain measure the mechanical laws read: ``"linear"``, or the one
    large-transformation measure every law of the domain shares. The laws
    declare it — what ``strain`` produces at rest carries its name (``E_*``,
    ``H_*``) — the loop only reads it. Refuses a domain mixing a measure of
    large transformations with anything else: bars, beams and shells have no
    large-transformation formulation yet."""
    rest = uniform(fespace.mesh(), displacement_names, 0.0)
    names = carried(element_field.strain(domain, rest, materials))
    prefixes = {name.split("_")[0] for name in names}
    finite = prefixes & set(_MEASURES)
    if not finite:
        return "linear"
    if len(prefixes) > 1:
        raise ValueError(
            "step_by_step: in large displacements every mechanical law reads the same "
            f"strain measure, and these read {sorted(prefixes)} — bars, beams and shells "
            "have no large-transformation formulation yet"
        )
    return _MEASURES[finite.pop()]


def _sub_models(model):
    return [model[i : i + 1] for i in range(len(model))]


def _refuse_in_large_displacements(mechanics, heat_domain):
    """What large displacements do not offer yet, refused with the reason."""
    if any("contact_gap" in sub[0].dual_vars() for sub in _sub_models(mechanics)):
        raise ValueError(
            "step_by_step: contact is not available in large displacements — its "
            "relations would have to follow the sliding of the projected point"
        )
    if heat_domain is not None:
        for sub in _sub_models(heat_domain):
            components = sub[0].material_components() or []
            if "k" not in components and any(c.startswith("k_") for c in components):
                raise ValueError(
                    "step_by_step: an oriented conductivity is not available in large "
                    "displacements — its material axes would have to turn with the "
                    "part (k = J⁻¹ F K₀ Fᵀ)"
                )


def _follower_split(mechanics):
    """The loads that follow the wall (``flux`` in its normal mode, density
    ``phi_n``) and the rest of the mechanical model, ``(follower, others)``;
    ``follower`` is ``None`` when there is none."""
    follower, others = None, None
    for sub in _sub_models(mechanics):
        if (sub[0].material_components() or []) == ["phi_n"]:
            follower = sub if follower is None else follower | sub
        else:
            others = sub if others is None else others | sub
    return follower, others


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
    mechanics: object | None
    mechanics_domain: object | None
    mechanics_constraints: object | None
    heat: object | None
    heat_domain: object | None
    heat_constraints: object | None
    fespace: object | None
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
    laws: object | None
    law_outputs: tuple
    transient_heat: bool
    radiating: bool
    weights: object | None
    indirect_domain: object | None = None
    # Large displacements: the strain measure the laws read (``"linear"``
    # otherwise), the reference and current configurations, the loads that
    # follow the wall — recomputed at every iterate on the current one — and
    # the model whose loads are computed once per step.
    strain_measure: str = "linear"
    configurations: object | None = None
    follower_loads: object | None = None
    step_loads: object | None = None


def describe_problem(data, options):
    """The ``Problem`` of ``data``: the model split by physics, its loads,
    materials and prescribed temperature as functions of time.

    Refuses a model with neither a mechanical nor a thermal domain, and a
    prescribed temperature given alongside a thermal model (the model then
    computes it).

    Large displacements are read off the laws: when the mechanical laws read a
    large-transformation measure (``strain_measure`` of their constructor), the
    problem carries that measure, the reference and current configurations
    (one more set in the ``Coords``, added here once) and its follower loads.
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
        strain_measure = _strain_measure(
            mechanics_domain, fespace, displacement_names, first_materials
        )
    else:
        fespace = heat_domain.fespace()
        displacement_names = multiplier_names = translation_names = rotation_names = ()
        force_names = moment_names = ()
        duality = {}
        weights = None
        strain_measure = "linear"

    large = strain_measure != "linear"
    if options.geometric_stiffness and not large:
        raise ValueError(
            "step_by_step: `geometric_stiffness` belongs to large displacements — in small "
            "ones the residual is written on the reference geometry, and K_σ would not be "
            "its derivative"
        )
    follower_loads, step_loads = None, mechanics
    if large:
        _refuse_in_large_displacements(mechanics, heat_domain)
        follower_loads, step_loads = _follower_split(mechanics)

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
        strain_measure=strain_measure,
        configurations=(configurations(fespace, displacement_names) if large else None),
        follower_loads=follower_loads,
        step_loads=step_loads,
    )
