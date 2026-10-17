"""Load control by a strain measure (Cast3M ``AUTOPILO``, ``PILOINDI``).

Two ways for the solver, rather than the user, to choose how much load a step
applies. Both measure the step by the **strain** it produces, read on the
model's own strain operator.
"""

from .. import element_field, field, node_field
from .fields import part

__all__ = [
    "AutomaticControl",
    "IndirectControl",
    "automatic_control",
    "indirect_control",
]

_BIG = 1e20


def _largest_strain(problem, increment, materials):
    """``max |ε|`` over the strain components the increment produces."""
    strain = element_field.strain(problem.mechanics_domain, increment, materials)
    names = [n for zone in strain for n in zone.components() if n.startswith("eps")]
    if not names:
        return 0.0
    magnitude = strain.abs()
    return max(magnitude.max(n) for n in set(names))


def automatic_control(problem, increment, elastic, fraction, materials):
    """The size of the step's displacement increment (Cast3M ``AUTOPILO``):
    the largest strain of ``Δu`` and of ``Δu − α Δu_el`` — the second
    measuring how far the step has left the elastic response, so that a
    yielding step counts its plastic part too. The caller scales ``Δu`` to
    bring this to the target."""
    total = _largest_strain(problem, increment, materials)
    if elastic is None:
        return total
    departure = increment - elastic * fraction
    return max(total, _largest_strain(problem, departure, materials))


def _components(field):
    return [n for zone in field for n in zone.components()]


def _spread(scalar, like):
    """The one-component ``scalar`` field copied into every component of
    ``like`` — what a division component by component needs."""
    (name,) = set(_components(scalar))
    copies = None
    for component in dict.fromkeys(_components(like)):
        copy = scalar.rename_component(name, component)
        copies = copy if copies is None else copies | copy
    return element_field.consolidate(copies)


def indirect_control(problem, solution, increment, pilot, target, materials):
    """The load multiplier change ``Δη`` that makes the step advance the strain
    by ``target`` along its own direction (Cast3M ``PILOINDI``).

    At each Gauss point of the piloting region, ``n`` is the direction of the
    strain reached ``ε(uₙ)`` — unit, or every component 1 when nothing is
    strained yet — ``a₀ = n : ε(Δu)`` what the step has done so far and
    ``a₁ = n : ε(δu_II)`` what the unit pilot load adds. The change that
    brings every point with ``a₁ > 0`` to at most ``target`` is
    ``Δη = min (target − a₀) / a₁`` over them. The products are the plain
    scalar products of the strain components.
    """
    domain = problem.indirect_domain
    reached = element_field.strain(domain, solution, materials)
    done = element_field.strain(domain, increment, materials)
    unit = element_field.strain(domain, pilot, materials)
    norm = field.psca(reached, reached) ** 0.5
    if norm.max() < 1e-20:
        direction = reached * 0.0 + 1.0
    else:
        direction = reached / _spread(norm, reached)
    a0 = field.psca(direction, done)
    a1 = field.psca(direction, unit)
    positive = element_field.mask(a1, gt=0.0)
    other = positive * -1.0 + 1.0
    ratio = (a0 * -1.0 + target) / (a1 * positive + other)
    return (ratio * positive + other * _BIG).min()


class AutomaticControl:
    """The automatic control of one substep (Cast3M ``AUTOMATIQUE``).

    The load applied is the fraction ``α`` of the step's increment. After each
    solve the displacement increment is scaled so that its strain measure
    (``automatic_control``) is ``target`` — divided by 3 every 20 iterations
    when it struggles; after each balance ``α`` is moved so that the residual
    becomes orthogonal to the increment, ``α ← α − (r·Δu)/(R·Δu)`` with ``R``
    the residual of the full increment. The step ends at the ``α`` reached.
    """

    def __init__(self, problem, fraction, target, elastic, nominal, work):
        self.problem = problem
        self.fraction = fraction
        self.target = target
        self.elastic = elastic
        self.nominal = nominal
        self.work = work
        self.reductions = 0
        self.divisor = 1.0
        self.scale = 1.0

    def after_solve(self, i, start, solution, materials):
        """``solution`` with its displacement increment rescaled."""
        names = self.problem.displacement_names
        u = part(solution, names)
        start_u = node_field.restrict_like(part(start, names), u)
        moved = u - start_u
        if self.reductions < i // 20:
            self.reductions += 1
            self.divisor *= 3.0
        size = automatic_control(
            self.problem, moved, self.elastic, self.fraction, materials
        )
        scale = (self.target / self.divisor) / max(size, 1e-300)
        if scale > 1.0 and self.fraction > 0.0:
            scale = min(scale, 1.0 / self.fraction)
        self.scale = scale
        rescaled = start_u + moved * scale
        multipliers = part(solution, self.problem.multiplier_names)
        return rescaled if multipliers is None else rescaled | multipliers

    def pending(self):
        """No level change before the balance: ``α`` moves after it."""
        return None

    def after_balance(self, start, solution, balance):
        """The new fraction, or ``None`` when it stays."""
        before = self.fraction
        if before == 1.0 and self.scale == 1.0:
            return None
        names = self.problem.displacement_names
        u = part(solution, names)
        moved = u - node_field.restrict_like(part(start, names), u)
        along = self.work(moved, self.nominal)
        if abs(along) < 1e-300:
            return None
        self.fraction = min(
            before - self.work(moved, balance.out_of_balance) / along, 1.0
        )
        return self.fraction if self.fraction != before else None


class IndirectControl:
    """The indirect control of one substep (Cast3M ``CHAR_PILO``).

    The loads are ``F + η F_p``: after each solve the multiplier moves by
    ``indirect_control``'s ``Δη`` and the displacement by ``Δη δu_II``,
    ``δu_II`` being the response of the operator to the unit pilot load.
    """

    def __init__(self, problem, pilot_response, target, multiplier):
        self.problem = problem
        self.response = pilot_response
        self.target = target
        self.multiplier = multiplier

    def after_solve(self, i, start, solution, materials):
        names = self.problem.displacement_names
        u = part(solution, names)
        start_u = node_field.restrict_like(part(start, names), u)
        response = node_field.restrict_like(part(self.response, names), u)
        change = indirect_control(
            self.problem, start_u, u - start_u, response, self.target, materials
        )
        self.multiplier += change
        moved = u + response * change
        multipliers = part(solution, self.problem.multiplier_names)
        return moved if multipliers is None else moved | multipliers

    def pending(self):
        """The multiplier just moved: the loads follow before the balance."""
        return self.multiplier

    def after_balance(self, start, solution, balance):
        return None
