"""Small compositions of pyrucast field operators shared by the procedures.

Each is two or three library calls under a name that says what the solver
means by it — no value is ever read or written one node at a time.
"""

from .. import node_field
from .._pyrucast import NodeField

__all__ = ["STRAIN_PREFIXES", "carried", "infinity_norm", "part", "shifted", "uniform"]

# What the loop reads as a strain: the measure each law reads — linear,
# Green–Lagrange or Hencky.
STRAIN_PREFIXES = ("eps", "E_", "H_")


def carried(field):
    """Every component ``field`` carries, in first-seen order."""
    seen = []
    for zone in field:
        for name in zone.components():
            if name not in seen:
                seen.append(name)
    return seen


def part(field, names):
    """The components ``names`` of ``field`` — ``None`` when it carries none."""
    if field is None or not any(n in carried(field) for n in names):
        return None
    return field.filter_components(list(names))


def shifted(base, increment):
    """``base + increment`` on the supports of ``increment``: ``base`` is
    reprojected there first, so the two always add zone by zone (absent from
    ``base`` reads as zero)."""
    if base is None:
        return increment
    if increment is None:
        return base
    return node_field.restrict_like(base, increment) + increment


def infinity_norm(field, names):
    """``max |v|`` over the components ``names`` of ``field`` (0 when it
    carries none of them)."""
    if field is None:
        return 0.0
    present = [n for n in names if n in carried(field)]
    if not present:
        return 0.0
    magnitude = field.abs()
    return max(magnitude.max(n) for n in present)


def uniform(mesh, names, value):
    """A nodal field on ``mesh`` carrying ``names``, every value ``value``."""
    out = NodeField(mesh, list(names))
    if value != 0.0:
        for zone in out:
            for name in names:
                zone.add_to_component(name, value)
    return out
