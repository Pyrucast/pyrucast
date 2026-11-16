"""Small compositions of pyrucast field operators shared by the procedures.

Each is two or three library calls under a name that says what the solver
means by it — no value is ever read or written one node at a time.
"""

from .._pyrucast import NodeField

__all__ = [
    "STRAIN_PREFIXES",
    "domain_mesh",
    "infinity_norm",
    "part",
    "uniform",
]

# What the loop reads as a strain: the measure each law reads — linear,
# Green–Lagrange or Hencky.
STRAIN_PREFIXES = ("eps", "E_", "H_")


def part(field, names):
    """The components ``names`` of ``field`` — ``None`` when it carries none."""
    if field is None or not any(n in field.components() for n in names):
        return None
    return field.filter_components(list(names))


def infinity_norm(field, names):
    """``max |v|`` over the components ``names`` of ``field`` (0 when it
    carries none of them)."""
    if field is None:
        return 0.0
    present = [n for n in names if n in field.components()]
    if not present:
        return 0.0
    magnitude = field.abs()
    return max(magnitude.max(n) for n in present)


def domain_mesh(fespace):
    """The mesh ``fespace`` spans, each zone once. ``fespace.mesh()`` gives one
    zone per subspace, and a model's space has one subspace per sub-model: two
    physics on one zone — a conduction and a radiation on the same rod, a fin
    radiating along its length — would carry it twice, and a field cannot be
    built on a zone listed twice. The union keeps each zone once."""
    mesh = fespace.mesh()
    return mesh | mesh


def uniform(mesh, names, value):
    """A nodal field on ``mesh`` carrying ``names``, every value ``value``."""
    out = NodeField(mesh, list(names))
    if value != 0.0:
        for zone in out:
            for name in names:
                zone.add_to_component(name, value)
    return out
