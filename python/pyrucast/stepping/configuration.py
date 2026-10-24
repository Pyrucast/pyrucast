"""The configurations of a computation in large displacements (Cast3M ``FORM``).

The body is computed on the **reference** configuration: its strain measure,
its law, its state and its internal forces, and its conduction. What acts on
its **current** shape — the Newton operator, a pressure that follows the wall,
the heat exchanges of the deformed part — is computed on a configuration laid
at ``x = X + u``.
Both live in the same ``Coords``: the current one is created once — the first
computation adds it, the next ones find it by its name — laid again in
absolute whenever ``u`` moves, and active only for the length of one
computation.
"""

from contextlib import contextmanager
from dataclasses import dataclass

from .. import coords as coords_ops

__all__ = ["CURRENT", "Configurations", "configurations"]


@dataclass(frozen=True)
class Configurations:
    """The reference and current configurations of one ``Coords``, by index,
    and the displacement components that lay the current one."""

    coords: object
    reference: int
    current: int
    displacement_names: tuple

    def lay(self, displacement):
        """Lay the current configuration at ``X + displacement``; the
        reference stays active."""
        coords_ops.displace_from(
            displacement, self.reference, self.current, list(self.displacement_names)
        )

    @contextmanager
    def on_current(self):
        """The block runs with the current configuration active, as last laid;
        the reference is active again after it, whatever happens inside."""
        self.coords.select(self.current)
        try:
            yield
        finally:
            self.coords.select(self.reference)

    @contextmanager
    def current_at(self, displacement):
        """The block runs with the current configuration laid at
        ``X + displacement`` and active (``lay``, then ``on_current``)."""
        self.lay(displacement)
        with self.on_current():
            yield


# The name of the current configuration in the ``Coords``: the set a
# computation lays at X + u, found again by the next one.
CURRENT = "stepping:current"


def configurations(fespace, displacement_names):
    """The configurations of the space ``fespace`` lives on: the reference is
    the configuration active now; the current one is the set named
    ``CURRENT`` in the ``Coords`` — added by the first computation, reused by
    the next ones, so that computing again does not grow the ``Coords``. It
    is always laid in absolute from the reference, so whatever an earlier
    computation left there does not matter."""
    coords = fespace.mesh().coords()
    reference = coords.active
    names = coords.names()
    current = names.index(CURRENT) if CURRENT in names else coords.add_config(CURRENT)
    return Configurations(coords, reference, current, tuple(displacement_names))
