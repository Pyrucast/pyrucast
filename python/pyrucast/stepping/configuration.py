"""The configurations of a computation in large displacements (Cast3M ``FORM``).

The body is computed on the **reference** configuration: its strain measure,
its law, its state and its internal forces. What acts on its **current**
shape — the Newton operator, a pressure that follows the wall, the conduction
on the deformed part — is computed on a configuration laid at ``x = X + u``.
Both live in the same ``Coords``: the current one is created once, laid again
in absolute whenever ``u`` moves, and active only for the length of one
computation.
"""

from contextlib import contextmanager
from dataclasses import dataclass

from .. import coords as coords_ops

__all__ = ["Configurations", "configurations"]


@dataclass(frozen=True)
class Configurations:
    """The reference and current configurations of one ``Coords``, by index,
    and the displacement components that lay the current one."""

    coords: object
    reference: int
    current: int
    displacement_names: tuple

    @contextmanager
    def current_at(self, displacement):
        """The block runs with the current configuration laid at
        ``X + displacement`` and active; the reference is active again after
        it, whatever happens inside."""
        coords_ops.displace_from(
            displacement, self.reference, self.current, list(self.displacement_names)
        )
        self.coords.select(self.current)
        try:
            yield
        finally:
            self.coords.select(self.reference)


def configurations(fespace, displacement_names):
    """The configurations of the space ``fespace`` lives on: the reference is
    the configuration active now, the current one is added to the ``Coords``
    — once for the whole computation."""
    coords = fespace.mesh().coords()
    reference = coords.active
    current = coords.add_config("current")
    return Configurations(coords, reference, current, tuple(displacement_names))
