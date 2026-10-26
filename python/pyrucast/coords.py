"""Operators on the coordinate store — mirror of ``ops::coords`` (Rust).

The only three operators that change the geometry: ``set`` lays down absolute
positions, ``displace`` adds an increment, ``displace_from`` lays one
configuration at another plus a displacement. They are the write face of
``node_field.positions``, which reads. Computing on another configuration is
the method ``Coords.in_config``, a context manager.
"""

from ._pyrucast import displace, displace_from
from ._pyrucast import set_positions as set

__all__ = [
    "displace",
    "displace_from",
    "set",
]
