"""Operators writing into the coordinate store — mirror of
``ops::coords`` (Rust).

The only three operators that change the geometry: ``set`` lays down absolute
positions, ``displace`` adds an increment, ``displace_from`` lays one
configuration at another plus a displacement. They are the write face of
``node_field.positions``, which reads.
"""

from ._pyrucast import displace as displace
from ._pyrucast import displace_from as displace_from
from ._pyrucast import set_positions as set  # noqa: A001 — `pyrucast.coords.set`

__all__ = [
    "displace",
    "displace_from",
    "set",
]
