"""Opérateurs écrivant dans le magasin de coordonnées — miroir de
``ops::coords`` (Rust).

Les deux seuls opérateurs qui modifient la géométrie : ``set`` repose des
positions absolues, ``displace`` ajoute un incrément. Ils sont la face
écriture de ``node_field.positions``, qui lit.
"""

from ._pyrucast import displace as displace
from ._pyrucast import set_positions as set  # noqa: A001 — `pyrucast.coords.set`

__all__ = [
    "displace",
    "set",
]
