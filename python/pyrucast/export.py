"""Export vers formats externes — miroir de ``ops::export`` (Rust).

Writes meshes and fields for third-party tools: legacy VTK for ParaView
(ASCII or binary, single files or time series), and the flat arrays every
exchange goes through (``to_arrays``). ``to_gmsh`` and ``to_medcoupling`` are
written in Python: they need the ``gmsh`` / ``medcoupling`` module, imported
only when they run, never at ``import pyrucast``.
"""

from ._gmsh import to_gmsh as to_gmsh
from ._med import to_medcoupling as to_medcoupling
from ._pyrucast import export_vtk as export_vtk
from ._pyrucast import to_arrays as to_arrays

__all__ = ["export_vtk", "to_arrays", "to_gmsh", "to_medcoupling"]
