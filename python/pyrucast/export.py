"""Export vers formats externes — miroir de ``ops::export`` (Rust).

Writes meshes and fields for third-party tools (legacy VTK for ParaView): the
side-effecting counterpart of the readers.
"""

from ._pyrucast import export_vtk as export_vtk

__all__ = ["export_vtk"]
