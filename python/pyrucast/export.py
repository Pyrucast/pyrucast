"""Export vers formats externes — miroir de ``ops::export`` (Rust).

Écrit maillages et champs pour des outils tiers (VTK legacy pour ParaView) :
contrepartie à effet de bord des lecteurs.
"""

from ._pyrucast import export_vtk as export_vtk

__all__ = ["export_vtk"]
