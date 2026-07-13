"""Intégration de la loi de comportement — miroir de ``ops::behavior`` (Rust).

Contrepartie exacte (éventuellement non linéaire) de la linéarisation
``assemble.stiffness`` : intègre la loi constitutive d'un ``Model``.
"""

from ._pyrucast import integrate_behavior as integrate_behavior

__all__ = ["integrate_behavior"]
