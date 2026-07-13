"""Répertoire de swap sur disque — miroir de ``store`` (Rust).

Configure et lit le répertoire où les gros conteneurs peuvent être délestés
sur disque.
"""

from ._pyrucast import (
    set_swap_dir as set_swap_dir,
    swap_dir as swap_dir,
)

__all__ = ["set_swap_dir", "swap_dir"]
