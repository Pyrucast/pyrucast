"""Construction de conteneurs — miroir de ``ops::build`` (Rust).

Fabrique les champs matériau (par zone, global, ou par sous-modèle) qui
alimentent l'assemblage et l'intégration du comportement.
"""

from ._pyrucast import (
    material_field as material_field,
    material_field_per_sub_model as material_field_per_sub_model,
    sub_material_field as sub_material_field,
)

__all__ = [
    "material_field",
    "material_field_per_sub_model",
    "sub_material_field",
]
