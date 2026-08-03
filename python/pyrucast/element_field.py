"""Opérateurs produisant un champ aux points de Gauss — miroir de
``ops::element_field``.

Cinématique (gradient, déformation et ses variantes structurales,
interpolation aux points de Gauss, dilatation thermique), données matériau,
et intégration de la loi de comportement.
"""

from ._pyrucast import (
    beam_deformation as beam_deformation,
    consolidate_element as consolidate,
    deformation as deformation,
    gradient as gradient,
    integrate_behavior as integrate_behavior,
    interp_to_gauss as interp_to_gauss,
    material_field as material_field,
    material_field_per_sub_model as material_field_per_sub_model,
    sub_material_field as sub_material_field,
    thermal_strain as thermal_strain,
)

__all__ = [
    "beam_deformation",
    "consolidate",
    "deformation",
    "gradient",
    "integrate_behavior",
    "interp_to_gauss",
    "material_field",
    "material_field_per_sub_model",
    "sub_material_field",
    "thermal_strain",
]
