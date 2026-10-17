"""Operators producing a field at the Gauss points — mirror of
``ops::element_field``.

Kinematics (gradient, strain and its structural variants, interpolation to the
Gauss points, thermal expansion), the elastic operator both ways, material
data, and the integration of the constitutive law.
"""

from ._pyrucast import (
    mask_element as mask,
    beam_deformation as beam_deformation,
    cell_integral as cell_integral,
    consolidate_element as consolidate,
    deformation as deformation,
    elastic_strain as elastic_strain,
    elastic_stress as elastic_stress,
    gradient as gradient,
    integrate_behavior as integrate_behavior,
    interp_to_gauss as interp_to_gauss,
    material_field as material_field,
    material_field_per_sub_model as material_field_per_sub_model,
    shell_deformation as shell_deformation,
    sub_material_field as sub_material_field,
    thermal_strain as thermal_strain,
)

__all__ = [
    "mask",
    "beam_deformation",
    "cell_integral",
    "consolidate",
    "deformation",
    "elastic_strain",
    "elastic_stress",
    "gradient",
    "integrate_behavior",
    "interp_to_gauss",
    "material_field",
    "material_field_per_sub_model",
    "shell_deformation",
    "sub_material_field",
    "thermal_strain",
]
