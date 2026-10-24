"""Operators producing a field at the Gauss points — mirror of
``ops::element_field``.

Kinematics (gradient, strain and its structural variants, interpolation to the
Gauss points, thermal expansion), the model's own strain operator ``strain``,
the elastic operator both ways, the conversion between stress measures,
material data, and the integration of the constitutive law.
"""

from ._pyrucast import (
    beam_deformation,
    cell_integral,
    deformation,
    elastic_strain,
    elastic_stress,
    gradient,
    integrate_behavior,
    interp_to_gauss,
    material_field,
    material_field_per_sub_model,
    shell_deformation,
    strain,
    stress_measure,
    sub_material_field,
    thermal_strain,
)
from ._pyrucast import (
    consolidate_element as consolidate,
)
from ._pyrucast import (
    mask_element as mask,
)

__all__ = [
    "beam_deformation",
    "cell_integral",
    "consolidate",
    "deformation",
    "elastic_strain",
    "elastic_stress",
    "gradient",
    "integrate_behavior",
    "interp_to_gauss",
    "mask",
    "material_field",
    "material_field_per_sub_model",
    "shell_deformation",
    "strain",
    "stress_measure",
    "sub_material_field",
    "thermal_strain",
]
