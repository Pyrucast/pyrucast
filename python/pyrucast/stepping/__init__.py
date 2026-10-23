"""Step-by-step computation — the mirror of Cast3M's ``PASAPAS`` and of the
procedures it calls, one function each.

==================  ===============================================
Cast3M              pyrucast.stepping
==================  ===============================================
PASAPAS             ``step_by_step``
PAS_DEFA            ``resolve_options``
PAS_MODL            ``describe_problem``
PAS_INIT            ``initial_state``
PAS_ETAT, PAS_MATE  ``materials_at``
PAS_RESU            ``store_step``
PAS_AJUSTE          ``adjust_time_step``
TRANSNON            ``heat_step``
TRANSLIN            ``heat.heat_step_linear``
DUPONT2             ``heat.heat_step_dupont``
UNPAS               ``mechanical_step``
ACT3                ``anderson.anderson_correction``
AUTOPILO, PILOINDI  ``control.automatic_control``, ``control.indirect_control``
==================  ===============================================

The parts of ``mechanical_step`` live in ``pyrucast.stepping.mechanics``:
``iteration_operator``, ``reference_forces``, ``nonlinear_predictor``,
``equilibrium_iterations``, ``integrate_increment``, ``convergence_status``.
Everything is composed of pyrucast operators; only Anderson's three scalars
are computed in Python.

Large displacements follow from the laws: when they read a large-transformation
strain measure (``strain_measure``), the body stays on the reference
configuration and what acts on its current shape is evaluated on a current
configuration, ``pyrucast.stepping.configuration`` (Cast3M ``FORM``).
"""

from . import anderson, control, heat, mechanics
from .driver import step_by_step
from .heat import heat_step
from .materials import materials_at
from .mechanics import mechanical_step
from .options import HOOKS, resolve_options
from .problem import describe_problem
from .results import store_step
from .state import initial_state
from .time_step import adjust_time_step

__all__ = [
    "HOOKS",
    "adjust_time_step",
    "anderson",
    "control",
    "describe_problem",
    "heat",
    "heat_step",
    "initial_state",
    "materials_at",
    "mechanical_step",
    "mechanics",
    "resolve_options",
    "step_by_step",
    "store_step",
]
