"""Step-by-step computation — a loop over time and the procedures it calls,
one function each:

- ``step_by_step``: the loop over time;
- ``resolve_options``: the settings, decided once;
- ``describe_problem``: what is being solved, read once from the input;
- ``initial_state``: the state at the first instant;
- ``materials_at``: the materials at a given instant and state;
- ``store_step``: the record of a step;
- ``adjust_time_step``: the time step chosen from the last one's difficulty;
- ``heat_step``, ``heat.heat_step_linear``, ``heat.heat_step_dupont``: the
  heat step, iterated θ-scheme, single linearised solve, or Dupont's
  three-level scheme;
- ``mechanical_step``: one mechanical step;
- ``anderson.anderson_correction``: Anderson acceleration of the iterations;
- ``control.automatic_control`` (or the hook ``automatic_measure``) and
  ``control.indirect_control``: load control by a strain measure.

The parts of ``mechanical_step`` live in ``pyrucast.stepping.mechanics``:
``iteration_operator``, ``reference_forces``, ``nonlinear_predictor``,
``equilibrium_iterations``, ``integrate_increment``, ``convergence_status``.
Everything is composed of pyrucast operators; only Anderson's three scalars
are computed in Python.

Large displacements follow from the laws: when they read a large-transformation
strain measure (``strain_measure``), the body stays on the reference
configuration and what acts on its current shape is evaluated on a current
configuration, ``pyrucast.stepping.configuration``.
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
