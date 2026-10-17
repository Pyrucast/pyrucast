"""The settings of a step-by-step computation, decided once (Cast3M ``PAS_DEFA``).

``resolve_options`` reads the input dictionary **once**: defaults, the
incompatibilities between settings, the features this package does not offer
(refused with a message rather than ignored). Nothing below it reads ``data``
for a setting again.
"""

from dataclasses import dataclass, field
from typing import Optional

__all__ = ["HOOKS", "Options", "resolve_options"]

# Keys of features deliberately outside this package, with the reason given to
# whoever asks for one. A key present in ``data`` is refused, never ignored.
_UNSUPPORTED = {
    "non_local": "non-local regularisation (Cast3M NON_LOCAL) is not available",
    "asymptotic": "the asymptotic numerical method (Cast3M MAN) is not available",
    "fefp": "multiplicative plasticity F = FeFp (Cast3M FEFP_FORMULATION) is not available",
    "substepping": "Gauss-point substepping (Cast3M SUBSTEPPING) is not available",
    "friction": "friction is not available; contact is frictionless",
    "cavity_radiation": "cavity radiation (view factors) is not available; radiation is to infinity",
    "phase_change": "phase change is not available",
    "evolving_model": "a model changing during the computation (Cast3M MODE/MATE) is not available",
    "consolidation": "consolidation and THM couplings are not available",
    "advection": "advection is not available",
    "automatic_augmentation": "automatic stiffness augmentation is not available; "
    "give an explicit `augmented_stiffness` matrix",
    "large_displacements": "large displacements arrive with the follower loads and "
    "configuration transport",
}

# The settings a caller may give, with their defaults (``None``: decided from
# the others in ``resolve_options``).
_DEFAULTS = {
    "precision": 1e-4,
    "strain_precision": 1e-4,
    "moment_precision": None,
    "max_iterations": None,
    "delta_iterations": None,
    "max_substeps": 100,
    "forced_convergence": None,
    "relaxation": 1.0,
    "acceleration": True,
    "stability": True,
    "tangent": False,
    "tangent_every": 1,
    "predictor": True,
    "adaptive_step": False,
    "dynamics": False,
    "heat_scheme": "nonlinear",
    "theta": 1.0,
    "dupont_relaxation": 0.25,
    "coherence": False,
    "coherence_passes": 10,
    "automatic": False,
    "automatic_target": 1e-3,
    "automatic_replays": 250,
    "t_ref": 0.0,
    "trace": False,
}

_HEAT_SCHEMES = ("nonlinear", "linear", "dupont")


# The user procedures the loop calls (Cast3M REEV_MEC, REEV_THE, PERSO1,
# CHARMECA, CHARTHER, PARATHER), by the name ``data["hooks"]`` gives them.
# Every hook receives ``(data, state)``, ``state`` being the start of the step;
# ``after_heat`` and ``after_mechanics`` also receive the step's outcome (what
# it reached, not yet committed); the two load hooks receive the time and
# return a ``NodeField`` added to the loads, or ``None``; the material hook
# returns the materials to use, or ``None`` to keep them.
HOOKS = (
    "after_mechanics",
    "after_heat",
    "after_step",
    "extra_mechanical_load",
    "extra_heat_load",
    "heat_material_update",
)


def _ignore(*_args, **_kwargs):
    return None


@dataclass(frozen=True)
class Options:
    """Every setting of the computation, resolved (see ``resolve_options``)."""

    times: tuple
    save_times: tuple
    precision: float
    strain_precision: float
    moment_precision: float
    max_iterations: int
    delta_iterations: int
    max_substeps: int
    forced_convergence: bool
    relaxation: float
    acceleration: bool
    stability: bool
    tangent: bool
    tangent_every: int
    predictor: bool
    adaptive_step: bool
    dynamics: bool
    heat_scheme: str
    theta: float
    dupont_relaxation: float
    coupling_passes: int
    coherence_tolerance: float
    automatic: bool
    automatic_target: float
    automatic_replays: int
    t_ref: float
    trace: bool
    indirect_control: Optional[dict] = None
    augmented_stiffness: object = None
    hooks: dict = field(default_factory=dict)

    def hook(self, name):
        """The user procedure called ``name``, or one that does nothing."""
        return self.hooks.get(name, _ignore)


def resolve_options(data):
    """The settings of ``data``, defaults filled in and conflicts settled
    (Cast3M ``PAS_DEFA``).

    Refuses — with the reason — a feature this package does not offer, an
    unknown heat scheme, and the combinations Cast3M itself forbids: coupled
    coherence passes with adaptive steps or automatic control. Returns an
    immutable ``Options``.
    """
    for key, reason in _UNSUPPORTED.items():
        if data.get(key):
            raise ValueError(f"step_by_step: `{key}` — {reason}")
    if "times" not in data or len(data["times"]) == 0:
        raise ValueError("step_by_step: `times` must list at least one instant")

    given = {k: data.get(k, v) for k, v in _DEFAULTS.items()}
    times = tuple(sorted(float(t) for t in data["times"]))
    save_times = tuple(sorted(float(t) for t in data.get("save_times", times)))

    if given["heat_scheme"] not in _HEAT_SCHEMES:
        raise ValueError(
            f"step_by_step: heat_scheme `{given['heat_scheme']}` — "
            f"expected one of {', '.join(_HEAT_SCHEMES)}"
        )
    coherence = bool(given["coherence"])
    if coherence and (given["adaptive_step"] or given["automatic"]):
        raise ValueError(
            "step_by_step: `coherence` (thermal-mechanical passes) cannot be combined "
            "with `adaptive_step` or `automatic` control"
        )
    automatic = bool(given["automatic"])
    indirect = data.get("indirect_control")
    if automatic and indirect is not None:
        raise ValueError(
            "step_by_step: `automatic` and `indirect_control` are two ways of choosing "
            "the load level — give one"
        )

    # Forced convergence is on by default, off with coherence passes (a pass
    # must converge for the coherence test to mean anything).
    forced = given["forced_convergence"]
    if forced is None:
        forced = not coherence
    # 49 iterations, 250 when forced convergence or a control may need them.
    max_iterations = given["max_iterations"]
    if max_iterations is None:
        controlled = automatic or indirect is not None or given["adaptive_step"]
        max_iterations = 250 if (forced or controlled) else 49
    delta = given["delta_iterations"]
    if delta is None:
        delta = max_iterations if (automatic or indirect is not None) else 10
    moment_precision = given["moment_precision"]
    if moment_precision is None:
        moment_precision = given["precision"]

    hooks = dict(data.get("hooks", {}))
    unknown = sorted(set(hooks) - set(HOOKS))
    if unknown:
        raise ValueError(
            f"step_by_step: unknown hook(s) {', '.join(unknown)} — expected among "
            f"{', '.join(HOOKS)}"
        )
    return Options(
        times=times,
        save_times=save_times,
        precision=float(given["precision"]),
        strain_precision=float(given["strain_precision"]),
        moment_precision=float(moment_precision),
        max_iterations=int(max_iterations),
        delta_iterations=int(delta),
        max_substeps=int(given["max_substeps"]),
        forced_convergence=bool(forced),
        relaxation=float(given["relaxation"]),
        acceleration=bool(given["acceleration"]),
        stability=bool(given["stability"]),
        tangent=bool(given["tangent"]),
        tangent_every=max(1, int(given["tangent_every"])),
        predictor=bool(given["predictor"]),
        adaptive_step=bool(given["adaptive_step"]),
        dynamics=bool(given["dynamics"]),
        heat_scheme=given["heat_scheme"],
        theta=float(given["theta"]),
        dupont_relaxation=float(given["dupont_relaxation"]),
        coupling_passes=int(given["coherence_passes"]) if coherence else 1,
        coherence_tolerance=float(given["precision"]),
        automatic=automatic,
        automatic_target=float(given["automatic_target"]),
        automatic_replays=int(given["automatic_replays"]),
        t_ref=float(given["t_ref"]),
        trace=bool(given["trace"]),
        indirect_control=indirect,
        augmented_stiffness=data.get("augmented_stiffness"),
        hooks=hooks,
    )
