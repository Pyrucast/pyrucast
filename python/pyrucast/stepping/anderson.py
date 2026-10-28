"""Anderson acceleration of a fixed-point iteration, memory 3 (Cast3M ``ACT3``).

The iteration ``x ← x + K⁻¹ r(x)`` with a constant ``K`` is a fixed point.
Every other iteration, the residual is corrected by the combination of the
last residuals that would have cancelled best: Anderson's method on the
residuals, as Cast3M's ``ACT3`` does it — the same normal equations, solved by
the same elimination, with the same fallbacks.

The algebra is three scalars and a few scalar products: it stays in Python.
The fields are only ever combined through pyrucast's operators.
"""

from .. import measure, node_field

__all__ = ["Accelerator", "anderson_correction"]

# Below this, a pivot or a squared norm is taken as zero (Cast3M XSZPRE, XPETIT).
_ROUNDING = 1e-15
_TINY = 1e-300


class _NormalEquations:
    """ACT3's three normal equations of ``min ‖F0 + Σ xⱼ (Fⱼ − F0)‖``, each
    scaled by its diagonal, and its three ways of solving them: with three
    directions, two, or one — each falling back to the next where ACT3 does."""

    def __init__(self, f11, f22, f33, f12, f13, f23, f01, f02, f03):
        self.a1, self.b1, self.c1, self.j1 = 1.0, f12 / f11, f13 / f11, -f01 / f11
        self.d1, self.e1, self.f1, self.k1 = f12 / f22, 1.0, f23 / f22, -f02 / f22
        self.g1, self.h1, self.l1 = f13 / f33, f23 / f33, -f03 / f33

    def three(self):
        """``x3`` eliminated, then ``x2``, then back-substitution."""
        a1, b1, c1, j1 = self.a1, self.b1, self.c1, self.j1
        a2, b2, j2 = a1 - self.g1 * c1, b1 - self.h1 * c1, j1 - self.l1 * c1
        d2 = self.d1 - self.g1 * self.f1
        e2 = self.e1 - self.h1 * self.f1
        k2 = self.k1 - self.l1 * self.f1
        if e2 <= self.e1 * _ROUNDING:
            return self.two()
        a3 = a2 - d2 * b2 / e2
        j3 = j2 - k2 * b2 / e2
        if a3 <= a1 * _ROUNDING or abs(a2) <= a1 * 1e-6:
            return self.two()
        xl = j3 / a3
        if abs(b2) <= abs(b1) * 1e-6:
            return self.two()
        xm = (j2 - a2 * xl) / b2
        if abs(c1) < _ROUNDING:
            return self.two()
        return (xl, xm, (j1 - a1 * xl - b1 * xm) / c1)

    def two(self):
        """Without ``F3``: ``x2`` eliminated from the first two equations."""
        a1, b1, j1 = self.a1, self.b1, self.j1
        a2 = a1 - self.d1 * b1 / self.e1
        j2 = j1 - self.k1 * b1 / self.e1
        if a2 <= a1 * _ROUNDING or abs(b1) < _ROUNDING:
            return self.one()
        xl = j2 / a2
        return (xl, (j1 - a1 * xl) / b1, 0.0)

    def one(self):
        """``F1`` alone."""
        if abs(self.a1) < _ROUNDING:
            return (0.0, 0.0, 0.0)
        return (self.j1 / self.a1, 0.0, 0.0)


def _near(weights, one_at):
    """Whether the weights are the unit vector ``one_at`` to 0.1: the
    combination would only return a point already visited."""
    return all(
        abs(w - (1.0 if k == one_at else 0.0)) < 0.1 for k, w in enumerate(weights)
    )


def anderson_correction(directions, residuals):
    """The correction to subtract from the residual (Cast3M ``ACT3``).

    ``residuals`` are the last four weighted residuals ``(F3, F2, F1, F0)``,
    oldest first; ``directions`` the matching ``(D2, D1, D0)``. The weights
    ``x`` minimise ``‖F0 + Σ xⱼ (Fⱼ − F0)‖``; the correction is
    ``(x1+x2+x3) D0 + (x2+x3) D1 + x3 D2``. ``None`` when nothing is gained.

    As ACT3 does: an oscillating sequence (``‖F1−F0‖`` well above
    ``‖F2−F0‖`` while comparable to ``‖F3−F0‖``) takes the mid-point of the
    flip-flop; a singular system keeps two directions, then one; weights that
    would land back on a point already visited are refused, one direction at
    a time.
    """
    f3, f2, f1, f0 = residuals
    d2, d1, d0 = directions
    g1, g2, g3 = f1 - f0, f2 - f0, f3 - f0
    f11 = max(measure.xtx(g1), _TINY * 1e2)
    f22 = max(measure.xtx(g2), _TINY * 1e2)
    f33 = max(measure.xtx(g3), _TINY * 1e2)
    equations = _NormalEquations(
        f11,
        f22,
        f33,
        measure.xty(g1, g2),
        measure.xty(g1, g3),
        measure.xty(g2, g3),
        measure.xty(f0, g1),
        measure.xty(f0, g2),
        measure.xty(f0, g3),
    )
    if f11 / f22 > 1.2 and 0.707 < f11 / f33 < 1.414:
        # Flip-flop between two states: their mid-point.
        weights = ((f11**0.5 + f22**0.5) / (2.0 * f11**0.5), 0.0, 0.0)
    else:
        weights = equations.three()
    # A combination landing back on a point already visited is no progress:
    # one direction fewer, down to none. Each fallback has fewer non-zero
    # weights than the one before, so this ends.
    while any(weights):
        if _near(weights, 0):
            weights = (0.0, 0.0, 0.0)
        elif _near(weights, 1):
            weights = equations.one()
        elif _near(weights, 2):
            weights = equations.two()
        else:
            break
    xl, xm, xn = weights
    if xl == 0.0 and xm == 0.0 and xn == 0.0:
        return None
    return d0 * (xl + xm + xn) + d1 * (xm + xn) + d2 * xn


class Accelerator:
    """The memory Cast3M's ``UNPAS`` keeps around ``ACT3``.

    Called once per iteration with the out-of-balance forces (constraint
    rows excluded): it records ``F = r·w`` (``w`` the nodal weights) and
    ``D = r − c`` (``c`` the correction applied at the previous iteration), and
    when asked to accelerate — and the four residuals are there — returns the
    correction to subtract. With ``stability``, a correction that would point
    backwards along the last increment is dropped, and one that would blow it
    up beyond ``1e5`` times is scaled down.
    """

    def __init__(self, weights, stability, pairing):
        self._weights = weights
        self._stability = stability
        self._pairing = pairing
        self._f = [None, None, None]  # F1, F2, F3
        self._d = [None, None]  # D1, D2
        self._applied = None
        self._support = None

    def __call__(self, residual, accelerate, last_increment):
        # Every residual is brought onto the first one's support: products and
        # differences across iterations then pair node with node.
        if self._support is None:
            self._support = residual
            if self._weights is not None:
                self._weights = node_field.restrict_like(self._weights, residual)
        else:
            residual = node_field.restrict_like(residual, self._support)
        weighted = residual * self._weights if self._weights is not None else residual
        direction = residual if self._applied is None else residual - self._applied
        correction = None
        ready = all(f is not None for f in self._f) and all(
            d is not None for d in self._d
        )
        if accelerate and ready:
            correction = anderson_correction(
                (self._d[1], self._d[0], direction),
                (self._f[2], self._f[1], self._f[0], weighted),
            )
            if (
                correction is not None
                and self._stability
                and last_increment is not None
            ):
                correction = self._guarded(residual, correction, last_increment)
        self._f = [weighted, self._f[0], self._f[1]]
        self._d = [direction, self._d[0]]
        self._applied = correction
        return correction

    def _guarded(self, residual, correction, increment):
        reference = self._pairing(increment, residual) + _TINY
        ratio = self._pairing(increment, residual - correction) / reference
        limit = 1.0 if reference < 0.0 else 1e5
        if ratio <= 0.0:
            return None  # retrograde: no acceleration
        if ratio > limit:
            return correction * (limit / ratio)
        return correction
