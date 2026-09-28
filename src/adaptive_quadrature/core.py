"""Adaptive Simpson's rule quadrature.

This module integrates a scalar real-valued function over a closed interval
[a, b] to a requested absolute tolerance using an adaptive, recursive
implementation of Simpson's rule. The adaptation is driven by the classic
Lyness-style estimate: on each subinterval we compare one Simpson estimate
(uses the midpoint) against the sum of two Simpson estimates on the left and
right halves. If they agree to within the tolerance scaled to that
subinterval, we accept the finer estimate; otherwise we recurse.

Design decisions worth stating plainly:

* We only support finite intervals with finite numeric endpoints. Infinite
  or semi-infinite intervals are out of scope; handling them well requires a
  change of variables or a quadrature weight class, which is a different
  library.
* The integrand must accept a single real number and return a real number.
  We do not try to vectorise; passing a vectorised callable is fine as long
  as it also works on scalars, but we never call it with an array.
* We cap the recursion depth to keep pathological integrands from blowing
  the C stack. When the cap is hit we raise rather than silently returning a
  wrong answer. This is a deliberate choice: a wrong answer that looks
  successful is the worst possible failure mode for a quadrature routine.
* Tolerance is absolute. We do not secretly switch to relative tolerance
  based on the magnitude of the integral, because that interacts badly with
  integrals whose true value is near zero. Callers who want relative
  accuracy can scale the tolerance themselves.
"""

from __future__ import annotations

import math
from typing import Callable

__all__ = ["integrate", "AdaptiveSimpsonError"]


class AdaptiveSimpsonError(Exception):
    """Raised when adaptive quadrature cannot meet the requested tolerance.

    This is raised for hard failures: non-finite integrand values, bad
    intervals, or hitting the recursion-depth cap. It is *not* raised merely
    because the estimated error is large; in that case the algorithm keeps
    subdividing until it either succeeds or hits the cap.
    """


def _simpson(f: Callable[[float], float], a: float, b: float) -> float:
    """Composite Simpson estimate on [a, b] using one parabola (3 points).

    This is the atomic estimate from which the adaptive rule is built. We use
    the closed form (b - a) / 6 * (f(a) + 4 f(m) + f(b)) rather than the
    two-panel composite form because it is cheaper and the adaptive rule
    composes these naturally by recursion.
    """
    m = 0.5 * (a + b)
    return (b - a) / 6.0 * (f(a) + 4.0 * f(m) + f(b))


def _adaptive(
    f: Callable[[float], float],
    a: float,
    b: float,
    whole: float,
    tol: float,
    depth: int,
    max_depth: int,
) -> float:
    """Recursive core of the adaptive Simpson integrator.

    ``whole`` is the Simpson estimate on [a, b] passed in from the caller so
    we do not recompute it. We split into [a, m] and [m, b], form their
    Simpson estimates, and compare ``left + right`` against ``whole``.

    The error estimate ``|left + right - whole|`` is a standard heuristic for
    Simpson's rule; it is not a rigorous bound, but it is what makes the
    method *adaptive* — regions where the integrand is hard get subdivided
    more than regions where it is smooth.

    We scale the tolerance by 0.5 at each level so that the sum of the
    tolerances over all leaves equals the original tolerance. This is the
    usual bookkeeping for recursive adaptive quadrature and keeps the global
    error budget honest.
    """
    m = 0.5 * (a + b)
    left = _simpson(f, a, m)
    right = _simpson(f, m, b)
    finer = left + right

    if depth >= max_depth:
        # Refusing to subdivide further is safer than returning a value we
        # know is wrong. The caller can raise ``max_depth`` if they have a
        # genuinely nasty integrand and are willing to pay for the stack.
        raise AdaptiveSimpsonError(
            f"recursion depth cap {max_depth} reached on interval "
            f"[{a!r}, {b!r}] with estimated error {abs(finer - whole)!r}; "
            "the integrand may be too singular or the tolerance too tight"
        )

    if abs(finer - whole) <= tol:
        # Richardson-style correction: the finer estimate is typically O(h^5)
        # better than the coarse one, so adding the difference back in shaves
        # a little error for free. The factor 1/15 comes from the ratio of
        # the leading error terms of Simpson's rule at two step sizes.
        return finer + (finer - whole) / 15.0

    return _adaptive(f, a, m, left, 0.5 * tol, depth + 1, max_depth) + _adaptive(
        f, m, b, right, 0.5 * tol, depth + 1, max_depth
    )


def integrate(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_depth: int = 50,
) -> float:
    """Integrate ``f`` over the closed interval ``[a, b]``.

    Parameters
    ----------
    f:
        Scalar real-valued function. Must accept a single float and return a
        float. Non-finite return values (NaN or inf) raise
        :class:`AdaptiveSimpsonError`.
    a, b:
        Finite interval endpoints. ``a`` may be greater than ``b``; the
        result is then the negative of the integral from ``b`` to ``a``, as
        is standard for oriented integrals.
    tol:
        Requested absolute tolerance on the integral. Must be positive and
        finite. The error estimate is a heuristic, not a guarantee, but in
        practice the returned value is within a small multiple of ``tol`` of
        the true integral for well-behaved integrands.
    max_depth:
        Maximum recursion depth. Each level halves the subinterval width,
        so ``max_depth=50`` permits subintervals as small as
        ``|b - a| * 2**-50``. Hitting the cap raises
        :class:`AdaptiveSimpsonError`.

    Returns
    -------
    float
        The estimated integral.

    Raises
    ------
    AdaptiveSimpsonError
        If the interval is degenerate (a == b is allowed and returns 0.0;
        NaN endpoints are not), if ``tol`` is out of range, if ``f`` returns
        a non-finite value, or if ``max_depth`` is reached.
    """
    if not math.isfinite(a) or not math.isfinite(b):
        raise AdaptiveSimpsonError(f"interval endpoints must be finite, got a={a!r}, b={b!r}")
    if not math.isfinite(tol) or tol <= 0.0:
        raise AdaptiveSimpsonError(f"tol must be a positive finite number, got {tol!r}")
    if not isinstance(max_depth, int) or max_depth < 1:
        raise AdaptiveSimpsonError(f"max_depth must be a positive int, got {max_depth!r}")

    if a == b:
        return 0.0

    # Orient the integral so the recursion always sees a < b. This keeps the
    # midpoint arithmetic clean and avoids sign mistakes in the error
    # estimate. We flip the sign at the end.
    if a > b:
        return -integrate(f, b, a, tol=tol, max_depth=max_depth)

    # Probe the integrand at the endpoints and midpoint once up front. This
    # turns "f returned NaN" into a clear error before we recurse, rather
    # than letting NaN propagate through the error estimate (where it would
    # silently defeat every tolerance check and produce a NaN result).
    fa = f(a)
    fb = f(b)
    fm = f(0.5 * (a + b))
    for label, val in (("f(a)", fa), ("f(b)", fb), ("f(m)", fm)):
        if not math.isfinite(val):
            raise AdaptiveSimpsonError(
                f"integrand returned non-finite value {val!r} at {label}"
            )

    # Build the top-level Simpson estimate from the already-computed values
    # so we don't evaluate f three extra times.
    m = 0.5 * (a + b)
    whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb)

    # Wrap f so that non-finite values inside the recursion raise cleanly.
    def guarded(x: float) -> float:
        v = f(x)
        if not math.isfinite(v):
            raise AdaptiveSimpsonError(
                f"integrand returned non-finite value {v!r} at x={x!r}"
            )
        return v

    return _adaptive(guarded, a, b, whole, tol, 0, max_depth)
