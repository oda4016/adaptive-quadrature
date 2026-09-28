# adaptive_quadrature

Numerically integrate a scalar real-valued function on a closed interval `[a, b]` to a requested absolute tolerance using an adaptive, recursive Simpson's rule.

```python
from adaptive_quadrature import integrate

# integrate exp(-x^2) on [-3, 3]
value = integrate(lambda x: __import__("math").exp(-x * x), -3.0, 3.0, tol=1e-12)
print(value)  # ~ 0.99997791 (= sqrt(pi) * erf(3))
```

The exported names are `integrate` and `AdaptiveSimpsonError`.

`integrate(f, a, b, tol=1e-10, max_depth=50)` returns a `float`. `a` may be greater than `b`, in which case the result is negated. `a == b` returns `0.0`. Non-finite integrand values, non-finite endpoints, non-positive `tol`, and hitting `max_depth` all raise `AdaptiveSimpsonError`.

## Why this exists

The problem is the one every numerical methods course poses: compute a definite integral to a desired accuracy without knowing in advance how many nodes the integrand will need. Fixed-order quadrature either wastes work on smooth regions or misses sharp features. Adaptive Simpson's rule subdivides where the integrand is hard and stops where it is easy, which is the trade-off most people actually want for smooth-to-moderately-singular integrands.

The trade-off made here is simplicity over generality. The library handles finite intervals and finite-valued integrands only. There is no support for infinite intervals, weighted quadrature, or vectorised integrands. The tolerance is absolute, not relative, because relative tolerance behaves badly for integrals whose true value is near zero; callers who want relative accuracy can scale `tol` themselves.

## The awkward edge

The error estimate (`|fine - coarse|` on each subinterval) is a heuristic, not a bound. For integrands that are merely continuous but not differentiable — `abs(x - c)` is the canonical case — Simpson's rule converges slowly and the estimator keeps asking for more subdivision. Rather than silently returning a wrong answer, the routine raises `AdaptiveSimpsonError` when it hits `max_depth`. If you have a genuinely nasty integrand, raise `max_depth`; if it is singular, transform variables first and use this library on the smooth result.
