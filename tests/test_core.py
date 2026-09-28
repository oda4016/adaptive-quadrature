import math
import unittest

from adaptive_quadrature import integrate, AdaptiveSimpsonError


class TestIntegrate(unittest.TestCase):
    def test_polynomial_exact_for_simpson(self):
        # Simpson's rule integrates cubics exactly on a single panel, so the
        # adaptive routine should nail this to within tolerance with no
        # subdivision.
        result = integrate(lambda x: x ** 3 - 2 * x ** 2 + x - 1, -1.0, 2.0, tol=1e-12)
        # analytic value
        expected = (2.0 ** 4 / 4 - 2 * 2.0 ** 3 / 3 + 2.0 ** 2 / 2 - 2.0) - (
            (-1.0) ** 4 / 4 - 2 * (-1.0) ** 3 / 3 + (-1.0) ** 2 / 2 - (-1.0)
        )
        self.assertAlmostEqual(result, expected, places=12)

    def test_gaussian(self):
        # exp(-x^2) on [-3, 3] is a smooth, non-polynomial integrand; the
        # true value is sqrt(pi) * erf(3).
        result = integrate(lambda x: math.exp(-x * x), -3.0, 3.0, tol=1e-12)
        expected = math.sqrt(math.pi) * math.erf(3.0)
        self.assertAlmostEqual(result, expected, places=10)

    def test_zero_interval(self):
        self.assertEqual(integrate(lambda x: x ** 2, 1.0, 1.0), 0.0)

    def test_reversed_interval(self):
        # Oriented integral: flipping the endpoints flips the sign.
        f = lambda x: math.sin(x) + 1.0
        forward = integrate(f, 0.0, 1.0, tol=1e-12)
        backward = integrate(f, 1.0, 0.0, tol=1e-12)
        self.assertAlmostEqual(forward, -backward, places=12)

    def test_constant_function(self):
        self.assertAlmostEqual(integrate(lambda x: 7.5, 0.0, 2.0, tol=1e-12), 15.0, places=12)

    def test_negative_integrand(self):
        self.assertAlmostEqual(
            integrate(lambda x: -math.exp(-x), 0.0, 1.0, tol=1e-12),
            math.exp(-1.0) - 1.0,
            places=12,
        )

    def test_sharp_peak_resolved(self):
        # A narrow Gaussian centered in the interval. This is the kind of
        # integrand that forces the adaptive subdivision to actually do
        # work; a non-adaptive rule with the same node count would miss it.
        center = 0.5
        width = 1e-2
        f = lambda x: math.exp(-((x - center) ** 2) / (2 * width ** 2))
        result = integrate(f, 0.0, 1.0, tol=1e-10)
        # True area of an unnormalised Gaussian over the whole line is
        # width * sqrt(2 pi). The tails outside [0, 1] are negligible.
        expected = width * math.sqrt(2.0 * math.pi)
        self.assertAlmostEqual(result, expected, places=6)

    def test_non_finite_integrand_raises(self):
        with self.assertRaises(AdaptiveSimpsonError):
            integrate(lambda x: math.nan, 0.0, 1.0)

    def test_integrand_returning_inf_raises(self):
        with self.assertRaises(AdaptiveSimpsonError):
            integrate(lambda x: math.inf if x > 0.5 else 0.0, 0.0, 1.0)

    def test_non_finite_endpoint_raises(self):
        with self.assertRaises(AdaptiveSimpsonError):
            integrate(lambda x: x, 0.0, math.inf)

    def test_bad_tolerance_raises(self):
        with self.assertRaises(AdaptiveSimpsonError):
            integrate(lambda x: x, 0.0, 1.0, tol=0.0)
        with self.assertRaises(AdaptiveSimpsonError):
            integrate(lambda x: x, 0.0, 1.0, tol=-1e-6)
        with self.assertRaises(AdaptiveSimpsonError):
            integrate(lambda x: x, 0.0, 1.0, tol=math.nan)

    def test_bad_max_depth_raises(self):
        with self.assertRaises(AdaptiveSimpsonError):
            integrate(lambda x: x, 0.0, 1.0, max_depth=0)
        with self.assertRaises(AdaptiveSimpsonError):
            integrate(lambda x: x, 0.0, 1.0, max_depth=-5)

    def test_recursion_cap_raises(self):
        # A function that is C^0 but not C^1 at a non-dyadic point defeats
        # Simpson's error model, so the routine will keep subdividing near
        # the kink until it hits the depth cap. We set max_depth very low to
        # make this cheap and deterministic. The kink must not lie on a
        # dyadic rational, or Simpson's rule becomes exact once the kink
        # sits at a panel endpoint and convergence is immediate.
        f = lambda x: abs(x - 0.3)
        with self.assertRaises(AdaptiveSimpsonError):
            integrate(f, 0.0, 1.0, tol=1e-14, max_depth=5)

    def test_result_is_float(self):
        result = integrate(lambda x: x * x, 0.0, 1.0)
        self.assertIsInstance(result, float)


if __name__ == "__main__":
    unittest.main()
