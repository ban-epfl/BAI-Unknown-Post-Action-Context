import numpy as np
from functools import lru_cache
from scipy.optimize import minimize_scalar
from scipy.special import lambertw, zeta


@lru_cache(maxsize=None)
def C_exp(x):
    # C^exp of Kaufmann and Koolen (2021) (Definition 15 of the paper), C_exp(x) = x + O(ln x)
    def g(xi, lam):
        return lam * (1 + xi) * np.log(2 * zeta(2) / np.log(1 + xi)**2) - np.log(1 - lam * (1 + xi))

    def min_over_lambda(xi):
        res = minimize_scalar(
            lambda lam: (g(xi, lam) + x) / lam,
            bounds=(1e-9, (1 - 1e-12) / (1 + xi)),
            method="bounded",
            options={"xatol": 1e-10}
        )
        return res.fun

    res = minimize_scalar(min_over_lambda, bounds=(1e-6, 0.5), method="bounded", options={"xatol": 1e-10})
    return res.fun


def W_bar(x):
    # W(x) = -W_{-1}(-e^{-x}) used in the sub-Gaussian threshold of Tuynman and Degenne (2025),
    # x + ln(x) <= W(x) <= x + ln(x) + 1/2
    return -lambertw(-np.exp(-x), k=-1).real


def context_initiator(n, k):
    # random context distributions of the n arms: k uniform variables per arm, normalized to sum to 1
    samples = np.random.uniform(low=0.0, high=1.0, size=(n, k))
    row_sums = samples.sum(axis=1, keepdims=True)
    normalized_samples = samples / row_sums

    return normalized_samples


def categorical_kl(p, q):
    return sum([p[i] * np.log(p[i]/q[i]) for i in range(len(p))])
