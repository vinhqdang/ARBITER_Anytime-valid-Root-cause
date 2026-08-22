"""Choosing the leakage budget against a mixed population.

Two results constrain this choice, and together they make it non-trivial:

  * Against a STRATEGIC adversary, conditional uniformity is optimal and any
    leakage is exploited: harm ~ sqrt(T) / ((1-eps) sqrt(p)). See
    monitor.policy. Monitor-private targeting gains nothing, because the
    adversary's behaviour cannot depend on what it cannot observe.

  * Against a NAIVE failure -- a miscalibrated or buggy agent that defects
    wherever its inputs lead, with no awareness of monitoring -- targeting is
    strictly better. Concentrating audits where failures actually are raises
    the audited flag rate and shortens detection.

So eps trades one population off against the other, and the optimum is
interior whenever both are present. This module computes it.

The practical reading: a monitor should target only as hard as its belief that
failures are accidental rather than adversarial justifies. That belief is the
real tuning knob, and it is a modelling commitment the operator must make
explicitly rather than inherit from a default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


from .eprocess import kl_bernoulli
from .policy import audited_flag_rate, optimal_evasion_horizon


@dataclass(frozen=True)
class TuningResult:
    """Optimal leakage and the costs it balances."""

    epsilon: float
    expected_harm: float
    strategic_harm: float
    naive_harm: float
    strategic_weight: float


def naive_harm(
    epsilon: float,
    audit_rate: float,
    pi_0: float,
    horizon: float,
    defect_rate: float,
    concentration: float,
    beta: float = 0.0,
    alpha: float = 0.05,
) -> float:
    """Harm from an unaware failure, under leakage `epsilon`.

    ``concentration`` in [0, 1] is how well the monitor's priority signal
    predicts where the naive failure actually occurs. At concentration 0 the
    signal is noise and targeting buys nothing; at 1 the failures sit exactly
    where the monitor looks, and the audited rate is amplified by the full
    targeting gain.

    A naive agent does not move in response to the allocation, so leakage is
    pure upside here -- which is precisely why the two populations pull in
    opposite directions.
    """
    # Targeting multiplies the audited defection rate by up to 1/(1-eps):
    # the same factor by which a strategic agent would DIVIDE it.
    gain = 1.0 + concentration * epsilon / max(1.0 - epsilon, 1e-9)
    q_audit = min(1.0, defect_rate * gain)
    flag_rate = audited_flag_rate(q_audit, pi_0, beta)

    if flag_rate <= pi_0 + 1e-15:
        return defect_rate * horizon

    audits = math.log(1.0 / alpha) / kl_bernoulli(flag_rate, pi_0)
    rounds = min(audits / audit_rate, horizon)
    return defect_rate * rounds


def strategic_harm(
    epsilon: float,
    audit_rate: float,
    pi_0: float,
    horizon: float,
    beta: float = 0.0,
    alpha: float = 0.05,
    thin_fraction: float = 0.9,
) -> float:
    """Harm from an adversary that best-responds to the allocation."""
    return optimal_evasion_horizon(
        pi_min=(1.0 - epsilon) * audit_rate,
        audit_rate=audit_rate,
        pi_0=pi_0,
        horizon=horizon,
        beta=beta,
        alpha=alpha,
        thin_fraction=thin_fraction,
    ).harm


def optimal_leakage(
    strategic_weight: float,
    audit_rate: float = 0.1,
    pi_0: float = 0.05,
    horizon: float = 1e6,
    naive_defect_rate: float = 0.05,
    concentration: float = 0.8,
    beta: float = 0.0,
    alpha: float = 0.05,
    grid: int = 400,
) -> TuningResult:
    """Minimise expected harm over the leakage budget.

    ``strategic_weight`` is the operator's prior that a failure, if it occurs,
    is adversarial rather than accidental. At weight 1 the optimum is eps = 0
    (uniform); at weight 0 it is eps -> 1 (target freely).
    """
    if not 0.0 <= strategic_weight <= 1.0:
        raise ValueError("strategic_weight must be in [0, 1]")

    best: TuningResult | None = None
    for step in range(grid):
        eps = step / grid  # excludes eps = 1, where harm is unbounded
        s_harm = strategic_harm(eps, audit_rate, pi_0, horizon, beta, alpha)
        n_harm = naive_harm(
            eps, audit_rate, pi_0, horizon, naive_defect_rate,
            concentration, beta, alpha,
        )
        expected = strategic_weight * s_harm + (1.0 - strategic_weight) * n_harm
        candidate = TuningResult(
            epsilon=eps,
            expected_harm=expected,
            strategic_harm=s_harm,
            naive_harm=n_harm,
            strategic_weight=strategic_weight,
        )
        if best is None or candidate.expected_harm < best.expected_harm:
            best = candidate

    assert best is not None
    return best


def sweep_weights(weights: tuple[float, ...], **kwargs: float) -> list[TuningResult]:
    """Optimal leakage across operator beliefs -- the tuning curve."""
    return [optimal_leakage(float(w), **kwargs) for w in weights]  # type: ignore[arg-type]


def exact_delta_star(
    audit_rate: float, pi_0: float, horizon: float, alpha: float = 0.05
) -> float:
    """The audited-flag-rate deviation at the strategic optimum, EXACTLY.

    At the crossover q* (Theorem 2's proof), the audited flag rate satisfies
    KL(pi_0 + delta*, pi_0) = log(1/alpha) / (horizon * audit_rate) exactly.
    The right side is a CONSTANT independent of epsilon, so delta* -- and
    hence the (1-eps)^-1 shape of B(eps) = delta*.horizon / (kappa.(1-eps))
    -- is exact, not a small-deviation artefact. Only this one transcendental
    equation needs solving; the quadratic-KL closed form in Theorem 2 is
    just its first-order solution.
    """
    kl_target = math.log(1.0 / alpha) / (horizon * audit_rate)
    lo, hi = 1e-12, 1.0 - pi_0 - 1e-9
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if kl_bernoulli(pi_0 + mid, pi_0) < kl_target:
            lo = mid
        else:
            hi = mid
    return hi


def w_star_lower_bound(
    audit_rate: float,
    pi_0: float,
    horizon: float,
    defect_rate: float,
    concentration: float,
    beta: float = 0.0,
    alpha: float = 0.05,
) -> float:
    """A certified LOWER BOUND on the true operating threshold w*.

    Derived from the sign of d/deps[w.B(eps) + (1-w).naive_harm(eps)] at
    eps=0, using EXACT (non-quadratic) derivatives on both sides:

      dB/deps|_0     = delta*.horizon/kappa    (exact, via exact_delta_star)
      d(naive)/deps|_0 = -defect_rate.log(1/alpha)/audit_rate
                          . dKL/dflag(flag_0) . kappa.defect_rate.concentration
                          / KL(flag_0, pi_0)^2                (exact)

    w := this ratio makes eps=0 a LOCAL minimum of expected harm. Because
    eps=0 becoming a local minimum is NECESSARY (not sufficient) for it to
    be the GLOBAL minimum, w_star_lower_bound <= true w*: for any w below
    this bound, targeting strictly helps, with no numerics required to know
    that. Confirmed numerically to sit below the true (grid-search) w* by
    roughly 15-20% at tested parameters -- a valid, cheap, one-sided
    guarantee, not a full closed form for w* itself.
    """
    kappa = 1.0 - beta - pi_0
    delta_star = exact_delta_star(audit_rate, pi_0, horizon, alpha)
    dB_deps = delta_star * horizon / kappa

    delta0 = kappa * defect_rate
    flag0 = pi_0 + delta0
    kl0 = kl_bernoulli(flag0, pi_0)
    dkl_dflag = math.log(flag0 * (1.0 - pi_0) / (pi_0 * (1.0 - flag0)))
    dnaive_deps = (
        -defect_rate
        * math.log(1.0 / alpha)
        / audit_rate
        * dkl_dflag
        * delta0
        * concentration
        / kl0**2
    )

    return (-dnaive_deps) / (dB_deps - dnaive_deps)
