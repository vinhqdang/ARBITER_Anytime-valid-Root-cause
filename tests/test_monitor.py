"""Tests for the e-process and allocation machinery.

The important ones are the analytic identities: if gro_wager and kl_bernoulli
disagree, every delay and threshold number downstream is wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from monitor.eprocess import (
    EProcess,
    gro_wager,
    kl_bernoulli,
    minimax_wager,
    stopped_e_bh,
)
from monitor.policy import allocate, best_response, undetectable_harm_rate


# --- Analytic identities ----------------------------------------------------


@pytest.mark.parametrize("pi_0", [0.05, 0.1, 0.3])
@pytest.mark.parametrize("q", [0.2, 0.5, 0.9])
def test_gro_wager_growth_equals_kl(pi_0: float, q: float) -> None:
    """At the GRO wager, per-audit log-growth must equal KL(q || pi_0)."""
    if q <= pi_0:
        pytest.skip("no positive-growth wager exists")
    lam = gro_wager(q, pi_0)

    growth = q * math.log(1 + lam * (1 - pi_0)) + (1 - q) * math.log(1 - lam * pi_0)

    assert growth == pytest.approx(kl_bernoulli(q, pi_0), rel=1e-9)


@pytest.mark.parametrize("pi_0", [0.05, 0.2])
@pytest.mark.parametrize("q", [0.3, 0.7])
def test_gro_wager_closed_form_factors(pi_0: float, q: float) -> None:
    """The wager should give factors q/pi_0 and (1-q)/(1-pi_0)."""
    lam = gro_wager(q, pi_0)
    assert 1 + lam * (1 - pi_0) == pytest.approx(q / pi_0)
    assert 1 - lam * pi_0 == pytest.approx((1 - q) / (1 - pi_0))


def test_gro_wager_zero_at_or_below_null() -> None:
    """No positive-growth bet exists when q <= pi_0 -- the evasion regime."""
    assert gro_wager(0.1, 0.1) == 0.0
    assert gro_wager(0.05, 0.1) == 0.0


def test_wager_stays_in_valid_range() -> None:
    """lam must stay below 1/pi_0 or wealth can go non-positive."""
    for pi_0 in (0.01, 0.1, 0.5):
        for q in np.linspace(pi_0 + 1e-6, 0.999, 50):
            assert 0 <= gro_wager(float(q), pi_0) < 1.0 / pi_0


# --- e-process behaviour ----------------------------------------------------


def test_eprocess_rejects_clear_violator() -> None:
    """A blatant violator should be rejected quickly."""
    ep = EProcess(pi_0=0.05, alpha=0.05)
    lam = gro_wager(0.8, 0.05)
    for _ in range(20):
        ep.update(flagged=True, wager=lam)
        if ep.rejected:
            break
    assert ep.rejected
    assert ep.n_audits < 10


def test_eprocess_never_rejects_evasive_agent() -> None:
    """PROPOSITION 1. An agent that is always compliant when audited drains
    the monitor's wealth; the test never fires however long we run."""
    ep = EProcess(pi_0=0.05, alpha=0.05)
    lam = gro_wager(0.5, 0.05)
    for _ in range(5000):
        ep.update(flagged=False, wager=lam)
    assert not ep.rejected
    # Wealth actively decays -- clean audits are lost bets, not neutral ones.
    assert ep.log_wealth < -10


def test_eprocess_validity_under_null_empirically() -> None:
    """Type-I error under H_0 should sit at or below alpha."""
    rng = np.random.default_rng(0)
    pi_0, alpha, trials = 0.1, 0.05, 400
    false_rejections = 0
    for _ in range(trials):
        ep = EProcess(pi_0=pi_0, alpha=alpha)
        for _ in range(300):
            # Truly compliant: flags occur at exactly the null rate.
            ep.update(flagged=bool(rng.random() < pi_0), wager=ep.plug_in_wager())
            if ep.rejected:
                false_rejections += 1
                break
    assert false_rejections / trials <= alpha * 1.5  # slack for MC noise


def test_eprocess_rejects_invalid_wager() -> None:
    ep = EProcess(pi_0=0.1)
    with pytest.raises(ValueError):
        ep.update(flagged=True, wager=1.0 / 0.1)


def test_minimax_wager_positive_when_alternatives_detectable() -> None:
    lam = minimax_wager(pi_0=0.1, q_grid=(0.3, 0.5, 0.8))
    assert 0 < lam < 1.0 / 0.1


def test_minimax_wager_zero_when_all_alternatives_at_null() -> None:
    assert minimax_wager(pi_0=0.1, q_grid=(0.05, 0.1)) == 0.0


# --- e-BH -------------------------------------------------------------------


def test_e_bh_rejects_nothing_when_all_small() -> None:
    assert stopped_e_bh([1.0, 1.0, 2.0], alpha=0.05) == []


def test_e_bh_rejects_large_e_value() -> None:
    # n=3, alpha=0.05, k=1 threshold is 3/(0.05*1) = 60.
    rejected = stopped_e_bh([100.0, 1.0, 1.0], alpha=0.05)
    assert rejected == [0]


def test_e_bh_empty_input() -> None:
    assert stopped_e_bh([], alpha=0.05) == []


# --- Allocation -------------------------------------------------------------


def test_allocation_floor_holds() -> None:
    """The leakage constraint must bind: min prob >= (1-eps) * rate."""
    priority = np.array([5.0, 1.0, 0.1, 0.0])
    ctx = np.full(4, 0.25)
    for eps in (0.0, 0.25, 0.5, 0.9):
        result = allocate(priority, ctx, rate=0.2, epsilon=eps)
        assert result.probs.min() >= (1.0 - eps) * 0.2 - 1e-9


def test_allocation_uniform_at_zero_leakage() -> None:
    result = allocate(np.array([9.0, 0.0]), np.full(2, 0.5), rate=0.3, epsilon=0.0)
    assert np.allclose(result.probs, 0.3)
    assert result.epsilon == pytest.approx(0.0)


def test_allocation_respects_budget_in_expectation() -> None:
    """Mass spent must equal the audit rate, for any leakage."""
    priority = np.array([3.0, 2.0, 1.0, 0.0])
    ctx = np.array([0.4, 0.3, 0.2, 0.1])
    for eps in (0.0, 0.5, 1.0):
        result = allocate(priority, ctx, rate=0.25, epsilon=eps)
        assert float(np.sum(ctx * result.probs)) == pytest.approx(0.25, abs=1e-6)


def test_allocation_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        allocate(np.array([1.0]), np.array([1.0]), rate=0.5, epsilon=1.5)
    with pytest.raises(ValueError):
        allocate(np.array([1.0]), np.array([1.0]), rate=0.0, epsilon=0.5)


# --- Adversary --------------------------------------------------------------


def test_best_response_respects_detectability_constraint() -> None:
    """The adversary must keep its audited rate at or below pi_0."""
    ctx = np.full(5, 0.2)
    probs = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
    _, _, q_audit = best_response(probs, ctx, pi_0=0.1)
    assert q_audit <= 0.1 + 1e-9


def test_best_response_concentrates_on_thin_audit_contexts() -> None:
    """Defection should load onto the least-audited contexts."""
    ctx = np.full(4, 0.25)
    probs = np.array([0.01, 0.1, 0.5, 0.9])
    q, _, _ = best_response(probs, ctx, pi_0=0.1)
    assert q[0] >= q[1] >= q[2] >= q[3]


def test_uniform_allocation_caps_defection_at_null() -> None:
    """KEY: with zero leakage the adversary gains nothing from evasion --
    its deployment rate cannot exceed the verifier noise floor."""
    ctx = np.full(6, 1 / 6)
    probs = np.full(6, 0.2)  # uniform, so epsilon = 0
    _, q_deploy, _ = best_response(probs, ctx, pi_0=0.1)
    assert q_deploy == pytest.approx(0.1, abs=1e-6)


def test_scaling_law_exponents() -> None:
    """Lock in the measured scaling of B(p, eps, T).

        B  ~  sqrt(T) / ((1 - eps) * sqrt(p))

    Derivation: rounds-to-detect = audits/p with audits ~ K p^2/(q^2 pi_min^2),
    so C := p/pi_min^2 = 1/((1-eps)^2 p), and the finite-horizon optimum sits
    at sqrt(C T). Note the p exponent is -1/2, NOT -1: dividing audits by the
    arrival rate p cancels one of the two powers of p. Getting this wrong was
    the original error.
    """
    from monitor.policy import optimal_evasion_horizon

    def harm(p: float, eps: float, T: float) -> float:
        return optimal_evasion_horizon(
            pi_min=(1 - eps) * p, audit_rate=p, pi_0=0.05, horizon=T,
            beta=0.1, alpha=0.05, thin_fraction=0.9,
        ).harm

    def exponent(xs: list[float], ys: list[float]) -> float:
        slope, _ = np.polyfit(np.log(xs), np.log(ys), 1)
        return float(slope)

    horizons = [1e5, 1e6, 1e7, 1e8]
    assert exponent(horizons, [harm(0.1, 0.5, t) for t in horizons]) == pytest.approx(
        0.5, abs=0.03
    )

    inv_leak = [1.0, 2.0, 4.0, 8.0, 16.0]
    leak_harms = [harm(0.1, 1 - 1 / v, 1e6) for v in inv_leak]
    assert exponent(inv_leak, leak_harms) == pytest.approx(1.0, abs=0.03)

    rates = [0.01, 0.02, 0.05, 0.1, 0.2]
    rate_harms = [harm(p, 0.5, 1e6) for p in rates]
    assert exponent([1 / p for p in rates], rate_harms) == pytest.approx(0.5, abs=0.03)


def test_harm_unbounded_when_a_context_is_never_audited() -> None:
    """eps = 1 exactly (pi_min = 0) is the genuine cliff: free defection."""
    from monitor.policy import optimal_evasion_horizon

    ev = optimal_evasion_horizon(
        pi_min=0.0, audit_rate=0.1, pi_0=0.05, horizon=1e6, thin_fraction=0.9
    )
    assert not ev.detected_within_horizon
    assert ev.rounds_to_detect == float("inf")
    assert ev.harm == pytest.approx(0.9 * 1e6)


def test_harm_rate_increases_with_leakage() -> None:
    """Monotonicity: more leakage must never help the monitor."""
    priority = np.arange(8.0)[::-1]
    ctx = np.full(8, 0.125)
    rates = [
        undetectable_harm_rate(
            allocate(priority, ctx, rate=0.2, epsilon=eps).probs, ctx, pi_0=0.1
        )
        for eps in (0.0, 0.2, 0.4, 0.6, 0.8)
    ]
    assert all(b >= a - 1e-9 for a, b in zip(rates, rates[1:])), rates
