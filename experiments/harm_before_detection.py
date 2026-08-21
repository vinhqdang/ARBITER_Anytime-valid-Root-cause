"""Measure B(p, eps): cumulative harm before detection, as a function of
allocation leakage.

Supersedes the undetectable-harm-rate framing, which rested on a mis-stated
constraint. With any audit-probability floor above zero every defector is
eventually caught, so the quantity of interest is not whether detection
happens but how much harm accrues first -- and how fast that grows as the
monitor targets its audits more aggressively.

The adversary's trade-off: hiding in thinly-audited contexts delays detection
but reduces harm per round. It picks the point on that curve that maximises
the product.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from monitor.policy import allocate, optimal_evasion  # noqa: E402

N_CONTEXTS = 200
AUDIT_RATE = 0.1
PI_0 = 0.05
BETA = 0.1  # verifier false-negative rate
ALPHA = 0.05

LEAKAGE_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999)


def sweep(beta: float = BETA, label: str = "") -> list[tuple[float, float]]:
    ctx = np.full(N_CONTEXTS, 1.0 / N_CONTEXTS)
    priority = np.arange(N_CONTEXTS, dtype=float)[::-1]

    print(f"\n{label or 'sweep'}: n={N_CONTEXTS} p={AUDIT_RATE} "
          f"pi_0={PI_0} beta={beta} alpha={ALPHA}")
    print(f"{'eps':>7} {'pi_min':>8} {'k*':>5} {'q_deploy':>9} {'q_audit':>8} "
          f"{'flagrate':>9} {'rounds':>10} {'B(p,eps)':>10} {'B/B0':>7}")
    print("-" * 82)

    results: list[tuple[float, float]] = []
    baseline = None
    for eps in LEAKAGE_GRID:
        alloc = allocate(priority, ctx, rate=AUDIT_RATE, epsilon=eps)
        ev = optimal_evasion(
            alloc.probs, ctx, PI_0, beta=beta, alpha=ALPHA,
            audit_rate=AUDIT_RATE,
        )
        rounds = (ev.expected_audits_to_detect / AUDIT_RATE
                  if np.isfinite(ev.expected_audits_to_detect) else float("inf"))
        if baseline is None:
            baseline = ev.harm_before_detection
        ratio = ev.harm_before_detection / baseline if baseline else float("nan")
        print(f"{eps:>7.3f} {alloc.probs.min():>8.4f} {ev.n_thin_contexts:>5} "
              f"{ev.q_deploy:>9.4f} {ev.q_audit:>8.4f} {ev.flag_rate:>9.4f} "
              f"{rounds:>10.1f} {ev.harm_before_detection:>10.1f} {ratio:>7.2f}")
        results.append((eps, ev.harm_before_detection))
    return results


def fit_blowup(results: list[tuple[float, float]]) -> None:
    """Is B(p, eps) ~ C / (1 - eps)^gamma? Estimate gamma by log-log fit."""
    pts = [(e, b) for e, b in results if 0 < e < 1 and np.isfinite(b)]
    if len(pts) < 3:
        print("\nnot enough finite points to fit")
        return
    x = np.log([1.0 / (1.0 - e) for e, _ in pts])
    y = np.log([b for _, b in pts])
    gamma, logc = np.polyfit(x, y, 1)
    resid = y - (gamma * x + logc)
    print(f"\nlog-log fit: B ~ {np.exp(logc):.2f} / (1-eps)^{gamma:.3f}   "
          f"(max resid {np.abs(resid).max():.3f})")


if __name__ == "__main__":
    main_results = sweep(BETA, "verifier with 10% false negatives")
    fit_blowup(main_results)

    # A perfect verifier is the cleanest case for the theory.
    sweep(0.0, "perfect verifier (beta=0)")

    # A weak verifier should make evasion much cheaper.
    sweep(0.5, "weak verifier (beta=0.5)")
