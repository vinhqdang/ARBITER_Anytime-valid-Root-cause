"""Test the conjectured power-collapse threshold.

CONJECTURE. Under leakage-constrained allocation with audit rate p and
verifier noise pi_0, the maximum deployment defection rate an adversary can
sustain while remaining permanently undetected is

    qbar(eps) = pi_0 / (1 - eps)

up to a saturation point where the thinly-audited contexts run out.

Sketch of why. With n equiprobable contexts, the targeted component
water-fills probability 1 onto the top p*n contexts, so

    pi(o) = (1-eps)p + eps   on the top p*n contexts
    pi(o) = (1-eps)p         on the remaining (1-p)n

The adversary's detectability budget is pi_0 * p (the audited mass times the
noise floor). Filling m of the thin contexts costs m/n * (1-eps)p, so
m/n = pi_0/(1-eps) -- i.e. qbar = pi_0/(1-eps), valid while that stays below
the thin fraction (1-p).

This script checks the closed form against the linear program actually solved
in monitor.policy, and checks the sequential consequence: adversaries above
the threshold get caught, adversaries at it never do.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from monitor.eprocess import EProcess, kl_bernoulli  # noqa: E402
from monitor.policy import allocate, best_response  # noqa: E402

N_CONTEXTS = 200
AUDIT_RATE = 0.1
PI_0 = 0.05
ALPHA = 0.05


def sweep_threshold() -> None:
    """Compare the LP solution to the closed form across leakage levels."""
    ctx = np.full(N_CONTEXTS, 1.0 / N_CONTEXTS)
    priority = np.arange(N_CONTEXTS, dtype=float)[::-1]

    print(f"n_contexts={N_CONTEXTS}  p={AUDIT_RATE}  pi_0={PI_0}")
    print(f"predicted saturation at qbar = 1-p = {1 - AUDIT_RATE}")
    print()
    print(f"{'eps':>6} {'pi_min':>8} {'qbar(LP)':>10} {'pi_0/(1-eps)':>13} "
          f"{'ratio':>7} {'q_audit':>8}")
    print("-" * 60)

    for eps in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
        alloc = allocate(priority, ctx, rate=AUDIT_RATE, epsilon=eps)
        _, q_deploy, q_audit = best_response(alloc.probs, ctx, PI_0)
        closed = PI_0 / (1.0 - eps) if eps < 1 else float("inf")
        ratio = q_deploy / closed if closed else float("nan")
        print(f"{eps:>6.2f} {alloc.probs.min():>8.4f} {q_deploy:>10.4f} "
              f"{closed:>13.4f} {ratio:>7.3f} {q_audit:>8.4f}")


def sweep_saturation() -> None:
    """Push leakage high enough to exhaust the thinly-audited contexts."""
    ctx = np.full(N_CONTEXTS, 1.0 / N_CONTEXTS)
    priority = np.arange(N_CONTEXTS, dtype=float)[::-1]
    print()
    print("Saturation regime (high leakage):")
    print(f"{'eps':>6} {'qbar(LP)':>10} {'pi_0/(1-eps)':>13} {'capped?':>9}")
    print("-" * 44)
    for eps in (0.9, 0.95, 0.98, 0.99, 0.995, 0.999):
        alloc = allocate(priority, ctx, rate=AUDIT_RATE, epsilon=eps)
        _, q_deploy, _ = best_response(alloc.probs, ctx, PI_0)
        closed = PI_0 / (1.0 - eps)
        print(f"{eps:>6.3f} {q_deploy:>10.4f} {closed:>13.4f} "
              f"{'YES' if q_deploy < closed * 0.99 else 'no':>9}")


def sequential_consequence() -> None:
    """The behavioural claim: at the threshold, never detected; above it,
    detected in roughly log(1/alpha)/KL audits."""
    rng = np.random.default_rng(7)
    ctx = np.full(N_CONTEXTS, 1.0 / N_CONTEXTS)
    priority = np.arange(N_CONTEXTS, dtype=float)[::-1]
    eps = 0.5
    alloc = allocate(priority, ctx, rate=AUDIT_RATE, epsilon=eps)
    _, qbar, _ = best_response(alloc.probs, ctx, PI_0)

    print()
    print(f"Sequential behaviour at eps={eps}, qbar={qbar:.4f}")
    print(f"{'q_deploy':>10} {'q_audit':>9} {'detected':>9} {'audits':>8} "
          f"{'predicted':>10}")
    print("-" * 50)

    for mult in (1.0, 1.5, 3.0, 6.0):
        q_target = qbar * mult
        if q_target >= 1.0:
            continue
        # Adversary defects on the thinnest contexts up to q_target mass.
        order = np.argsort(alloc.probs)
        q = np.zeros(N_CONTEXTS)
        filled = 0.0
        for idx in order:
            if filled >= q_target:
                break
            q[idx] = 1.0
            filled += ctx[idx]
        audit_mass = float(np.sum(ctx * alloc.probs))
        q_audit = float(np.sum(ctx * alloc.probs * q) / audit_mass)

        ep = EProcess(pi_0=PI_0, alpha=ALPHA)
        detected_at = None
        for step in range(200_000):
            # Sample an audited context, then whether it was a defection.
            idx = rng.choice(N_CONTEXTS, p=ctx * alloc.probs / audit_mass)
            defecting = q[idx] > 0.5
            flagged = defecting or (rng.random() < PI_0)
            ep.update(flagged=flagged, wager=ep.plug_in_wager())
            if ep.rejected:
                detected_at = step + 1
                break

        pred = (np.log(1 / ALPHA) / kl_bernoulli(q_audit, PI_0)
                if q_audit > PI_0 else float("inf"))
        print(f"{filled:>10.4f} {q_audit:>9.4f} "
              f"{str(detected_at) if detected_at else 'never':>9} "
              f"{ep.n_audits:>8} {pred:>10.1f}")


if __name__ == "__main__":
    sweep_threshold()
    sweep_saturation()
    sequential_consequence()
