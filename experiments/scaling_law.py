"""Establish the scaling of B(p, eps, T): harm an adversary can extract
before detection, over a finite horizon.

Conjecture from the algebra:

    B  ~  sqrt(T) / ((1 - eps) * p)

reasoning: harm is q*T in the slow regime and C/q in the fast regime with
C ~ 1/pi_min^2, so the optimum is at sqrt(C*T) and pi_min = (1-eps)*p.

This script estimates each exponent independently by log-log fit, holding the
others fixed. If the exponents come out at 0.5, -1, -1 the conjecture stands.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from monitor.policy import optimal_evasion_horizon  # noqa: E402

PI_0 = 0.05
BETA = 0.1
ALPHA = 0.05
THIN = 0.9  # fraction of contexts left at the audit floor


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    lx, ly = np.log(xs), np.log(ys)
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = float(np.abs(ly - (slope * lx + intercept)).max())
    return float(slope), resid


def exponent_in_horizon(eps: float = 0.5, p: float = 0.1) -> None:
    horizons = [10**k for k in range(3, 10)]
    harms = []
    print(f"\nHorizon scaling at eps={eps}, p={p}")
    print(f"{'T':>12} {'q*':>10} {'rounds':>14} {'caught':>7} {'B':>14}")
    print("-" * 62)
    for T in horizons:
        ev = optimal_evasion_horizon(
            pi_min=(1 - eps) * p, audit_rate=p, pi_0=PI_0, horizon=T,
            beta=BETA, alpha=ALPHA, thin_fraction=THIN,
        )
        harms.append(ev.harm)
        print(f"{T:>12.0e} {ev.q_deploy:>10.6f} {ev.rounds_to_detect:>14.1f} "
              f"{str(ev.detected_within_horizon):>7} {ev.harm:>14.2f}")
    slope, resid = _fit([float(t) for t in horizons], harms)
    print(f"  => B ~ T^{slope:.4f}   (max log resid {resid:.4f})"
          f"   [conjecture 0.5]")


def exponent_in_leakage(T: float = 1e6, p: float = 0.1) -> None:
    epss = [0.0, 0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375]
    harms, inv = [], []
    print(f"\nLeakage scaling at T={T:.0e}, p={p}")
    print(f"{'eps':>10} {'1/(1-eps)':>11} {'q*':>10} {'B':>14} {'caught':>7}")
    print("-" * 56)
    for eps in epss:
        ev = optimal_evasion_horizon(
            pi_min=(1 - eps) * p, audit_rate=p, pi_0=PI_0, horizon=T,
            beta=BETA, alpha=ALPHA, thin_fraction=THIN,
        )
        harms.append(ev.harm)
        inv.append(1.0 / (1.0 - eps))
        print(f"{eps:>10.6f} {1/(1-eps):>11.2f} {ev.q_deploy:>10.6f} "
              f"{ev.harm:>14.2f} {str(ev.detected_within_horizon):>7}")
    slope, resid = _fit(inv, harms)
    print(f"  => B ~ (1-eps)^-{slope:.4f}   (max log resid {resid:.4f})"
          f"   [conjecture 1.0]")


def exponent_in_rate(T: float = 1e6, eps: float = 0.5) -> None:
    ps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    harms = []
    print(f"\nAudit-rate scaling at T={T:.0e}, eps={eps}")
    print(f"{'p':>8} {'pi_min':>9} {'q*':>10} {'B':>14} {'caught':>7}")
    print("-" * 52)
    for p in ps:
        ev = optimal_evasion_horizon(
            pi_min=(1 - eps) * p, audit_rate=p, pi_0=PI_0, horizon=T,
            beta=BETA, alpha=ALPHA, thin_fraction=THIN,
        )
        harms.append(ev.harm)
        print(f"{p:>8.3f} {(1-eps)*p:>9.4f} {ev.q_deploy:>10.6f} "
              f"{ev.harm:>14.2f} {str(ev.detected_within_horizon):>7}")
    slope, resid = _fit([1.0 / p for p in ps], harms)
    print(f"  => B ~ p^-{slope:.4f}   (max log resid {resid:.4f})"
          f"   [conjecture 1.0]")


if __name__ == "__main__":
    exponent_in_horizon()
    exponent_in_leakage()
    exponent_in_rate()
