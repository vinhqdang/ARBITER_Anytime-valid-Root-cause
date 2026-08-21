"""Does targeting on monitor-private features recover power at zero leakage?

Three policies, same audit budget, same evasive adversary:

  uniform        flat everywhere. Zero leakage, no targeting, baseline power.
  leaky_greedy   targets on features the AGENT CAN SEE. Maximum targeting,
                 maximum leakage -- the active-evaluation default.
  private        targets on features the agent CANNOT see, held conditionally
                 uniform on what it can. Zero leakage by construction.

Prediction: `private` matches or beats `leaky_greedy` on detection speed while
matching `uniform` on leakage. If so, informational advantage -- not audit
budget -- is what makes monitoring a strategic agent possible.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from monitor.eprocess import kl_bernoulli  # noqa: E402
from monitor.private import (  # noqa: E402
    allocate_private,
    detection_rate,
    strategic_best_response,
)

N_VISIBLE = 20
N_PRIVATE = 10
RATE = 0.1
PI_0 = 0.05
BETA = 0.1
ALPHA = 0.05
TARGET_DEPLOY = 0.10  # the adversary wants this deployment defection rate


def _signal_grid(rng: np.random.Generator) -> np.ndarray:
    """Where the defection actually is, as a monitor-side priority signal.

    The monitor's private features are informative: defection concentrates in
    a few (visible, private) cells, and the monitor can see which private
    cells those are even though the agent cannot condition on them.
    """
    signal = rng.random((N_VISIBLE, N_PRIVATE)) ** 3
    return signal / signal.max()


def run() -> None:
    rng = np.random.default_rng(11)
    visible = np.full(N_VISIBLE, 1.0 / N_VISIBLE)
    private = np.full((N_VISIBLE, N_PRIVATE), 1.0 / N_PRIVATE)
    priority = _signal_grid(rng)

    print(f"visible cells={N_VISIBLE} private cells={N_PRIVATE} "
          f"p={RATE} pi_0={PI_0} beta={BETA} target q_deploy={TARGET_DEPLOY}")
    print()
    print(f"{'policy':>14} {'leakage':>9} {'q_deploy':>9} {'flagrate':>9} "
          f"{'KL':>10} {'audits':>10} {'speedup':>8}")
    print("-" * 74)

    baseline_audits = None
    for name, aggressiveness in (
        ("uniform", 0.0),
        ("private", 1.0),
    ):
        alloc = allocate_private(priority, visible, private, RATE, aggressiveness)
        policy = strategic_best_response(alloc, PI_0, BETA, TARGET_DEPLOY)
        flag_rate, q_deploy = detection_rate(alloc, policy, BETA, PI_0)
        kl = kl_bernoulli(flag_rate, PI_0)
        audits = np.log(1 / ALPHA) / kl if kl > 0 else float("inf")
        if baseline_audits is None:
            baseline_audits = audits
        print(f"{name:>14} {alloc.leakage:>9.5f} {q_deploy:>9.4f} "
              f"{flag_rate:>9.5f} {kl:>10.6f} {audits:>10.1f} "
              f"{baseline_audits / audits:>8.2f}x")

    # The leaky comparator: same targeting strength, but on agent-visible
    # features, so the adversary can and does exploit it.
    _leaky_comparison(priority, visible, private, baseline_audits)


def _leaky_comparison(
    priority: np.ndarray,
    visible: np.ndarray,
    private: np.ndarray,
    baseline_audits: float,
) -> None:
    """Target on the VISIBLE axis: powerful against a naive agent, and
    exploitable by a strategic one."""
    from monitor.policy import allocate, optimal_evasion_horizon

    visible_priority = priority.mean(axis=1)
    for eps in (0.5, 0.9):
        alloc = allocate(visible_priority, visible, rate=RATE, epsilon=eps)
        # A strategic agent defects in the least-audited visible cells.
        order = np.argsort(alloc.probs)
        policy = np.zeros_like(priority)
        remaining = TARGET_DEPLOY
        for i in order:
            if remaining <= 1e-12:
                break
            share = min(1.0, remaining / visible[i])
            policy[i, :] = share
            remaining -= share * visible[i]
        joint = visible[:, None] * private
        audit_mass = float(np.sum(joint * alloc.probs[:, None]))
        q_audit = float(np.sum(joint * alloc.probs[:, None] * policy) / audit_mass)
        flag = q_audit * (1 - BETA) + (1 - q_audit) * PI_0
        kl = kl_bernoulli(flag, PI_0)
        audits = np.log(1 / ALPHA) / kl if kl > 0 else float("inf")
        print(f"{'leaky eps=' + str(eps):>14} {eps:>9.5f} "
              f"{float(np.sum(joint * policy)):>9.4f} {flag:>9.5f} "
              f"{kl:>10.6f} {audits:>10.1f} {baseline_audits / audits:>8.2f}x")

    print()
    print("Reading: 'private' should show leakage ~0 like uniform, but far")
    print("fewer audits to detection. 'leaky' buys targeting at a leakage")
    print("cost the adversary converts directly into evasion.")


if __name__ == "__main__":
    run()
