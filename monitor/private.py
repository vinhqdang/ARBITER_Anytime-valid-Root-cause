"""Audit allocation that targets on monitor-private information.

The result that motivates this module: under a worst-case evasive adversary,
harm before detection scales as B ~ sqrt(T) / ((1-eps) sqrt(p)), monotone in
the leakage eps. So a minimax monitor should set eps = 0 -- uniform random
auditing -- and every bit of targeting is actively harmful. That contradicts
the active-evaluation literature, but as an algorithm "audit uniformly" is
not much of one.

The way out is that leakage happens only through features the agent can
observe. Factor the context as

    o = (o_a, m)      o_a observable to the agent,  m private to the monitor

and require the policy to be conditionally uniform in what the agent sees:

    E_m[ pi(audit | o_a, m) ] = p        for every o_a                  (*)

Under (*) the agent's inferable audit probability is p everywhere, so it has
nothing to exploit and eps = 0 exactly -- while the monitor is free to target
hard along m. Statistical power is therefore governed by the monitor's
INFORMATIONAL ADVANTAGE over the agent, not by its audit budget.

Realistic sources of m in a multiagent LLM deployment: cross-agent
correlations the agent cannot see, the monitor's own e-process state, and
system history the agent does not carry between episodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrivateAllocation:
    """Audit probabilities over a grid of (agent-visible, private) cells.

    ``probs[i, j]`` is pi(audit | o_a = i, m = j).
    """

    probs: np.ndarray  # shape (n_visible, n_private)
    visible_probs: np.ndarray  # P(o_a = i)
    private_probs: np.ndarray  # P(m = j | o_a), shape (n_visible, n_private)
    rate: float

    @property
    def agent_inferable(self) -> np.ndarray:
        """What the agent can compute: E_m[pi | o_a], one value per o_a.

        Constraint (*) makes this constant at `rate`, which is exactly why
        the agent cannot target its defection.
        """
        return np.sum(self.private_probs * self.probs, axis=1)

    @property
    def leakage(self) -> float:
        """eps as the agent experiences it -- based on what it can infer.

        Note this uses `agent_inferable`, NOT the raw per-cell minimum: the
        agent cannot condition on m, so per-cell variation is invisible to it.
        """
        inferable = self.agent_inferable
        return 1.0 - float(inferable.min()) / self.rate

    def check(self) -> None:
        if np.any(self.probs < -1e-9) or np.any(self.probs > 1.0 + 1e-9):
            raise ValueError("audit probabilities outside [0, 1]")
        spent = float(
            np.sum(
                self.visible_probs[:, None] * self.private_probs * self.probs
            )
        )
        if abs(spent - self.rate) > 1e-6:
            raise ValueError(f"budget {spent:.6f} != rate {self.rate:.6f}")


def allocate_private(
    priority: np.ndarray,
    visible_probs: np.ndarray,
    private_probs: np.ndarray,
    rate: float,
    aggressiveness: float = 1.0,
) -> PrivateAllocation:
    """Target within each agent-visible cell, uniformly across cells.

    ``priority[i, j]`` is the monitor's value of auditing cell (i, j) --
    typically expected e-value growth. ``aggressiveness`` in [0, 1] scales how
    hard the monitor targets *within* a visible cell; it does NOT create
    leakage, because the budget for each visible cell is held at `rate`
    regardless.

    This is the algorithm: informative where the agent is blind, flat where it
    can see.
    """
    if not 0.0 <= aggressiveness <= 1.0:
        raise ValueError(f"aggressiveness must be in [0, 1], got {aggressiveness}")
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"rate must be in (0, 1], got {rate}")
    if priority.shape != private_probs.shape:
        raise ValueError("priority and private_probs shape mismatch")

    n_visible, _ = priority.shape
    probs = np.zeros_like(priority, dtype=float)

    for i in range(n_visible):
        # Each visible cell gets exactly `rate` of audit mass, so nothing the
        # agent can condition on varies. Within the cell, water-fill by
        # priority up to the aggressiveness weight.
        targeted = _waterfill(priority[i], private_probs[i], rate)
        uniform = np.full(priority.shape[1], rate)
        probs[i] = (1.0 - aggressiveness) * uniform + aggressiveness * targeted

    alloc = PrivateAllocation(
        probs=probs,
        visible_probs=visible_probs,
        private_probs=private_probs,
        rate=rate,
    )
    alloc.check()
    return alloc


def _waterfill(priority: np.ndarray, weights: np.ndarray, rate: float) -> np.ndarray:
    """Spend `rate` of conditional mass on the highest-priority cells."""
    probs = np.zeros(len(priority))
    remaining = rate
    for idx in np.argsort(priority)[::-1]:
        if remaining <= 1e-12:
            break
        take = min(1.0, remaining / max(weights[idx], 1e-12))
        probs[idx] = take
        remaining -= take * weights[idx]
    return probs


def detection_rate(
    alloc: PrivateAllocation,
    defect_cells: np.ndarray,
    beta: float = 0.0,
    pi_0: float = 0.05,
) -> tuple[float, float]:
    """Audited flag rate and deployment defection rate for a given adversary.

    ``defect_cells[i, j]`` in [0, 1] is the agent's defection probability. A
    *strategic* agent cannot condition on j, so its policy must be constant
    along the private axis -- which is what makes targeting on m effective.

    Returns (audited flag rate, deployment defection rate).
    """
    joint = alloc.visible_probs[:, None] * alloc.private_probs
    audit_mass = float(np.sum(joint * alloc.probs))
    if audit_mass <= 0:
        return pi_0, float(np.sum(joint * defect_cells))

    q_audit = float(np.sum(joint * alloc.probs * defect_cells) / audit_mass)
    q_deploy = float(np.sum(joint * defect_cells))
    flag_rate = q_audit * (1.0 - beta) + (1.0 - q_audit) * pi_0
    return flag_rate, q_deploy


def strategic_best_response(
    alloc: PrivateAllocation,
    pi_0: float,
    beta: float = 0.0,
    target_deploy: float = 0.1,
) -> np.ndarray:
    """The agent's best evasive policy, constrained to what it can observe.

    Because the agent cannot condition on m, its policy is constant along the
    private axis. It ranks visible cells by inferable audit probability and
    defects in the thinnest ones -- but under constraint (*) those are all
    equal, so it has no preference and must spread. That is the mechanism by
    which private targeting recovers power at zero leakage.
    """
    n_visible, n_private = alloc.probs.shape
    inferable = alloc.agent_inferable

    policy = np.zeros((n_visible, n_private))
    remaining = target_deploy
    for i in np.argsort(inferable):
        if remaining <= 1e-12:
            break
        share = min(1.0, remaining / max(alloc.visible_probs[i], 1e-12))
        policy[i, :] = share  # constant along the private axis
        remaining -= share * alloc.visible_probs[i]
    return policy
