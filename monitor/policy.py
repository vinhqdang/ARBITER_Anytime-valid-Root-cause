"""Audit allocation under a leakage constraint, and the adversary's best
response to it.

The central object is the tension the allocation policy creates. A monitor
that targets audits informatively gains statistical power, but the targeting
is a function of contexts the audited agent also observes -- so the agent can
compute its own audit probability and defect where that probability is
lowest. Power and unpredictability are in direct conflict.

Leakage is defined by the floor, because the adversary only cares about the
thinnest-audited context:

    pi_min := min_o pi(audit | o),    epsilon := 1 - pi_min / p

epsilon = 0 is uniform allocation (nothing to infer); epsilon = 1 leaves some
context never audited (free evasion). The constraint pi(.) >= (1-eps)*p is
met exactly by mixtures of uniform and targeted allocation with weight
(1-eps) on uniform, which is what `allocate` builds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AllocationResult:
    """An audit policy over contexts, with its realised leakage."""

    probs: np.ndarray  # pi(audit | o) for each context
    epsilon: float  # realised leakage, 1 - min(probs)/rate
    rate: float  # marginal audit rate p

    def check(self) -> None:
        """Assert the policy is a valid probability allocation."""
        if np.any(self.probs < -1e-9) or np.any(self.probs > 1.0 + 1e-9):
            raise ValueError("audit probabilities outside [0, 1]")


def allocate(
    priority: np.ndarray,
    context_probs: np.ndarray,
    rate: float,
    epsilon: float,
) -> AllocationResult:
    """Leakage-constrained allocation.

    ``priority`` is the monitor's per-context value of auditing -- typically
    expected e-value growth. ``epsilon`` is the leakage budget: 0 forces
    uniform, 1 allows fully greedy targeting.

    The policy is (1-eps) * uniform + eps * targeted, where the targeted part
    is proportional to priority. Mixing is what makes the floor hold by
    construction rather than by projection.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"rate must be in (0, 1], got {rate}")

    n = len(priority)
    if len(context_probs) != n:
        raise ValueError("priority and context_probs length mismatch")

    uniform = np.full(n, rate)

    # Targeted component: spend the whole budget on the highest-priority
    # contexts, respecting per-context probability <= 1 and total budget.
    targeted = _greedy_targeted(priority, context_probs, rate)

    probs = (1.0 - epsilon) * uniform + epsilon * targeted
    realised = 1.0 - float(probs.min()) / rate
    result = AllocationResult(probs=probs, epsilon=realised, rate=rate)
    result.check()
    return result


def _greedy_targeted(
    priority: np.ndarray, context_probs: np.ndarray, rate: float
) -> np.ndarray:
    """Water-fill the audit budget onto the highest-priority contexts.

    Budget is a *mass* constraint: sum_o P(o) * pi(o) = rate.
    """
    n = len(priority)
    probs = np.zeros(n)
    remaining = rate
    for idx in np.argsort(priority)[::-1]:
        if remaining <= 1e-12:
            break
        # Filling context idx to probability 1 costs context_probs[idx] mass.
        take = min(1.0, remaining / max(context_probs[idx], 1e-12))
        probs[idx] = take
        remaining -= take * context_probs[idx]
    return probs


def best_response(
    audit_probs: np.ndarray,
    context_probs: np.ndarray,
    pi_0: float,
    max_defect: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """The adversary's optimal evasive defection policy.

    Maximise deployment defection rate

        q_deploy = sum_o P(o) q(o)

    subject to keeping the *audited* flag rate at or below the verifier's
    noise floor, which is exactly the condition for the e-process to have no
    positive-growth wager available:

        q_audit = sum_o P(o) pi(o) q(o) / sum_o P(o) pi(o)  <=  pi_0

    This is a fractional knapsack: fill q(o) = max_defect on the contexts
    with the lowest audit probability first, since they cost least against
    the constraint per unit of harm gained.

    Returns (q per context, q_deploy, q_audit).
    """
    n = len(audit_probs)
    audit_mass = float(np.sum(context_probs * audit_probs))
    if audit_mass <= 0:
        # Nothing is audited anywhere: defect everywhere, never detected.
        return np.full(n, max_defect), max_defect, 0.0

    budget = pi_0 * audit_mass  # allowed sum_o P(o) pi(o) q(o)
    q = np.zeros(n)
    remaining = budget

    # Cheapest first: lowest audit probability buys the most deployment
    # defection per unit of the detectability budget.
    for idx in np.argsort(audit_probs):
        if remaining <= 1e-15:
            break
        cost_per_unit = context_probs[idx] * audit_probs[idx]
        if cost_per_unit <= 1e-15:
            # Never audited: free defection, no constraint cost.
            q[idx] = max_defect
            continue
        affordable = remaining / cost_per_unit
        q[idx] = min(max_defect, affordable)
        remaining -= q[idx] * cost_per_unit

    q_deploy = float(np.sum(context_probs * q))
    q_audit = float(np.sum(context_probs * audit_probs * q) / audit_mass)
    return q, q_deploy, q_audit


def undetectable_harm_rate(
    audit_probs: np.ndarray,
    context_probs: np.ndarray,
    pi_0: float,
    harm_per_defection: float = 1.0,
) -> float:
    """DEPRECATED -- kept only so the earlier result is reproducible.

    This computes the adversary's deployment rate under the constraint
    q_audit <= pi_0, which was the WRONG constraint: it treats the defection
    rate as the thing that must hide under the noise floor, when in fact
    defection is itself the signal the verifier flags. The audited FLAG rate
    is q_audit*(1-beta) + (1-q_audit)*pi_0, which exceeds pi_0 for any
    q_audit > 0 unless the verifier is useless (beta >= 1 - pi_0).

    Consequence: with any audit-probability floor above zero, there is no
    permanently-undetectable defection rate. Use `optimal_evasion` instead,
    which maximises harm accrued BEFORE detection.
    """
    _, q_deploy, _ = best_response(audit_probs, context_probs, pi_0)
    return harm_per_defection * q_deploy


def audited_flag_rate(q_audit: float, pi_0: float, beta: float) -> float:
    """Probability an audit flags, given the audited-round defection rate.

    A defection is caught with probability 1 - beta (beta = verifier false
    negative rate); a compliant round false-flags with probability pi_0.
    """
    return q_audit * (1.0 - beta) + (1.0 - q_audit) * pi_0


@dataclass(frozen=True)
class HorizonEvasion:
    """Optimal evasion against a finite deployment horizon.

    Without a horizon the adversary's problem is degenerate: harm before
    detection is q * rounds, rounds ~ 1/KL(pi_0 + delta || pi_0) ~ 1/delta^2,
    and delta ~ q, so the product diverges as q -> 0. An adversary defecting
    arbitrarily rarely survives arbitrarily long. Capping at a horizon T
    makes the problem well posed: defect fast and get caught early, or defect
    slowly and accrue q*T over the whole horizon. The optimum sits at the
    crossover.
    """

    q_deploy: float
    q_audit: float
    flag_rate: float
    rounds_to_detect: float
    detected_within_horizon: bool
    harm: float


def optimal_evasion_horizon(
    pi_min: float,
    audit_rate: float,
    pi_0: float,
    horizon: float,
    beta: float = 0.0,
    alpha: float = 0.05,
    harm_per_defection: float = 1.0,
    thin_fraction: float = 1.0,
    grid: int = 20_000,
) -> HorizonEvasion:
    """Maximise harm over a finite horizon, in closed parametric form.

    The adversary concentrates defection on the thinnest-audited contexts, so
    for a deployment rate q its audited rate is q * pi_min / audit_rate. That
    reduces the whole problem to a one-dimensional search over q, avoiding
    the context-discretisation artefact entirely.

    Expected scaling: in the slow regime harm = q*T (rising in q); in the fast
    regime harm = q * C/q^2 = C/q (falling in q). The maximum is at the
    crossover, giving harm ~ sqrt(C*T) with C ~ 1/pi_min^2 -- i.e.

        B ~ sqrt(T) / (pi_min * something)

    which is what the experiment checks.
    """

    if pi_min <= 0:
        # A never-audited context: defect there freely, forever.
        return HorizonEvasion(
            q_deploy=thin_fraction,
            q_audit=0.0,
            flag_rate=pi_0,
            rounds_to_detect=float("inf"),
            detected_within_horizon=False,
            harm=harm_per_defection * thin_fraction * horizon,
        )

    # Vectorised: the scalar loop was ~20k Python iterations per call, which
    # made the leakage-tuning sweep (a call per grid point per weight)
    # intractable.
    q = thin_fraction * np.arange(1, grid + 1, dtype=float) / grid
    q_audit = np.minimum(1.0, q * pi_min / audit_rate)
    flag_rate = q_audit * (1.0 - beta) + (1.0 - q_audit) * pi_0

    detectable = flag_rate > pi_0 + 1e-15
    rounds = np.full(grid, np.inf)
    if np.any(detectable):
        fr = np.clip(flag_rate[detectable], 1e-12, 1.0 - 1e-12)
        p0 = min(max(pi_0, 1e-12), 1.0 - 1e-12)
        kl = fr * np.log(fr / p0) + (1.0 - fr) * np.log((1.0 - fr) / (1.0 - p0))
        rounds[detectable] = math.log(1.0 / alpha) / kl / audit_rate

    harm = harm_per_defection * q * np.minimum(rounds, horizon)
    best_i = int(np.argmax(harm))

    return HorizonEvasion(
        q_deploy=float(q[best_i]),
        q_audit=float(q_audit[best_i]),
        flag_rate=float(flag_rate[best_i]),
        rounds_to_detect=float(rounds[best_i]),
        detected_within_horizon=bool(rounds[best_i] <= horizon),
        harm=float(harm[best_i]),
    )


@dataclass(frozen=True)
class EvasionOutcome:
    """The adversary's optimal evasive strategy against a given allocation."""

    n_thin_contexts: int  # how many of the thinnest contexts it defects in
    q_deploy: float  # deployment defection rate
    q_audit: float  # defection rate among audited rounds
    flag_rate: float  # audited flag rate
    expected_audits_to_detect: float
    harm_before_detection: float


def optimal_evasion(
    audit_probs: np.ndarray,
    context_probs: np.ndarray,
    pi_0: float,
    beta: float = 0.0,
    alpha: float = 0.05,
    harm_per_defection: float = 1.0,
    audit_rate: float | None = None,
) -> EvasionOutcome:
    """Maximise cumulative harm accrued before detection.

    The adversary faces a genuine trade-off. Defecting only in thinly-audited
    contexts keeps the audited flag rate near pi_0, so evidence accrues
    slowly and detection is far off -- but the deployment defection rate, and
    hence harm per round, is small. Defecting everywhere earns more per round
    and gets caught almost immediately.

    Total harm is (harm per round) x (rounds until detection), and the
    adversary picks the number of thin contexts to defect in so as to
    maximise that product. This is the quantity B(p, eps) in the theorem.
    """
    from .eprocess import kl_bernoulli  # local import avoids a cycle

    n = len(audit_probs)
    order = np.argsort(audit_probs)
    audit_mass = float(np.sum(context_probs * audit_probs))
    rate = audit_rate if audit_rate is not None else audit_mass

    best: EvasionOutcome | None = None
    cum_deploy = 0.0
    cum_audited = 0.0

    for k in range(1, n + 1):
        idx = order[k - 1]
        cum_deploy += float(context_probs[idx])
        cum_audited += float(context_probs[idx] * audit_probs[idx])

        q_audit = cum_audited / audit_mass if audit_mass > 0 else 0.0
        flag_rate = audited_flag_rate(q_audit, pi_0, beta)

        if flag_rate <= pi_0 + 1e-15:
            # No positive-growth wager exists: never detected.
            audits_needed = float("inf")
            harm = float("inf")
        else:
            growth = kl_bernoulli(flag_rate, pi_0)
            audits_needed = math.log(1.0 / alpha) / growth
            # Audits arrive at `rate` per agent-round, so rounds until
            # detection is audits_needed / rate.
            rounds = audits_needed / rate if rate > 0 else float("inf")
            harm = harm_per_defection * cum_deploy * rounds

        candidate = EvasionOutcome(
            n_thin_contexts=k,
            q_deploy=cum_deploy,
            q_audit=q_audit,
            flag_rate=flag_rate,
            expected_audits_to_detect=audits_needed,
            harm_before_detection=harm,
        )
        if best is None or candidate.harm_before_detection > best.harm_before_detection:
            best = candidate

    assert best is not None
    return best
