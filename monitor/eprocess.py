"""Anytime-valid e-processes for per-agent compliance testing.

The null for agent i is H_0: P(verifier flags) <= pi_0, where pi_0 is the
verifier's false-flag rate on a compliant agent. Wealth is the betting
martingale

    W <- W * (1 + lam * (X - pi_0)),    X in {0, 1},  lam in [0, 1/pi_0)

which is a non-negative supermartingale under H_0, so Ville's inequality
gives P(exists t : W_t >= 1/alpha) <= alpha -- valid at every stopping time,
with no pre-specified sample size.

The upper limit on lam is 1/pi_0, not 1/(1-pi_0): the binding case is X = 0,
where the factor is 1 - lam*pi_0 and must stay positive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Below this, treat a rate as exactly the boundary to avoid divide-by-zero
# in the GRO wager and in KL.
EPS = 1e-12


def gro_wager(q: float, pi_0: float) -> float:
    """Growth-rate-optimal wager against a known alternative rate q.

    Maximising q*log(1 + lam*(1-pi_0)) + (1-q)*log(1 - lam*pi_0) gives

        lam* = (q - pi_0) / (pi_0 * (1 - pi_0))

    at which 1 + lam*(1-pi_0) = q/pi_0 and 1 - lam*pi_0 = (1-q)/(1-pi_0),
    so the per-audit log-growth equals KL(Bern(q) || Bern(pi_0)) exactly.

    Returns 0 when q <= pi_0: no positive-growth bet exists, which is
    precisely the regime an evasive adversary tries to sit in.
    """
    if q <= pi_0:
        return 0.0
    return (q - pi_0) / (pi_0 * (1.0 - pi_0))


def kl_bernoulli(q: float, p: float) -> float:
    """KL(Bern(q) || Bern(p)) in nats -- the per-audit evidence growth rate."""
    q = min(max(q, EPS), 1.0 - EPS)
    p = min(max(p, EPS), 1.0 - EPS)
    return q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))


def minimax_wager(pi_0: float, q_grid: tuple[float, ...]) -> float:
    """Wager maximising the worst-case growth over a set of alternatives.

    The monitor does not know q, and an adversary picks q knowing the wager.
    Betting GRO against a guessed q is fragile: if the adversary sits just
    below that q, growth goes negative. This maximises the minimum growth
    over q_grid instead, which is the honest objective when the alternative
    is chosen adversarially.
    """
    candidates = [q for q in q_grid if q > pi_0]
    if not candidates:
        return 0.0

    best_lam, best_worst = 0.0, -math.inf
    # Wagers are bounded by 1/pi_0; scan the interior on a fixed grid.
    upper = 1.0 / pi_0
    for step in range(1, 200):
        lam = upper * step / 200.0
        worst = min(_log_growth(lam, q, pi_0) for q in candidates)
        if worst > best_worst:
            best_lam, best_worst = lam, worst
    return best_lam if best_worst > 0 else 0.0


def _log_growth(lam: float, q: float, pi_0: float) -> float:
    up = 1.0 + lam * (1.0 - pi_0)
    down = 1.0 - lam * pi_0
    if up <= 0 or down <= 0:
        return -math.inf
    return q * math.log(up) + (1.0 - q) * math.log(down)


@dataclass
class EProcess:
    """Per-agent betting martingale, tracked in log space.

    Log space matters: an evasive adversary drives wealth toward zero over
    thousands of audits, which underflows a float product long before the
    experiment ends.
    """

    pi_0: float
    alpha: float = 0.05
    log_wealth: float = 0.0
    n_audits: int = 0
    n_flags: int = 0
    # Running estimate of the audited flag rate, used by the plug-in wager.
    _flag_rate_prior: float = 0.5

    def update(self, flagged: bool, wager: float) -> None:
        """Apply one audit outcome."""
        if not 0.0 <= wager < 1.0 / self.pi_0:
            raise ValueError(
                f"wager {wager} outside [0, 1/pi_0) = [0, {1.0 / self.pi_0})"
            )
        self.n_audits += 1
        self.n_flags += int(flagged)
        factor = 1.0 + wager * ((1.0 if flagged else 0.0) - self.pi_0)
        self.log_wealth += math.log(factor)

    @property
    def wealth(self) -> float:
        """e-value. May underflow to 0.0 for a heavily-drained process."""
        return math.exp(self.log_wealth) if self.log_wealth > -700 else 0.0

    @property
    def rejected(self) -> bool:
        return self.log_wealth >= math.log(1.0 / self.alpha)

    def plug_in_wager(self) -> float:
        """GRO wager against the empirical audited flag rate.

        Laplace-smoothed so the first few audits do not produce a wild bet.
        This is what a monitor can actually do: it never observes q.
        """
        q_hat = (self.n_flags + self._flag_rate_prior) / (
            self.n_audits + 2.0 * self._flag_rate_prior
        )
        return gro_wager(q_hat, self.pi_0)


def stopped_e_bh(e_values: list[float], alpha: float) -> list[int]:
    """e-BH on e-values; indices of rejected agents.

    Valid at arbitrary stopping times under arbitrary dependence between the
    agents' e-processes, which is what we need -- agent e-processes are
    dependent by construction, since they share the audit budget.

    Sort descending, find the largest k with e_(k) >= n / (alpha * k), and
    reject the top k.
    """
    n = len(e_values)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: e_values[i], reverse=True)
    k_best = 0
    for k in range(1, n + 1):
        threshold = n / (alpha * k)
        if e_values[order[k - 1]] >= threshold:
            k_best = k
    return order[:k_best]
