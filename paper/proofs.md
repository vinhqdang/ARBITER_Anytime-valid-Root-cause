# Proofs (working notes → to be typeset in `arbiter.tex`)

## Theorem 1 (Conditional uniformity is complete)

**Setup.** A context `o ∈ O` is drawn each round with `o = (o_a, m)`, where
`o_a` is observable to the agent under audit and `m` is private to the
monitor. The monitor's audit policy is `π(audit | o) ∈ [0,1]`, with marginal
rate `E[π] = p`. An agent is *strategic* if its action — in particular its
defection indicator `q(o) ∈ [0,1]` — is a measurable function of `o_a` alone:
it cannot condition on `m`, because it does not observe it.

Say `π` is **conditionally uniform** if `E_m[π(audit | o_a, m)] = p` for
(almost) every `o_a`.

**Claim.** For any strategic agent policy `q`, the audited defection rate
equals the deployment defection rate,

  `q_audit := E[q(o) · π(audit | o)] / p = E[q(o)] =: q_deploy`,

**if and only if** `π` is conditionally uniform. Consequently every
conditionally-uniform policy induces the same, unbiased audited sample —
none is more informative than uniform allocation — and any policy that is
*not* conditionally uniform admits a strategic `q` for which `q_audit <
q_deploy` strictly.

**Proof (sufficiency).** Write the audited rate as an iterated expectation
over `o_a` then `m`:

  `E[q(o)π(o)] = E_{o_a}[ q(o_a) · E_m[π(o_a, m) | o_a] ]`

using that `q` depends only on `o_a` and so is constant inside the inner
expectation. If `π` is conditionally uniform, `E_m[π | o_a] = p` for every
`o_a`, so the bracket equals `q(o_a) · p` and

  `E[q(o)π(o)] = p · E_{o_a}[q(o_a)] = p · q_deploy`.

Dividing by `p` gives `q_audit = q_deploy`. ∎

**Proof (necessity).** Suppose `π` is not conditionally uniform: there is a
set `A` of positive measure on which `f(o_a) := E_m[π | o_a]` differs from
`p`, WLOG `f(o_a) < p` on `A` (the case `f > p` is symmetric, using the
complement of `A`). A strategic agent that defects only on `A`, i.e.
`q = 1_A`, gets

  `q_audit = E[1_A(o_a) π(o)] / p = P(A) · E[f | A] / p < P(A) = q_deploy`,

since `E[f | A] < p` by construction. So `q_audit < q_deploy` strictly, and
the audit sample is biased downward relative to deployment — exactly the
mechanism a strategic agent exploits. ∎

**Corollary (monitor-private information is worthless).** Since the
*constraint* is only ever on `E_m[π | o_a]`, the actual values of
`π(o_a, m)` across `m` — however informatively they are chosen using
monitor-private signals — do not affect `q_audit` for ANY strategic `q`.
Every conditionally-uniform policy is equally powerful against a strategic
adversary, and uniform allocation (`π ≡ p`) already attains the class
optimum. This matches `test_conditional_uniformity_gives_unbiased_audit` and
`test_violating_conditional_uniformity_is_exploitable` in
`tests/test_monitor.py`, and is the result that killed the
monitor-private-targeting design in `monitor/private.py`.

*Remark.* This is the honest reading of "leakage": `ε := 1 - min_{o_a}
f(o_a)/p` is a summary of how far `π` is from conditional uniformity, not an
independent design freedom. Theorem 2's scaling law is the price of moving
`ε` away from 0.

---

## Proposition 1 (Vacuity, restated precisely)

Let agent `i`'s e-process be `W_t = ∏_{s≤t} (1 + λ_s(X_s - π_0))` with the
audited outcome `X_s ∈ {0,1}`. If the agent complies whenever audited
(`X_s = 0` for all audited `s`), then for any wager sequence `λ_s ∈ [0,
1/π_0)` measurable w.r.t. the history,

  `W_t = ∏_{s≤t} (1 - λ_s π_0) → 0` a.s. as `t → ∞`,

provided `Σ_s λ_s = ∞` (satisfied by GRO wagers against any hypothesised
`q > π_0`, and by the plug-in wager once enough clean audits have driven the
empirical rate below `π_0`). So `P(reject) → 0`: the test is valid
(trivially, since `W_t ≥ 1/α` never happens) and **powerless**. This is
`test_eprocess_never_rejects_evasive_agent`.

---

## Theorem 2 (Scaling law — status: DERIVED and numerically confirmed)

**Setup.** Adversary defects on a mass-`q` set of the thinnest-audited
contexts. Under `π_min = (1-ε)p`, the audited defection rate is
`q_audit = q·(1-ε)` (Section on leakage above), and the audited flag rate is

  `flag(q) = q_audit(1-β) + (1-q_audit)π_0 = π_0 + κ(1-ε)q`,  where `κ := 1-β-π_0`.

**Step 1 (small-deviation KL).** Writing `δ := flag(q) - π_0 = κ(1-ε)q`, the
standard second-order expansion of the binary KL divergence around `π_0`
gives, for small `δ`,

  `KL(π_0+δ ‖ π_0) ≈ δ² / (2π_0(1-π_0))`.

**Step 2 (rounds to detection).** The e-process needs
`audits ≈ log(1/α)/KL` audited samples to reject, and audits arrive at rate
`p` per round, so

  `rounds(q) ≈ audits/p = 2π_0(1-π_0)log(1/α) / (κ²(1-ε)²p·q²) =: K/q²`,

  `K := 2π_0(1-π_0)log(1/α) / (κ²(1-ε)²p)`.

**Step 3 (optimal evasion rate under a horizon `T`).** Harm is
`h(q) = q · min(rounds(q), T) = q · min(K/q², T)`. For `q < √(K/T)` this is
`qT`, increasing; for `q > √(K/T)` it is `K/q`, decreasing. The maximum is
at the crossover `q* = √(K/T)`, where

  `B := h(q*) = q*·T = T·√(K/T) = √(KT)`.

**Result.**

  `B(p, ε, T) = √( 2π_0(1-π_0)·log(1/α)·T ) / ( (1-β-π_0)·(1-ε)·√p )`.

**Numerical confirmation** (`π_0=0.05, β=0.1, α=0.05`, closed form vs.
`optimal_evasion_horizon`'s grid search):

| ε | p | T | closed form | measured (grid) | ratio |
|---|---|---|---|---|---|
| 0.50 | 0.10 | 1e5 | 1255.23 | 1274.35 | 0.985 |
| 0.50 | 0.10 | 1e6 | 3969.40 | 3976.14 | 0.998 |
| 0.50 | 0.10 | 1e7 | 12552.33 | 12547.05 | 1.000 |
| 0.75 | 0.10 | 1e6 | 7938.79 | 7965.00 | 0.997 |
| 0.50 | 0.02 | 1e6 | 8875.84 | 8961.95 | 0.990 |

Agreement within 1.7% throughout, confirming both the exponents (`T^{1/2}`,
`(1-ε)^{-1}`, `p^{-1/2}`) AND the constant. The approximation's own
consistency condition — `δ = κ(1-ε)q*` should be small relative to `π_0` —
holds at every row above (e.g. row 2: `δ ≈ 0.0017 ≪ π_0 = 0.05`), so the
small-deviation step is self-validating in the regime that matters (evasive
adversaries operate at low audited defection rates by construction — that is
the whole point of evasion).

*Status: this is now a real closed-form theorem, not a numerology fit.
Remaining polish: an exact (non-asymptotic) statement replacing the ≈ steps
with explicit error bounds, which the 8-page draft can defer to
supplementary if needed.*

## Theorem 3 (Targeting threshold — status: MEASURED, no closed form yet)

The same three-step method applies in principle: `naive_harm(ε)` has a
similar `flag(q)` structure but with `q` FIXED (not optimized) and a
targeting *gain* factor `1 + cε/(1-ε)` that saturates at `q_audit = 1`. The
critical `w*` solves `w·∂B/∂ε|_{ε=0} = (1-w)·(-∂(naive_harm)/∂ε)|_{ε=0}`.
Both derivatives are well-defined in closed form from the pieces above; the
saturation nonlinearity in `naive_harm` is the remaining complication before
this can be written as cleanly as Theorem 2. Left as numeric for the current
draft (`monitor/tuning.py`); closed-form derivation is a good target if page
budget and time allow before the 8 October deadline, otherwise stays
empirical with the sensitivity table already in `project_arbiter...` memory.

## Theorem 3 (Targeting threshold — status: MEASURED, no closed form yet)

Critical strategic-prior weight `w*` above which `ε* = 0` is optimal, ranges
0.025–0.39 across tested parameters (`monitor/tuning.py`,
`experiments/...` sweeps), falling with horizon `T`. No closed form derived
yet; candidate route is the same small-`δ` KL approximation applied to both
`strategic_harm` and `naive_harm` simultaneously.
