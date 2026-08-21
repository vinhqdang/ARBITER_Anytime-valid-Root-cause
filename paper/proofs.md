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

## Theorem 2 (Scaling law — status: MEASURED, not yet derived in closed form)

**Measured (see `experiments/scaling_law.py`, locked in
`test_scaling_law_exponents`):**

  `B(p, ε, T) ≍ C · sqrt(T) / ((1-ε) · sqrt(p))`,   exponents 0.4876 / 1.0016 / 0.5049.

**TODO before this can be stated as a theorem:** derive `C` from first
principles via the KL small-deviation approximation
`KL(π_0+δ‖π_0) ≈ δ²/(2π_0(1-π_0))` and the optimality condition
`d/dq [q · min(rounds(q), T)] = 0` at the interior optimum, matching the
constant against `optimal_evasion_horizon`'s numerics. Also needs the
`β` (verifier false-negative rate) dependence made explicit — currently
folded into the numerics via `audited_flag_rate`, not yet in the closed form
above.

## Theorem 3 (Targeting threshold — status: MEASURED, no closed form yet)

Critical strategic-prior weight `w*` above which `ε* = 0` is optimal, ranges
0.025–0.39 across tested parameters (`monitor/tuning.py`,
`experiments/...` sweeps), falling with horizon `T`. No closed form derived
yet; candidate route is the same small-`δ` KL approximation applied to both
`strategic_harm` and `naive_harm` simultaneously.
