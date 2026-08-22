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

### Upgrade: the `(1-ε)^{-1}` shape is EXACT, not asymptotic

The small-deviation step above is only needed to get a *closed form* for the
constant; the *shape* of `B` in `ε` is exact. At the crossover `q*`, the
optimality condition is `rounds(q*) = T`, i.e.
`KL(flag(q*), π_0) = log(1/α)/(Tp) =: KL_target` — and the right side does
not depend on `ε`. Since `KL(x, π_0)` is monotone in `x`, this equation has
a UNIQUE solution `flag* = π_0 + δ*`, and `δ*` is therefore a **constant
independent of ε**. Because `flag(q) = π_0 + κ(1-ε)q` (exact, no
approximation), the optimal `q*(ε) = δ*/(κ(1-ε))`, and

  `B(ε) = q*(ε)·T = δ*·T/κ · 1/(1-ε)`

**exactly** — the `1/(1-ε)` factor is algebra, not approximation. Only `δ*`
needs solving, via the one-line bisection `exact_delta_star` in
`monitor/tuning.py` (solves `KL(π_0+δ, π_0) = KL_target` directly). The
small-deviation quadratic KL approximation used above is recoverable as
`δ*`'s first-order solution to this same equation — confirmed numerically to
agree with the exact bisection to ~2% (`test_exact_delta_star_matches_quadratic_approximation`).
This strengthens Theorem 2's statement: replace `κ(1-ε)√p` in the
denominator with `κ` implicit inside `δ*` solved exactly, and the theorem
holds with no `o(1)` hedge on the *shape*, only on which value of `δ*` is
used for the constant.

## Theorem 3 (Targeting threshold)

### A certified lower bound (status: DERIVED, verified as a genuine one-sided bound)

The critical `w*` solves `w·∂B/∂ε|_{ε=0} = (1-w)·(-∂(naive_harm)/∂ε)|_{ε=0}`,
and **both derivatives can be computed exactly, without the small-deviation
approximation**:

- `∂B/∂ε|_0 = δ*·T/κ` — immediate from the exact shape above.
- `∂(naive_harm)/∂ε|_0`: `naive_harm(ε) = q_0 · rounds(ε)` with
  `flag(ε) = π_0 + κq_0·g(ε)`, `g(ε) = 1 + cε/(1-ε)`, `g'(0) = c`. Using the
  exact (non-quadratic) derivative of the binary KL,
  `∂KL(x,π_0)/∂x = \log\frac{x(1-π_0)}{π_0(1-x)}`, the chain rule gives
  a closed form (implemented as `w_star_lower_bound` in `monitor/tuning.py`)
  — confirmed to match a direct finite difference of `naive_harm` to 3
  significant figures (`test_naive_harm_derivative_matches_finite_difference`;
  finite differences are trustworthy here because, unlike `strategic_harm`,
  `naive_harm` has no internal grid search to introduce discretisation
  noise — this distinction cost real debugging time: an earlier attempt to
  validate `∂B/∂ε` by finite-differencing `strategic_harm` directly
  produced a value ≈2× too large at every step size from `1e-2` to `1e-5`,
  which turned out to be the derivative of a single FROZEN grid point in
  `optimal_evasion_horizon`'s internal argmax, not the continuous envelope
  — resolved by differentiating the validated closed form analytically
  instead of finite-differencing the discretised numerical optimum).

**What this threshold proves, precisely.** Call it `w_lb`. Because
`eps=0` becoming a local minimum of expected harm is *necessary but not
sufficient* for it to be the *global* minimum, `w_lb ≤ w*` (true global
threshold) always — a rigorous one-sided guarantee: **for any `w < w_lb`,
uniform allocation is provably not optimal; some targeting strictly
reduces expected harm.** It is not a full closed form for `w*` itself.

**Numerically confirmed as a genuine (non-vacuous) bound**, three parameter
regimes, `test_w_star_lower_bound_is_a_true_lower_bound`:

| audit_rate | π₀ | T | q₀ | c | β | `w_lb` | true `w*` (fine bisection) | ratio |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.05 | 1e6 | 0.05 | 0.8 | 0.0 | 0.0608 | 0.0729 | 0.834 |
| 0.1 | 0.05 | 1e4 | 0.05 | 0.8 | 0.0 | 0.3823 | 0.4308 | 0.887 |
| 0.2 | 0.10 | 1e5 | 0.02 | 0.5 | 0.1 | 0.2479 | 0.2723 | 0.910 |

Consistently 83–91% of the true value across regimes — a real, useful bound,
not a coincidence at one parameter point. The remaining gap is the honest
cost of a *local* (tangent-at-`ε=0`) criterion versus the *global*
optimum; closing it fully would require characterising the entire harm
curve's shape (not just its slope at one point), which does not appear to
admit a clean closed form and is left as numeric (`optimal_leakage`).

## Theorem 3 (Targeting threshold — status: MEASURED, no closed form yet)

Critical strategic-prior weight `w*` above which `ε* = 0` is optimal, ranges
0.025–0.39 across tested parameters (`monitor/tuning.py`,
`experiments/...` sweeps), falling with horizon `T`. No closed form derived
yet; candidate route is the same small-`δ` KL approximation applied to both
`strategic_harm` and `naive_harm` simultaneously.
