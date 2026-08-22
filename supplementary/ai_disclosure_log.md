# AI-Assistance Disclosure: Methodology Log

This is a good-faith summary of the AI assistant's involvement in this
paper's research process, prepared to satisfy the AAMAS 2027 policy on
AI-assisted technologies. It is a curated record of key decision points and
representative prompts, not a verbatim transcript of the full working
session (which ran to several hundred exchanges over multiple days).

## 1. Framing search and rejection (hypothesis formation)

The research question — anytime-valid monitoring of strategic multi-agent
LLM systems — went through five substantially different framings before
arriving at the one in this paper. Each was proposed by the assistant,
checked against prior literature via adversarial search, and either
rejected or retained:

1. **Passive parent-conditional e-processes + stopped e-BH on the agent
   graph.** Rejected as an incremental combination of existing statistical
   machinery (e-processes, e-BH, SCM conditioning) applied to a new domain,
   with no new algorithmic content.
2. **Intervention makes blame identifiable** (active causal probing of
   agents). Rejected: found to be substantially pre-empted by the active
   diagnosis literature (Sampath et al. 1998), adaptive probing (Rish et al.
   2005), and a large 2025–2026 LLM multi-agent counterfactual-replay
   literature (DoVer, Causal Agent Replay, and others).
3. **On-demand resampling of an agent's conditional distribution as a
   structural edge.** Rejected: found already done empirically (FAMAS,
   FSE 2026).
4. **Adversarial e-process design against a test-aware opponent** (an
   agent that detects and responds to the monitor's statistical test
   itself). Initially promising; ultimately subsumed by the finding below.
5. **Leakage-constrained audit allocation with a certified harm bound**
   (this paper). Survived adversarial prior-art search across active
   diagnosis, inspection games, AI control, sandbagging, and adjacent
   anytime-valid-inference literatures. Two of the three nearest
   competitors (Griffin et al., "Games for AI Control"; Capitaine et al.,
   "Test-then-Punish") were found, on close reading, to explicitly flag the
   gap this paper fills as outside their own model's scope.

Representative prompt (paraphrased): *"but it is just a combination? usually
it will be rejected if we just combine existing things"* — this user
pushback after framing (1) directly motivated the adversarial-search
discipline applied to framings (2)–(5).

## 2. Literature verification

Approximately 15 adversarial prior-art searches were run (via parallel
sub-agent dispatch) against arXiv, OpenAlex, Crossref, DBLP, and direct
primary-source fetches, explicitly instructed to find the paper that would
kill each candidate framing rather than to confirm it. This surfaced:

- Two citations initially asserted from memory that do not exist
  ("TRAJDEBUG", "Before Agents Speak") — recorded as a `DO NOT CITE` list
  in the bibliography rather than silently dropped.
- One paper's stated result initially mischaracterised from its abstract
  alone; a full-text re-read (Gauthier et al., arXiv:2601.05427) found a
  theorem (FDR control over per-agent hypotheses at a data-dependent
  stopping time) that closed one of the paper's original claimed
  contributions, forcing a re-scope.
- Two venue claims (Who&When at ICML 2025; LumiMAS at AAMAS 2026) initially
  recorded as unconfirmed by the arXiv abstract pages themselves, later
  independently confirmed via PMLR, DBLP, Semantic Scholar, and Crossref.

## 3. Theorem derivation and numerical validation

All four main results (Proposition 1, Theorems 1–3) were derived through an
iterative process of symbolic derivation followed by numerical
cross-checking against a from-scratch code implementation
(`monitor/eprocess.py`, `monitor/policy.py`, `monitor/tuning.py`), with 40+
unit and property tests. Two corrections of note:

- The original formalisation of the adversary's constraint
  (`q_audit <= pi_0`) was found, via simulation, to be logically wrong: it
  treated the defection rate as the quantity that must hide below the
  noise floor, when defection is itself the signal the verifier flags. This
  was caught by the simulation producing an adversary "detected in 11
  audits" against a theoretical prediction of "never detected" — a direct
  contradiction that forced re-derivation.
- Theorem 3's threshold derivation initially produced a finite-difference
  check that disagreed with the closed form by a factor of ~2, converging
  smoothly across five orders of magnitude in step size (which read as a
  legitimate measurement). This was eventually traced to the internal
  optimisation routine's discrete grid search: perturbations smaller than
  one grid cell's width do not move the selected optimum, so the measured
  "derivative" was the slope of one frozen grid point rather than the
  continuous envelope. Resolved by differentiating the analytically
  validated closed form directly and cross-checking against a
  grid-search-free function (`naive_harm`), where finite differences agreed
  to 5 significant figures.

## 4. Experimental design

The empirical stress test (Section 6.3) went through three failed task
designs before arriving at the one reported, each diagnosed and fixed in
turn:

1. An 18-step arithmetic dependency chain: gave a floor effect (zero
   exploitation in every cell) because verification was cheap for a
   reasoning model, and separately revealed a token-budget cliff rather
   than a graded effort/accuracy trade-off.
2. A field-extraction task with a near-miss distractor: still zero
   exploitation, traced to an explicit textual "tell" (a disqualifying
   label) that let the model abstain correctly without any real search
   effort.
3. A far-apart contradiction-detection task (the one reported): required
   fixing two ground-truth construction bugs before it produced trustworthy
   data — a filler-template reuse bug that created spurious
   self-contradictions in both experimental arms, and a "planted
   contradiction" that was, on inspection, not actually a logical
   contradiction (a stricter deadline is compatible with a looser backstop;
   the model's "NONE" verdict on it was correct, not a miss).

A separate data-integrity bug was caught during multi-model comparison
runs: HTTP 503 errors from an overloaded free-tier model endpoint were
initially being counted as "unparsed" responses, which is statistically
indistinguishable from a genuine abstention. This was fixed to exclude and
separately report provider errors before any conclusions were drawn from
that data.

## 5. Authorial oversight

Every theorem, numerical claim, and experimental result in the submitted
paper was checked against the underlying code and, where applicable,
against primary literature sources before inclusion. The assistant's
identification of its own errors above (the constraint bug, the
finite-difference artefact, the task-design failures) was accepted only
after independent numerical confirmation, not taken on assertion. The
authors are responsible for all claims in the final manuscript.
