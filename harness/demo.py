"""End-to-end smoke test of the harness. Run directly."""

import sys
from pathlib import Path

# Model output contains non-cp1252 characters (arrows, dashes); the default
# Windows console encoding raises on them mid-print.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from .llm_client import LLMClient
from .mas import Fault, build_topology, replay_agent, run_trajectory

CLEAN_CONTEXT = "Reference: the project deadline is 8 October and the budget is $40,000."
POISONED_CONTEXT = "Reference: the project deadline is 8 December and the budget is $40,000."

TASK = "State the project deadline and compute the monthly budget if spread evenly to the deadline from 21 August."


def main() -> None:
    specs = build_topology("chain", 3)
    print("graph:", [(s.agent_id, s.parents) for s in specs])

    # Cache ON for the observational run: reproducible and cheap.
    cached = LLMClient(cache_dir=Path("runs/cache"))

    clean = run_trajectory(
        TASK, specs, cached, topology="chain", clean_context=CLEAN_CONTEXT
    )
    print(f"\n[clean] culprit={clean.culprit}")
    print("  final:", clean.steps[-1].output[:120].replace("\n", " "))

    # poison_context corrupts shared state, so NO message in the trace is
    # itself corrupted -- the case that defeats message-only monitoring.
    fault = Fault(target="a0", kind="poison_context", payload=POISONED_CONTEXT)
    faulted = run_trajectory(
        TASK, specs, cached, topology="chain", fault=fault, clean_context=CLEAN_CONTEXT
    )
    print(f"\n[faulted] culprit={faulted.culprit} kind={fault.kind}")
    for step in faulted.steps:
        print(f"  {step.agent_id}: {step.output[:90].replace(chr(10), ' ')}")

    # Interventional primitive: re-run the last agent with the clean context.
    # Cache OFF so repeated draws are genuinely independent.
    fresh = LLMClient(cache_dir=None)
    draws = replay_agent(
        faulted, specs, "a2", fresh, shared_context=CLEAN_CONTEXT, n_samples=2
    )
    print(f"\n[replay a2 under clean context] {len(draws)} draws")
    for i, d in enumerate(draws):
        print(f"  draw {i}: {d.output[:90].replace(chr(10), ' ')}")

    print("\ncached ledger:", cached.ledger.summary())
    print("fresh ledger: ", fresh.ledger.summary())


if __name__ == "__main__":
    main()
