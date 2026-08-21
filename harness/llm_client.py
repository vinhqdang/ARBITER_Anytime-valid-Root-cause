"""OpenRouter client for the MAS harness.

Deliberately small: one blocking call, explicit errors, on-disk response cache.

Two quirks of the reasoning models we target, both learned the hard way:

1. ``content`` can be ``None`` while ``finish_reason == "length"``. The model
   spent the whole budget on internal reasoning and never emitted an answer.
   Silently logging that as an empty agent turn corrupts a trajectory, so it
   is raised as ``TruncatedReasoning``.
2. Responses carry a ``reasoning`` field alongside ``content``. It is captured
   because a monitor that reads reasoning is a different experimental
   condition from one that reads only messages.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

API_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "stealth/ox-alpha"

# Reasoning models need headroom: the budget is shared between hidden
# reasoning and the visible answer, so a cap sized for the answer alone
# yields content=None.
DEFAULT_MAX_TOKENS = 4096

MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 300

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """Base class for client failures."""


class TruncatedReasoning(LLMError):
    """Model consumed its token budget on reasoning and emitted no content."""


class APIRefused(LLMError):
    """Model returned a refusal rather than an answer."""


@dataclass(frozen=True)
class Completion:
    """One model response."""

    text: str
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    cost: float
    model: str
    cached: bool = False


@dataclass
class UsageLedger:
    """Running totals, so a corpus run can report its own cost."""

    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

    def record(self, completion: Completion) -> None:
        self.calls += 1
        if completion.cached:
            self.cache_hits += 1
            return
        self.prompt_tokens += completion.prompt_tokens
        self.completion_tokens += completion.completion_tokens
        self.cost += completion.cost

    def summary(self) -> str:
        return (
            f"{self.calls} calls ({self.cache_hits} cached), "
            f"{self.prompt_tokens} prompt + {self.completion_tokens} completion tokens, "
            f"${self.cost:.4f}"
        )


def load_api_key(env_path: Path | None = None) -> str:
    """Read the key from the environment, falling back to a .env file.

    The .env file is gitignored. The key is never logged or echoed.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key

    path = env_path or Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        raise LLMError(
            "No OPENROUTER_API_KEY in environment and no .env file at "
            f"{path}. Set the variable or create the file."
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == "OPENROUTER_API_KEY":
            return value.strip()

    raise LLMError(f"OPENROUTER_API_KEY not found in {path}")


@dataclass
class LLMClient:
    """Blocking chat client with an optional on-disk cache.

    The cache is keyed on the full request, so replaying an agent on identical
    inputs is free. That matters for two reasons: it keeps corpus regeneration
    cheap, and it makes runs reproducible. It must be DISABLED for any
    experiment that needs fresh samples from an agent's conditional
    distribution -- caching would return the same draw every time and silently
    destroy the resampling.
    """

    model: str = DEFAULT_MODEL
    api_key: str = field(default_factory=load_api_key, repr=False)
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 1.0
    cache_dir: Path | None = None
    ledger: UsageLedger = field(default_factory=UsageLedger)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> Completion:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if seed is not None:
            payload["seed"] = seed

        cache_path = self._cache_path(payload)
        if cache_path is not None and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            completion = Completion(**cached, cached=True)
            self.ledger.record(completion)
            return completion

        completion = self._request(payload)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                k: v for k, v in completion.__dict__.items() if k != "cached"
            }
            cache_path.write_text(json.dumps(record), encoding="utf-8")
        self.ledger.record(completion)
        return completion

    def _cache_path(self, payload: dict[str, object]) -> Path | None:
        if self.cache_dir is None:
            return None
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _request(self, payload: dict[str, object]) -> Completion:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None

        for attempt in range(MAX_ATTEMPTS):
            if attempt:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

            request = urllib.request.Request(
                API_URL,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(
                    request, timeout=REQUEST_TIMEOUT_SECONDS
                ) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in RETRYABLE_STATUS:
                    detail = exc.read().decode("utf-8", errors="replace")[:400]
                    raise LLMError(f"HTTP {exc.code}: {detail}") from exc
                continue
            except OSError as exc:
                # Covers URLError and TimeoutError (both OSError subclasses)
                # plus bare ConnectionResetError, which the provider throws
                # on long uncached runs and which killed the first corpus run.
                last_error = exc
                continue

            return self._parse(data)

        raise LLMError(
            f"Request failed after {MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    def _parse(self, data: dict) -> Completion:
        if "error" in data:
            raise LLMError(json.dumps(data["error"])[:400])

        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"No choices in response: {json.dumps(data)[:300]}")

        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content")
        finish = choice.get("finish_reason")

        if message.get("refusal"):
            raise APIRefused(str(message["refusal"])[:300])

        if text is None or text == "":
            if finish == "length":
                raise TruncatedReasoning(
                    "Model emitted no content; the token budget went to "
                    f"reasoning. Raise max_tokens above {payload_max(data)}."
                )
            raise LLMError(f"Empty content with finish_reason={finish!r}")

        usage = data.get("usage") or {}
        return Completion(
            text=text,
            reasoning=message.get("reasoning"),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            cost=float(usage.get("cost", 0.0)),
            model=data.get("model", self.model),
        )


def payload_max(data: dict) -> int:
    """Best-effort echo of the completion budget, for error messages."""
    usage = data.get("usage") or {}
    return int(usage.get("completion_tokens", 0))
