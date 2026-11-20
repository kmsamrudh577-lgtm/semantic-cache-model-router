"""
pricing.py — central model registry, per-1M-token pricing and cost helpers.

Every cost number in the project (actual cost, 100%-strong-model baseline,
savings) is computed here so there is exactly one place to update prices.

Prices are USD per 1M tokens (input / output). They are reasonable public
list prices at time of writing — verify against your provider before quoting
numbers in a report, and override via env vars if needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str            # model id sent to the API
    tier: str            # "cheap" | "strong"
    input_per_1m: float  # USD per 1M input tokens
    output_per_1m: float # USD per 1M output tokens


# Known price list (USD / 1M tokens). Add models here to make them selectable.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    # Google Gemini (via OpenAI-compatible endpoint)
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    # Local / self-hosted (e.g. Ollama) — treat as free compute
    "llama3.2": (0.0, 0.0),
    "llama3.1:70b": (0.0, 0.0),
}

# Default model pair per provider: (cheap, strong)
PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "openai": ("gpt-4o-mini", "gpt-4o"),
    "gemini": ("gemini-2.0-flash", "gemini-2.5-pro"),
    "ollama": ("llama3.2", "llama3.1:70b"),
}

# Route labels used across the whole project
ROUTE_CACHE = "CACHE_HIT"
ROUTE_CHEAP = "CHEAP_MODEL"
ROUTE_STRONG = "STRONG_MODEL"

# Local sentence-transformers embeddings cost nothing per call.
EMBEDDING_COST_PER_1M = 0.0


def _model(name: str, tier: str) -> ModelConfig:
    inp, out = PRICE_TABLE.get(name, (0.0, 0.0))
    # Optional per-tier env overrides, e.g. CHEAP_INPUT_PRICE=0.2
    inp = float(os.getenv(f"{tier.upper()}_INPUT_PRICE", inp))
    out = float(os.getenv(f"{tier.upper()}_OUTPUT_PRICE", out))
    return ModelConfig(name=name, tier=tier, input_per_1m=inp, output_per_1m=out)


def get_models(provider: str | None = None) -> dict[str, ModelConfig]:
    """Return {"cheap": ModelConfig, "strong": ModelConfig} for a provider.

    Model names can be overridden with CHEAP_MODEL / STRONG_MODEL env vars.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    cheap_default, strong_default = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])
    return {
        "cheap": _model(os.getenv("CHEAP_MODEL", cheap_default), "cheap"),
        "strong": _model(os.getenv("STRONG_MODEL", strong_default), "strong"),
    }


def compute_cost(model: ModelConfig, input_tokens: int, output_tokens: int) -> float:
    """USD cost of one call."""
    return (input_tokens * model.input_per_1m + output_tokens * model.output_per_1m) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token for English).

    Live API calls use the provider's exact usage numbers instead.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def savings(actual_cost: float, baseline_cost: float) -> tuple[float, float]:
    """Return (saved_usd, saved_pct) relative to the all-strong-model baseline."""
    saved = baseline_cost - actual_cost
    pct = (saved / baseline_cost * 100.0) if baseline_cost > 0 else 0.0
    return saved, pct


if __name__ == "__main__":
    for prov in PROVIDER_DEFAULTS:
        m = get_models(prov)
        print(prov, m["cheap"], m["strong"], sep="\n  ")
