"""
llm_client.py — unified LLM wrapper.

* Live mode: any OpenAI-compatible Chat Completions endpoint via the `openai`
  SDK — OpenAI itself, Gemini (Google's OpenAI-compatible endpoint), Ollama,
  Groq, vLLM, LM Studio ... selected with LLM_PROVIDER / OPENAI_BASE_URL.
* Mock mode: a deterministic simulated LLM. Same prompt + same model ->
  same text, same token counts, same simulated latency. Lets you run the
  full pipeline, telemetry and eval.py with no API key and no internet.

If a live call fails and `fallback_to_mock` is on, the client transparently
answers with the mock and marks the response provider as "mock-fallback".
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field

from pricing import ModelConfig, estimate_tokens, get_models

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


PROVIDER_ENDPOINTS = {
    # provider: (base_url or None for SDK default, env var holding the key)
    "openai": (None, "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}

CLASSIFIER_SYSTEM_PROMPT = (
    "You are a query-complexity classifier for an LLM router. "
    "Reply with exactly one word: SIMPLE if the request is a short factual lookup, "
    "definition, translation, unit conversion or trivial arithmetic; COMPLEX if it "
    "needs multi-step reasoning, proofs, system design, code writing/debugging, "
    "analysis or comparison of several factors."
)


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    provider: str              # "openai" | "gemini" | "ollama" | "mock" | "mock-fallback"
    success: bool = True
    error: str | None = None
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Deterministic mock helpers
# --------------------------------------------------------------------------- #
def _seed(*parts: str) -> int:
    """Stable integer seed from strings (Python's hash() is salted per process)."""
    return int(hashlib.sha256("||".join(parts).encode()).hexdigest()[:12], 16)


_HARD_WORDS = re.compile(
    r"\b(prove|proof|derive|design|architect|analy[sz]e|evaluate|compare|trade-?offs?|"
    r"optimi[sz]e|debug|complexity|algorithm|dynamic programming|step[- ]by[- ]step|"
    r"implement|refactor|why does|explain why|scalab\w*|concurren\w*|distributed)\b",
    re.I,
)

_FILLER = [
    "The key idea is to break the problem into smaller, well-defined parts.",
    "In practice, the right choice depends on the constraints of your use case.",
    "A common pitfall is to overlook edge cases such as empty inputs or very large values.",
    "This approach balances simplicity with reasonable performance.",
    "It is worth validating the result with a small, concrete example.",
    "Production systems usually add monitoring and fallbacks around this logic.",
    "Several well-known references discuss this topic in more depth.",
    "The trade-off is between latency, cost and accuracy.",
    "Each step builds directly on the result of the previous one.",
    "Formally, this follows from the definitions stated above.",
]


def mock_difficulty(prompt: str) -> float:
    """The mock model's *hidden* notion of how hard a prompt is (0..1).

    Deliberately independent of router.py so routing mistakes are possible
    and the escalation path can be demonstrated.
    """
    words = len(prompt.split())
    hits = len(_HARD_WORDS.findall(prompt))
    code = 1 if re.search(r"[{}();=\[\]]|def |class |SELECT ", prompt) else 0
    score = min(1.0, 0.15 * hits + 0.012 * words + 0.25 * code)
    return round(score, 3)


class LLMClient:
    """Provider-agnostic chat client with deterministic mock fallback."""

    def __init__(
        self,
        provider: str | None = None,
        mock: bool | None = None,
        fallback_to_mock: bool = True,
        simulate_latency: bool = False,
        timeout: float = 30.0,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        self.models = get_models(self.provider)
        self.fallback_to_mock = fallback_to_mock
        self.simulate_latency = simulate_latency  # actually sleep in mock mode
        self.timeout = timeout
        self._client = None

        base_url, key_env = PROVIDER_ENDPOINTS.get(self.provider, (None, "OPENAI_API_KEY"))
        self.base_url = os.getenv("OPENAI_BASE_URL", base_url)
        self.api_key = os.getenv(key_env) if key_env else "ollama"
        # Mock if explicitly asked, or if no credentials are available.
        self.mock = bool(mock) if mock is not None else not self.has_credentials

    # ------------------------------------------------------------------ #
    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key)

    @property
    def mode_label(self) -> str:
        return "mock" if self.mock else f"live:{self.provider}"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # imported lazily so mock mode needs no SDK

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, tier: str = "cheap", system: str | None = None,
                 max_tokens: int = 700) -> LLMResponse:
        """Generate a completion with the cheap or strong model."""
        model = self.models[tier]
        if self.mock:
            return self._mock_generate(prompt, model)
        try:
            return self._live_generate(prompt, model, system, max_tokens)
        except Exception as exc:  # network error, bad key, rate limit ...
            if not self.fallback_to_mock:
                return LLMResponse("", model.name, 0, 0, 0.0, self.provider, False, str(exc))
            resp = self._mock_generate(prompt, model)
            resp.provider = "mock-fallback"
            resp.error = f"live call failed, used mock: {exc}"
            return resp

    def classify(self, prompt: str) -> tuple[str, LLMResponse]:
        """Ask the cheap model to label a prompt SIMPLE / COMPLEX."""
        model = self.models["cheap"]
        if not self.mock:
            try:
                resp = self._live_generate(prompt, model, CLASSIFIER_SYSTEM_PROMPT, max_tokens=3)
                label = "COMPLEX" if "COMPLEX" in resp.text.upper() else "SIMPLE"
                return label, resp
            except Exception as exc:
                if not self.fallback_to_mock:
                    raise
                err = str(exc)
        else:
            err = None
        label, resp = self._mock_classify(prompt, model)
        if err:
            resp.provider, resp.error = "mock-fallback", err
        return label, resp

    # ------------------------------------------------------------------ #
    def _live_generate(self, prompt: str, model: ModelConfig, system: str | None,
                       max_tokens: int) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        t0 = time.perf_counter()
        out = self._get_client().chat.completions.create(
            model=model.name, messages=messages, max_tokens=max_tokens, temperature=0.2
        )
        latency = (time.perf_counter() - t0) * 1000
        text = (out.choices[0].message.content or "").strip()
        usage = getattr(out, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", None) or estimate_tokens((system or "") + prompt)
        out_tok = getattr(usage, "completion_tokens", None) or estimate_tokens(text)
        return LLMResponse(text, model.name, int(in_tok), int(out_tok), latency, self.provider)

    def _mock_generate(self, prompt: str, model: ModelConfig) -> LLMResponse:
        rng = _seed(model.name, prompt)
        difficulty = mock_difficulty(prompt)
        strong = model.tier == "strong"

        # Simulated latency: strong models are slower; longer answers are slower.
        base = 650 if strong else 180
        latency = base + (rng % 400 if strong else rng % 160) + difficulty * (900 if strong else 250)

        if not strong and difficulty > 0.45:
            # Cheap model "struggles": short, hedged answer -> fails validation.
            text = (f"[{model.name} · simulated] I'm not entirely sure, but it may depend "
                    f"on the details.")
        else:
            n_sent = (4 + int(difficulty * 10)) if strong else (2 + int(difficulty * 4))
            body = " ".join(_FILLER[(rng + i * 7) % len(_FILLER)] for i in range(n_sent))
            topic = prompt.strip().rstrip("?.!")
            if len(topic) > 90:
                topic = topic[:87] + "..."
            text = (f"[{model.name} · simulated answer]\n\n"
                    f"Regarding \"{topic}\": {body}")
            if strong and difficulty > 0.3:
                text += "\n\nStep-by-step summary:\n" + "\n".join(
                    f"{i + 1}. {_FILLER[(rng + i * 3) % len(_FILLER)]}" for i in range(3)
                )

        if self.simulate_latency:
            time.sleep(latency / 1000)
        return LLMResponse(
            text=text,
            model=model.name,
            input_tokens=estimate_tokens(prompt) + 12,  # + chat-format overhead
            output_tokens=estimate_tokens(text),
            latency_ms=round(latency, 1),
            provider="mock",
            meta={"difficulty": difficulty},
        )

    def _mock_classify(self, prompt: str, model: ModelConfig) -> tuple[str, LLMResponse]:
        # A simulated small-model classifier: uses the hidden difficulty with a
        # slightly different cut-off than the cheap model's capability limit.
        difficulty = mock_difficulty(prompt)
        label = "COMPLEX" if difficulty >= 0.3 else "SIMPLE"
        latency = 120 + _seed("clf", prompt) % 80
        if self.simulate_latency:
            time.sleep(latency / 1000)
        resp = LLMResponse(
            text=label, model=model.name,
            input_tokens=estimate_tokens(CLASSIFIER_SYSTEM_PROMPT + prompt) + 12,
            output_tokens=1, latency_ms=float(latency), provider="mock",
            meta={"difficulty": difficulty},
        )
        return label, resp


if __name__ == "__main__":
    c = LLMClient(mock=True)
    for p in ["What is the capital of France?",
              "Design a distributed rate limiter and analyze the trade-offs."]:
        print(c.classify(p)[0], c.generate(p, "cheap").text[:80], sep=" | ")
