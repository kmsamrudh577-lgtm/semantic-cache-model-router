"""
router.py — cost-aware SIMPLE / COMPLEX routing + response validation.

Strategies
----------
heuristic       zero-cost rules: length, reasoning verbs, code/math markers,
                multi-part questions vs. "what is / define / translate" cues.
llm_classifier  asks the CHEAP model to label the prompt (costs a few tokens).
hybrid          heuristic first; only prompts in the uncertainty band
                (score in [band_low, band_high]) are sent to the classifier.

Escalation
----------
If a SIMPLE route's answer fails `validate_response` (empty, too short,
hedging/refusal, truncated), the gateway re-asks the STRONG model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from llm_client import LLMClient

STRATEGIES = ("heuristic", "llm_classifier", "hybrid")

_REASONING = re.compile(
    r"\b(why|how does|prove|proof|derive|design|architect\w*|analy[sz]e|analysis|evaluate|"
    r"compare|contrast|trade-?offs?|optimi[sz]e|debug|complexity|algorithm\w*|"
    r"step[- ]by[- ]step|implement|refactor|strategy|scalab\w*|pros and cons|"
    r"critique|justify|dynamic programming|recursion|concurren\w*|distributed)\b",
    re.I,
)
_SIMPLE_CUES = re.compile(
    r"^(what is|what's|what are|who (is|was|wrote)|when (is|was|did)|where is|define|"
    r"translate|convert|list|name|spell|how many|what does .* stand for)\b",
    re.I,
)
_CODE = re.compile(r"```|[{};=]|\bdef \w+\(|\bclass \w+|\bfunction\b|\bSELECT\b.*\bFROM\b|=>")
_MATH = re.compile(r"[∑∫√≤≥≠]|\\frac|\bintegral\b|\bderivative\b|\bmatrix\b|\bprobability\b", re.I)
_CONSTRAINTS = re.compile(r"\b(must|at least|at most|without using|constraints?|ensure|"
                          r"edge cases?|in O\()", re.I)


@dataclass
class RouteDecision:
    label: str                       # "SIMPLE" | "COMPLEX"
    strategy: str
    score: float                     # heuristic complexity score 0..1
    reasons: list[str] = field(default_factory=list)
    classifier_used: bool = False
    classifier_input_tokens: int = 0
    classifier_output_tokens: int = 0
    classifier_latency_ms: float = 0.0

    @property
    def tier(self) -> str:
        return "strong" if self.label == "COMPLEX" else "cheap"


def heuristic_score(prompt: str) -> tuple[float, list[str]]:
    """Complexity score in [0, 1] plus the human-readable reasons behind it."""
    score, reasons = 0.0, []
    words = len(prompt.split())
    if words > 60:
        score += 0.30; reasons.append(f"long prompt ({words} words)")
    elif words > 25:
        score += 0.15; reasons.append(f"medium-length prompt ({words} words)")

    hits = sorted({m.group(0).lower() for m in _REASONING.finditer(prompt)})
    if hits:
        score += min(0.6, 0.3 * len(hits)); reasons.append(f"reasoning cues: {', '.join(hits[:5])}")
    if _CODE.search(prompt):
        score += 0.25; reasons.append("code markers")
    if _MATH.search(prompt):
        score += 0.20; reasons.append("math markers")
    if _CONSTRAINTS.search(prompt):
        score += 0.10; reasons.append("explicit constraints")
    questions = prompt.count("?") + len(re.findall(r"\b(and then|also|additionally)\b", prompt, re.I))
    if questions >= 2:
        score += 0.15; reasons.append("multi-part question")
    if re.search(r"\b(and|then) (explain|evaluate|compare|analy[sz]e|justify)\b", prompt, re.I):
        score += 0.15; reasons.append("compound task")
    if _SIMPLE_CUES.search(prompt.strip()) and words <= 15:
        score -= 0.25; reasons.append("simple lookup phrasing")
    score = max(0.0, min(1.0, score))
    if not reasons:
        reasons.append("no complexity signals")
    return round(score, 3), reasons


class ModelRouter:
    def __init__(self, llm: LLMClient, threshold: float = 0.5,
                 band_low: float = 0.25, band_high: float = 0.6):
        self.llm = llm
        self.threshold = threshold          # heuristic SIMPLE/COMPLEX cut-off
        self.band_low, self.band_high = band_low, band_high  # hybrid uncertainty band

    def route(self, prompt: str, strategy: str = "hybrid") -> RouteDecision:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown routing strategy {strategy!r}; choose from {STRATEGIES}")
        score, reasons = heuristic_score(prompt)
        heuristic_label = "COMPLEX" if score >= self.threshold else "SIMPLE"

        use_classifier = strategy == "llm_classifier" or (
            strategy == "hybrid" and self.band_low <= score <= self.band_high
        )
        if not use_classifier:
            if strategy == "hybrid":
                reasons = reasons + ["hybrid: heuristic confident, classifier skipped"]
            return RouteDecision(heuristic_label, strategy, score, reasons)

        label, resp = self.llm.classify(prompt)
        note = ("hybrid: score in uncertainty band -> " if strategy == "hybrid" else "") + \
               f"LLM classifier ({resp.model}) said {label}"
        return RouteDecision(label, strategy, score, reasons + [note], True,
                             resp.input_tokens, resp.output_tokens, resp.latency_ms)


_HEDGES = re.compile(
    r"(i'?m not (entirely )?sure|i don'?t know|i cannot|i can'?t help|as an ai|"
    r"unable to (answer|help)|it may depend on the details)", re.I
)


def validate_response(text: str, prompt: str, min_chars: int = 40) -> tuple[bool, str]:
    """Basic quality gate for a cheap-model answer. Returns (ok, reason)."""
    if not text or not text.strip():
        return False, "empty response"
    body = re.sub(r"^\[.*?\]\s*", "", text.strip())   # ignore mock "[model · simulated]" tag
    # Expected minimum length grows a little with prompt length.
    needed = min_chars + min(160, len(prompt) // 4)
    if _HEDGES.search(body):
        return False, "hedging / refusal language"
    if len(body) < needed and len(prompt.split()) > 8:
        return False, f"too short ({len(body)} < {needed} chars)"
    if body.endswith(("...", ",", " and", " the")):
        return False, "looks truncated"
    return True, "ok"


if __name__ == "__main__":
    for p in ["What is the capital of France?",
              "Design a scalable URL shortener and explain the trade-offs.",
              "Prove that sqrt(2) is irrational step by step."]:
        print(heuristic_score(p), "|", p)
