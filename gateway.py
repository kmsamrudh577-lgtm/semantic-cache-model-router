"""
gateway.py — the orchestration layer used by both app.py and eval.py.

    prompt ─► validate/normalize ─► semantic cache ──hit──► CACHE_HIT
                                         │miss
                                         ▼
                                 router (heuristic | llm_classifier | hybrid)
                                   │SIMPLE                 │COMPLEX
                                   ▼                       ▼
                              CHEAP_MODEL ──fails──► STRONG_MODEL (escalation)
                                   │                       │
                                   └──► store in cache ◄───┘ ─► telemetry log
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from cache import SemanticCache, normalize_tags, validate_prompt
from llm_client import LLMClient, LLMResponse
from pricing import ROUTE_CACHE, ROUTE_CHEAP, ROUTE_STRONG, compute_cost, estimate_tokens
from router import ModelRouter, RouteDecision, validate_response
from telemetry import RequestRecord, TelemetryLogger


@dataclass
class GatewayConfig:
    similarity_threshold: float = 0.88
    routing_strategy: str = "hybrid"     # heuristic | llm_classifier | hybrid
    escalation: bool = True              # retry with strong model if cheap answer fails checks
    qualifier_guard: bool = True         # veto cache hits whose numbers/units/entities differ


@dataclass
class GatewayResult:
    query: str
    response: str
    route: str
    model: str | None
    similarity: float
    matched_query: str | None
    latency_ms: float
    input_tokens: int
    output_tokens: int
    actual_cost: float
    baseline_cost: float
    cost_saved: float
    escalated: bool = False
    decision: RouteDecision | None = None
    blocked_reason: str | None = None
    provider: str = "mock"
    notes: list[str] = field(default_factory=list)

    @property
    def saved_pct(self) -> float:
        return 100 * self.cost_saved / self.baseline_cost if self.baseline_cost > 0 else 0.0


class Gateway:
    def __init__(self, cache: SemanticCache, llm: LLMClient, telemetry: TelemetryLogger,
                 config: GatewayConfig | None = None):
        self.cache = cache
        self.llm = llm
        self.router = ModelRouter(llm)
        self.telemetry = telemetry
        self.config = config or GatewayConfig()

    # ------------------------------------------------------------------ #
    def process(self, prompt: str, tags: dict | None = None,
                config: GatewayConfig | None = None) -> GatewayResult:
        cfg = config or self.config
        t0 = time.perf_counter()
        prompt = validate_prompt(prompt)          # raises ValueError on bad input
        tags = normalize_tags(tags)
        strong, cheap = self.llm.models["strong"], self.llm.models["cheap"]
        llm_ms = 0.0                              # simulated LLM time (mock w/o sleep)

        # 1) Semantic cache --------------------------------------------------
        lookup = self.cache.lookup(prompt, tags, cfg.similarity_threshold, cfg.qualifier_guard)
        if lookup.hit:
            e = lookup.entry
            in_tok = estimate_tokens(prompt) + 12
            baseline = compute_cost(strong, in_tok, e.output_tokens)
            res = GatewayResult(prompt, e.response, ROUTE_CACHE, e.model, lookup.similarity,
                                e.query, 0.0, in_tok, e.output_tokens, 0.0, baseline, baseline,
                                provider="cache",
                                notes=[f"served from cache entry #{e.id} (hits={e.hits})"])
            return self._finish(res, t0, llm_ms, cfg, tags)

        notes = []
        if lookup.blocked_reason:
            notes.append(f"cache candidate rejected by qualifier guard ({lookup.blocked_reason})")

        # 2) Route -------------------------------------------------------------
        decision = self.router.route(prompt, cfg.routing_strategy)
        clf_cost = compute_cost(cheap, decision.classifier_input_tokens,
                                decision.classifier_output_tokens) if decision.classifier_used else 0.0
        llm_ms += decision.classifier_latency_ms

        # 3) Call the chosen model (+ escalation) ---------------------------
        resp: LLMResponse = self.llm.generate(prompt, decision.tier)
        llm_ms += resp.latency_ms
        actual = clf_cost + compute_cost(self.llm.models[decision.tier],
                                         resp.input_tokens, resp.output_tokens)
        route = ROUTE_STRONG if decision.tier == "strong" else ROUTE_CHEAP
        escalated, model_label = False, resp.model
        if resp.error:
            notes.append(resp.error)

        if decision.tier == "cheap":
            ok, why = validate_response(resp.text, prompt)
            if not ok and cfg.escalation:
                notes.append(f"cheap answer failed validation ({why}) -> escalated to strong model")
                resp = self.llm.generate(prompt, "strong")
                llm_ms += resp.latency_ms
                actual += compute_cost(strong, resp.input_tokens, resp.output_tokens)
                route, escalated, model_label = ROUTE_STRONG, True, f"{cheap.name} -> {resp.model}"
            elif not ok:
                notes.append(f"cheap answer failed validation ({why}); escalation disabled")

        # Baseline = what the same request would cost on the strong model alone.
        baseline = compute_cost(strong, resp.input_tokens, resp.output_tokens)

        # 4) Store good answers ------------------------------------------------
        if resp.success and resp.text and (route == ROUTE_STRONG or validate_response(resp.text, prompt)[0]):
            self.cache.add(prompt, resp.text, resp.model, tags, resp.input_tokens,
                           resp.output_tokens, embedding=lookup.embedding)

        res = GatewayResult(
            prompt, resp.text, route, model_label, lookup.similarity,
            lookup.entry.query if lookup.entry else None, 0.0,
            resp.input_tokens + decision.classifier_input_tokens,
            resp.output_tokens + decision.classifier_output_tokens,
            actual, baseline, baseline - actual, escalated, decision,
            lookup.blocked_reason, resp.provider, notes,
        )
        return self._finish(res, t0, llm_ms, cfg, tags)

    # ------------------------------------------------------------------ #
    def _finish(self, res: GatewayResult, t0: float, llm_ms: float, cfg: GatewayConfig,
                tags: dict) -> GatewayResult:
        wall = (time.perf_counter() - t0) * 1000
        # In mock mode without real sleeping, add the simulated LLM time so
        # latency numbers stay realistic; otherwise wall-clock already has it.
        simulated = self.llm.mock and not self.llm.simulate_latency
        res.latency_ms = round(wall + (llm_ms if simulated else 0.0), 2)
        d = res.decision
        self.telemetry.log(RequestRecord(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            query=res.query, tags=json.dumps(tags), matched_query=res.matched_query,
            similarity=round(res.similarity, 4), route=res.route, model=res.model,
            strategy=cfg.routing_strategy, complexity_label=d.label if d else None,
            escalated=res.escalated, latency_ms=res.latency_ms,
            input_tokens=res.input_tokens, output_tokens=res.output_tokens,
            actual_cost=res.actual_cost, baseline_cost=res.baseline_cost,
            cost_saved=res.cost_saved, threshold=cfg.similarity_threshold,
            provider=res.provider, blocked_reason=res.blocked_reason,
            note="; ".join(res.notes) or None,
        ))
        return res
