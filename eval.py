"""
eval.py — repeatable benchmark + threshold sweep.

For every configuration (threshold x qualifier-guard x routing strategy) the
script starts from an EMPTY in-memory cache, replays the same ordered prompt
suite through the real Gateway, and measures:

  * paraphrase hit rate     — category A prompts that were served from cache
  * near-miss false hits    — category B prompts wrongly served from cache
  * cache precision         — correct hits / all hits (A + B)
  * routing accuracy        — category C prompts sent to the expected tier
  * routing split, escalations, avg / p95 latency, $ and % saved vs all-strong

Usage:
  python eval.py                                   # default sweep, mock LLM
  python eval.py --thresholds 0.8 0.85 0.9 0.95
  python eval.py --strategies heuristic hybrid --live
Results -> results/eval_summary.{csv,json}, results/eval_details.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from cache import SemanticCache, load_embedder
from gateway import Gateway, GatewayConfig
from llm_client import LLMClient
from router import STRATEGIES
from telemetry import TelemetryLogger, summarize

# --------------------------------------------------------------------------- #
# Test suite. Order matters: SEED prompts populate the cache first.
#   expect_hit: True (should be served from cache) / False (must NOT be)
#   expect_route: SIMPLE / COMPLEX for routing-accuracy prompts
# --------------------------------------------------------------------------- #
SUITE: list[dict] = [
    # ---- Seeds (first time seen -> always misses, then cached) -------------
    {"id": "S1", "cat": "seed", "prompt": "What is the capital of France?", "tags": {"topic": "geography"}},
    {"id": "S2", "cat": "seed", "prompt": "How do I reverse a list in Python?", "tags": {"topic": "python"}},
    {"id": "S3", "cat": "seed", "prompt": "Explain what a REST API is.", "tags": {"topic": "web"}},
    {"id": "S4", "cat": "seed", "prompt": "Convert 5 kilometers to miles.", "tags": {"topic": "math"}},
    {"id": "S5", "cat": "seed", "prompt": "What are the health benefits of green tea?", "tags": {"topic": "health"}},
    {"id": "S6", "cat": "seed", "prompt": "Write a SQL query to find the top 5 customers by total sales.", "tags": {"topic": "sql"}},
    {"id": "S7", "cat": "seed", "prompt": "What is the boiling point of water at sea level in Celsius?", "tags": {"topic": "science"}},

    # ---- A: exact repeats & close paraphrases (expected HITS) ------------
    {"id": "A1", "cat": "A_paraphrase", "expect_hit": True, "prompt": "What is the capital of France?", "tags": {"topic": "geography"}},
    {"id": "A2", "cat": "A_paraphrase", "expect_hit": True, "prompt": "what's the capital city of France", "tags": {"topic": "geography"}},
    {"id": "A3", "cat": "A_paraphrase", "expect_hit": True, "prompt": "Tell me the capital of France.", "tags": {"topic": "geography"}},
    {"id": "A4", "cat": "A_paraphrase", "expect_hit": True, "prompt": "How can I reverse a list in Python?", "tags": {"topic": "python"}},
    {"id": "A5", "cat": "A_paraphrase", "expect_hit": True, "prompt": "explain what a REST API is", "tags": {"topic": "web"}},
    {"id": "A6", "cat": "A_paraphrase", "expect_hit": True, "prompt": "Can you explain what REST APIs are?", "tags": {"topic": "web"}},
    {"id": "A7", "cat": "A_paraphrase", "expect_hit": True, "prompt": "Convert 5 km to miles", "tags": {"topic": "math"}},
    {"id": "A8", "cat": "A_paraphrase", "expect_hit": True, "prompt": "What health benefits does green tea have?", "tags": {"topic": "health"}},
    {"id": "A9", "cat": "A_paraphrase", "expect_hit": True, "prompt": "Write an SQL query that finds the top 5 customers by total sales.", "tags": {"topic": "sql"}},
    {"id": "A10", "cat": "A_paraphrase", "expect_hit": True, "prompt": "At sea level, what is the boiling point of water in Celsius?", "tags": {"topic": "science"}},

    # ---- B: near-miss / tricky prompts (expected MISSES) -------------------
    {"id": "B1", "cat": "B_near_miss", "expect_hit": False, "prompt": "What is the capital of Germany?", "tags": {"topic": "geography"}},
    {"id": "B2", "cat": "B_near_miss", "expect_hit": False, "prompt": "How do I reverse a string in Python?", "tags": {"topic": "python"}},
    {"id": "B3", "cat": "B_near_miss", "expect_hit": False, "prompt": "Convert 50 kilometers to miles.", "tags": {"topic": "math"}},
    {"id": "B4", "cat": "B_near_miss", "expect_hit": False, "prompt": "Convert 5 miles to kilometers.", "tags": {"topic": "math"}},
    {"id": "B5", "cat": "B_near_miss", "expect_hit": False, "prompt": "Write a SQL query to find the top 10 customers by total sales.", "tags": {"topic": "sql"}},
    {"id": "B6", "cat": "B_near_miss", "expect_hit": False, "prompt": "What is the boiling point of water at sea level in Fahrenheit?", "tags": {"topic": "science"}},
    {"id": "B7", "cat": "B_near_miss", "expect_hit": False, "prompt": "What are the side effects of green tea?", "tags": {"topic": "health"}},
    {"id": "B8", "cat": "B_near_miss", "expect_hit": False, "prompt": "Explain what a SOAP API is.", "tags": {"topic": "web"}},
    {"id": "B9", "cat": "B_near_miss", "expect_hit": False, "prompt": "How do I reverse a list in Python?", "tags": {"topic": "javascript"},
     "why": "identical text, different context tag -> metadata filter must block"},
    {"id": "B10", "cat": "B_near_miss", "expect_hit": False, "prompt": "What was the capital of France in 1940?", "tags": {"topic": "geography"}},

    # ---- C: easy vs hard reasoning (routing accuracy) ----------------------
    {"id": "C1", "cat": "C_routing", "expect_route": "SIMPLE", "prompt": "Define photosynthesis in one sentence."},
    {"id": "C2", "cat": "C_routing", "expect_route": "SIMPLE", "prompt": "Translate 'good morning' into Spanish."},
    {"id": "C3", "cat": "C_routing", "expect_route": "SIMPLE", "prompt": "What is 15% of 200?"},
    {"id": "C4", "cat": "C_routing", "expect_route": "SIMPLE", "prompt": "Who wrote Romeo and Juliet?"},
    {"id": "C5", "cat": "C_routing", "expect_route": "SIMPLE", "prompt": "List three primary colors."},
    {"id": "C6", "cat": "C_routing", "expect_route": "SIMPLE", "prompt": "What does HTTP stand for?"},
    {"id": "C7", "cat": "C_routing", "expect_route": "COMPLEX", "prompt": "Design a scalable URL shortener system and explain the trade-offs between consistency and availability."},
    {"id": "C8", "cat": "C_routing", "expect_route": "COMPLEX", "prompt": "Prove that the square root of 2 is irrational, step by step."},
    {"id": "C9", "cat": "C_routing", "expect_route": "COMPLEX", "prompt": "Analyze the time and space complexity of merge sort versus quicksort, and explain when each is preferable."},
    {"id": "C10", "cat": "C_routing", "expect_route": "COMPLEX", "prompt": "Debug this Python code and explain why it raises a KeyError: d = {}; d['a'] += 1"},
    {"id": "C11", "cat": "C_routing", "expect_route": "COMPLEX", "prompt": "Compare the economic causes of the 2008 financial crisis with those of the Great Depression, and evaluate which policy responses were most effective."},
    {"id": "C12", "cat": "C_routing", "expect_route": "COMPLEX", "prompt": "Write a Python function to find the longest palindromic substring using dynamic programming, and explain its complexity."},
]


def run_config(embedder, threshold: float, guard: bool, strategy: str, live: bool) -> tuple[dict, list[dict]]:
    cache = SemanticCache(":memory:", embedder=embedder)
    telemetry = TelemetryLogger(":memory:")
    llm = LLMClient(mock=not live)
    gw = Gateway(cache, llm, telemetry)
    cfg = GatewayConfig(threshold, strategy, escalation=True, qualifier_guard=guard)

    rows = []
    for case in SUITE:
        r = gw.process(case["prompt"], case.get("tags"), cfg)
        hit = r.route == "CACHE_HIT"
        row = {
            "threshold": threshold, "guard": guard, "strategy": strategy,
            "id": case["id"], "category": case["cat"], "prompt": case["prompt"],
            "route": r.route, "similarity": round(r.similarity, 4),
            "matched_query": r.matched_query, "blocked_reason": r.blocked_reason,
            "complexity_label": r.decision.label if r.decision else None,
            "escalated": r.escalated, "latency_ms": r.latency_ms,
            "actual_cost": r.actual_cost, "baseline_cost": r.baseline_cost,
        }
        if "expect_hit" in case:
            row["expect_hit"] = case["expect_hit"]
            row["correct"] = hit == case["expect_hit"]
        if "expect_route" in case:
            row["expect_route"] = case["expect_route"]
            row["correct"] = (r.decision is not None and r.decision.label == case["expect_route"])
        rows.append(row)

    d = pd.DataFrame(rows)
    a, b, c = (d[d.category == k] for k in ("A_paraphrase", "B_near_miss", "C_routing"))
    a_hits = int((a.route == "CACHE_HIT").sum())
    b_hits = int((b.route == "CACHE_HIT").sum())
    s = summarize(telemetry.to_dataframe())
    rc = s["route_counts"]
    summary = {
        "threshold": threshold,
        "guard": "on" if guard else "off",
        "strategy": strategy,
        "hit_rate_%": round(s["cache_hit_rate"], 1),
        "paraphrase_recall_%": round(100 * a_hits / len(a), 1),
        "near_miss_false_hits": f"{b_hits}/{len(b)}",
        "false_hit_rate_%": round(100 * b_hits / len(b), 1),
        "cache_precision_%": round(100 * a_hits / (a_hits + b_hits), 1) if a_hits + b_hits else 100.0,
        "routing_acc_%": round(100 * c["correct"].mean(), 1),
        "cache/cheap/strong": f"{rc.get('CACHE_HIT', 0)}/{rc.get('CHEAP_MODEL', 0)}/{rc.get('STRONG_MODEL', 0)}",
        "escalations": s["escalations"],
        "avg_latency_ms": round(s["avg_latency_ms"], 1),
        "p95_latency_ms": round(s["p95_latency_ms"], 1),
        "cost_$": round(s["total_cost"], 6),
        "baseline_$": round(s["baseline_cost"], 6),
        "saved_%": round(s["cost_saved_pct"], 1),
    }
    return summary, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic cache + router benchmark")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.80, 0.88, 0.93])
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    ap.add_argument("--guard", choices=["both", "on", "off"], default="both",
                    help="qualifier guard setting(s) to sweep")
    ap.add_argument("--live", action="store_true", help="use the real LLM API instead of the mock")
    ap.add_argument("--embedder", choices=["auto", "hashing"], default="auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    embedder = load_embedder(args.embedder)
    guards = {"both": [False, True], "on": [True], "off": [False]}[args.guard]
    n_prompts = len(SUITE)
    print(f"Embedder: {embedder.name} | prompts: {n_prompts} | LLM: {'live' if args.live else 'mock'}")

    summaries, details = [], []
    t0 = time.perf_counter()
    # Threshold x guard sweep on the hybrid router, plus a strategy comparison
    # at the middle threshold (guard on) to keep the run short.
    base = "hybrid" if "hybrid" in args.strategies else args.strategies[0]
    mid = sorted(args.thresholds)[len(args.thresholds) // 2]
    configs = [(t, g, base) for t in args.thresholds for g in guards]
    configs += [(mid, guards[-1], s) for s in args.strategies if s != base]

    for t, g, s in configs:
        summary, rows = run_config(embedder, t, g, s, args.live)
        summaries.append(summary)
        details.extend(rows)

    os.makedirs(args.out, exist_ok=True)
    sdf = pd.DataFrame(summaries)
    sdf.to_csv(os.path.join(args.out, "eval_summary.csv"), index=False)
    with open(os.path.join(args.out, "eval_summary.json"), "w") as f:
        json.dump({"embedder": embedder.name, "llm": "live" if args.live else "mock",
                   "n_prompts": n_prompts, "results": summaries}, f, indent=2)
    pd.DataFrame(details).to_csv(os.path.join(args.out, "eval_details.csv"), index=False)

    with pd.option_context("display.max_columns", None, "display.width", 250):
        print("\n" + sdf.to_string(index=False))

    # Per-prompt view of the near-miss category at the middle threshold, guard off vs on.
    det = pd.DataFrame(details)
    view = det[(det.category == "B_near_miss") & (det.threshold == mid) & (det.strategy == base)]
    if not view.empty:
        print(f"\nNear-miss detail @ threshold {mid} ({base}):")
        piv = view.pivot_table(index=["id", "prompt"], columns="guard", values="route", aggfunc="first")
        piv.columns = [f"guard_{'on' if c else 'off'}" for c in piv.columns]
        sims = view.groupby("id")["similarity"].first()
        piv.insert(0, "similarity", piv.index.get_level_values(0).map(sims))
        with pd.option_context("display.width", 250, "display.max_colwidth", 70):
            print(piv.to_string())

    print(f"\nSaved results to {args.out}/ in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
