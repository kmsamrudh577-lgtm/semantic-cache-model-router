"""
app.py — Streamlit dashboard for the Semantic Cache & Cost-Aware Model Router.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from cache import SemanticCache, load_embedder
from gateway import Gateway, GatewayConfig
from llm_client import LLMClient
from pricing import PROVIDER_DEFAULTS, get_models
from router import STRATEGIES
from telemetry import TelemetryLogger

st.set_page_config(page_title="Semantic Cache & Model Router", page_icon="⚡", layout="wide")

DATA_DIR = os.getenv("DATA_DIR", "data")
BADGE_COLORS = {"CACHE_HIT": "#16a34a", "CHEAP_MODEL": "#2563eb", "STRONG_MODEL": "#9333ea"}
SAMPLE_PROMPTS = [
    "",
    "What is the capital of France?",
    "what's the capital city of France",
    "What is the capital of Germany?",
    "Convert 5 kilometers to miles.",
    "Convert 50 kilometers to miles.",
    "How do I reverse a list in Python?",
    "Define photosynthesis in one sentence.",
    "Design a scalable URL shortener and explain the trade-offs between consistency and availability.",
    "Prove that the square root of 2 is irrational, step by step.",
]


# --------------------------------------------------------------------------- #
# Heavy, long-lived resources (shared across reruns)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model ...")
def get_cache() -> SemanticCache:
    return SemanticCache(os.path.join(DATA_DIR, "semantic_cache.db"), embedder=load_embedder())


@st.cache_resource
def get_telemetry() -> TelemetryLogger:
    return TelemetryLogger(os.path.join(DATA_DIR, "telemetry.db"))


def badge(route: str) -> str:
    color = BADGE_COLORS.get(route, "#64748b")
    return (f"<span style='background:{color};color:white;padding:4px 12px;border-radius:999px;"
            f"font-weight:600;font-size:0.9rem'>{route}</span>")


cache, telemetry = get_cache(), get_telemetry()

# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Controls")
    provider = st.selectbox("Provider", list(PROVIDER_DEFAULTS),
                            index=list(PROVIDER_DEFAULTS).index(os.getenv("LLM_PROVIDER", "openai"))
                            if os.getenv("LLM_PROVIDER", "openai") in PROVIDER_DEFAULTS else 0)
    probe = LLMClient(provider=provider)
    mock_mode = st.toggle("Offline / mock LLM mode", value=not probe.has_credentials,
                          help="Deterministic simulated LLM. Turned on automatically when no API key is set.")
    if not mock_mode and not probe.has_credentials:
        st.warning("No API key found for this provider — falling back to mock mode.")
        mock_mode = True
    simulate_latency = st.toggle("Simulate real latency (sleep)", value=False, disabled=not mock_mode)

    threshold = st.slider("similarity_threshold", 0.70, 0.98, 0.88, 0.01)
    strategy = st.selectbox("routing_strategy", STRATEGIES, index=STRATEGIES.index("hybrid"))
    escalation = st.toggle("Escalate failed cheap answers", value=True)
    guard = st.toggle("Qualifier guard (numbers/units/entities)", value=True,
                      help="Rejects cache hits when numbers, units, negations or named entities differ.")

    models = get_models(provider)
    st.caption(f"**Cheap:** `{models['cheap'].name}` — ${models['cheap'].input_per_1m}/"
               f"${models['cheap'].output_per_1m} per 1M tok")
    st.caption(f"**Strong:** `{models['strong'].name}` — ${models['strong'].input_per_1m}/"
               f"${models['strong'].output_per_1m} per 1M tok")
    st.caption(f"**Embedder:** `{cache.embedder.name}` · **Cache size:** {len(cache)}")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Clear cache", use_container_width=True):
        cache.clear(); st.session_state.pop("last", None); st.toast("Cache cleared")
    if c2.button("♻️ Reset logs", use_container_width=True):
        telemetry.reset(); st.toast("Telemetry reset")

llm = LLMClient(provider=provider, mock=mock_mode, simulate_latency=simulate_latency)
gateway = Gateway(cache, llm, telemetry)
config = GatewayConfig(threshold, strategy, escalation, guard)

st.title("⚡ Semantic Cache & Cost-Aware Model Router")
st.caption(f"LLM mode: **{llm.mode_label}** · every request is checked against the semantic cache "
           f"first, then routed to the cheapest model that can handle it.")

tab_play, tab_tel, tab_cache = st.tabs(["🧪 Playground", "📊 Telemetry", "🗂️ Cache inspector"])

# --------------------------------------------------------------------------- #
# Playground
# --------------------------------------------------------------------------- #
with tab_play:
    sample = st.selectbox("Quick demo prompt (optional)", SAMPLE_PROMPTS,
                          format_func=lambda s: s or "— type your own —")
    with st.form("ask"):
        prompt = st.text_area("Prompt", value=sample, height=110,
                              placeholder="Ask anything, e.g. 'What is the capital of France?'")
        t1, t2 = st.columns(2)
        topic = t1.text_input("topic / context_tag (optional)",
                              help="Entries with different tags never match each other.")
        language = t2.selectbox("language (optional)", ["", "en", "es", "fr", "de", "hi"])
        submitted = st.form_submit_button("🚀 Send", type="primary", use_container_width=True)

    if submitted:
        try:
            with st.spinner("Routing ..."):
                st.session_state["last"] = gateway.process(
                    prompt, {"topic": topic, "language": language}, config)
        except ValueError as exc:
            st.error(f"Invalid prompt: {exc}")

    res = st.session_state.get("last")
    if res:
        st.markdown(badge(res.route) + (" &nbsp;⤴️ <b>escalated</b>" if res.escalated else ""),
                    unsafe_allow_html=True)
        m = st.columns(3)
        m[0].metric("Similarity", f"{res.similarity:.3f}")
        m[1].metric("Latency", f"{res.latency_ms:,.0f} ms")
        m[2].metric("Model", res.model or "-")
        m = st.columns(3)
        m[0].metric("Actual cost", f"${res.actual_cost:.6f}")
        m[1].metric("Baseline (all-strong)", f"${res.baseline_cost:.6f}")
        m[2].metric("Saved", f"${res.cost_saved:.6f}", f"{res.saved_pct:.0f}%")
        st.markdown("**Response**")
        st.info(res.response)
        with st.expander("Routing details", expanded=False):
            st.write({
                "model": res.model,
                "matched_query": res.matched_query,
                "qualifier_guard_block": res.blocked_reason,
                "complexity_label": res.decision.label if res.decision else None,
                "heuristic_score": res.decision.score if res.decision else None,
                "reasons": res.decision.reasons if res.decision else None,
                "tokens (in/out)": f"{res.input_tokens}/{res.output_tokens}",
                "provider": res.provider,
                "notes": res.notes,
            })

# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #
with tab_tel:
    df = telemetry.to_dataframe()
    s = telemetry.summary()
    if s["requests"] == 0:
        st.info("No requests yet — send a few prompts from the Playground.")
    else:
        k = st.columns(3)
        k[0].metric("Requests", s["requests"])
        k[1].metric("Cache hit rate", f"{s['cache_hit_rate']:.1f}%")
        k[2].metric("Total saved", f"${s['cost_saved']:.5f}", f"{s['cost_saved_pct']:.1f}%")
        k = st.columns(3)
        k[0].metric("Avg latency", f"{s['avg_latency_ms']:,.0f} ms")
        k[1].metric("p95 latency", f"{s['p95_latency_ms']:,.0f} ms")
        k[2].metric("Escalations", s["escalations"])
        st.caption(f"Actual spend ${s['total_cost']:.5f} vs. all-strong baseline "
                   f"${s['baseline_cost']:.5f} · escalations: {s['escalations']}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Routing distribution")
            dist = (pd.Series(s["route_counts"]).reindex(BADGE_COLORS.keys()).fillna(0)
                    .rename("requests").to_frame())
            st.bar_chart(dist, color="#2563eb")
        with c2:
            st.subheader("Latency per request (ms)")
            st.line_chart(df.set_index(df.index + 1)["latency_ms"])

        st.subheader("Cumulative cost: actual vs. all-strong baseline ($)")
        cum = pd.DataFrame({"actual": df["actual_cost"].cumsum(),
                            "baseline (100% strong)": df["baseline_cost"].cumsum()})
        cum.index = cum.index + 1
        st.line_chart(cum)

        st.subheader("Recent requests")
        cols = ["timestamp", "query", "route", "model", "similarity", "matched_query",
                "latency_ms", "input_tokens", "output_tokens", "actual_cost", "baseline_cost",
                "cost_saved", "escalated", "blocked_reason"]
        st.dataframe(df[cols].iloc[::-1].head(50), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download full log (CSV)", df.to_csv(index=False),
                           "telemetry.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Cache inspector
# --------------------------------------------------------------------------- #
with tab_cache:
    recs = cache.to_records()
    st.write(f"{len(recs)} cached entries (embedder `{cache.embedder.name}`)")
    if recs:
        st.dataframe(pd.DataFrame(recs).iloc[::-1], use_container_width=True, hide_index=True)
