"""
telemetry.py — per-request SQLite logger + metric aggregation.

Every request gets one row: timestamp, query, matched query, similarity,
route, latency, tokens, actual cost, all-strong baseline cost and savings.
`summary()` turns the log into the dashboard KPIs (hit rate, routing split,
avg/p95 latency, $ and % saved).
"""
from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class RequestRecord:
    timestamp: str
    query: str
    tags: str
    matched_query: str | None
    similarity: float
    route: str                  # CACHE_HIT | CHEAP_MODEL | STRONG_MODEL
    model: str | None
    strategy: str
    complexity_label: str | None
    escalated: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    actual_cost: float
    baseline_cost: float
    cost_saved: float
    threshold: float
    provider: str
    blocked_reason: str | None = None
    note: str | None = None


COLUMNS = list(RequestRecord.__dataclass_fields__.keys())


class TelemetryLogger:
    def __init__(self, db_path: str = "data/telemetry.db"):
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, query TEXT, tags TEXT, matched_query TEXT, similarity REAL,
                route TEXT, model TEXT, strategy TEXT, complexity_label TEXT, escalated INTEGER,
                latency_ms REAL, input_tokens INTEGER, output_tokens INTEGER,
                actual_cost REAL, baseline_cost REAL, cost_saved REAL, threshold REAL,
                provider TEXT, blocked_reason TEXT, note TEXT)"""
        )
        self._conn.commit()

    def log(self, record: RequestRecord) -> None:
        row = asdict(record)
        row["escalated"] = int(row["escalated"])
        with self._lock:
            self._conn.execute(
                f"INSERT INTO requests ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                [row[c] for c in COLUMNS],
            )
            self._conn.commit()

    def to_dataframe(self, limit: int | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM requests ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
        with self._lock:
            df = pd.read_sql_query(sql, self._conn)
        return df.iloc[::-1].reset_index(drop=True)   # chronological order

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM requests")
            self._conn.commit()

    def export_csv(self, path: str) -> str:
        self.to_dataframe().to_csv(path, index=False)
        return path

    def summary(self) -> dict:
        return summarize(self.to_dataframe())


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate KPIs from a telemetry dataframe (also used by eval.py)."""
    n = len(df)
    if n == 0:
        return {"requests": 0, "cache_hit_rate": 0.0, "route_counts": {}, "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0, "total_cost": 0.0, "baseline_cost": 0.0,
                "cost_saved": 0.0, "cost_saved_pct": 0.0, "escalations": 0}
    counts = df["route"].value_counts().to_dict()
    total, base = float(df["actual_cost"].sum()), float(df["baseline_cost"].sum())
    return {
        "requests": n,
        "cache_hit_rate": 100.0 * counts.get("CACHE_HIT", 0) / n,
        "route_counts": counts,
        "avg_latency_ms": float(df["latency_ms"].mean()),
        "p95_latency_ms": float(np.percentile(df["latency_ms"], 95)),
        "total_cost": total,
        "baseline_cost": base,
        "cost_saved": base - total,
        "cost_saved_pct": (100.0 * (base - total) / base) if base > 0 else 0.0,
        "escalations": int(df["escalated"].astype(bool).sum()),
    }
