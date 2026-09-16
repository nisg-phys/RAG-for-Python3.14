from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel


OBSERVABILITY_DIR = Path(os.getenv("RAGBOT_OBSERVABILITY_DIR", "storage/observability"))
EVENTS_PATH = OBSERVABILITY_DIR / "events.jsonl"


def _ensure_storage() -> None:
    OBSERVABILITY_DIR.mkdir(parents=True, exist_ok=True)


def _serialize(payload: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, BaseModel):
        return payload.model_dump()
    return payload


def record_event(event_type: str, payload: BaseModel | dict[str, Any]) -> None:
    _ensure_storage()
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "payload": _serialize(payload),
    }
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def load_events() -> list[dict[str, Any]]:
    if not EVENTS_PATH.exists():
        return []

    events = []
    with EVENTS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil((percentile / 100) * len(ordered)) - 1
    index = max(0, min(index, len(ordered) - 1))
    return round(ordered[index], 4)


def build_summary() -> dict[str, Any]:
    events = load_events()
    request_payloads = [event["payload"] for event in events if event["event_type"] == "rag_query"]
    ingestion_payloads = [event["payload"] for event in events if event["event_type"] == "ingestion"]
    evaluation_payloads = [event["payload"] for event in events if event["event_type"] == "evaluation"]

    total_latencies = [float(payload.get("total_latency_ms", 0.0)) for payload in request_payloads]
    retrieval_latencies = [
        float(payload.get("retrieval", {}).get("retrieval_latency_ms", 0.0))
        for payload in request_payloads
    ]
    generation_latencies = [
        float(payload.get("generation", {}).get("generation_latency_ms", 0.0))
        for payload in request_payloads
    ]
    returned_chunks = [
        int(payload.get("retrieval", {}).get("returned_chunks", 0))
        for payload in request_payloads
    ]

    return {
        "request_metrics": {
            "total_requests": len(request_payloads),
            "avg_total_latency_ms": _average(total_latencies),
            "p95_total_latency_ms": _percentile(total_latencies, 95),
            "avg_retrieval_latency_ms": _average(retrieval_latencies),
            "avg_generation_latency_ms": _average(generation_latencies),
            "avg_returned_chunks": _average([float(value) for value in returned_chunks]),
        },
        "latest_ingestion": ingestion_payloads[-1] if ingestion_payloads else None,
        "latest_evaluation": evaluation_payloads[-1] if evaluation_payloads else None,
    }
