"""FastAPI wrapper for the Supply Chain GenAI Agent.

Exposes the LangGraph workflow as HTTP endpoints so ERP/TMS systems, dashboards,
or schedulers can drive disruption resolution without invoking the CLI.

Endpoints:
  POST /resolve           — Run an order through the agent; returns either a final
                            resolution or a pending_approval state if HITL is required.
  POST /approve/{thread_id} — Resume a paused workflow with a human decision.
  POST /feedback          — Record actual outcomes for later retraining.
  GET  /health            — Liveness probe.
  GET  /ready             — Readiness probe (verifies model + index artifacts loaded).

Run with:
  uvicorn api:app --host 0.0.0.0 --port 8000

For production:
  - Replace MemorySaver with SqliteSaver or PostgresSaver so paused HITL threads
    survive process restarts.
  - Put a real auth layer (API key header or OAuth) in front of /resolve and /approve.
  - Add LangSmith / Langfuse tracing via env vars.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

# Load .env in local development before any env-var reads.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from supply_chain_genai_agent_groq_hf import (
    _LANGGRAPH_AVAILABLE,
    compile_graph,
    ensure_required_artifacts,
    load_cost_predictor,
    load_disruption_classifier,
    load_hybrid_retriever,
    load_vector_store,
    record_outcome,
)

logger = logging.getLogger("supply_chain_api")

# A single compiled graph is reused across requests. The underlying MemorySaver
# checkpointer keys workflows by thread_id, so concurrent orders are isolated.
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        if not _LANGGRAPH_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LangChain/LangGraph dependencies are not installed.",
            )
        _graph = compile_graph()
    return _graph


# ---------------------------------------------------------------------------
# Lifespan: replaces the deprecated @app.on_event("startup") pattern.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load artifacts on startup so the first request has no cold-start tax."""
    try:
        load_disruption_classifier()
        load_cost_predictor()
        load_vector_store()
        load_hybrid_retriever()
        _get_graph()
        logger.info("Startup warmup completed.")
    except Exception as exc:
        logger.warning("Warmup encountered a non-fatal error: %s", exc)
    yield
    # Nothing to release on shutdown (models are in-process, FAISS is in-memory).


app = FastAPI(
    title="Supply Chain GenAI Agent API",
    description=(
        "HTTP wrapper around the LangGraph disruption-resolution workflow. "
        "Combines XGBoost disruption detection, FAISS RAG, and Groq LLM reasoning "
        "to produce auditable ERP/TMS execution payloads."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class OrderRequest(BaseModel):
    order: dict[str, Any] = Field(..., description="Order payload matching CSV columns.")
    order_id: str | None = Field(None, description="Optional override for the order ID.")
    replay_mode: bool = Field(
        False,
        description=(
            "Treat Disruption_Event as a known historical alert. "
            "Use only when replaying historical CSV rows, never for real-time inference."
        ),
    )

    model_config = {"json_schema_extra": {"example": {
        "order": {
            "Order_ID": "DEMO-001",
            "Route_Type": "Suez",
            "Product_Category": "Pharmaceuticals",
            "Transportation_Mode": "Sea",
            "Geopolitical_Risk_Index": 0.82,
            "Weather_Severity_Index": 7.5,
            "Inflation_Rate_Pct": 5.2,
            "Scheduled_Lead_Time_Days": 24,
            "Base_Lead_Time_Days": 18,
            "Order_Weight_Kg": 1200,
            "Shipping_Cost_USD": 18500,
        },
        "replay_mode": False,
    }}}


class ApprovalRequest(BaseModel):
    approved: bool
    action: str | None = Field(
        None,
        description="Action chosen by the human reviewer. Defaults to the recommended action.",
    )
    reviewer: str | None = Field(None, description="Reviewer identifier for audit logs.")


class FeedbackRequest(BaseModel):
    order_id: str
    recommended_action: str
    actual_action: str
    actual_delay_days: float
    actual_cost_usd: float
    predicted_cost_usd: float | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Observability"])
async def health() -> dict[str, str]:
    """Liveness probe — returns 200 as long as the process is alive."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/ready", tags=["Observability"])
async def ready() -> dict[str, Any]:
    """Readiness probe — reports which artifacts are loaded."""
    report = ensure_required_artifacts()
    is_ready = (
        report["disruption_classifier"]
        and report["cost_predictor"]
        and report["langgraph_langchain_installed"]
    )
    return {"ready": is_ready, "artifacts": report}


# ---------------------------------------------------------------------------
# Core resolution endpoints
# ---------------------------------------------------------------------------
@app.post("/resolve", tags=["Workflow"])
async def resolve(req: OrderRequest) -> dict[str, Any]:
    """Run an order through the disruption-resolution workflow.

    Returns one of two shapes:

    - **resolved**: workflow completed; `result` contains the full state including
      `execution_payload`, `audit_log`, and `reasoning_trace`.
    - **pending_approval**: workflow paused at the HITL node; call
      `POST /approve/{thread_id}` to resume.
    """
    graph = _get_graph()

    order_id = str(
        req.order_id
        or req.order.get("Order_ID")
        or f"ORDER-{int(datetime.now(timezone.utc).timestamp())}"
    )
    config = {"configurable": {"thread_id": order_id}}

    # replay_mode is passed through state so concurrent requests don't race on
    # a shared os.environ key (previous approach was not thread-safe).
    initial_state = {
        "order": req.order,
        "order_id": order_id,
        "replay_mode": req.replay_mode,
        "audit_log": [],
        "reasoning_trace": [],
    }

    try:
        result = graph.invoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("Resolve failed for order %s", order_id)
        raise HTTPException(status_code=500, detail=f"Workflow error: {exc}") from exc

    # When LangGraph hits an interrupt() the graph state is saved and invoke() returns
    # without a final_decision. Surface that to the caller as a pending state.
    if result.get("requires_human") and not result.get("final_decision"):
        return {
            "status": "pending_approval",
            "thread_id": order_id,
            "order_id": order_id,
            "disruption_detected": result.get("disruption_detected"),
            "disruption_type": result.get("disruption_type"),
            "severity": (result.get("risk_assessment") or {}).get("severity"),
            "recommended_action": result.get("recommended_action"),
            "confidence": result.get("confidence"),
            "cost_options": (result.get("cost_analysis") or {}).get("options", []),
            "reasoning_trace": result.get("reasoning_trace", []),
        }

    return {"status": "resolved", "thread_id": order_id, "result": result}


@app.post("/approve/{thread_id}", tags=["Workflow"])
async def approve(thread_id: str, req: ApprovalRequest) -> dict[str, Any]:
    """Resume a paused HITL workflow with a human decision.

    `thread_id` must match the value returned by a prior `pending_approval` response.
    """
    from langgraph.types import Command

    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    resume_payload = {
        "approved": req.approved,
        "action": req.action,
        "reviewer": req.reviewer,
    }

    try:
        result = graph.invoke(Command(resume=resume_payload), config=config)
    except Exception as exc:
        logger.exception("Approval failed for thread %s", thread_id)
        raise HTTPException(status_code=500, detail=f"Resume error: {exc}") from exc

    return {"status": "resolved", "thread_id": thread_id, "result": result}


@app.post("/feedback", tags=["Retraining"])
async def feedback(req: FeedbackRequest) -> dict[str, Any]:
    """Persist an actual outcome for later model retraining."""
    path = record_outcome(
        order_id=req.order_id,
        recommended_action=req.recommended_action,
        actual_action=req.actual_action,
        actual_delay_days=req.actual_delay_days,
        actual_cost_usd=req.actual_cost_usd,
        predicted_cost_usd=req.predicted_cost_usd,
        notes=req.notes,
    )
    return {"status": "recorded", "path": str(path)}
