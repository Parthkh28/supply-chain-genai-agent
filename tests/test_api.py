"""FastAPI endpoint tests using TestClient (no real LLM or model calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_artifacts(monkeypatch):
    """Patch all heavy imports and model loads so tests run without GPU/models."""
    monkeypatch.setattr(
        "supply_chain_genai_agent_groq_hf._LANGGRAPH_AVAILABLE", True
    )

    fake_classifier = MagicMock()
    fake_classifier.predict_proba.return_value = [[0.8, 0.2]]

    fake_cost_model = MagicMock()
    fake_cost_model.predict.return_value = [12000.0]

    with (
        patch("supply_chain_genai_agent_groq_hf.load_disruption_classifier", return_value=fake_classifier),
        patch("supply_chain_genai_agent_groq_hf.load_cost_predictor", return_value=fake_cost_model),
        patch("supply_chain_genai_agent_groq_hf.load_vector_store", return_value=MagicMock()),
        patch("supply_chain_genai_agent_groq_hf.load_hybrid_retriever", return_value=MagicMock()),
        patch("api.load_disruption_classifier", return_value=fake_classifier),
        patch("api.load_cost_predictor", return_value=fake_cost_model),
        patch("api.load_vector_store", return_value=MagicMock()),
        patch("api.load_hybrid_retriever", return_value=MagicMock()),
        patch("api._get_graph", return_value=MagicMock()),
        patch("api.ensure_required_artifacts", return_value={
            "disruption_classifier": True,
            "cost_predictor": True,
            "langgraph_langchain_installed": True,
            "faiss_index": True,
        }),
    ):
        yield


@pytest.fixture()
def client(mock_artifacts):
    from api import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_has_status_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_has_timestamp(self, client):
        body = client.get("/health").json()
        assert "timestamp" in body


# ---------------------------------------------------------------------------
# /ready
# ---------------------------------------------------------------------------
class TestReady:
    def test_returns_200(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200

    def test_ready_is_true_when_all_artifacts_present(self, client):
        body = client.get("/ready").json()
        assert body["ready"] is True

    def test_artifacts_key_present(self, client):
        body = client.get("/ready").json()
        assert "artifacts" in body


# ---------------------------------------------------------------------------
# /resolve — request validation
# ---------------------------------------------------------------------------
DEMO_ORDER = {
    "Order_ID": "TEST-001",
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
}


class TestResolveValidation:
    def test_missing_order_field_returns_422(self, client):
        resp = client.post("/resolve", json={})
        assert resp.status_code == 422

    def test_empty_order_dict_accepted(self, client, mock_artifacts):
        """An empty order dict is valid — the agent has defaults for every feature."""
        with patch("api._get_graph") as mock_graph:
            mock_graph.return_value.invoke.return_value = {
                "disruption_detected": False,
                "requires_human": False,
                "final_decision": "No Action",
            }
            resp = client.post("/resolve", json={"order": {}})
        assert resp.status_code == 200

    def test_replay_mode_false_by_default(self, client, mock_artifacts):
        with patch("api._get_graph") as mock_graph:
            captured = {}

            def fake_invoke(state, config=None):
                captured["replay_mode"] = state.get("replay_mode")
                return {"disruption_detected": False, "requires_human": False, "final_decision": "No Action"}

            mock_graph.return_value.invoke.side_effect = fake_invoke
            client.post("/resolve", json={"order": DEMO_ORDER})
        assert captured.get("replay_mode") is False

    def test_replay_mode_passed_through_state(self, client, mock_artifacts):
        with patch("api._get_graph") as mock_graph:
            captured = {}

            def fake_invoke(state, config=None):
                captured["replay_mode"] = state.get("replay_mode")
                return {"disruption_detected": False, "requires_human": False, "final_decision": "No Action"}

            mock_graph.return_value.invoke.side_effect = fake_invoke
            client.post("/resolve", json={"order": DEMO_ORDER, "replay_mode": True})
        assert captured.get("replay_mode") is True

    def test_resolved_status_in_response(self, client, mock_artifacts):
        with patch("api._get_graph") as mock_graph:
            mock_graph.return_value.invoke.return_value = {
                "disruption_detected": True,
                "requires_human": False,
                "final_decision": "Re-routing",
            }
            resp = client.post("/resolve", json={"order": DEMO_ORDER})
        assert resp.status_code == 200
        assert resp.json()["status"] == "resolved"

    def test_pending_approval_when_hitl_required(self, client, mock_artifacts):
        with patch("api._get_graph") as mock_graph:
            mock_graph.return_value.invoke.return_value = {
                "disruption_detected": True,
                "requires_human": True,
                "final_decision": None,
                "recommended_action": "Re-routing",
                "confidence": 0.45,
                "risk_assessment": {"severity": "HIGH"},
                "disruption_type": "Port Congestion",
                "cost_analysis": {"options": []},
                "reasoning_trace": [],
            }
            resp = client.post("/resolve", json={"order": DEMO_ORDER})
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending_approval"
        assert "thread_id" in resp.json()


# ---------------------------------------------------------------------------
# /feedback
# ---------------------------------------------------------------------------
class TestFeedback:
    def test_missing_required_fields_returns_422(self, client):
        resp = client.post("/feedback", json={})
        assert resp.status_code == 422

    def test_valid_feedback_recorded(self, client, mock_artifacts):
        payload = {
            "order_id": "TEST-001",
            "recommended_action": "Re-routing",
            "actual_action": "Re-routing",
            "actual_delay_days": 3.0,
            "actual_cost_usd": 17500.0,
            "predicted_cost_usd": 18200.0,
            "notes": "matched",
        }
        with patch("api.record_outcome", return_value="/tmp/outcomes.jsonl"):
            resp = client.post("/feedback", json=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"
