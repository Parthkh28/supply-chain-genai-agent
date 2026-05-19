#!/usr/bin/env python3
"""
Supply Chain GenAI Agent - inference/RAG/LangGraph only.

This file is intentionally separated from the Kaggle model-training notebook.
It does NOT train XGBoost models. It only:
  1. loads trained ML artifacts produced by supply_chain_kaggle_model_training.ipynb,
  2. builds/loads the RAG vector index from the generated knowledge-base text,
  3. runs the LangGraph multi-agent workflow for disruption resolution,
  4. produces an execution payload/audit trail.

Expected artifact layout, either next to this file or configured by environment variables:

  supply_chain_agent_training_outputs/
    models/
      disruption_classifier.pkl
      cost_predictor.pkl
      feature_config.json
    knowledge_base/
      mitigation_playbook.txt
      faiss_index/              # created by: python supply_chain_genai_agent.py index

Optional environment variables:
  SUPPLY_CHAIN_MODELS_DIR       - directory containing model .pkl files and feature_config.json
  SUPPLY_CHAIN_KB_DIR           - directory containing mitigation_playbook.txt/faiss_index
  SUPPLY_CHAIN_DATA_PATH        - CSV path used only for local demo resolve mode
  GROQ_MODEL                    - defaults to llama3-70b-8192
  GROQ_API_KEY                  - required for Groq LLM calls
  SUPPLY_CHAIN_REPLAY_MODE      - set to 1 only when replaying historical CSV rows
"""

from __future__ import annotations

import argparse
import json
import logging
import operator
import os
from difflib import get_close_matches
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Skip HuggingFace Hub network checks when the model is already cached locally.
# On a fresh machine the model downloads once; every run after that is instant.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
    from langgraph.types import interrupt

    _LANGGRAPH_AVAILABLE = True
    _LANGGRAPH_IMPORT_ERROR: str | None = None
except ImportError as _exc:
    _LANGGRAPH_AVAILABLE = False
    _LANGGRAPH_IMPORT_ERROR = repr(_exc)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("supply_chain_genai_agent")

if not _LANGGRAPH_AVAILABLE:
    logger.warning(
        "LangChain/LangGraph optional imports failed: %s. "
        "FAISS indexing and the LangGraph workflow will be unavailable until the missing package is installed.",
        _LANGGRAPH_IMPORT_ERROR,
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent


def _first_existing_dir(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_OUTPUT_DIR = ROOT / "supply_chain_agent_training_outputs"
MODELS_DIR = Path(
    os.getenv(
        "SUPPLY_CHAIN_MODELS_DIR",
        str(_first_existing_dir(DEFAULT_OUTPUT_DIR / "models", ROOT / "models")),
    )
)
KB_DIR = Path(
    os.getenv(
        "SUPPLY_CHAIN_KB_DIR",
        str(_first_existing_dir(DEFAULT_OUTPUT_DIR / "knowledge_base", ROOT / "knowledge_base")),
    )
)
DATA_PATH = Path(
    os.getenv("SUPPLY_CHAIN_DATA_PATH", str(ROOT / "global_supply_chain_disruption_v1.csv"))
)


# ---------------------------------------------------------------------------
# Default config fallback. In normal use, feature_config.json from the Kaggle
# notebook should override these values.
# ---------------------------------------------------------------------------
DEFAULT_DISRUPTION_PROB_THRESHOLD = 0.35

DEFAULT_ROUTE_RISK_MAP: dict[str, float] = {
    "Suez": 0.91,
    "Pacific": 0.74,
    "Atlantic": 0.58,
    "Intra-Asia": 0.45,
    "Commodity": 0.28,
}

DEFAULT_PRODUCT_CRITICALITY_MAP: dict[str, float] = {
    "Pharmaceuticals": 1.00,
    "Perishable Foods": 0.95,
    "Semiconductors": 0.70,
    "Consumer Electronics": 0.55,
    "Auto Parts": 0.50,
    "Raw Materials": 0.25,
    "Textiles": 0.15,
}

DISRUPTION_LABEL_MAP: dict[str, int] = {
    "No Disruption": 0,
    "Port Congestion": 1,
    "Geopolitical Conflict (Route Diversion)": 2,
    "Severe Weather (Typhoon/Storm)": 3,
}

NO_DISRUPTION_VALUES = {
    "",
    "none",
    "nan",
    "no disruption",
    "no_disruption",
    "no-disruption",
    "null",
}

# Production-safe default: only pre-event features. If feature_config.json is present,
# the active training feature list from the notebook overrides this fallback.
LEAKAGE_PRONE_FEATURES = {"delay_ratio", "Shipping_Cost_USD", "log_shipping_cost", "cost_percentile"}

DEFAULT_CLASSIFIER_FEATURES: list[str] = [
    "Geopolitical_Risk_Index",
    "Weather_Severity_Index",
    "Inflation_Rate_Pct",
    "Scheduled_Lead_Time_Days",
    "lead_time_buffer",
    "composite_risk_score",
    "geo_weather_interaction",
    "route_risk_score",
    "product_criticality",
    "mode_Sea",
    "mode_Air",
    "route_Suez",
    "route_Pacific",
    "route_Atlantic",
    "route_Intra-Asia",
    "route_Commodity",
    "product_Pharmaceuticals",
    "product_Perishable Foods",
    "product_Semiconductors",
    "product_Auto Parts",
]

DEFAULT_COST_MODEL_FEATURES: list[str] = [
    "Geopolitical_Risk_Index",
    "Weather_Severity_Index",
    "Scheduled_Lead_Time_Days",
    "route_risk_score",
    "product_criticality",
    "mode_Sea",
    "mode_Air",
    "geo_weather_interaction",
    "route_Suez",
    "route_Pacific",
    "route_Atlantic",
]

DEFAULT_ENHANCED_COST_MODEL_FEATURES: list[str] = [
    "Geopolitical_Risk_Index",
    "Weather_Severity_Index",
    "Scheduled_Lead_Time_Days",
    "Base_Lead_Time_Days",
    "lead_time_buffer",
    "Order_Weight_Kg",
    "route_risk_score",
    "product_criticality",
    "geo_weather_interaction",
    "mode_Sea",
    "mode_Air",
    "route_Suez",
    "route_Pacific",
    "route_Atlantic",
    "route_Intra-Asia",
    "route_Commodity",
    "product_Pharmaceuticals",
    "product_Perishable Foods",
    "product_Semiconductors",
    "product_Auto Parts",
    "product_Consumer Electronics",
    "product_Raw Materials",
    "product_Textiles",
]

COST_MULTIPLIERS: dict[str, float] = {
    "Expedited Air Freight": 12.5,
    "Re-routing": 1.6,
    "Standard Shipping": 1.0,
    "Delay Accepted": 1.0,
}
VALID_ACTION_NAMES = tuple(COST_MULTIPLIERS.keys())
ACTION_ALIASES: dict[str, str] = {
    "air freight": "Expedited Air Freight",
    "airfreight": "Expedited Air Freight",
    "expedite": "Expedited Air Freight",
    "expedited": "Expedited Air Freight",
    "reroute": "Re-routing",
    "rerouting": "Re-routing",
    "re-route": "Re-routing",
    "partial re-routing": "Re-routing",
    "alternative route": "Re-routing",
    "standard": "Standard Shipping",
    "normal shipping": "Standard Shipping",
    "continue": "Standard Shipping",
    "accept delay": "Delay Accepted",
    "delay accepted": "Delay Accepted",
    "wait": "Delay Accepted",
}

CONFIDENCE_THRESHOLD = 0.60
COST_DELTA_THRESHOLD = 50_000.0
UTILITY_ALPHA = 0.60
UTILITY_BETA = 0.40


# ---------------------------------------------------------------------------
# Load feature config generated by the training notebook.
# ---------------------------------------------------------------------------
def load_feature_config(models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    """Load feature_config.json produced by the Kaggle training notebook."""
    path = models_dir / "feature_config.json"
    if not path.exists():
        logger.warning(
            "feature_config.json not found at %s. Using built-in fallback feature lists.",
            path,
        )
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


FEATURE_CONFIG = load_feature_config()

DISRUPTION_PROB_THRESHOLD: float = float(
    FEATURE_CONFIG.get("disruption_probability_threshold", DEFAULT_DISRUPTION_PROB_THRESHOLD)
)
ROUTE_RISK_MAP: dict[str, float] = dict(
    FEATURE_CONFIG.get("route_risk_map", DEFAULT_ROUTE_RISK_MAP)
)
PRODUCT_CRITICALITY_MAP: dict[str, float] = dict(
    FEATURE_CONFIG.get("product_criticality_map", DEFAULT_PRODUCT_CRITICALITY_MAP)
)
ROUTE_COST_REFERENCE_STATS: dict[str, dict[str, float]] = dict(
    FEATURE_CONFIG.get("route_cost_reference_stats", {})
)
CLASSIFIER_FEATURES: list[str] = list(
    FEATURE_CONFIG.get("classifier_features_active", DEFAULT_CLASSIFIER_FEATURES)
)

_leaky_active_features = sorted(set(CLASSIFIER_FEATURES) & LEAKAGE_PRONE_FEATURES)
if _leaky_active_features:
    logger.warning(
        "Active classifier features include likely post-event/leakage fields: %s. "
        "Retrain with USE_LEAKAGE_SAFE_CLASSIFIER=True for real-time inference.",
        _leaky_active_features,
    )

# Default to the agent-compatible cost model. To use the enhanced model, set:
#   SUPPLY_CHAIN_COST_MODEL_FILENAME=cost_predictor_enhanced.pkl
#   SUPPLY_CHAIN_COST_FEATURE_KEY=cost_model_features_enhanced
COST_MODEL_FILENAME = os.getenv("SUPPLY_CHAIN_COST_MODEL_FILENAME", "cost_predictor.pkl")
COST_FEATURE_KEY = os.getenv("SUPPLY_CHAIN_COST_FEATURE_KEY", "cost_model_features_agent_compatible")
COST_MODEL_FEATURES: list[str] = list(
    FEATURE_CONFIG.get(COST_FEATURE_KEY, DEFAULT_COST_MODEL_FEATURES)
)
if COST_FEATURE_KEY == "cost_model_features_enhanced" and not FEATURE_CONFIG.get(COST_FEATURE_KEY):
    COST_MODEL_FEATURES = DEFAULT_ENHANCED_COST_MODEL_FEATURES

ALL_EXPECTED_FEATURES = sorted(
    set(CLASSIFIER_FEATURES)
    | set(DEFAULT_CLASSIFIER_FEATURES)
    | set(COST_MODEL_FEATURES)
    | set(DEFAULT_COST_MODEL_FEATURES)
    | set(DEFAULT_ENHANCED_COST_MODEL_FEATURES)
)


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output.
# ---------------------------------------------------------------------------
class RiskAssessmentSchema(BaseModel):
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    reasoning: str = Field(..., max_length=700)
    key_risk_factor: str = Field(..., max_length=250)


class MitigationOptionSchema(BaseModel):
    action: str
    description: str
    estimated_delay_reduction_days: float = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class MitigationPlanSchema(BaseModel):
    options: list[MitigationOptionSchema] = Field(..., min_length=1, max_length=5)


class CostOptionResult(BaseModel):
    action: str
    estimated_total_cost: float
    cost_delta: float
    confidence: float
    utility: float = 0.0
    requires_hitl: bool = False


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _audit(agent: str, order_id: str, msg: str, data: dict | None = None) -> dict:
    return {
        "timestamp": _ts(),
        "agent": agent,
        "order_id": order_id,
        "message": msg,
        "data": data or {},
    }


def _to_python(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to native Python types for msgpack serialization."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _safe(d: dict, key: str, default: float = 0.0) -> float:
    value = d.get(key, default)
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_disruption_event(value: Any) -> Any:
    """Normalize no-disruption placeholders to NaN and keep canonical labels."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in NO_DISRUPTION_VALUES:
        return np.nan
    canonical = {key.lower(): key for key in DISRUPTION_LABEL_MAP}
    return canonical.get(text.lower(), text)


def has_disruption_event(value: Any) -> bool:
    return not pd.isna(normalize_disruption_event(value))


def is_replay_mode_enabled() -> bool:
    """Return True only when historical labels should be treated as known alerts."""
    return os.getenv("SUPPLY_CHAIN_REPLAY_MODE", "0").strip().lower() in {"1", "true", "yes", "y"}


def normalize_action_name(action: Any, *, fallback: str = "Standard Shipping") -> str:
    """Normalize LLM action text to one of COST_MULTIPLIERS keys."""
    raw = "" if action is None else str(action).strip()
    if not raw:
        logger.warning("Empty mitigation action received. Falling back to %s.", fallback)
        return fallback

    raw_lower = raw.lower()
    for valid in VALID_ACTION_NAMES:
        if raw_lower == valid.lower():
            return valid

    for alias, valid in ACTION_ALIASES.items():
        if alias in raw_lower:
            logger.info("Normalized mitigation action %r -> %r", raw, valid)
            return valid

    close = get_close_matches(raw, VALID_ACTION_NAMES, n=1, cutoff=0.65)
    if close:
        logger.info("Fuzzy-normalized mitigation action %r -> %r", raw, close[0])
        return close[0]

    logger.warning(
        "Unrecognized mitigation action %r. Expected one of %s. Falling back to %s.",
        raw,
        list(VALID_ACTION_NAMES),
        fallback,
    )
    return fallback


def action_multiplier(action: Any) -> tuple[str, float]:
    """Return normalized action and its cost multiplier."""
    normalized = normalize_action_name(action)
    return normalized, COST_MULTIPLIERS[normalized]


def estimate_route_cost_percentile(order: dict[str, Any]) -> float:
    """Estimate a route-level shipping-cost percentile from training reference stats.

    The training notebook saves route-level cost quantiles in feature_config.json.
    This is only an approximation, but it is better than using a static 0.5 for every
    real-time order. If stats are unavailable, returns 0.5.
    """
    if "cost_percentile" in order and not pd.isna(order.get("cost_percentile")):
        return _safe(order, "cost_percentile", 0.5)

    shipping_cost = _safe(order, "Shipping_Cost_USD", np.nan)
    if pd.isna(shipping_cost):
        return 0.5

    route = str(order.get("Route_Type", ""))
    stats = ROUTE_COST_REFERENCE_STATS.get(route) or ROUTE_COST_REFERENCE_STATS.get("__global__")
    if not stats:
        return 0.5

    # Piecewise-linear interpolation across saved quantiles.
    points: list[tuple[float, float]] = []
    for pct_key, q in [("min", 0.0), ("p10", 0.10), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("max", 1.0)]:
        if pct_key in stats:
            try:
                points.append((float(stats[pct_key]), q))
            except (TypeError, ValueError):
                pass
    points = sorted(set(points), key=lambda item: item[0])
    if len(points) < 2:
        return 0.5

    if shipping_cost <= points[0][0]:
        return points[0][1]
    if shipping_cost >= points[-1][0]:
        return points[-1][1]

    for (cost_lo, pct_lo), (cost_hi, pct_hi) in zip(points, points[1:]):
        if cost_lo <= shipping_cost <= cost_hi:
            if cost_hi == cost_lo:
                return pct_hi
            frac = (shipping_cost - cost_lo) / (cost_hi - cost_lo)
            return float(pct_lo + frac * (pct_hi - pct_lo))

    return 0.5


def ensure_required_artifacts() -> dict[str, bool]:
    """Return a simple artifact availability report."""
    report = {
        "models_dir_exists": MODELS_DIR.exists(),
        "disruption_classifier": (MODELS_DIR / "disruption_classifier.pkl").exists(),
        "cost_predictor": (MODELS_DIR / COST_MODEL_FILENAME).exists(),
        "feature_config": (MODELS_DIR / "feature_config.json").exists(),
        "kb_dir_exists": KB_DIR.exists(),
        "mitigation_playbook": (KB_DIR / "mitigation_playbook.txt").exists(),
        "faiss_index": (KB_DIR / "faiss_index").exists(),
        "langgraph_langchain_installed": _LANGGRAPH_AVAILABLE,
        "replay_mode_enabled": is_replay_mode_enabled(),
    }
    return report


def print_artifact_report() -> None:
    print("Artifact check")
    print("-" * 60)
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"KB_DIR    : {KB_DIR}")
    print(f"DATA_PATH : {DATA_PATH}")
    for key, value in ensure_required_artifacts().items():
        print(f"{key:32s}: {value}")


# ---------------------------------------------------------------------------
# Inference-only feature engineering.
# Mirrors the training notebook, but does not train any model.
# ---------------------------------------------------------------------------
def enrich_order(raw: dict[str, Any]) -> dict[str, Any]:
    """Add engineered fields and one-hot columns required by trained models."""
    order = dict(raw)

    event = order.get("Disruption_Event")
    order["Disruption_Event"] = normalize_disruption_event(event)

    scheduled_lead_time = _safe(order, "Scheduled_Lead_Time_Days")
    base_lead_time = _safe(order, "Base_Lead_Time_Days")
    delay_days = _safe(order, "Delay_Days")
    geo = _safe(order, "Geopolitical_Risk_Index")
    weather = _safe(order, "Weather_Severity_Index")
    inflation = _safe(order, "Inflation_Rate_Pct")
    shipping_cost = _safe(order, "Shipping_Cost_USD")

    order["lead_time_buffer"] = scheduled_lead_time - base_lead_time
    order["delay_ratio"] = delay_days / (scheduled_lead_time + 1e-6)
    order["geo_weather_interaction"] = geo * weather
    order["composite_risk_score"] = (
        geo * 0.50 + (weather / 10.0) * 0.30 + (inflation / 10.0) * 0.20
    )
    order["route_risk_score"] = ROUTE_RISK_MAP.get(order.get("Route_Type", ""), 0.5)
    order["product_criticality"] = PRODUCT_CRITICALITY_MAP.get(
        order.get("Product_Category", ""), 0.5
    )
    order["log_shipping_cost"] = np.log1p(shipping_cost)

    # Exact percentile is dataset-level. For real-time inference, estimate it from
    # route-level reference stats saved by the training notebook when available.
    order["cost_percentile"] = estimate_route_cost_percentile(order)
    order["air_viable"] = int(order["product_criticality"] >= 0.5 and delay_days >= 5)

    mode = str(order.get("Transportation_Mode", ""))
    order["mode_Sea"] = int(mode == "Sea")
    order["mode_Air"] = int(mode == "Air")

    route = str(order.get("Route_Type", ""))
    for route_name in ROUTE_RISK_MAP:
        order[f"route_{route_name}"] = int(route == route_name)

    product = str(order.get("Product_Category", ""))
    for product_name in PRODUCT_CRITICALITY_MAP:
        order[f"product_{product_name}"] = int(product == product_name)

    for feature in ALL_EXPECTED_FEATURES:
        order.setdefault(feature, 0)

    return _to_python(order)

def infer_disruption_type(order: dict[str, Any]) -> str:
    weather = _safe(order, "Weather_Severity_Index")
    geo = _safe(order, "Geopolitical_Risk_Index")
    route = str(order.get("Route_Type", ""))
    if weather >= 7:
        return "Severe Weather (Typhoon/Storm)"
    if geo >= 0.65 or route == "Suez":
        return "Geopolitical Conflict (Route Diversion)"
    return "Port Congestion"

def _feature_frame(order: dict[str, Any], features: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{feature: _safe(order, feature) for feature in features}], columns=features)


def classifier_feature_frame(order: dict[str, Any]) -> pd.DataFrame:
    return _feature_frame(order, CLASSIFIER_FEATURES)


def cost_feature_frame(order: dict[str, Any], action: str = "") -> pd.DataFrame:
    order_for_cost = dict(order)
    normalized_action = normalize_action_name(action, fallback="Standard Shipping")
    action_lower = normalized_action.lower()
    is_air = int("air" in action_lower or "expedite" in action_lower)
    if "delay accepted" in action_lower or "standard shipping" in action_lower:
        is_air = int(order_for_cost.get("Transportation_Mode") == "Air")

    order_for_cost["mode_Air"] = is_air
    order_for_cost["mode_Sea"] = 1 - is_air
    return _feature_frame(order_for_cost, COST_MODEL_FEATURES)


# ---------------------------------------------------------------------------
# Trained artifact loading. These functions intentionally load only; no fit().
# ---------------------------------------------------------------------------
def _require_file(path: Path, purpose: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {purpose}: {path}\n"
            "Run supply_chain_kaggle_model_training.ipynb first, then place/copy "
            "the output artifacts next to this script or set SUPPLY_CHAIN_MODELS_DIR "
            "and SUPPLY_CHAIN_KB_DIR."
        )
    return path


@lru_cache(maxsize=1)
def load_disruption_classifier():
    path = _require_file(MODELS_DIR / "disruption_classifier.pkl", "disruption classifier")
    logger.info("Loading disruption classifier from %s", path)
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_cost_predictor():
    path = _require_file(MODELS_DIR / COST_MODEL_FILENAME, "cost predictor")
    logger.info("Loading cost predictor from %s", path)
    return joblib.load(path)


# ---------------------------------------------------------------------------
# RAG utilities. The Kaggle notebook writes mitigation_playbook.txt. This file
# builds/loads FAISS for retrieval, but does not build ML models.
# ---------------------------------------------------------------------------
def load_playbook_documents(playbook_path: Path | None = None) -> list[str]:
    path = playbook_path or (KB_DIR / "mitigation_playbook.txt")
    path = _require_file(path, "knowledge-base playbook")
    text = path.read_text(encoding="utf-8")
    docs = [chunk.strip() for chunk in text.split("\n\n---\n\n") if chunk.strip()]
    if not docs:
        raise ValueError(f"No documents found in playbook: {path}")
    return docs


def build_faiss_index(
    playbook_path: Path | None = None,
    index_dir: Path | None = None,
) -> None:
    """Create a FAISS index from mitigation_playbook.txt using local HuggingFace embeddings."""
    if not _LANGGRAPH_AVAILABLE:
        raise RuntimeError(
            "LangChain/LangGraph dependencies are not installed. Install langchain-groq, langchain-huggingface, sentence-transformers, langgraph, and faiss-cpu before building FAISS."
        )

    docs = load_playbook_documents(playbook_path)
    index_dir = index_dir or (KB_DIR / "faiss_index")
    index_dir.parent.mkdir(parents=True, exist_ok=True)

    lc_docs = [
        Document(page_content=text, metadata={"source": text.splitlines()[0].strip()})
        for text in docs
    ]
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(lc_docs, embeddings)
    vector_store.save_local(str(index_dir))
    logger.info("Saved FAISS index with %d documents to %s", len(lc_docs), index_dir)


@lru_cache(maxsize=1)
def load_vector_store():
    if not _LANGGRAPH_AVAILABLE:
        return None
    index_dir = KB_DIR / "faiss_index"
    if not index_dir.exists():
        logger.warning(
            "FAISS index not found at %s. Falling back to simple keyword retrieval.",
            index_dir,
        )
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve RAG context. Uses FAISS when available, otherwise simple text scoring."""
    vector_store = load_vector_store()
    if vector_store is not None:
        docs = vector_store.similarity_search(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs)

    # Fallback for local demos before the FAISS index is built.
    try:
        docs = load_playbook_documents()
    except FileNotFoundError:
        return "No historical playbook context available."

    query_terms = {term.lower() for term in query.replace("/", " ").split() if len(term) > 2}
    scored: list[tuple[int, str]] = []
    for doc in docs:
        lower_doc = doc.lower()
        score = sum(1 for term in query_terms if term in lower_doc)
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return "\n\n".join(doc for score, doc in scored[:k] if score > 0) or "\n\n".join(
        doc for _, doc in scored[:k]
    )


def get_llm() -> ChatGroq:
    if not _LANGGRAPH_AVAILABLE:
        raise RuntimeError("LangChain/LangGraph dependencies are not installed.")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ---------------------------------------------------------------------------
# LangGraph agent state and nodes.
# ---------------------------------------------------------------------------
if _LANGGRAPH_AVAILABLE:
    from typing import TypedDict

    class SupplyChainState(TypedDict, total=False):
        order: dict
        order_id: str

        disruption_detected: bool
        disruption_probability: float
        disruption_type: str

        risk_assessment: dict
        risk_score: float

        mitigation_options: list[dict]

        cost_analysis: dict
        recommended_action: str
        confidence: float

        requires_human: bool
        human_approved: bool
        final_decision: str

        execution_payload: dict

        audit_log: Annotated[list[dict], operator.add]
        reasoning_trace: Annotated[list[str], operator.add]


    def detect_exception(state: SupplyChainState) -> dict:
        """Use the pre-trained classifier plus an optional historical replay override.

        In real-time mode, `Disruption_Event` is treated as a label-like field and is
        ignored for detection. Set SUPPLY_CHAIN_REPLAY_MODE=1 only when replaying
        historical CSV rows where a known event should force the workflow path.
        """
        order_id = state["order_id"]
        order = enrich_order(state["order"])

        classifier = load_disruption_classifier()
        features = classifier_feature_frame(order)
        probability = float(classifier.predict_proba(features)[0][1])

        raw_event = state["order"].get("Disruption_Event")
        replay_mode = is_replay_mode_enabled()
        rule_hit = has_disruption_event(raw_event) if replay_mode else False
        normalized_event = normalize_disruption_event(raw_event)
        if rule_hit:
            probability = max(probability, 0.85)

        disrupted = probability >= DISRUPTION_PROB_THRESHOLD or rule_hit
        if disrupted and rule_hit:
            disruption_type = str(normalized_event)
        elif disrupted:
            disruption_type = infer_disruption_type(order)
        else:
            disruption_type = "None"
        

        logger.info(
            "[DETECTOR] %s probability=%.3f rule_hit=%s replay_mode=%s result=%s",
            order_id,
            probability,
            rule_hit,
            replay_mode,
            "DISRUPTED" if disrupted else "NORMAL",
        )
        return _to_python({
            "order": order,
            "disruption_detected": bool(disrupted),
            "disruption_probability": round(float(probability), 4),
            "disruption_type": disruption_type,
            "audit_log": [
                _audit(
                    "EXCEPTION_DETECTOR",
                    order_id,
                    f"{'DISRUPTION DETECTED' if disrupted else 'No disruption'} "
                    f"(prob={probability:.3f}, rule={rule_hit}, replay={replay_mode})",
                    {
                        "probability": float(probability),
                        "rule_triggered": bool(rule_hit),
                        "replay_mode": bool(replay_mode),
                        "type": disruption_type,
                    },
                )
            ],
            "reasoning_trace": [
                f"[DETECTOR] Order {order_id}: prob={probability:.3f}, "
                f"rule_hit={rule_hit}, replay_mode={replay_mode}, type={disruption_type}"
            ],
        })


    def assess_risk(state: SupplyChainState) -> dict:
        """Compute business risk and ask the LLM for structured severity."""
        order_id = state["order_id"]
        order = state["order"]

        geo = _safe(order, "Geopolitical_Risk_Index")
        weather = _safe(order, "Weather_Severity_Index")
        product_criticality = PRODUCT_CRITICALITY_MAP.get(order.get("Product_Category", ""), 0.5)
        route_risk = ROUTE_RISK_MAP.get(order.get("Route_Type", ""), 0.5)
        delay = _safe(order, "Delay_Days")

        score = geo * 0.35 + (weather / 10.0) * 0.25 + product_criticality * 0.25 + route_risk * 0.15

        query = (
            f"{state['disruption_type']} on {order.get('Route_Type')} route carrying "
            f"{order.get('Product_Category')}"
        )
        rag_context = retrieve_context(query, k=3)

        prompt = (
            "You are a senior supply-chain risk analyst.\n\n"
            "ORDER DETAILS:\n"
            f"  Order ID            : {order_id}\n"
            f"  Disruption Type     : {state['disruption_type']}\n"
            f"  Route               : {order.get('Route_Type')}\n"
            f"  Product             : {order.get('Product_Category')}\n"
            f"  Current Delay       : {delay} days\n"
            f"  Geopolitical Risk   : {geo}\n"
            f"  Weather Severity    : {weather}\n"
            f"  Composite Risk Score: {score:.3f}\n\n"
            "HISTORICAL CONTEXT:\n"
            f"{rag_context}\n\n"
            "Return severity as exactly one of CRITICAL / HIGH / MEDIUM / LOW. "
            "Explain the reasoning briefly and identify the single most important risk factor."
        )

        try:
            assessment = get_llm().with_structured_output(RiskAssessmentSchema).invoke(prompt)
        except Exception as exc:
            logger.warning("[RISK] LLM call failed (%s). Using rule fallback.", exc)
            severity = (
                "CRITICAL"
                if score > 0.75
                else "HIGH"
                if score > 0.55
                else "MEDIUM"
                if score > 0.35
                else "LOW"
            )
            assessment = RiskAssessmentSchema(
                severity=severity,
                reasoning=f"Rule-based fallback because LLM call failed. Composite score={score:.3f}.",
                key_risk_factor="composite_risk_score",
            )

        risk_assessment = {
            "severity": assessment.severity,
            "reasoning": assessment.reasoning,
            "key_risk_factor": assessment.key_risk_factor,
            "geo_risk_index": geo,
            "weather_severity": weather,
            "product_criticality": product_criticality,
            "route_risk": route_risk,
            "composite_risk_score": round(score, 4),
        }

        logger.info("[RISK] %s severity=%s score=%.3f", order_id, assessment.severity, score)
        return _to_python({
            "risk_assessment": risk_assessment,
            "risk_score": round(float(score), 4),
            "audit_log": [
                _audit(
                    "RISK_ASSESSOR",
                    order_id,
                    f"Severity={assessment.severity} score={score:.3f}",
                    risk_assessment,
                )
            ],
            "reasoning_trace": [
                f"[RISK] {order_id}: severity={assessment.severity}, score={score:.3f}, "
                f"key_factor={assessment.key_risk_factor}"
            ],
        })


    def plan_mitigation(state: SupplyChainState) -> dict:
        """Ask the LLM to propose mitigation options grounded in retrieved context."""
        order_id = state["order_id"]
        order = state["order"]
        risk_assessment = state["risk_assessment"]

        query = (
            f"Best mitigation for {state['disruption_type']} affecting "
            f"{order.get('Product_Category')} on {order.get('Route_Type')} route"
        )
        rag_context = retrieve_context(query, k=4)

        prompt = (
            "You are a supply-chain mitigation planner.\n\n"
            "SITUATION:\n"
            f"  Disruption   : {state['disruption_type']}\n"
            f"  Severity     : {risk_assessment['severity']}\n"
            f"  Route        : {order.get('Route_Type')}\n"
            f"  Product      : {order.get('Product_Category')} "
            f"(criticality {risk_assessment['product_criticality']:.2f})\n"
            f"  Current Delay: {order.get('Delay_Days', 0)} days\n\n"
            "HISTORICAL MITIGATION OUTCOMES:\n"
            f"{rag_context}\n\n"
            "Propose exactly 3 mitigation options ranked best-first. For each option, "
            "use one of these action names exactly: Expedited Air Freight, Re-routing, "
            "Standard Shipping, Delay Accepted. Include estimated delay reduction, "
            "confidence from 0 to 1, and rationale."
        )

        try:
            plan = get_llm().with_structured_output(MitigationPlanSchema).invoke(prompt)
            options = [option.model_dump() for option in plan.options]
        except Exception as exc:
            logger.warning("[PLANNER] LLM call failed (%s). Using fallback options.", exc)
            options = [
                {
                    "action": "Re-routing",
                    "description": "Divert via an alternative corridor where capacity is available.",
                    "estimated_delay_reduction_days": 3.0,
                    "confidence": 0.55,
                    "rationale": "Rule fallback: moderate cost and moderate recovery potential.",
                },
                {
                    "action": "Expedited Air Freight",
                    "description": "Move urgent goods by air to recover the schedule.",
                    "estimated_delay_reduction_days": 7.0,
                    "confidence": 0.70,
                    "rationale": "Rule fallback: high recovery potential for critical products.",
                },
                {
                    "action": "Standard Shipping",
                    "description": "Continue with current plan and monitor status.",
                    "estimated_delay_reduction_days": 0.0,
                    "confidence": 0.40,
                    "rationale": "Rule fallback: lowest incremental cost but no active recovery.",
                },
            ]

        for option in options:
            original_action = option.get("action", "")
            normalized_action = normalize_action_name(original_action)
            if normalized_action != original_action:
                option["original_action"] = original_action
                option["action"] = normalized_action

        options.sort(
            key=lambda option: (
                float(option.get("confidence", 0)),
                float(option.get("estimated_delay_reduction_days", 0)),
            ),
            reverse=True,
        )
        top_action = options[0]["action"] if options else "None"

        logger.info("[PLANNER] %s generated %d options top=%s", order_id, len(options), top_action)
        return {
            "mitigation_options": options,
            "audit_log": [
                _audit(
                    "MITIGATION_PLANNER",
                    order_id,
                    f"Generated {len(options)} options; top={top_action}",
                    {"options": [option["action"] for option in options]},
                )
            ],
            "reasoning_trace": [
                f"[PLANNER] {order_id}: {len(options)} options. Top={top_action} "
                f"(conf={options[0]['confidence']:.2f})"
                if options
                else f"[PLANNER] {order_id}: no options generated"
            ],
        }


    def evaluate_cost(state: SupplyChainState) -> dict:
        """Use the pre-trained cost model plus business-rule multipliers."""
        order_id = state["order_id"]
        order = state["order"]
        current_cost = _safe(order, "Shipping_Cost_USD")
        options = state.get("mitigation_options", [])
        risk_assessment = state["risk_assessment"]

        cost_model = load_cost_predictor()
        evaluated: list[dict[str, Any]] = []

        for option in options:
            raw_action = option.get("action", "")
            action, multiplier = action_multiplier(raw_action)
            if action != raw_action:
                option["original_action"] = raw_action
                option["action"] = action
            rule_cost = current_cost * multiplier
            rule_delta = rule_cost - current_cost

            cost_features = cost_feature_frame(order, action)
            ml_cost = float(cost_model.predict(cost_features)[0])

            blended_cost = 0.5 * rule_cost + 0.5 * max(ml_cost, 0.0)
            blended_delta = blended_cost - current_cost
            utility = (
                UTILITY_ALPHA * float(option.get("confidence", 0.5))
                - UTILITY_BETA * (max(blended_delta, 0.0) / max(COST_DELTA_THRESHOLD, 1.0))
            )

            evaluated.append(
                CostOptionResult(
                    action=action,
                    estimated_total_cost=round(blended_cost, 2),
                    cost_delta=round(blended_delta, 2),
                    confidence=float(option.get("confidence", 0.5)),
                    utility=round(utility, 4),
                    requires_hitl=blended_delta > COST_DELTA_THRESHOLD,
                ).model_dump()
            )

        evaluated.sort(key=lambda item: item["utility"], reverse=True)
        best = evaluated[0] if evaluated else {}
        best_action = best.get("action", "None")
        best_confidence = float(best.get("confidence", 0.0))

        requires_human = (
            bool(best.get("requires_hitl", False))
            or best_confidence < CONFIDENCE_THRESHOLD
            or risk_assessment["severity"] == "CRITICAL"
        )

        cost_analysis = {
            "current_cost": current_cost,
            "options": evaluated,
            "ranking_method": f"utility(alpha={UTILITY_ALPHA}, beta={UTILITY_BETA})",
        }


        logger.info(
            "[COST] %s best=%s delta=%s HITL=%s",
            order_id,
            best_action,                           
            f"{best.get('cost_delta', 0):,.0f}",
            requires_human,
        )
        return _to_python({
            "cost_analysis": cost_analysis,
            "recommended_action": best_action,       # ← underscore
            "confidence": float(best_confidence),
            "requires_human": bool(requires_human),
            "audit_log": [_audit("COST_EVALUATOR", order_id,
                f"Best={best_action} delta={best.get('cost_delta',0):.0f} HITL={requires_human}",  # ← underscore
                cost_analysis)],
            "reasoning_trace": [
            f"[COST] {order_id}: best={best_action}, "
            f"delta=${float(best.get('cost_delta', 0)):,.0f}, "
            f"utility={float(best.get('utility', 0)):.3f}, "
            f"HITL={requires_human}"
        ],  
        })


    def human_review(state: SupplyChainState) -> dict:
        """Pause for approval when the risk/cost policy requires HITL."""
        order_id = state["order_id"]
        summary = {
            "order_id": order_id,
            "disruption": state["disruption_type"],
            "severity": state["risk_assessment"]["severity"],
            "recommended_action": state["recommended_action"],
            "cost_options": state["cost_analysis"]["options"],
            "confidence": state["confidence"],
            "reasoning_so_far": state.get("reasoning_trace", []),
        }

        human_input = interrupt(
            value={
                "message": "High-risk exception requires human approval.",
                "summary": summary,
            }
        )
        if not isinstance(human_input, dict):
            logger.warning("[HITL] Unexpected resume payload type: %s", type(human_input).__name__)
            human_input = {}
        approved = bool(human_input.get("approved", False))
        final_action = normalize_action_name(human_input.get("action", state["recommended_action"]))

        logger.info("[HITL] %s approved=%s action=%s", order_id, approved, final_action)
        return {
            "human_approved": approved,
            "final_decision": final_action if approved else "ESCALATED - Awaiting resubmission",
            "audit_log": [
                _audit(
                    "HUMAN_REVIEW",
                    order_id,
                    f"{'APPROVED' if approved else 'REJECTED'}: {final_action}",
                    {"approved": approved, "chosen_action": final_action},
                )
            ],
            "reasoning_trace": [
                f"[HITL] {order_id}: human {'approved' if approved else 'rejected'} -> {final_action}"
            ],
        }


    def execute_action(state: SupplyChainState) -> dict:
        """Build final ERP/TMS write-back payload."""
        order_id = state["order_id"]
        final = state.get("final_decision") or state.get("recommended_action") or "NO_ACTION"
        selected = next(
            (
                option
                for option in state.get("cost_analysis", {}).get("options", [])
                if option.get("action") == final
            ),
            None,
        )

        payload = {
            "order_id": order_id,
            "decision": final,
            "auto_resolved": not state.get("requires_human", False),
            "cost_delta_usd": selected["cost_delta"] if selected else 0.0,
            "confidence": state.get("confidence", 0.0),
            "severity": state.get("risk_assessment", {}).get("severity", "UNKNOWN"),
            "resolved_at": _ts(),
            "system": "SUPPLY_CHAIN_GENAI_AGENT",
        }

        logger.info("[EXECUTE] %s action=%s auto=%s", order_id, final, payload["auto_resolved"])
        return {
            "final_decision": final,
            "execution_payload": payload,
            "audit_log": [
                _audit(
                    "ACTION_EXECUTOR",
                    order_id,
                    f"Executed: {final} auto={payload['auto_resolved']}",
                    payload,
                )
            ],
            "reasoning_trace": [
                f"[EXECUTE] {order_id}: {final} executed. "
                f"Auto-resolved={payload['auto_resolved']}, "
                f"delta=${payload['cost_delta_usd']:,.0f}"
            ],
        }


    def _route_after_detection(state: SupplyChainState) -> str:
        return "disrupted" if state.get("disruption_detected", False) else "no_disruption"


    def _route_after_cost(state: SupplyChainState) -> str:
        return "human_review" if state.get("requires_human", False) else "execute_action"


    def build_graph() -> StateGraph:
        graph = StateGraph(SupplyChainState)
        graph.add_node("detect_exception", detect_exception)
        graph.add_node("assess_risk", assess_risk)
        graph.add_node("plan_mitigation", plan_mitigation)
        graph.add_node("evaluate_cost", evaluate_cost)
        graph.add_node("human_review", human_review)
        graph.add_node("execute_action", execute_action)

        graph.set_entry_point("detect_exception")
        graph.add_conditional_edges(
            "detect_exception",
            _route_after_detection,
            {"no_disruption": END, "disrupted": "assess_risk"},
        )
        graph.add_edge("assess_risk", "plan_mitigation")
        graph.add_edge("plan_mitigation", "evaluate_cost")
        graph.add_conditional_edges(
            "evaluate_cost",
            _route_after_cost,
            {"human_review": "human_review", "execute_action": "execute_action"},
        )
        graph.add_edge("human_review", "execute_action")
        graph.add_edge("execute_action", END)
        return graph


    def compile_graph():
        memory = MemorySaver()
        return build_graph().compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Public runner helpers
# ---------------------------------------------------------------------------
def resolve_order(order: dict[str, Any], order_id: str | None = None) -> dict[str, Any]:
    """Run one order through the GenAI agent workflow."""
    if not _LANGGRAPH_AVAILABLE:
        raise RuntimeError("LangChain/LangGraph dependencies are required for resolve_order().")

    app = compile_graph()
    enriched_id = str(order_id or order.get("Order_ID") or f"ORDER-{int(datetime.now().timestamp())}")
    initial_state = {
        "order": order,
        "order_id": enriched_id,
        "audit_log": [],
        "reasoning_trace": [],
    }
    config = {"configurable": {"thread_id": enriched_id}}
    return app.invoke(initial_state, config=config)


def load_order_from_csv(csv_path: Path = DATA_PATH, order_idx: int = 0) -> dict[str, Any]:
    path = _require_file(csv_path, "demo CSV")
    df = pd.read_csv(path)
    if order_idx < 0 or order_idx >= len(df):
        raise IndexError(f"order_idx={order_idx} is outside CSV row range 0..{len(df)-1}")
    return df.iloc[order_idx].to_dict()


def print_resolution_result(result: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("  SUPPLY CHAIN GENAI AGENT RESULT")
    print("=" * 72)
    print(f"  Order ID            : {result.get('order_id')}")
    print(f"  Disruption Detected : {result.get('disruption_detected')}")
    print(f"  Disruption Prob.    : {result.get('disruption_probability')}")
    print(f"  Disruption Type     : {result.get('disruption_type')}")
    if result.get("disruption_detected"):
        print(f"  Severity            : {result.get('risk_assessment', {}).get('severity')}")
        print(f"  Recommended Action  : {result.get('recommended_action')}")
        print(f"  Confidence          : {result.get('confidence')}")
        print(f"  Requires HITL       : {result.get('requires_human')}")
        print(f"  Final Decision      : {result.get('final_decision')}")
        print(f"  Execution Payload   : {json.dumps(result.get('execution_payload', {}), indent=2)}")
    print("\nReasoning Trace")
    print("-" * 72)
    for line in result.get("reasoning_trace", []):
        print(f"  {line}")
    print("\nAudit Log")
    print("-" * 72)
    for entry in result.get("audit_log", []):
        print(f"  [{entry['timestamp']}] {entry['agent']}: {entry['message']}")
    print("=" * 72)


def run_index_mode(args: argparse.Namespace) -> None:
    playbook_path = Path(args.playbook) if args.playbook else KB_DIR / "mitigation_playbook.txt"
    index_dir = Path(args.index_dir) if args.index_dir else KB_DIR / "faiss_index"
    build_faiss_index(playbook_path=playbook_path, index_dir=index_dir)


def run_resolve_mode(args: argparse.Namespace) -> None:
    if getattr(args, "replay_mode", False):
        os.environ["SUPPLY_CHAIN_REPLAY_MODE"] = "1"

    order = (
        json.loads(Path(args.order_json).read_text(encoding="utf-8"))
        if args.order_json
        else load_order_from_csv(DATA_PATH, args.order_idx)
    )

    app = compile_graph()
    order_id = str(
        args.order_id or order.get("Order_ID")
        or f"ORDER-{int(datetime.now().timestamp())}"
    )
    config = {"configurable": {"thread_id": order_id}}
    initial_state = {"order": order, "order_id": order_id, "audit_log": [], "reasoning_trace": []}

    result = app.invoke(initial_state, config=config)

    if (
        getattr(args, "auto_approve", False)
        and result.get("requires_human")
        and not result.get("final_decision")
    ):
        from langgraph.types import Command
        result = app.invoke(
            Command(resume={"approved": True, "action": result.get("recommended_action")}),
            config=config,
        )

    print_resolution_result(result)

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supply Chain GenAI Agent. No model training is performed here."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Check required artifacts and dependency availability.")

    index_parser = subparsers.add_parser(
        "index",
        help="Build FAISS index from knowledge_base/mitigation_playbook.txt.",
    )
    index_parser.add_argument("--playbook", default=None, help="Path to mitigation_playbook.txt")
    index_parser.add_argument("--index-dir", default=None, help="Output FAISS index directory")

    resolve_parser = subparsers.add_parser(
        "resolve",
        help="Resolve one order using the GenAI agent workflow.",
    )
    resolve_parser.add_argument(
        "--order-idx",
        type=int,
        default=0,
        help="CSV row index for local demo mode. Ignored when --order-json is supplied.",
    )
    resolve_parser.add_argument(
        "--order-json",
        default=None,
        help="Path to a JSON file containing one order dict.",
    )
    resolve_parser.add_argument("--order-id", default=None, help="Optional override order id.")
    resolve_parser.add_argument(
        "--replay-mode",
        action="store_true",
        help="Treat Disruption_Event in historical inputs as a known alert override.",
    )
    resolve_parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve HITL interrupt for CLI/demo mode.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "resolve" and _LANGGRAPH_AVAILABLE:
        load_disruption_classifier()
        load_cost_predictor()
        load_vector_store()

    if args.command == "check":
        print_artifact_report()
    elif args.command == "index":
        run_index_mode(args)
    elif args.command == "resolve":
        run_resolve_mode(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
