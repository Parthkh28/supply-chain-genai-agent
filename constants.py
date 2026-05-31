"""Shared constants and default configuration for the supply-chain agent."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "supply_chain_agent_training_outputs"

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_COST_MODEL_FILENAME = "cost_predictor_enhanced.pkl"
DEFAULT_COST_FEATURE_KEY = "cost_model_features_enhanced"
LEGACY_COST_MODEL_FILENAME = "cost_predictor.pkl"
LEGACY_COST_FEATURE_KEY = "cost_model_features_agent_compatible"


def first_existing_dir(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


MODELS_DIR = Path(
    os.getenv(
        "SUPPLY_CHAIN_MODELS_DIR",
        str(first_existing_dir(DEFAULT_OUTPUT_DIR / "models", ROOT / "models")),
    )
)
KB_DIR = Path(
    os.getenv(
        "SUPPLY_CHAIN_KB_DIR",
        str(first_existing_dir(DEFAULT_OUTPUT_DIR / "knowledge_base", ROOT / "knowledge_base")),
    )
)
DATA_PATH = Path(
    os.getenv("SUPPLY_CHAIN_DATA_PATH", str(ROOT / "data" / "global_supply_chain_disruption_v1.csv"))
)

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
