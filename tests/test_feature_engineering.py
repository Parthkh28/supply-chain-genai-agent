"""Tests for inference-time feature engineering in supply_chain_genai_agent_groq_hf.py."""

import math
import pytest

from supply_chain_genai_agent_groq_hf import (
    _extract_temporal_features,
    _safe,
    _to_python,
    action_multiplier,
    early_warning_band,
    enrich_order,
    has_disruption_event,
    is_replay_mode_enabled,
    normalize_action_name,
    normalize_disruption_event,
    operational_exception_signals,
)


# ---------------------------------------------------------------------------
# _to_python
# ---------------------------------------------------------------------------
class TestToPython:
    def test_converts_numpy_int(self):
        import numpy as np
        assert _to_python(np.int64(42)) == 42
        assert isinstance(_to_python(np.int64(42)), int)

    def test_converts_numpy_float(self):
        import numpy as np
        assert _to_python(np.float32(3.14)) == pytest.approx(3.14, abs=1e-4)

    def test_nan_becomes_none(self):
        import numpy as np
        assert _to_python(float("nan")) is None
        assert _to_python(np.float64("nan")) is None

    def test_inf_becomes_none(self):
        assert _to_python(float("inf")) is None

    def test_nested_dict(self):
        import numpy as np
        result = _to_python({"a": np.int32(1), "b": [np.float64(2.0)]})
        assert result == {"a": 1, "b": [2.0]}


# ---------------------------------------------------------------------------
# _safe
# ---------------------------------------------------------------------------
class TestSafe:
    def test_returns_float(self):
        assert _safe({"x": 5}, "x") == 5.0

    def test_missing_key_returns_default(self):
        assert _safe({}, "missing", default=99.0) == 99.0

    def test_nan_value_returns_default(self):
        import math
        assert _safe({"x": float("nan")}, "x", default=1.0) == 1.0

    def test_string_number_coerces(self):
        assert _safe({"x": "3.5"}, "x") == 3.5

    def test_non_numeric_string_returns_default(self):
        assert _safe({"x": "abc"}, "x", default=0.0) == 0.0


# ---------------------------------------------------------------------------
# normalize_disruption_event / has_disruption_event
# ---------------------------------------------------------------------------
class TestNormalizeDisruption:
    @pytest.mark.parametrize("value", ["none", "None", "nan", "no disruption", "no_disruption", ""])
    def test_no_disruption_placeholders_become_nan(self, value):
        import numpy as np
        result = normalize_disruption_event(value)
        assert result is np.nan or (isinstance(result, float) and math.isnan(result))

    def test_valid_event_preserved(self):
        result = normalize_disruption_event("Port Congestion")
        assert result == "Port Congestion"

    def test_case_insensitive_canonical_lookup(self):
        result = normalize_disruption_event("port congestion")
        assert result == "Port Congestion"

    def test_has_disruption_false_for_none(self):
        assert has_disruption_event(None) is False

    def test_has_disruption_true_for_event(self):
        assert has_disruption_event("Port Congestion") is True

    def test_has_disruption_false_for_placeholder(self):
        assert has_disruption_event("no disruption") is False


# ---------------------------------------------------------------------------
# normalize_action_name / action_multiplier
# ---------------------------------------------------------------------------
class TestNormalizeAction:
    def test_exact_match(self):
        assert normalize_action_name("Re-routing") == "Re-routing"

    def test_alias_air_freight(self):
        assert normalize_action_name("air freight") == "Expedited Air Freight"

    def test_alias_reroute(self):
        assert normalize_action_name("rerouting") == "Re-routing"

    def test_empty_falls_back(self):
        result = normalize_action_name("", fallback="Standard Shipping")
        assert result == "Standard Shipping"

    def test_fuzzy_match(self):
        result = normalize_action_name("Expedited Air Freigh")  # typo
        assert result == "Expedited Air Freight"

    def test_action_multiplier_returns_pair(self):
        action, mult = action_multiplier("Re-routing")
        assert action == "Re-routing"
        assert mult == pytest.approx(1.6)


# ---------------------------------------------------------------------------
# _extract_temporal_features
# ---------------------------------------------------------------------------
class TestTemporalFeatures:
    def test_missing_date_returns_zeros(self):
        result = _extract_temporal_features({})
        assert result["order_month"] == 0
        assert result["order_quarter"] == 0

    def test_august_is_typhoon_season(self):
        result = _extract_temporal_features({"Order_Date": "2024-08-15"})
        assert result["is_typhoon_season"] == 1

    def test_december_is_peak_shipping(self):
        result = _extract_temporal_features({"Order_Date": "2024-12-01"})
        assert result["is_peak_shipping"] == 1

    def test_january_is_lunar_new_year(self):
        result = _extract_temporal_features({"Order_Date": "2024-01-20"})
        assert result["is_lunar_new_year_window"] == 1

    def test_quarter_derivation(self):
        result = _extract_temporal_features({"Order_Date": "2024-05-01"})
        assert result["order_quarter"] == 2

    def test_invalid_date_returns_zeros(self):
        result = _extract_temporal_features({"Order_Date": "not-a-date"})
        assert result["order_month"] == 0


# ---------------------------------------------------------------------------
# operational_exception_signals
# ---------------------------------------------------------------------------
class TestOperationalExceptionSignals:
    def test_no_signals_for_on_time_order(self):
        order = {"Delivery_Status": "On Time", "observed_delay_days": 0}
        assert operational_exception_signals(order) == []

    def test_delay_produces_signal(self):
        order = {"observed_delay_days": 5.0}
        signals = operational_exception_signals(order)
        assert any("observed_delay_days" in s for s in signals)

    def test_late_delivery_status_produces_signal(self):
        order = {"Delivery_Status": "Delayed", "observed_delay_days": 0}
        signals = operational_exception_signals(order)
        assert any("Delivery_Status" in s for s in signals)

    def test_actual_vs_scheduled_lead_time_computes_delay(self):
        order = {
            "Actual_Lead_Time_Days": 30,
            "Scheduled_Lead_Time_Days": 20,
        }
        signals = operational_exception_signals(order)
        assert len(signals) > 0


# ---------------------------------------------------------------------------
# early_warning_band
# ---------------------------------------------------------------------------
class TestEarlyWarningBand:
    def test_high_probability_is_high_watchlist(self):
        assert early_warning_band(0.50, {}) == "HIGH_WATCHLIST"

    def test_low_probability_is_low(self):
        assert early_warning_band(0.05, {"composite_risk_score": 0.1}) == "LOW"

    def test_high_composite_risk_elevates_band(self):
        # composite_risk_score above 0.75 should return HIGH_WATCHLIST
        assert early_warning_band(0.10, {"composite_risk_score": 0.80}) == "HIGH_WATCHLIST"


# ---------------------------------------------------------------------------
# enrich_order — smoke test
# ---------------------------------------------------------------------------
class TestEnrichOrder:
    BASE_ORDER = {
        "Order_ID": "TEST-001",
        "Route_Type": "Suez",
        "Product_Category": "Pharmaceuticals",
        "Transportation_Mode": "Sea",
        "Geopolitical_Risk_Index": 0.8,
        "Weather_Severity_Index": 7.0,
        "Inflation_Rate_Pct": 4.0,
        "Scheduled_Lead_Time_Days": 20,
        "Base_Lead_Time_Days": 15,
        "Order_Weight_Kg": 1000,
        "Shipping_Cost_USD": 15000,
    }

    def test_enriched_order_has_route_risk_score(self):
        enriched = enrich_order(self.BASE_ORDER)
        assert "route_risk_score" in enriched

    def test_enriched_order_has_product_criticality(self):
        enriched = enrich_order(self.BASE_ORDER)
        assert "product_criticality" in enriched

    def test_enriched_order_has_one_hot_mode(self):
        enriched = enrich_order(self.BASE_ORDER)
        assert "mode_Sea" in enriched
        assert enriched["mode_Sea"] == 1

    def test_enriched_order_has_one_hot_route(self):
        enriched = enrich_order(self.BASE_ORDER)
        assert "route_Suez" in enriched
        assert enriched["route_Suez"] == 1

    def test_enriched_order_has_composite_risk_score(self):
        enriched = enrich_order(self.BASE_ORDER)
        score = enriched.get("composite_risk_score", None)
        assert score is not None and 0.0 <= score <= 1.0

    def test_missing_fields_do_not_raise(self):
        minimal = {"Order_ID": "MIN-001"}
        enriched = enrich_order(minimal)
        assert isinstance(enriched, dict)

    def test_disruption_event_normalised(self):
        order = {**self.BASE_ORDER, "Disruption_Event": "no disruption"}
        enriched = enrich_order(order)
        import math
        val = enriched.get("Disruption_Event")
        assert val is None or (isinstance(val, float) and math.isnan(val))


# ---------------------------------------------------------------------------
# is_replay_mode_enabled — env var behaviour
# ---------------------------------------------------------------------------
class TestReplayMode:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("SUPPLY_CHAIN_REPLAY_MODE", raising=False)
        assert is_replay_mode_enabled() is False

    def test_enabled_when_set_to_1(self, monkeypatch):
        monkeypatch.setenv("SUPPLY_CHAIN_REPLAY_MODE", "1")
        assert is_replay_mode_enabled() is True

    def test_enabled_when_set_to_true(self, monkeypatch):
        monkeypatch.setenv("SUPPLY_CHAIN_REPLAY_MODE", "true")
        assert is_replay_mode_enabled() is True
