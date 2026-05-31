"""Tests for constants.py — validates that every business-rule value is sane."""

import pytest
from constants import (
    ACTION_ALIASES,
    CONFIDENCE_THRESHOLD,
    COST_DELTA_THRESHOLD,
    COST_MULTIPLIERS,
    DEFAULT_DISRUPTION_PROB_THRESHOLD,
    DEFAULT_PRODUCT_CRITICALITY_MAP,
    DEFAULT_ROUTE_RISK_MAP,
    DISRUPTION_LABEL_MAP,
    LEAKAGE_PRONE_FEATURES,
    NO_DISRUPTION_VALUES,
    UTILITY_ALPHA,
    UTILITY_BETA,
    VALID_ACTION_NAMES,
)


class TestCostMultipliers:
    def test_all_actions_have_multiplier(self):
        for name in VALID_ACTION_NAMES:
            assert name in COST_MULTIPLIERS, f"{name} missing from COST_MULTIPLIERS"

    def test_multipliers_are_positive(self):
        for action, mult in COST_MULTIPLIERS.items():
            assert mult > 0, f"{action} has non-positive multiplier {mult}"

    def test_expedited_air_is_most_expensive(self):
        assert COST_MULTIPLIERS["Expedited Air Freight"] == max(COST_MULTIPLIERS.values())

    def test_aliases_resolve_to_valid_actions(self):
        for alias, target in ACTION_ALIASES.items():
            assert target in COST_MULTIPLIERS, (
                f"Alias '{alias}' points to unknown action '{target}'"
            )


class TestRiskMaps:
    def test_route_risk_values_in_range(self):
        for route, risk in DEFAULT_ROUTE_RISK_MAP.items():
            assert 0.0 <= risk <= 1.0, f"{route} risk={risk} out of [0,1]"

    def test_product_criticality_in_range(self):
        for product, crit in DEFAULT_PRODUCT_CRITICALITY_MAP.items():
            assert 0.0 <= crit <= 1.0, f"{product} criticality={crit} out of [0,1]"

    def test_suez_is_highest_risk_route(self):
        assert DEFAULT_ROUTE_RISK_MAP["Suez"] == max(DEFAULT_ROUTE_RISK_MAP.values())

    def test_pharmaceuticals_is_highest_criticality(self):
        assert DEFAULT_PRODUCT_CRITICALITY_MAP["Pharmaceuticals"] == max(
            DEFAULT_PRODUCT_CRITICALITY_MAP.values()
        )


class TestDisruptionLabels:
    def test_no_disruption_is_zero(self):
        assert DISRUPTION_LABEL_MAP["No Disruption"] == 0

    def test_all_labels_are_non_negative_ints(self):
        for label, value in DISRUPTION_LABEL_MAP.items():
            assert isinstance(value, int) and value >= 0, f"{label}={value}"

    def test_no_disruption_values_are_lowercase(self):
        for val in NO_DISRUPTION_VALUES:
            assert val == val.lower(), f"'{val}' is not lowercase in NO_DISRUPTION_VALUES"


class TestThresholds:
    def test_disruption_prob_threshold_in_range(self):
        assert 0.0 < DEFAULT_DISRUPTION_PROB_THRESHOLD < 1.0

    def test_confidence_threshold_in_range(self):
        assert 0.0 < CONFIDENCE_THRESHOLD < 1.0

    def test_cost_delta_threshold_is_positive(self):
        assert COST_DELTA_THRESHOLD > 0

    def test_utility_weights_sum_to_one(self):
        assert abs(UTILITY_ALPHA + UTILITY_BETA - 1.0) < 1e-9

    def test_leakage_features_are_strings(self):
        for feat in LEAKAGE_PRONE_FEATURES:
            assert isinstance(feat, str)
