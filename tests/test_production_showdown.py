"""Tests for the production-vs-enriched showdown runner."""

from __future__ import annotations

import pandas as pd

from src.models import run_production_showdown


def _enriched_results() -> dict:
    return {
        "leaderboard": [
            {
                "feature_set": "enriched_all",
                "model_name": "catboost",
                "feature_count": 161,
                "metrics": {
                    "accuracy": 0.6513,
                    "log_loss": 0.6165,
                    "brier_score": 0.2138,
                    "roc_auc": 0.7181,
                    "calibration_error": 0.0417,
                },
            },
            {
                "feature_set": "enriched_all",
                "model_name": "lightgbm",
                "feature_count": 161,
                "metrics": {
                    "accuracy": 0.6554,
                    "log_loss": 0.6169,
                    "brier_score": 0.2140,
                    "roc_auc": 0.7164,
                    "calibration_error": 0.0192,
                },
            },
        ]
    }


def _production_results() -> dict:
    return {
        "production_neural_full_fusion": {
            "status": "scored",
            "metrics": {
                "accuracy": 0.6485,
                "log_loss": 0.6169,
                "brier_score": 0.2140,
                "roc_auc": 0.7150,
                "calibration_error": 0.0300,
            },
            "test_slices": {
                "high_context_confidence": {
                    "game_count": 200,
                    "accuracy": 0.64,
                    "log_loss": 0.6200,
                    "brier_score": 0.2150,
                    "roc_auc": 0.70,
                    "calibration_error": 0.03,
                    "positive_rate": 0.55,
                }
            },
        },
        "production_ensemble_raw": {
            "status": "scored",
            "metrics": {
                "accuracy": 0.6562,
                "log_loss": 0.6152,
                "brier_score": 0.2133,
                "roc_auc": 0.7250,
                "calibration_error": 0.0280,
            },
            "test_slices": {
                "high_context_confidence": {
                    "game_count": 200,
                    "accuracy": 0.66,
                    "log_loss": 0.6120,
                    "brier_score": 0.2130,
                    "roc_auc": 0.73,
                    "calibration_error": 0.02,
                    "positive_rate": 0.55,
                }
            },
        },
        "production_ensemble_calibrated": {
            "status": "scored",
            "metrics": {
                "accuracy": 0.6570,
                "log_loss": 0.6207,
                "brier_score": 0.2155,
                "roc_auc": 0.7250,
                "calibration_error": 0.0180,
            },
            "test_slices": {
                "high_context_confidence": {
                    "game_count": 200,
                    "accuracy": 0.65,
                    "log_loss": 0.6180,
                    "brier_score": 0.2140,
                    "roc_auc": 0.73,
                    "calibration_error": 0.015,
                    "positive_rate": 0.55,
                }
            },
        },
    }


def test_build_combined_leaderboard_orders_by_log_loss():
    """The combined leaderboard should mix families and rank by log loss first."""
    rows = run_production_showdown.build_combined_leaderboard(
        _enriched_results(),
        _production_results(),
    )

    assert rows[0]["label"] == "production ensemble raw"
    assert rows[1]["label"] == "enriched_all / catboost"
    assert any(row["family"] == "production_stack" for row in rows)
    assert any(row["family"] == "enriched_benchmark" for row in rows)


def test_covered_test_game_ids_uses_intersection_in_expected_order():
    """Production scoring should use rows covered by every saved prediction artifact."""
    expected = pd.Series(["g1", "g2", "g3", "g4"])
    neural = pd.DataFrame({"game_id": ["g1", "g3", "g4"]})
    ensemble = pd.DataFrame({"game_id": ["g1", "g2", "g4"]})

    covered = run_production_showdown._covered_test_game_ids(expected, neural, ensemble)

    assert covered.to_list() == ["g1", "g4"]


def test_coverage_summary_counts_missing_rows():
    """Coverage metadata should make skipped playoff rows visible in reports."""
    summary = run_production_showdown._coverage_summary(
        pd.Series(["g1", "g2", "g3"]),
        pd.Series(["g1"]),
    )

    assert summary == {
        "expected_test_game_count": 3,
        "scored_game_count": 1,
        "missing_game_count": 2,
    }


def test_build_showdown_markdown_includes_verdict_and_slices():
    """The showdown markdown should surface the combined ranking and slice table."""
    combined = run_production_showdown.build_combined_leaderboard(
        _enriched_results(),
        _production_results(),
    )
    results = {
        "generated_at_utc": "2026-05-17T11:00:00+00:00",
        "combined_leaderboard": combined,
        "production_results": _production_results(),
        "verdict": {
            "best_overall": combined[0],
            "best_production": combined[0],
            "best_enriched": combined[1],
            "best_calibrated": combined[-1],
            "production_minus_enriched_log_loss": -0.0013,
        },
    }

    markdown = run_production_showdown.build_showdown_markdown(results)

    assert "# Production Stack Showdown" in markdown
    assert "Combined Leaderboard" in markdown
    assert "Verdict" in markdown
    assert "production ensemble raw" in markdown
    assert "high_context_confidence" in markdown
