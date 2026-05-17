"""Tests for the next-generation promotion validation gate."""

from __future__ import annotations

import pandas as pd

from src.models import run_nextgen_validation


def _test_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["0022300001", "0042300002", "0022400003", "0022400004"],
            "season": ["2023-24", "2023-24", "2024-25", "2024-25"],
            "home_projected_player_value_missing": [0.0, 2.0, 0.0, 1.0],
            "away_projected_player_value_missing": [0.0, 0.0, 3.0, 0.0],
            "home_projected_top8_confidence_mean": [0.90, 0.70, 0.80, 0.60],
            "away_projected_top8_confidence_mean": [0.85, 0.75, 0.82, 0.65],
            "home_back_to_back": [0, 1, 0, 0],
            "away_back_to_back": [0, 0, 1, 0],
            "home_games_last_7": [2, 4, 3, 5],
            "away_games_last_7": [2, 3, 4, 5],
        }
    )


def _scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["0022300001", "0042300002", "0022400003", "0022400004"],
            "actual_home_win": [1, 0, 1, 0],
            "production_raw_prob": [0.60, 0.45, 0.55, 0.48],
            "nextgen_raw_prob": [0.70, 0.30, 0.65, 0.35],
        }
    )


def test_build_validation_slices_tracks_playoffs_missing_players_and_seasons():
    """Validation slices should include promotion-critical coverage masks."""
    slices = run_nextgen_validation.build_validation_slices(_test_frame())

    assert int(slices["all_test"].sum()) == 4
    assert int(slices["playoffs"].sum()) == 1
    assert int(slices["regular_season"].sum()) == 3
    assert int(slices["missing_player_impact"].sum()) == 3
    assert int(slices["season_2024_25"].sum()) == 2


def test_evaluate_model_comparison_scores_slices_and_deltas():
    """Slice comparison should report production, next-gen, and deltas."""
    slices = run_nextgen_validation.build_validation_slices(_test_frame())
    results = run_nextgen_validation.evaluate_model_comparison(_scored_frame(), slices)

    assert results["all_test"]["status"] == "scored"
    assert results["all_test"]["game_count"] == 4
    assert results["all_test"]["delta"]["log_loss"] < 0
    assert results["playoffs"]["game_count"] == 1


def test_build_promotion_verdict_blocks_without_required_coverage():
    """A small aggregate win should not promote without playoff and injury coverage."""
    slice_results = {
        "all_test": {
            "status": "scored",
            "game_count": 4,
            "delta": {"log_loss": -0.002},
        },
        "playoffs": {"status": "no_data", "game_count": 0},
        "missing_player_impact": {"status": "no_data", "game_count": 0},
        "season_2024_25": {
            "status": "scored",
            "game_count": 2,
            "delta": {"log_loss": 0.001},
        },
    }

    verdict = run_nextgen_validation.build_promotion_verdict(slice_results)

    assert verdict["status"] == "blocked"
    blocked_checks = {check["name"] for check in verdict["checks"] if not check["passed"]}
    assert "playoff_coverage" in blocked_checks
    assert "missing_player_coverage" in blocked_checks


def test_build_summary_markdown_reports_blocked_gate():
    """The markdown summary should be readable for the experiment log."""
    results = {
        "generated_at_utc": "2026-05-17T12:00:00+00:00",
        "verdict": {
            "status": "blocked",
            "recommendation": "Promote only after coverage gates pass.",
            "checks": [
                {
                    "name": "playoff_coverage",
                    "status": "block",
                    "detail": "Need playoff games.",
                }
            ],
        },
        "slice_results": {
            "all_test": {
                "status": "scored",
                "game_count": 4,
                "production": {"log_loss": 0.62, "accuracy": 0.65},
                "nextgen": {"log_loss": 0.61, "accuracy": 0.66},
                "delta": {"log_loss": -0.01},
            },
            "playoffs": {"status": "no_data", "game_count": 0},
        },
    }

    markdown = run_nextgen_validation.build_summary_markdown(results)

    assert "# Next-Gen Promotion Gate" in markdown
    assert "Status: `blocked`" in markdown
    assert "playoff_coverage" in markdown
