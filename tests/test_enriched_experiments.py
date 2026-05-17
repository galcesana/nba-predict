"""Tests for the enriched matchup experiment runner."""

from __future__ import annotations

import pandas as pd

from src.models import run_enriched_experiments


def _legacy_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "0022300001",
                "date": "2024-01-01",
                "season": "2023-24",
                "season_type": "Regular Season",
                "home_team_idx": 1,
                "away_team_idx": 2,
                "home_metric": 1.0,
                "away_metric": 0.5,
                "debug_label": "not-a-feature",
                "target_home_win": 1,
            }
        ]
    )


def _enriched_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "0022300001",
                "date": "2024-01-01",
                "season": "2023-24",
                "season_type": "Regular Season",
                "home_team_idx": 1,
                "away_team_idx": 2,
                "home_metric": 1.0,
                "away_metric": 0.5,
                "debug_label": "not-a-feature",
                "home_expected_starter_continuity": 0.8,
                "away_expected_starter_continuity": 0.7,
                "diff_expected_starter_continuity": 0.1,
                "home_projected_player_value_available": 20.0,
                "away_projected_player_value_available": 17.5,
                "diff_projected_player_value_available": 2.5,
                "home_projected_player_value_missing": 1.0,
                "away_projected_player_value_missing": 4.0,
                "diff_projected_player_value_missing": -3.0,
                "home_projected_top8_confidence_mean": 0.9,
                "away_projected_top8_confidence_mean": 0.7,
                "diff_projected_top8_confidence_mean": 0.2,
                "home_projected_top8_availability_mean": 0.95,
                "away_projected_top8_availability_mean": 0.8,
                "diff_projected_top8_availability_mean": 0.15,
                "target_home_win": 1,
            }
        ]
    )


def _slice_df() -> pd.DataFrame:
    rows = []
    for idx, game_id, season, missing, confidence in [
        (0, "0042400001", "2023-24", 20.0, 0.95),
        (1, "0022400002", "2023-24", 1.0, 0.25),
        (2, "0042500003", "2024-25", 15.0, 0.85),
        (3, "0022500004", "2024-25", 0.0, 0.10),
    ]:
        rows.append(
            {
                "game_id": game_id,
                "date": f"2024-0{idx + 1}-01",
                "season": season,
                "home_team_idx": 1,
                "away_team_idx": 2,
                "home_metric": 1.0 + idx,
                "away_metric": 0.5 + idx,
                "home_projected_player_value_missing": missing,
                "away_projected_player_value_missing": missing / 2,
                "home_projected_top8_confidence_mean": confidence,
                "away_projected_top8_confidence_mean": max(confidence - 0.1, 0.0),
                "target_home_win": idx % 2,
            }
        )
    return pd.DataFrame(rows)


def test_get_experiment_feature_sets_builds_round_two_ablations():
    """Round-two feature families should isolate the intended M1 column groups."""
    feature_sets = run_enriched_experiments.get_experiment_feature_sets(
        _legacy_df(),
        _enriched_df(),
    )

    _, legacy_cols = feature_sets["legacy"]
    _, enriched_cols = feature_sets["enriched_all"]
    _, m1_only_cols = feature_sets["m1_only"]
    _, value_cols = feature_sets["enriched_value_only"]
    _, availability_cols = feature_sets["enriched_availability_only"]
    _, lineup_cols = feature_sets["enriched_lineup_only"]
    _, no_conf_cols = feature_sets["enriched_no_confidence"]

    assert "home_metric" in legacy_cols
    assert "season_type" not in legacy_cols
    assert "debug_label" not in legacy_cols
    assert "home_projected_player_value_available" in enriched_cols
    assert "home_projected_player_value_available" in m1_only_cols
    assert "home_metric" not in m1_only_cols
    assert "home_projected_player_value_available" in value_cols
    assert "home_projected_top8_confidence_mean" in availability_cols
    assert "home_expected_starter_continuity" in lineup_cols
    assert "home_projected_top8_confidence_mean" not in no_conf_cols


def test_build_test_slice_masks_includes_playoff_and_context_slices():
    """Slice generation should expose playoff, missing-value, and confidence masks."""
    slice_masks = run_enriched_experiments.build_test_slice_masks(_slice_df())

    assert "all_test" in slice_masks
    assert "playoffs" in slice_masks
    assert "regular_season" in slice_masks
    assert "high_missing_value" in slice_masks
    assert "low_context_confidence" in slice_masks
    assert bool(slice_masks["playoffs"].loc["0042400001"])
    assert not bool(slice_masks["playoffs"].loc["0022400002"])


def test_artifact_game_universe_check_rejects_stale_regular_season_cache():
    """Cached matchup artifacts should rebuild when games gain playoff rows."""
    games = pd.DataFrame(
        {
            "game_id": ["0022400001", "0042400002"],
            "season_type": ["Regular Season", "Playoffs"],
        }
    )
    stale_artifact = pd.DataFrame({"game_id": ["0022400001"]})

    assert not run_enriched_experiments._artifact_matches_game_universe(
        stale_artifact,
        games,
        label="test artifact",
        exact_game_ids=True,
    )


def test_artifact_game_universe_check_accepts_multiraw_cache_with_all_regimes():
    """Multi-row feature artifacts need regime coverage, not exact game-id equality."""
    games = pd.DataFrame(
        {
            "game_id": ["0022400001", "0042400002"],
            "season_type": ["Regular Season", "Playoffs"],
        }
    )
    feature_artifact = pd.DataFrame(
        {
            "game_id": ["0022400001", "0022400001", "0042400002"],
            "player_id": [1, 2, 1],
        }
    )

    assert run_enriched_experiments._artifact_matches_game_universe(
        feature_artifact,
        games,
        label="test features",
        exact_game_ids=False,
    )


def test_build_summary_markdown_mentions_best_model_and_unavailable_models():
    """The markdown summary should reflect leaderboard, slices, and missing libs."""
    results = {
        "generated_at_utc": "2026-05-17T12:00:00+00:00",
        "feature_sets": {
            "legacy": {
                "feature_count": 2,
                "logistic_regression": {
                    "status": "trained",
                    "test": {
                        "accuracy": 0.60,
                        "log_loss": 0.64,
                        "brier_score": 0.23,
                        "roc_auc": 0.61,
                        "calibration_error": 0.04,
                    },
                },
                "xgboost": {
                    "status": "trained",
                    "test": {
                        "accuracy": 0.61,
                        "log_loss": 0.63,
                        "brier_score": 0.22,
                        "roc_auc": 0.62,
                        "calibration_error": 0.03,
                    },
                },
                "lightgbm": {
                    "status": "unavailable",
                    "reason": "lightgbm is not installed",
                },
                "catboost": {
                    "status": "unavailable",
                    "reason": "catboost is not installed",
                },
            },
            "enriched_all": {
                "feature_count": 5,
                "xgboost_platt": {
                    "status": "trained",
                    "test": {
                        "accuracy": 0.64,
                        "log_loss": 0.60,
                        "brier_score": 0.21,
                        "roc_auc": 0.66,
                        "calibration_error": 0.02,
                    },
                    "test_slices": {
                        "playoffs": {
                            "game_count": 12,
                            "accuracy": 0.67,
                            "log_loss": 0.58,
                            "brier_score": 0.20,
                            "roc_auc": 0.68,
                            "calibration_error": 0.02,
                            "positive_rate": 0.5,
                        }
                    },
                },
            },
        },
        "leaderboard": [
            {
                "feature_set": "enriched_all",
                "model_name": "xgboost_platt",
                "feature_count": 5,
                "metrics": {
                    "accuracy": 0.64,
                    "log_loss": 0.60,
                    "brier_score": 0.21,
                    "roc_auc": 0.66,
                    "calibration_error": 0.02,
                },
            },
            {
                "feature_set": "legacy",
                "model_name": "xgboost",
                "feature_count": 2,
                "metrics": {
                    "accuracy": 0.61,
                    "log_loss": 0.63,
                    "brier_score": 0.22,
                    "roc_auc": 0.62,
                    "calibration_error": 0.03,
                },
            },
        ],
        "best_test_model": {
            "feature_set": "enriched_all",
            "model_name": "xgboost_platt",
            "metrics": {
                "accuracy": 0.64,
                "log_loss": 0.60,
                "brier_score": 0.21,
                "roc_auc": 0.66,
                "calibration_error": 0.02,
            },
            "test_slices": {
                "playoffs": {
                    "game_count": 12,
                    "accuracy": 0.67,
                    "log_loss": 0.58,
                    "brier_score": 0.20,
                    "roc_auc": 0.68,
                    "calibration_error": 0.02,
                    "positive_rate": 0.5,
                }
            },
        },
    }

    markdown = run_enriched_experiments.build_summary_markdown(results)

    assert "# M1 Enriched Matchup Experiments" in markdown
    assert "Test Leaderboard" in markdown
    assert "xgboost_platt" in markdown
    assert "Best Test Model" in markdown
    assert "Unavailable Models" in markdown
    assert "playoffs" in markdown
