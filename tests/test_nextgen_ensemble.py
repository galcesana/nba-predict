"""Tests for the next-generation enriched ensemble runner."""

from __future__ import annotations

import json

import pandas as pd

from src.models import run_nextgen_ensemble


def _production_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "actual_home_win": [1, 0, 1, 0],
            "neural_prob": [0.75, 0.35, 0.60, 0.45],
            "xgboost_prob": [0.70, 0.40, 0.55, 0.42],
            "elo_prob": [0.65, 0.45, 0.58, 0.50],
        }
    )


def _enriched_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "actual_home_win": [1, 0, 1, 0],
            "enriched_catboost_prob": [0.80, 0.30, 0.62, 0.38],
            "enriched_lightgbm_prob": [0.78, 0.32, 0.64, 0.41],
        }
    )


def _legacy_matchup_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "date": ["2025-01-01", "2025-01-02"],
            "season": ["2024-25", "2024-25"],
            "target_home_win": [1, 0],
            "base_rating_gap": [0.5, -0.3],
        }
    )


def _enriched_matchup_frame() -> pd.DataFrame:
    frame = _legacy_matchup_frame().copy()
    frame["home_projected_player_value_available"] = [4.0, 3.0]
    frame["away_projected_top8_value_confidence_mean"] = [0.7, 0.8]
    frame["home_expected_starter_continuity"] = [0.9, 0.6]
    return frame


def test_build_enriched_input_feature_config_uses_model_specific_feature_sets():
    """Next-gen inputs should train each enriched learner on its selected family."""
    config = run_nextgen_ensemble.build_enriched_input_feature_config(
        _enriched_matchup_frame(),
        legacy_df=_legacy_matchup_frame(),
    )

    catboost_cols = config["catboost"]["feature_cols"]
    lightgbm_cols = config["lightgbm"]["feature_cols"]

    assert config["catboost"]["feature_set"] == "enriched_value_only"
    assert config["lightgbm"]["feature_set"] == "enriched_all"
    assert "home_projected_player_value_available" in catboost_cols
    assert "home_expected_starter_continuity" not in catboost_cols
    assert "home_expected_starter_continuity" in lightgbm_cols


def test_feature_column_cache_rejects_legacy_shared_schema(tmp_path, monkeypatch):
    """Old shared-column next-gen caches should be rebuilt after feature-family tuning."""
    path = tmp_path / "enriched_feature_columns.json"
    path.write_text(json.dumps(["base_rating_gap"]), encoding="utf-8")
    monkeypatch.setattr(run_nextgen_ensemble, "ENRICHED_FEATURE_COLUMNS_PATH", path)
    config = run_nextgen_ensemble.build_enriched_input_feature_config(
        _enriched_matchup_frame(),
        legacy_df=_legacy_matchup_frame(),
    )

    assert not run_nextgen_ensemble._feature_column_cache_matches(config)


def test_feature_column_cache_accepts_model_specific_schema(tmp_path, monkeypatch):
    """Current next-gen caches should preserve the model-specific feature selections."""
    path = tmp_path / "enriched_feature_columns.json"
    config = run_nextgen_ensemble.build_enriched_input_feature_config(
        _enriched_matchup_frame(),
        legacy_df=_legacy_matchup_frame(),
    )
    path.write_text(
        json.dumps(run_nextgen_ensemble._feature_config_payload(config)),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_nextgen_ensemble, "ENRICHED_FEATURE_COLUMNS_PATH", path)

    assert run_nextgen_ensemble._feature_column_cache_matches(config)


def test_assemble_meta_frames_merges_production_and_enriched_inputs():
    """Meta frames should contain production and enriched probability columns."""
    val_frame, test_frame = run_nextgen_ensemble.assemble_meta_frames(
        _production_frame(),
        _production_frame(),
        _enriched_frame(),
        _enriched_frame(),
    )

    assert "neural_prob" in val_frame.columns
    assert "enriched_catboost_prob" in val_frame.columns
    assert "enriched_lightgbm_prob" in test_frame.columns
    assert not val_frame[run_nextgen_ensemble.ENRICHED_INPUT_COLS].isna().any().any()


def test_prediction_cache_matches_split_rejects_stale_game_ids():
    """Enriched input prediction caches should rebuild when split ids change."""
    split_df = pd.DataFrame({"game_id": ["g1", "g2"]})
    stale_predictions = pd.DataFrame(
        {
            "game_id": ["g1"],
            "enriched_catboost_prob": [0.6],
            "enriched_lightgbm_prob": [0.55],
        }
    )

    assert not run_nextgen_ensemble._prediction_cache_matches_split(
        stale_predictions,
        split_df,
        label="test cache",
    )


def test_prediction_cache_matches_split_accepts_current_input_columns():
    """Current enriched prediction caches should match exact split ids and columns."""
    split_df = pd.DataFrame({"game_id": ["g1", "g2"]})
    predictions = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "enriched_catboost_prob": [0.6, 0.4],
            "enriched_lightgbm_prob": [0.55, 0.45],
        }
    )

    assert run_nextgen_ensemble._prediction_cache_matches_split(
        predictions,
        split_df,
        label="test cache",
    )


def test_build_variant_leaderboard_ranks_nextgen_and_production_rows():
    """Leaderboard should mix saved production rows with next-gen variants."""
    production_results = {
        "production_ensemble_raw": {
            "metrics": {
                "accuracy": 0.65,
                "log_loss": 0.615,
                "brier_score": 0.213,
                "roc_auc": 0.72,
                "calibration_error": 0.04,
            }
        },
        "production_neural_full_fusion": {
            "metrics": {
                "accuracy": 0.64,
                "log_loss": 0.620,
                "brier_score": 0.216,
                "roc_auc": 0.71,
                "calibration_error": 0.03,
            }
        },
        "production_ensemble_calibrated": {
            "metrics": {
                "accuracy": 0.65,
                "log_loss": 0.621,
                "brier_score": 0.216,
                "roc_auc": 0.72,
                "calibration_error": 0.02,
            }
        },
    }
    variant_results = {
        "nextgen_full": {
            "raw": {
                "metrics": {
                    "accuracy": 0.66,
                    "log_loss": 0.610,
                    "brier_score": 0.210,
                    "roc_auc": 0.73,
                    "calibration_error": 0.03,
                }
            },
            "calibrated": {
                "metrics": {
                    "accuracy": 0.65,
                    "log_loss": 0.618,
                    "brier_score": 0.215,
                    "roc_auc": 0.73,
                    "calibration_error": 0.02,
                }
            },
        }
    }

    rows = run_nextgen_ensemble.build_variant_leaderboard(
        production_results,
        variant_results,
    )

    assert rows[0]["label"] == "nextgen_full / raw"
    assert any(row["family"] == "production_stack" for row in rows)
    assert any(row["family"] == "nextgen_ensemble" for row in rows)


def test_build_summary_markdown_reports_verdict_and_weights():
    """The markdown report should include the leaderboard, verdict, and weights."""
    leaderboard = [
        {
            "family": "nextgen_ensemble",
            "label": "nextgen_full / raw",
            "variant": "nextgen_full",
            "mode": "raw",
            "metrics": {
                "accuracy": 0.66,
                "log_loss": 0.610,
                "brier_score": 0.210,
                "roc_auc": 0.73,
                "calibration_error": 0.03,
            },
        },
        {
            "family": "production_stack",
            "label": "production ensemble raw",
            "variant": "production_ensemble_raw",
            "mode": "saved",
            "metrics": {
                "accuracy": 0.65,
                "log_loss": 0.615,
                "brier_score": 0.213,
                "roc_auc": 0.72,
                "calibration_error": 0.04,
            },
        },
    ]
    results = {
        "generated_at_utc": "2026-05-17T12:00:00+00:00",
        "leaderboard": leaderboard,
        "variants": {
            "nextgen_full": {
                "model_weights": {
                    "neural_prob": 0.4,
                    "enriched_catboost_prob": 0.8,
                }
            }
        },
        "verdict": {
            "best_overall": leaderboard[0],
            "best_nextgen": leaderboard[0],
            "production_baseline": leaderboard[1],
            "nextgen_minus_production_log_loss": -0.005,
        },
    }

    markdown = run_nextgen_ensemble.build_summary_markdown(results)

    assert "# Next-Generation Ensemble Results" in markdown
    assert "nextgen_full / raw" in markdown
    assert "Next-Gen Full Weights" in markdown
    assert "enriched_catboost_prob" in markdown
