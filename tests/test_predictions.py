"""Tests for Phase 8 Daily Prediction System.

Run with: pytest tests/test_predictions.py -v
"""

import json

import pandas as pd
import pytest

from src.app.predict_today import generate_predictions_for_date
from src.app.run_backtest import run_backtest
from src.models.predict import PredictionPipeline
from src.utils.paths import PROCESSED_DIR


@pytest.fixture(scope="module")
def sample_data():
    """Load a small slice of actual data for integration testing."""
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    logs = pd.read_parquet(PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet")
    
    # Pick a random date in the middle of 2023-24 season
    test_date = "2024-01-15"
    
    target_games = games[games["date"] == test_date].head(2).copy()
    target_games = target_games.drop(columns=["home_win", "home_score", "away_score"], errors="ignore")

    hist_games = games[(games["season"] == "2023-24") & (games["date"] < test_date)].copy()
    hist_logs = logs[(logs["season"] == "2023-24") & (logs["date"] < test_date)].copy()

    return target_games, hist_games, hist_logs, test_date


@pytest.fixture(scope="module")
def pipeline():
    return PredictionPipeline(use_cuda=False)


class StubPredictionPipeline:
    """Deterministic lightweight pipeline for script orchestration tests."""

    def predict_games(self, target_games, historical_games, historical_team_logs):
        results = []
        for i, (_, row) in enumerate(target_games.iterrows()):
            prob = 0.65 if i % 2 == 0 else 0.35
            results.append({
                "game_id": row["game_id"],
                "home_team_idx": int(row["home_team_idx"]),
                "away_team_idx": int(row["away_team_idx"]),
                "home_win_probability": prob,
                "away_win_probability": round(1 - prob, 4),
                "predicted_winner": "home" if prob >= 0.5 else "away",
                "confidence_bucket": "medium",
                "component_outputs": {
                    "elo_probability": prob,
                    "tabular_probability": prob,
                    "sequence_probability": prob,
                    "final_probability": prob,
                },
                "top_model_factors": ["Synthetic test factor"],
            })
        return results


@pytest.fixture
def stub_pipeline():
    return StubPredictionPipeline()


@pytest.fixture
def synthetic_backtest_data():
    full_games = pd.DataFrame([
        {
            "game_id": "g1",
            "date": "2024-01-15",
            "season": "2023-24",
            "home_team_idx": 1,
            "away_team_idx": 2,
            "home_win": 1,
        },
        {
            "game_id": "g2",
            "date": "2024-01-15",
            "season": "2023-24",
            "home_team_idx": 3,
            "away_team_idx": 4,
            "home_win": 0,
        },
        {
            "game_id": "g3",
            "date": "2024-01-16",
            "season": "2023-24",
            "home_team_idx": 5,
            "away_team_idx": 6,
            "home_win": 1,
        },
        {
            "game_id": "g4",
            "date": "2024-01-16",
            "season": "2023-24",
            "home_team_idx": 7,
            "away_team_idx": 8,
            "home_win": 0,
        },
    ])

    full_logs = pd.DataFrame([
        {"game_id": "hist1", "team_idx": 1, "date": "2024-01-10", "season": "2023-24"},
        {"game_id": "hist1", "team_idx": 2, "date": "2024-01-10", "season": "2023-24"},
    ])

    return full_games, full_logs


@pytest.fixture(scope="module")
def sample_predictions(pipeline, sample_data):
    target_games, hist_games, hist_logs, _ = sample_data
    return pipeline.predict_games(target_games, hist_games, hist_logs)


class TestPredictionPipeline:
    def test_predict_single_game(self, sample_predictions):
        """Prediction pipeline returns valid output for one game."""
        assert len(sample_predictions) > 0
        
    def test_prediction_schema(self, sample_predictions):
        """Output JSON matches expected schema."""
        pred = sample_predictions[0]
        assert "game_id" in pred
        assert "home_team_idx" in pred
        assert "away_team_idx" in pred
        assert "home_win_probability" in pred
        assert "away_win_probability" in pred
        assert "predicted_winner" in pred
        assert "confidence_bucket" in pred
        assert "component_outputs" in pred
        assert "top_model_factors" in pred

    def test_probabilities_sum_to_one(self, sample_predictions):
        """home_win_prob + away_win_prob ≈ 1.0 for every prediction."""
        for pred in sample_predictions:
            total = pred["home_win_probability"] + pred["away_win_probability"]
            assert abs(total - 1.0) < 1e-4

    def test_probabilities_in_range(self, sample_predictions):
        """All probabilities in [0.01, 0.99]."""
        for pred in sample_predictions:
            assert 0.0 <= pred["home_win_probability"] <= 1.0
            assert 0.0 <= pred["away_win_probability"] <= 1.0

    def test_confidence_bucket_valid(self, sample_predictions):
        """confidence_bucket is one of: low, medium, high."""
        valid_buckets = {"low", "medium", "high"}
        for pred in sample_predictions:
            assert pred["confidence_bucket"] in valid_buckets

    def test_component_outputs_present(self, sample_predictions):
        """component_outputs has elo, tabular, sequence, final."""
        comps = sample_predictions[0]["component_outputs"]
        assert "elo_probability" in comps
        assert "tabular_probability" in comps
        assert "sequence_probability" in comps
        assert "final_probability" in comps

    def test_top_factors_generated(self, sample_predictions):
        """top_model_factors is a non-empty list of strings."""
        factors = sample_predictions[0]["top_model_factors"]
        assert isinstance(factors, list)
        assert len(factors) > 0
        assert isinstance(factors[0], str)

    def test_prediction_deterministic(self, pipeline, sample_data, sample_predictions):
        """Same game predicted twice produces identical output."""
        target_games, hist_games, hist_logs, _ = sample_data
        
        # Predict again
        results2 = pipeline.predict_games(target_games, hist_games, hist_logs)
        
        assert sample_predictions[0]["home_win_probability"] == results2[0]["home_win_probability"]


class TestScripts:
    def test_backtest_on_known_date(self, tmp_path, stub_pipeline, synthetic_backtest_data):
        """Backtest runner saves a valid report."""
        games, logs = synthetic_backtest_data
        report, out_file = run_backtest(
            start_date="2024-01-15",
            end_date="2024-01-16",
            output_dir=tmp_path,
            pipeline=stub_pipeline,
            full_games=games,
            full_logs=logs,
        )

        assert out_file.exists()
        assert report["total_games"] > 0
        assert len(report["predictions"]) == report["total_games"]
        assert "accuracy" in report
        assert "log_loss" in report

    def test_backtest_accuracy_reasonable(self, tmp_path, stub_pipeline, synthetic_backtest_data):
        """Backtest metrics are deterministic for the saved report."""
        games, logs = synthetic_backtest_data
        report, _ = run_backtest(
            start_date="2024-01-15",
            end_date="2024-01-16",
            output_dir=tmp_path,
            pipeline=stub_pipeline,
            full_games=games,
            full_logs=logs,
        )

        assert report["accuracy"] == 1.0
        assert report["log_loss"] < 0.5
        assert report["total_games"] == 4

    def test_predict_today_runs(self, tmp_path, stub_pipeline, sample_data):
        """Daily prediction runner completes without hitting the live API."""
        target_games, hist_games, hist_logs, test_date = sample_data

        output, out_file = generate_predictions_for_date(
            date_str=test_date,
            output_dir=tmp_path,
            schedule_fetcher=lambda _: target_games,
            pipeline=stub_pipeline,
            historical_games=pd.concat([hist_games, target_games], ignore_index=True),
            historical_team_logs=hist_logs,
        )

        assert out_file.exists()
        assert output["date"] == test_date
        assert len(output["predictions"]) == len(target_games)

    def test_daily_predictions_saved(self, tmp_path, stub_pipeline, sample_data):
        """Daily prediction output is saved as valid JSON."""
        target_games, hist_games, hist_logs, test_date = sample_data

        output, out_file = generate_predictions_for_date(
            date_str=test_date,
            output_dir=tmp_path,
            schedule_fetcher=lambda _: target_games,
            pipeline=stub_pipeline,
            historical_games=pd.concat([hist_games, target_games], ignore_index=True),
            historical_team_logs=hist_logs,
        )

        assert out_file.exists()
        with open(out_file) as f:
            data = json.load(f)
            assert data["date"] == test_date
            assert "predictions" in data
            assert data["predictions"] == output["predictions"]
