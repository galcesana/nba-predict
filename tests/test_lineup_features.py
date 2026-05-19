"""Tests for lineup and rotation feature building."""

from __future__ import annotations

import pandas as pd

from src.features import lineup_features


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "game-2",
                "date": "2026-05-17",
                "season": "2025-26",
                "home_team_idx": 8,
                "away_team_idx": 5,
            }
        ]
    )


def _player_logs() -> pd.DataFrame:
    cavs_rows = [
        {"game_id": "hist-1", "date": "2026-05-10", "team_idx": 5, "player_id": 101, "minutes": 36},
        {"game_id": "hist-1", "date": "2026-05-10", "team_idx": 5, "player_id": 102, "minutes": 34},
        {"game_id": "hist-1", "date": "2026-05-10", "team_idx": 5, "player_id": 103, "minutes": 32},
        {"game_id": "hist-1", "date": "2026-05-10", "team_idx": 5, "player_id": 104, "minutes": 30},
        {"game_id": "hist-1", "date": "2026-05-10", "team_idx": 5, "player_id": 105, "minutes": 28},
        {"game_id": "hist-1", "date": "2026-05-10", "team_idx": 5, "player_id": 106, "minutes": 20},
        {"game_id": "hist-1", "date": "2026-05-10", "team_idx": 5, "player_id": 107, "minutes": 18},
        {"game_id": "hist-1", "date": "2026-05-10", "team_idx": 5, "player_id": 108, "minutes": 16},
        {"game_id": "hist-2", "date": "2026-05-12", "team_idx": 5, "player_id": 101, "minutes": 37},
        {"game_id": "hist-2", "date": "2026-05-12", "team_idx": 5, "player_id": 102, "minutes": 33},
        {"game_id": "hist-2", "date": "2026-05-12", "team_idx": 5, "player_id": 103, "minutes": 31},
        {"game_id": "hist-2", "date": "2026-05-12", "team_idx": 5, "player_id": 104, "minutes": 29},
        {"game_id": "hist-2", "date": "2026-05-12", "team_idx": 5, "player_id": 105, "minutes": 27},
        {"game_id": "hist-2", "date": "2026-05-12", "team_idx": 5, "player_id": 106, "minutes": 22},
        {"game_id": "hist-2", "date": "2026-05-12", "team_idx": 5, "player_id": 107, "minutes": 19},
        {"game_id": "hist-2", "date": "2026-05-12", "team_idx": 5, "player_id": 108, "minutes": 15},
    ]
    pistons_rows = [
        {"game_id": "hist-3", "date": "2026-05-12", "team_idx": 8, "player_id": 201, "minutes": 35},
    ]
    return pd.DataFrame(
        cavs_rows + pistons_rows
    )


def _availability() -> pd.DataFrame:
    rows = []
    role_values = {
        101: 40.0,
        102: 35.0,
        103: 30.0,
        104: 28.0,
        105: 26.0,
        106: 18.0,
        107: 16.0,
        108: 14.0,
        201: 33.0,
    }
    for player_id, role_score in role_values.items():
        rows.append(
            {
                "game_id": "game-2",
                "date": "2026-05-17",
                "season": "2025-26",
                "team_idx": 5 if player_id < 200 else 8,
                "player_id": player_id,
                "player_idx": player_id,
                "player_name": f"Player {player_id}",
                "status": "AVAILABLE" if player_id != 101 else "QUESTIONABLE",
                "availability_score": 1.0 if player_id != 101 else 0.5,
                "projection_confidence": 0.8,
                "source_type": "historical_recent_role",
                "source_timestamp": "2026-05-16T16:00:00Z",
                "report_reason": None,
                "recent_games_played": 2,
                "expected_minutes": role_score,
                "replacement_risk_score": 0.6 if player_id == 101 else 0.0,
                "role_score": role_score,
            }
        )
    return pd.DataFrame(rows)


def test_build_lineup_features_captures_missing_starter_value():
    """Reduced availability for a top player should increase missing starter value."""
    features = lineup_features.build_lineup_features(
        _games(),
        _player_logs(),
        _availability(),
    )

    team_row = features[features["team_idx"] == 5].iloc[0]
    assert team_row["expected_missing_starter_value"] > 0
    assert team_row["expected_missing_replacement_risk"] > 0
    assert team_row["available_top8_players"] == 8
    assert team_row["projected_available_rotation_value"] < sum(
        _availability()[_availability()["team_idx"] == 5]["role_score"]
    )


def test_build_lineup_features_uses_only_prior_games():
    """Continuity should be based on the last prior game, not the target game itself."""
    features = lineup_features.build_lineup_features(
        _games(),
        _player_logs(),
        _availability(),
    )

    team_row = features[features["team_idx"] == 5].iloc[0]
    assert team_row["expected_starter_continuity"] > 0
    assert 0.0 <= team_row["expected_top8_continuity"] <= 1.0


def test_build_lineup_features_counts_out_rotation_players():
    """Fully unavailable rotation players should reduce the projected available count."""
    availability = _availability()
    availability.loc[availability["player_id"] == 106, "status"] = "OUT"
    availability.loc[availability["player_id"] == 106, "availability_score"] = 0.0

    features = lineup_features.build_lineup_features(
        _games(),
        _player_logs(),
        availability,
    )

    team_row = features[features["team_idx"] == 5].iloc[0]

    assert team_row["available_top8_players"] == 7
    assert team_row["expected_missing_rotation_value"] > team_row["expected_missing_starter_value"]
