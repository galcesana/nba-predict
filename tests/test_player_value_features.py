"""Tests for pregame player value feature building."""

from __future__ import annotations

import pandas as pd

from src.features import player_value_features


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
        {
            "game_id": "hist-1",
            "date": "2026-05-10",
            "season": "2025-26",
            "team_idx": 5,
            "opponent_team_idx": 8,
            "player_id": 101,
            "player_idx": 1,
            "player_name": "Donovan Mitchell",
            "minutes": 36,
            "points": 28,
            "rebounds": 5,
            "assists": 6,
            "plus_minus": 8,
            "fantasy_points": 48.0,
        },
        {
            "game_id": "hist-1",
            "date": "2026-05-10",
            "season": "2025-26",
            "team_idx": 5,
            "opponent_team_idx": 8,
            "player_id": 102,
            "player_idx": 2,
            "player_name": "Jarrett Allen",
            "minutes": 32,
            "points": 16,
            "rebounds": 11,
            "assists": 2,
            "plus_minus": 5,
            "fantasy_points": 38.0,
        },
        {
            "game_id": "hist-1",
            "date": "2026-05-10",
            "season": "2025-26",
            "team_idx": 5,
            "opponent_team_idx": 8,
            "player_id": 109,
            "player_idx": 9,
            "player_name": "Depth Wing",
            "minutes": 12,
            "points": 4,
            "rebounds": 2,
            "assists": 1,
            "plus_minus": 1,
            "fantasy_points": 11.0,
        },
        {
            "game_id": "hist-2",
            "date": "2026-05-12",
            "season": "2025-26",
            "team_idx": 5,
            "opponent_team_idx": 8,
            "player_id": 101,
            "player_idx": 1,
            "player_name": "Donovan Mitchell",
            "minutes": 34,
            "points": 26,
            "rebounds": 4,
            "assists": 7,
            "plus_minus": 6,
            "fantasy_points": 46.0,
        },
        {
            "game_id": "hist-2",
            "date": "2026-05-12",
            "season": "2025-26",
            "team_idx": 5,
            "opponent_team_idx": 8,
            "player_id": 102,
            "player_idx": 2,
            "player_name": "Jarrett Allen",
            "minutes": 31,
            "points": 14,
            "rebounds": 10,
            "assists": 1,
            "plus_minus": 3,
            "fantasy_points": 34.0,
        },
        {
            "game_id": "hist-3",
            "date": "2026-05-13",
            "season": "2025-26",
            "team_idx": 8,
            "opponent_team_idx": 5,
            "player_id": 201,
            "player_idx": 3,
            "player_name": "Cade Cunningham",
            "minutes": 37,
            "points": 27,
            "rebounds": 7,
            "assists": 9,
            "plus_minus": 7,
            "fantasy_points": 49.0,
        },
    ]
    return pd.DataFrame(cavs_rows)


def _team_logs_for_absence_risk() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "hist-1",
                "date": "2026-05-10",
                "season": "2025-26",
                "team_idx": 5,
                "net_rating": 12.0,
                "point_diff": 14,
            },
            {
                "game_id": "hist-2",
                "date": "2026-05-12",
                "season": "2025-26",
                "team_idx": 5,
                "net_rating": 9.0,
                "point_diff": 10,
            },
            {
                "game_id": "hist-3",
                "date": "2026-05-14",
                "season": "2025-26",
                "team_idx": 5,
                "net_rating": -13.0,
                "point_diff": -15,
            },
        ]
    )


def test_build_player_value_features_uses_only_prior_games():
    """Pregame player values should summarize only historical games before the target date."""
    features = player_value_features.build_player_value_features(
        _games(),
        _player_logs(),
        recent_team_games=5,
        max_players=5,
    )

    mitchell = features[features["player_id"] == 101].iloc[0]

    assert mitchell["recent_games_played"] == 2
    assert round(mitchell["recent_minutes_avg"], 2) == 35.00
    assert round(mitchell["last_game_minutes"], 2) == 34.00
    assert mitchell["recent_usage_proxy"] > 0
    assert mitchell["recent_value_per_minute"] > 0
    assert mitchell["player_value_model_version"] == (
        player_value_features.PLAYER_VALUE_FEATURE_VERSION
    )


def test_build_player_value_features_capture_role_stability():
    """Players appearing in fewer recent games should have lower role stability."""
    features = player_value_features.build_player_value_features(
        _games(),
        _player_logs(),
        recent_team_games=2,
        max_players=5,
    )

    mitchell = features[features["player_id"] == 101].iloc[0]
    depth_wing = features[features["player_id"] == 109].iloc[0]

    assert mitchell["recent_role_stability"] == 1.0
    assert depth_wing["recent_role_stability"] == 0.5
    assert mitchell["recent_starter_rate"] > depth_wing["recent_starter_rate"]
    assert mitchell["value_confidence"] > depth_wing["value_confidence"]


def test_build_player_value_features_rank_top_players_above_depth():
    """The player value score should rank core rotation players above fringe minutes."""
    features = player_value_features.build_player_value_features(
        _games(),
        _player_logs(),
        recent_team_games=5,
        max_players=5,
    )

    cavs = features[features["team_idx"] == 5].sort_values("rotation_rank")

    assert cavs.iloc[0]["player_id"] == 101
    assert cavs.iloc[0]["player_value_score"] > cavs.iloc[-1]["player_value_score"]
    assert cavs.iloc[0]["role_tier"] == 3


def test_build_player_value_features_scores_replacement_risk_from_prior_absences():
    """Team drop-off in prior missed games should become a leakage-safe risk signal."""
    logs = pd.concat(
        [
            _player_logs(),
            pd.DataFrame(
                [
                    {
                        "game_id": "hist-3",
                        "date": "2026-05-14",
                        "season": "2025-26",
                        "team_idx": 5,
                        "opponent_team_idx": 8,
                        "player_id": 102,
                        "player_idx": 2,
                        "player_name": "Jarrett Allen",
                        "minutes": 33,
                        "points": 17,
                        "rebounds": 12,
                        "assists": 2,
                        "plus_minus": -4,
                        "fantasy_points": 39.0,
                    },
                    {
                        "game_id": "hist-3",
                        "date": "2026-05-14",
                        "season": "2025-26",
                        "team_idx": 5,
                        "opponent_team_idx": 8,
                        "player_id": 109,
                        "player_idx": 9,
                        "player_name": "Depth Wing",
                        "minutes": 18,
                        "points": 6,
                        "rebounds": 3,
                        "assists": 1,
                        "plus_minus": -6,
                        "fantasy_points": 13.0,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    features = player_value_features.build_player_value_features(
        _games(),
        logs,
        team_game_logs=_team_logs_for_absence_risk(),
        recent_team_games=5,
        max_players=5,
    )

    mitchell = features[features["player_id"] == 101].iloc[0]
    allen = features[features["player_id"] == 102].iloc[0]

    assert mitchell["recent_absence_games"] == 1
    assert mitchell["recent_absence_net_rating_delta"] > 0
    assert mitchell["replacement_risk_score"] > 0
    assert allen["replacement_risk_score"] == 0
