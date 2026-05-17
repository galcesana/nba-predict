"""Tests for processed player-game log building."""

from __future__ import annotations

import pandas as pd

from src.data import fetch_player_logs


def test_build_player_game_logs_normalizes_raw_frame(monkeypatch):
    """Raw league game log rows should be converted into the processed schema."""
    monkeypatch.setattr(
        fetch_player_logs,
        "player_id_to_idx",
        lambda player_id: {201143: 7}[player_id],
    )
    monkeypatch.setattr(
        fetch_player_logs,
        "team_abbr_to_idx",
        lambda abbr: {"BOS": 1, "NYK": 12}[abbr],
    )

    raw = pd.DataFrame(
        [
            {
                "SEASON_ID": "22024",
                "PLAYER_ID": 201143,
                "PLAYER_NAME": "Al Horford",
                "TEAM_ABBREVIATION": "BOS",
                "GAME_ID": "0022400061",
                "GAME_DATE": "2024-10-22",
                "MATCHUP": "BOS vs. NYK",
                "WL": "W",
                "MIN": 26,
                "PTS": 11,
                "REB": 3,
                "AST": 5,
                "STL": 1,
                "BLK": 1,
                "TOV": 0,
                "FG_PCT": 0.571,
                "FG3_PCT": 0.6,
                "FT_PCT": None,
                "PLUS_MINUS": 19,
                "FANTASY_PTS": 28.1,
            }
        ]
    )

    result = fetch_player_logs.build_player_game_logs(raw)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["season"] == "2024-25"
    assert row["team_idx"] == 1
    assert row["opponent_team_idx"] == 12
    assert row["player_idx"] == 7
    assert row["is_home"] == 1
    assert row["won"] == 1
    assert row["ft_pct"] == 0.0
    assert row["available_for_game"] == 1


def test_build_player_game_logs_returns_empty_schema_for_empty_input():
    """Empty raw frames should still return the expected processed columns."""
    result = fetch_player_logs.build_player_game_logs(pd.DataFrame())

    assert result.empty
    assert {"player_id", "player_idx", "available_for_game"}.issubset(result.columns)
