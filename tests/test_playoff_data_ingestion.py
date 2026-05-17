"""Tests for regular-season plus playoff data ingestion support."""

from __future__ import annotations

import pandas as pd

from src.data import fetch_games


def _raw_team_game_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SEASON_ID": "22024",
                "TEAM_ABBREVIATION": "BOS",
                "GAME_ID": "0022400001",
                "GAME_DATE": "2024-10-22",
                "MATCHUP": "BOS vs. NYK",
                "WL": "W",
                "PTS": 132,
                "FGM": 48,
                "FGA": 95,
                "FG3M": 29,
                "FTA": 19,
                "OREB": 11,
                "DREB": 35,
                "REB": 46,
                "AST": 33,
                "TOV": 10,
                "STL": 7,
                "BLK": 6,
                "MIN": 240,
                "FG_PCT": 0.505,
                "PLUS_MINUS": 23,
            },
            {
                "SEASON_ID": "22024",
                "TEAM_ABBREVIATION": "NYK",
                "GAME_ID": "0022400001",
                "GAME_DATE": "2024-10-22",
                "MATCHUP": "NYK @ BOS",
                "WL": "L",
                "PTS": 109,
                "FGM": 43,
                "FGA": 94,
                "FG3M": 11,
                "FTA": 16,
                "OREB": 14,
                "DREB": 34,
                "REB": 48,
                "AST": 20,
                "TOV": 12,
                "STL": 5,
                "BLK": 3,
                "MIN": 240,
                "FG_PCT": 0.457,
                "PLUS_MINUS": -23,
            },
            {
                "SEASON_ID": "22024",
                "TEAM_ABBREVIATION": "BOS",
                "GAME_ID": "0042400101",
                "GAME_DATE": "2025-04-20",
                "MATCHUP": "BOS vs. MIA",
                "WL": "W",
                "PTS": 114,
                "FGM": 42,
                "FGA": 88,
                "FG3M": 16,
                "FTA": 17,
                "OREB": 9,
                "DREB": 32,
                "REB": 41,
                "AST": 25,
                "TOV": 9,
                "STL": 8,
                "BLK": 4,
                "MIN": 240,
                "FG_PCT": 0.477,
                "PLUS_MINUS": 11,
            },
            {
                "SEASON_ID": "22024",
                "TEAM_ABBREVIATION": "MIA",
                "GAME_ID": "0042400101",
                "GAME_DATE": "2025-04-20",
                "MATCHUP": "MIA @ BOS",
                "WL": "L",
                "PTS": 103,
                "FGM": 39,
                "FGA": 90,
                "FG3M": 12,
                "FTA": 15,
                "OREB": 10,
                "DREB": 31,
                "REB": 41,
                "AST": 22,
                "TOV": 11,
                "STL": 6,
                "BLK": 3,
                "MIN": 240,
                "FG_PCT": 0.433,
                "PLUS_MINUS": -11,
            },
        ]
    )


def test_build_games_table_infers_regular_season_and_playoffs():
    """Game rows should carry season_type even when old raw caches do not."""
    games = fetch_games.build_games_table(_raw_team_game_rows())

    by_game = games.set_index("game_id")["season_type"].to_dict()
    assert by_game["0022400001"] == fetch_games.REGULAR_SEASON
    assert by_game["0042400101"] == fetch_games.PLAYOFFS


def test_build_team_game_logs_preserves_playoff_season_type():
    """Team logs should expose season_type for regime-aware feature evaluation."""
    logs = fetch_games.build_team_game_logs(_raw_team_game_rows())

    assert "season_type" in logs.columns
    playoff_rows = logs[logs["game_id"] == "0042400101"]
    assert set(playoff_rows["season_type"]) == {fetch_games.PLAYOFFS}
    assert len(playoff_rows) == 2
