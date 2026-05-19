"""Tests for matchup-row enrichment with player and lineup features."""

from __future__ import annotations

import pandas as pd

from src.features.build_matchup_dataset import (
    append_enriched_features,
    build_enriched_matchup_dataset,
    build_matchup_dataset,
)
from src.features.lineup_features import build_lineup_features
from src.features.player_value_features import build_player_value_features
from src.features.projected_availability import build_projected_availability


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "hist-1",
                "date": "2026-05-10",
                "season": "2025-26",
                "home_team_idx": 5,
                "away_team_idx": 8,
                "home_win": 1,
            },
            {
                "game_id": "hist-2",
                "date": "2026-05-12",
                "season": "2025-26",
                "home_team_idx": 8,
                "away_team_idx": 5,
                "home_win": 0,
            },
            {
                "game_id": "game-3",
                "date": "2026-05-17",
                "season": "2025-26",
                "home_team_idx": 8,
                "away_team_idx": 5,
                "home_win": 0,
            },
        ]
    )


def _team_logs() -> pd.DataFrame:
    rows = [
            {
                "game_id": "hist-1",
                "date": "2026-05-10",
                "season": "2025-26",
                "team_idx": 5,
                "won": 1,
                "point_diff": 8,
                "net_rating": 6.0,
                "off_rating": 116.0,
                "def_rating": 110.0,
                "pace": 99.0,
                "ts_pct": 0.58,
                "efg_pct": 0.56,
                "turnover_pct": 0.12,
                "assist_pct": 0.63,
            },
            {
                "game_id": "hist-1",
                "date": "2026-05-10",
                "season": "2025-26",
                "team_idx": 8,
                "won": 0,
                "point_diff": -8,
                "net_rating": -6.0,
                "off_rating": 108.0,
                "def_rating": 114.0,
                "pace": 99.0,
                "ts_pct": 0.52,
                "efg_pct": 0.50,
                "turnover_pct": 0.15,
                "assist_pct": 0.57,
            },
            {
                "game_id": "hist-2",
                "date": "2026-05-12",
                "season": "2025-26",
                "team_idx": 5,
                "won": 1,
                "point_diff": 5,
                "net_rating": 4.0,
                "off_rating": 113.0,
                "def_rating": 109.0,
                "pace": 98.0,
                "ts_pct": 0.57,
                "efg_pct": 0.55,
                "turnover_pct": 0.11,
                "assist_pct": 0.61,
            },
            {
                "game_id": "hist-2",
                "date": "2026-05-12",
                "season": "2025-26",
                "team_idx": 8,
                "won": 0,
                "point_diff": -5,
                "net_rating": -4.0,
                "off_rating": 109.0,
                "def_rating": 113.0,
                "pace": 98.0,
                "ts_pct": 0.53,
                "efg_pct": 0.51,
                "turnover_pct": 0.14,
                "assist_pct": 0.56,
            },
            {
                "game_id": "game-3",
                "date": "2026-05-17",
                "season": "2025-26",
                "team_idx": 5,
                "won": 0,
                "point_diff": 0,
                "net_rating": 0.0,
                "off_rating": 0.0,
                "def_rating": 0.0,
                "pace": 0.0,
                "ts_pct": 0.0,
                "efg_pct": 0.0,
                "turnover_pct": 0.0,
                "assist_pct": 0.0,
            },
            {
                "game_id": "game-3",
                "date": "2026-05-17",
                "season": "2025-26",
                "team_idx": 8,
                "won": 0,
                "point_diff": 0,
                "net_rating": 0.0,
                "off_rating": 0.0,
                "def_rating": 0.0,
                "pace": 0.0,
                "ts_pct": 0.0,
                "efg_pct": 0.0,
                "turnover_pct": 0.0,
                "assist_pct": 0.0,
            },
        ]
    defaults = {
        "free_throw_rate": 0.22,
        "off_rebound_pct": 0.26,
        "def_rebound_pct": 0.74,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _player_logs() -> pd.DataFrame:
    rows = []
    for game_id, game_date, team_idx, players in [
        ("hist-1", "2026-05-10", 5, [(101, 36, 48.0), (102, 32, 38.0), (103, 18, 18.0)]),
        ("hist-1", "2026-05-10", 8, [(201, 37, 49.0), (202, 33, 35.0), (203, 21, 20.0)]),
        ("hist-2", "2026-05-12", 5, [(101, 34, 46.0), (102, 31, 34.0), (103, 16, 16.0)]),
        ("hist-2", "2026-05-12", 8, [(201, 35, 45.0), (202, 32, 33.0), (203, 22, 22.0)]),
    ]:
        for player_id, minutes, fantasy_points in players:
            rows.append(
                {
                    "game_id": game_id,
                    "date": game_date,
                    "season": "2025-26",
                    "team_idx": team_idx,
                    "opponent_team_idx": 8 if team_idx == 5 else 5,
                    "player_id": player_id,
                    "player_idx": player_id,
                    "player_name": f"Player {player_id}",
                    "minutes": minutes,
                    "points": round(fantasy_points / 1.7),
                    "rebounds": 5,
                    "assists": 4,
                    "plus_minus": 6 if player_id in {101, 201} else 2,
                    "fantasy_points": fantasy_points,
                }
            )
    return pd.DataFrame(rows)


def test_build_enriched_matchup_dataset_adds_player_and_lineup_columns():
    """The enriched matchup dataset should expose the new M1 team-level summaries."""
    games = _games()
    team_logs = _team_logs()
    player_logs = _player_logs()

    value_features = build_player_value_features(
        games,
        player_logs,
        recent_team_games=5,
        max_players=5,
    )
    projected, _ = build_projected_availability(
        games,
        player_logs,
        player_value_features=value_features,
        recent_team_games=5,
        max_players=5,
    )
    projected.loc[
        (projected["game_id"] == "game-3") & (projected["player_id"] == 101),
        ["status", "availability_score"],
    ] = ["OUT", 0.0]
    lineup_rows = build_lineup_features(games, player_logs, projected, recent_team_games=5)

    enriched = build_enriched_matchup_dataset(
        games,
        team_logs,
        player_logs,
        player_value_features=value_features,
        projected_availability=projected,
        lineup_features_df=lineup_rows,
    )

    target = enriched[enriched["game_id"] == "game-3"].iloc[0]

    assert "home_projected_player_value_available" in enriched.columns
    assert "home_projected_top8_value_confidence_mean" in enriched.columns
    assert "away_projected_top8_minutes_missing" in enriched.columns
    assert "away_expected_missing_starter_value" in enriched.columns
    assert "diff_projected_player_value_missing" in enriched.columns
    assert target["away_projected_player_value_missing"] > 0
    assert target["away_projected_top8_minutes_missing"] > 0


def test_build_enriched_matchup_dataset_preserves_base_row_count():
    """Adding player-aware columns should not change the one-row-per-game contract."""
    games = _games()
    enriched = build_enriched_matchup_dataset(
        games,
        _team_logs(),
        _player_logs(),
    )

    assert len(enriched) == len(games)
    assert set(enriched["game_id"]) == set(games["game_id"])


def test_append_enriched_features_matches_full_enriched_builder():
    """Live inference can reuse a base matchup row and attach the same M1 columns."""
    games = _games()
    team_logs = _team_logs()
    player_logs = _player_logs()

    value_features = build_player_value_features(
        games,
        player_logs,
        recent_team_games=5,
        max_players=5,
    )
    projected, _ = build_projected_availability(
        games,
        player_logs,
        player_value_features=value_features,
        recent_team_games=5,
        max_players=5,
    )
    lineup_rows = build_lineup_features(games, player_logs, projected, recent_team_games=5)

    base_matchup = build_matchup_dataset(games, team_logs)
    appended = append_enriched_features(
        base_matchup,
        games,
        projected_availability=projected,
        lineup_features_df=lineup_rows,
    )
    full = build_enriched_matchup_dataset(
        games,
        team_logs,
        player_logs,
        player_value_features=value_features,
        projected_availability=projected,
        lineup_features_df=lineup_rows,
    )

    pd.testing.assert_frame_equal(
        full.sort_index(axis=1),
        appended.sort_index(axis=1),
    )
