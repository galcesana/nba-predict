"""Tests for projected availability building."""

from __future__ import annotations

import pandas as pd

from src.features import projected_availability


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
    return pd.DataFrame(
        [
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
                "fantasy_points": 48.0,
                "plus_minus": 8,
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
                "fantasy_points": 38.0,
                "plus_minus": 5,
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
                "fantasy_points": 46.0,
                "plus_minus": 6,
            },
            {
                "game_id": "hist-3",
                "date": "2026-05-13",
                "season": "2025-26",
                "team_idx": 5,
                "opponent_team_idx": 8,
                "player_id": 103,
                "player_idx": 4,
                "player_name": "Darius Garland",
                "minutes": 35,
                "fantasy_points": 41.0,
                "plus_minus": 4,
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
                "fantasy_points": 49.0,
                "plus_minus": 7,
            },
        ]
    )


def _season_metadata() -> dict[str, pd.DataFrame]:
    return {
        "2025-26": pd.DataFrame(
            [
                {
                    "player_id": 101,
                    "player_idx": 1,
                    "player_name": "Donovan Mitchell",
                    "team_idx": 5,
                    "aliases": ["DONOVAN MITCHELL", "MITCHELL, DONOVAN"],
                },
                {
                    "player_id": 201,
                    "player_idx": 3,
                    "player_name": "Cade Cunningham",
                    "team_idx": 8,
                    "aliases": ["CADE CUNNINGHAM", "CUNNINGHAM, CADE"],
                },
                {
                    "player_id": 103,
                    "player_idx": 4,
                    "player_name": "Darius Garland",
                    "team_idx": 5,
                    "aliases": ["DARIUS GARLAND", "GARLAND, DARIUS"],
                },
            ]
        )
    }


def test_build_projected_availability_resolves_official_injury_rows():
    """Official report names should resolve to player ids and override availability."""
    reports = pd.DataFrame(
        [
            {
                "game_id": "game-2",
                "team_idx": 5,
                "player_name": "Mitchell, Donovan",
                "status": "Questionable",
                "reason": "Left ankle sprain",
                "report_generated_at": "2026-05-16T16:30:00-0400",
            }
        ]
    )

    projected, unresolved = projected_availability.build_projected_availability(
        _games(),
        _player_logs(),
        injury_reports=reports,
        metadata_by_season=_season_metadata(),
        recent_team_games=5,
        max_players=5,
    )

    mitchell = projected[projected["player_id"] == 101].iloc[0]
    assert mitchell["status"] == "QUESTIONABLE"
    assert mitchell["availability_score"] == 0.5
    assert mitchell["source_type"] == "official_injury_report"
    assert unresolved.empty


def test_build_projected_availability_keeps_recent_role_baseline_without_reports():
    """Recent player roles should seed availability even when no report exists."""
    projected, unresolved = projected_availability.build_projected_availability(
        _games(),
        _player_logs(),
        injury_reports=pd.DataFrame(),
        metadata_by_season=_season_metadata(),
        recent_team_games=5,
        max_players=5,
        use_historical_absence_proxy=False,
    )

    assert unresolved.empty
    assert not projected.empty
    cavs = projected[projected["team_idx"] == 5]
    assert set(cavs["player_id"]) == {101, 102, 103}
    assert cavs["availability_score"].eq(1.0).all()
    assert {
        "expected_usage_proxy",
        "value_confidence",
        "role_tier",
        "projected_minutes",
        "projected_value_available",
        "projected_value_missing",
    }.issubset(projected.columns)
    assert cavs["projected_minutes"].gt(0).any()
    assert cavs["projected_value_available"].gt(0).any()


def test_build_projected_availability_marks_prior_absence_without_target_leakage():
    """A rotation player missing prior games should receive a non-live absence proxy."""
    games = pd.DataFrame(
        [
            {
                "game_id": "game-5",
                "date": "2026-05-20",
                "season": "2025-26",
                "home_team_idx": 5,
                "away_team_idx": 8,
            }
        ]
    )
    rows = []
    for game_id, game_date, players in [
        ("hist-1", "2026-05-10", [101, 102]),
        ("hist-2", "2026-05-12", [101, 102]),
        ("hist-3", "2026-05-14", [101, 103]),
        ("hist-4", "2026-05-16", [101, 103]),
    ]:
        for player_id in players:
            rows.append(
                {
                    "game_id": game_id,
                    "date": game_date,
                    "season": "2025-26",
                    "team_idx": 5,
                    "opponent_team_idx": 8,
                    "player_id": player_id,
                    "player_idx": player_id,
                    "player_name": f"Player {player_id}",
                    "minutes": 35 if player_id == 102 else 30,
                    "fantasy_points": 44.0 if player_id == 102 else 30.0,
                    "plus_minus": 4,
                    "points": 20,
                    "assists": 4,
                    "rebounds": 6,
                }
            )
    rows.append(
        {
            "game_id": "hist-4",
            "date": "2026-05-16",
            "season": "2025-26",
            "team_idx": 8,
            "opponent_team_idx": 5,
            "player_id": 201,
            "player_idx": 201,
            "player_name": "Player 201",
            "minutes": 34,
            "fantasy_points": 40.0,
            "plus_minus": 3,
            "points": 22,
            "assists": 5,
            "rebounds": 7,
        }
    )

    projected, unresolved = projected_availability.build_projected_availability(
        games,
        pd.DataFrame(rows),
        team_game_logs=pd.DataFrame(
            [
                {
                    "game_id": "hist-1",
                    "date": "2026-05-10",
                    "season": "2025-26",
                    "team_idx": 5,
                    "net_rating": 10.0,
                    "point_diff": 11,
                },
                {
                    "game_id": "hist-2",
                    "date": "2026-05-12",
                    "season": "2025-26",
                    "team_idx": 5,
                    "net_rating": 8.0,
                    "point_diff": 9,
                },
                {
                    "game_id": "hist-3",
                    "date": "2026-05-14",
                    "season": "2025-26",
                    "team_idx": 5,
                    "net_rating": -12.0,
                    "point_diff": -14,
                },
                {
                    "game_id": "hist-4",
                    "date": "2026-05-16",
                    "season": "2025-26",
                    "team_idx": 5,
                    "net_rating": -15.0,
                    "point_diff": -16,
                },
            ]
        ),
        recent_team_games=5,
        max_players=5,
    )

    absent_player = projected[
        (projected["game_id"] == "game-5") & (projected["player_id"] == 102)
    ].iloc[0]

    assert unresolved.empty
    assert absent_player["status"] == "PROJECTED_ABSENT"
    assert absent_player["source_type"] == "historical_absence_proxy"
    assert absent_player["availability_score"] < 1.0
    assert absent_player["projected_value_missing"] > 0
    assert absent_player["replacement_risk_score"] > 0
    assert (
        absent_player["projected_replacement_value_missing"]
        > absent_player["projected_value_missing"]
    )
    assert absent_player["projected_minutes"] < absent_player["expected_minutes"]
    assert absent_player["availability_model_version"] == (
        projected_availability.PROJECTED_AVAILABILITY_VERSION
    )


def test_build_projected_availability_records_unmatched_injury_names():
    """Unmatched report names should be surfaced in an audit table."""
    reports = pd.DataFrame(
        [
            {
                "game_id": "game-2",
                "team_idx": 5,
                "player_name": "Unknown, Player",
                "status": "Out",
                "reason": "Not on roster",
                "report_generated_at": "2026-05-16T16:30:00-0400",
            }
        ]
    )

    projected, unresolved = projected_availability.build_projected_availability(
        _games(),
        _player_logs(),
        injury_reports=reports,
        metadata_by_season=_season_metadata(),
        recent_team_games=5,
        max_players=5,
    )

    assert not projected.empty
    assert len(unresolved) == 1
    assert unresolved.iloc[0]["resolution_status"] == "unmatched_name"


def test_build_projected_availability_adds_report_only_players_with_role_context():
    """Resolved report-only players should keep historical role value when added fresh."""
    reports = pd.DataFrame(
        [
            {
                "game_id": "game-2",
                "team_idx": 5,
                "player_name": "Garland, Darius",
                "status": "Out",
                "reason": "Toe sprain",
                "report_generated_at": "2026-05-16T16:30:00-0400",
            }
        ]
    )

    projected, _ = projected_availability.build_projected_availability(
        _games(),
        _player_logs(),
        injury_reports=reports,
        metadata_by_season=_season_metadata(),
        recent_team_games=2,
        max_players=2,
    )

    garland = projected[projected["player_id"] == 103].iloc[0]

    assert garland["status"] == "OUT"
    assert garland["availability_score"] == 0.0
    assert garland["recent_games_played"] == 1
    assert garland["expected_minutes"] > 0
    assert garland["role_score"] > 0
    assert garland["projected_value_available"] == 0.0
    assert garland["projected_value_missing"] > 0
