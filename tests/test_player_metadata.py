"""Tests for player metadata and identity utilities."""

from __future__ import annotations

import pandas as pd

from src.anonymization import player_mapping
from src.data import player_metadata
from src.data.providers.nba_api_provider import NbaApiProvider


class _FakeProvider:
    def fetch_player_info(self, season: str) -> pd.DataFrame:
        assert season == "2024-25"
        return pd.DataFrame(
            [
                {
                    "player_id": 201143,
                    "player_name": "Al Horford",
                    "team_abbr": "BOS",
                    "position": "C-F",
                    "height": "6-9",
                    "weight": "240",
                    "birth_date": "JUN 03, 1986",
                    "experience": "17",
                },
                {
                    "player_id": 203954,
                    "player_name": "Joel Embiid",
                    "team_abbr": "PHI",
                    "position": "C",
                    "height": "7-0",
                    "weight": "280",
                    "birth_date": "MAR 16, 1994",
                    "experience": "8",
                },
            ]
        )


def test_build_player_mapping_is_stable():
    """Player mappings should be sorted by player_id for stable anonymous indices."""
    mapping = player_mapping.build_player_to_idx_mapping(
        [
            {"id": 30},
            {"id": 10},
            {"id": 20},
        ]
    )
    assert mapping == {10: 0, 20: 1, 30: 2}


def test_normalize_player_name_handles_punctuation():
    """Normalized player names should be case-insensitive and punctuation-light."""
    assert player_mapping.normalize_player_name("T.J. McConnell") == "TJ MCCONNELL"
    assert player_mapping.normalize_player_name("  Al  Horford ") == "AL HORFORD"


def test_build_player_metadata_for_season_enriches_aliases():
    """Season metadata should merge roster info with static player aliases."""
    metadata = player_metadata.build_player_metadata_for_season("2024-25", _FakeProvider())

    assert not metadata.empty
    assert {"player_id", "player_idx", "team_abbr", "aliases"}.issubset(metadata.columns)
    horford = metadata[metadata["player_id"] == 201143].iloc[0]
    assert "AL HORFORD" in horford["aliases"]


def test_resolve_player_name_accepts_last_first_alias():
    """Alias lookup should resolve injury-style 'Last, First' names."""
    metadata = pd.DataFrame(
        [
            {
                "player_id": 201143,
                "aliases": ["AL HORFORD", "HORFORD, AL"],
            }
        ]
    )
    assert player_metadata.resolve_player_name("Horford, Al", metadata) == 201143


def test_resolve_player_name_prefers_team_filtered_roster():
    """Team-aware resolution should disambiguate shared aliases across rosters."""
    metadata = pd.DataFrame(
        [
            {
                "player_id": 101,
                "team_idx": 5,
                "aliases": ["J WILLIAMS", "WILLIAMS, J"],
            },
            {
                "player_id": 202,
                "team_idx": 8,
                "aliases": ["J WILLIAMS", "WILLIAMS, J"],
            },
        ]
    )

    assert player_metadata.resolve_player_name("Williams, J", metadata, team_idx=5) == 101
    assert player_metadata.resolve_player_name("Williams, J", metadata, team_idx=8) == 202


def test_fetch_player_info_normalizes_team_roster_rows(monkeypatch, tmp_path):
    """The provider should normalize CommonTeamRoster output into the project schema."""

    class _FakeRosterEndpoint:
        def __init__(self, team_id: int, season: str):
            assert team_id == 1610612738
            assert season == "2024-25"

        def get_data_frames(self) -> list[pd.DataFrame]:
            return [
                pd.DataFrame(
                    [
                        {
                            "PLAYER_ID": 201143,
                            "PLAYER": "Al Horford",
                            "POSITION": "C-F",
                            "HEIGHT": "6-9",
                            "WEIGHT": "240",
                            "BIRTH_DATE": "JUN 03, 1986",
                            "EXP": "17",
                            "HOW_ACQUIRED": "Free Agent",
                        }
                    ]
                )
            ]

    monkeypatch.setattr(
        "src.data.providers.nba_api_provider.nba_teams.get_teams",
        lambda: [{"id": 1610612738, "abbreviation": "BOS"}],
    )
    monkeypatch.setattr(
        "src.data.providers.nba_api_provider.commonteamroster.CommonTeamRoster",
        _FakeRosterEndpoint,
    )

    provider = NbaApiProvider(cache_dir=tmp_path)
    roster = provider.fetch_player_info("2024-25")

    assert list(roster["team_abbr"].unique()) == ["BOS"]
    assert roster.iloc[0]["player_id"] == 201143
    assert roster.iloc[0]["player_name"] == "Al Horford"
