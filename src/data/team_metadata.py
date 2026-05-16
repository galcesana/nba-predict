"""Helpers for mapping NBA team ids, names, and display aliases."""

from __future__ import annotations

from functools import lru_cache

from nba_api.stats.static import teams as nba_teams

from src.anonymization.team_mapping import load_team_to_idx

_EXTRA_TEAM_ALIASES = {
    "LAC": {"LOS ANGELES CLIPPERS"},
    "LAL": {"LOS ANGELES LAKERS"},
    "NOP": {"NEW ORLEANS", "NEW ORLEANS PELS"},
    "NYK": {"NEW YORK"},
    "GSW": {"GOLDEN STATE"},
    "OKC": {"OKLAHOMA CITY"},
    "PHX": {"PHOENIX"},
    "SAS": {"SAN ANTONIO"},
}


@lru_cache(maxsize=1)
def load_team_metadata_by_idx() -> dict[int, dict[str, object]]:
    """Return current NBA team metadata keyed by anonymous team index."""
    abbr_to_idx = load_team_to_idx()
    metadata: dict[int, dict[str, object]] = {}

    for team in nba_teams.get_teams():
        abbr = str(team["abbreviation"]).upper()
        if abbr not in abbr_to_idx:
            continue

        idx = int(abbr_to_idx[abbr])
        full_name = str(team["full_name"])
        city = str(team["city"])
        nickname = str(team["nickname"])
        aliases = {
            abbr,
            full_name.upper(),
            city.upper(),
            nickname.upper(),
            f"{city} {nickname}".upper(),
            *{alias.upper() for alias in _EXTRA_TEAM_ALIASES.get(abbr, set())},
        }
        metadata[idx] = {
            "team_idx": idx,
            "abbreviation": abbr,
            "full_name": full_name,
            "city": city,
            "nickname": nickname,
            "aliases": sorted(aliases),
        }

    return metadata


@lru_cache(maxsize=1)
def load_team_name_lookup() -> dict[str, int]:
    """Return an uppercase name/alias lookup to anonymous team index."""
    lookup: dict[str, int] = {}
    for team_idx, metadata in load_team_metadata_by_idx().items():
        for alias in metadata["aliases"]:
            lookup[str(alias).upper()] = team_idx
    return lookup
