"""Helpers for loading, resolving, and enriching NBA player metadata."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import pandas as pd
from nba_api.stats.static import players as nba_players

from src.anonymization.player_mapping import ensure_player_id_mapping, normalize_player_name
from src.anonymization.team_mapping import team_abbr_to_idx


class PlayerInfoProvider(Protocol):
    """Protocol for providers that can load season roster info."""

    def fetch_player_info(self, season: str) -> pd.DataFrame: ...


def _player_aliases(row: pd.Series) -> set[str]:
    aliases = {
        normalize_player_name(row.get("full_name", "")),
    }
    first_name = normalize_player_name(row.get("first_name", ""))
    last_name = normalize_player_name(row.get("last_name", ""))
    if first_name and last_name:
        aliases.add(f"{last_name}, {first_name}")
        aliases.add(f"{first_name} {last_name}")
    return {alias for alias in aliases if alias}


@lru_cache(maxsize=1)
def load_static_player_metadata() -> pd.DataFrame:
    """Return normalized static player metadata from nba_api."""
    rows = pd.DataFrame(nba_players.get_players())
    rows = rows.rename(columns={"id": "player_id", "full_name": "full_name"})
    rows["player_id"] = rows["player_id"].astype(int)
    player_mapping = ensure_player_id_mapping(rows["player_id"].tolist())
    rows["player_idx"] = rows["player_id"].map(player_mapping)
    rows["normalized_name"] = rows["full_name"].map(normalize_player_name)
    rows["aliases"] = rows.apply(_player_aliases, axis=1)
    return rows.sort_values("player_id").reset_index(drop=True)


def build_player_metadata_for_season(
    season: str,
    provider: PlayerInfoProvider,
) -> pd.DataFrame:
    """Build season roster metadata enriched with static player aliases."""
    roster = provider.fetch_player_info(season).copy()
    if roster.empty:
        return pd.DataFrame()

    roster["player_id"] = roster["player_id"].astype(int)
    player_mapping = ensure_player_id_mapping(roster["player_id"].tolist())
    static_columns = [
        "player_id",
        "player_idx",
        "full_name",
        "first_name",
        "last_name",
        "is_active",
        "aliases",
    ]
    merged = roster.merge(
        load_static_player_metadata()[static_columns],
        on="player_id",
        how="left",
    )
    merged["player_name"] = merged["player_name"].fillna(merged["full_name"])
    merged["player_idx"] = merged["player_id"].map(player_mapping)
    merged["team_idx"] = merged["team_abbr"].map(team_abbr_to_idx)
    merged["normalized_name"] = merged["player_name"].map(normalize_player_name)
    merged["aliases"] = merged.apply(
        lambda row: sorted(
            set(row["aliases"] if isinstance(row["aliases"], set) else set())
            .union(_player_aliases(row))
            .union({row["normalized_name"]})
        ),
        axis=1,
    )
    return merged.sort_values(["team_abbr", "player_name"]).reset_index(drop=True)


def build_player_name_lookup(metadata: pd.DataFrame) -> dict[str, int]:
    """Build normalized alias -> player_id lookup."""
    lookup: dict[str, int] = {}
    for _, row in metadata.iterrows():
        player_id = int(row["player_id"])
        for alias in row.get("aliases", []):
            lookup[normalize_player_name(alias)] = player_id
    return lookup


def resolve_player_name(
    name: str,
    metadata: pd.DataFrame,
    *,
    team_idx: int | None = None,
) -> int | None:
    """Resolve a raw player name to player_id using season-aware aliases."""
    candidate_metadata = metadata
    if team_idx is not None and "team_idx" in metadata.columns:
        candidate_metadata = metadata[metadata["team_idx"] == int(team_idx)]
        if candidate_metadata.empty:
            candidate_metadata = metadata

    lookup = build_player_name_lookup(candidate_metadata)
    return lookup.get(normalize_player_name(name))
