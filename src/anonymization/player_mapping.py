"""Player identity to anonymous index mapping utilities."""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from nba_api.stats.static import players as nba_players

from src.utils.paths import MAPPINGS_DIR

PLAYER_MAPPING_PATH = MAPPINGS_DIR / "player_to_idx.json"


def normalize_player_name(name: str) -> str:
    """Normalize a player name for stable alias matching."""
    normalized = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.upper().strip()
    normalized = normalized.replace(".", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def build_player_to_idx_mapping(
    player_rows: Iterable[dict[str, object]] | None = None,
) -> dict[int, int]:
    """Build a stable player_id -> anonymous index mapping."""
    rows = list(player_rows) if player_rows is not None else list(nba_players.get_players())
    player_ids = sorted(
        {
            int(row["id"])
            for row in rows
            if row.get("id") is not None
        }
    )
    return {player_id: idx for idx, player_id in enumerate(player_ids)}


def extend_player_to_idx_mapping(
    player_ids: Iterable[int],
    *,
    base_mapping: dict[int, int] | None = None,
) -> dict[int, int]:
    """Extend a mapping with observed player ids, preserving existing assignments."""
    mapping = dict(base_mapping) if base_mapping is not None else load_player_to_idx()
    next_idx = max(mapping.values(), default=-1) + 1
    for player_id in sorted({int(player_id) for player_id in player_ids}):
        if player_id not in mapping:
            mapping[player_id] = next_idx
            next_idx += 1
    return mapping


def _write_mapping(mapping: dict[int, int], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(player_id): idx for player_id, idx in mapping.items()}
    target.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_player_mapping(
    path: Path | None = None,
    *,
    extra_player_ids: Iterable[int] | None = None,
) -> Path:
    """Persist the current player mapping to disk."""
    target = path or PLAYER_MAPPING_PATH
    mapping = build_player_to_idx_mapping()
    if extra_player_ids is not None:
        mapping = extend_player_to_idx_mapping(extra_player_ids, base_mapping=mapping)
    return _write_mapping(mapping, target)


@lru_cache(maxsize=1)
def load_player_to_idx(path: str | None = None) -> dict[int, int]:
    """Load player_id -> anonymous index mapping."""
    mapping_path = Path(path) if path is not None else PLAYER_MAPPING_PATH
    if mapping_path.exists():
        with open(mapping_path, encoding="utf-8") as handle:
            mapping = json.load(handle)
        return {int(player_id): int(idx) for player_id, idx in mapping.items()}
    return build_player_to_idx_mapping()


@lru_cache(maxsize=1)
def load_idx_to_player(path: str | None = None) -> dict[int, int]:
    """Load anonymous index -> player_id reverse mapping."""
    return {idx: player_id for player_id, idx in load_player_to_idx(path).items()}


def player_id_to_idx(player_id: int, path: str | None = None) -> int:
    """Convert player_id to anonymous index."""
    mapping = load_player_to_idx(path)
    player_id = int(player_id)
    if player_id not in mapping:
        raise KeyError(f"Unknown player_id: {player_id}")
    return mapping[player_id]


def ensure_player_id_mapping(
    player_ids: Iterable[int],
    *,
    path: Path | None = None,
) -> dict[int, int]:
    """Ensure the persisted mapping includes the provided player ids."""
    target = path or PLAYER_MAPPING_PATH
    existing = load_player_to_idx(str(target) if path is not None else None)
    extended = extend_player_to_idx_mapping(player_ids, base_mapping=existing)
    if extended != existing or not target.exists():
        _write_mapping(extended, target)
        load_player_to_idx.cache_clear()
        load_idx_to_player.cache_clear()
    return load_player_to_idx(str(target) if path is not None else None)


def idx_to_player_id(idx: int, path: str | None = None) -> int:
    """Convert anonymous index back to player_id."""
    reverse = load_idx_to_player(path)
    idx = int(idx)
    if idx not in reverse:
        raise KeyError(f"Unknown player index: {idx}")
    return reverse[idx]
