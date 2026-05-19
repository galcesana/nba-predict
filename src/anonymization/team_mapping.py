"""Team name ↔ anonymous index mapping utilities.

Loads team_to_idx.json and provides bidirectional lookups.
"""

import json
from functools import lru_cache

from src.utils.paths import MAPPINGS_DIR


@lru_cache(maxsize=1)
def load_team_to_idx() -> dict[str, int]:
    """Load team abbreviation → index mapping."""
    path = MAPPINGS_DIR / "team_to_idx.json"
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_idx_to_team() -> dict[int, str]:
    """Load index → team abbreviation mapping (reverse of team_to_idx)."""
    return {v: k for k, v in load_team_to_idx().items()}


def team_abbr_to_idx(abbr: str) -> int:
    """Convert team abbreviation to anonymous index.

    Args:
        abbr: Team abbreviation (e.g., "BOS").

    Returns:
        Integer index (0-29).

    Raises:
        KeyError: If abbreviation not found.
    """
    mapping = load_team_to_idx()
    if abbr not in mapping:
        raise KeyError(f"Unknown team abbreviation: {abbr}. Known: {sorted(mapping.keys())}")
    return mapping[abbr]


def idx_to_team_abbr(idx: int) -> str:
    """Convert anonymous index to team abbreviation.

    Args:
        idx: Team index (0-29).

    Returns:
        Team abbreviation string.

    Raises:
        KeyError: If index not found.
    """
    mapping = load_idx_to_team()
    if idx not in mapping:
        raise KeyError(f"Unknown team index: {idx}. Valid range: 0-29.")
    return mapping[idx]
