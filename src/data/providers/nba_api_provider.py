"""NBA API data provider implementation.

Wraps the nba_api library with:
- Rate limiting (1 request/second)
- Exponential backoff on failures
- Response schema validation
- Parquet caching for raw responses
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams as nba_teams

from .base import DataProvider

logger = logging.getLogger(__name__)


class NbaApiProvider(DataProvider):
    """DataProvider implementation using the nba_api library.

    All methods include rate limiting and retry logic.
    Raw responses are cached as Parquet files in the cache directory.
    """

    REQUEST_DELAY: float = 1.0  # seconds between API requests
    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: int = 2
    MAX_BACKOFF: int = 60  # seconds

    def __init__(self, cache_dir: Path | str | None = None):
        """Initialize provider.

        Args:
            cache_dir: Directory to cache raw API responses as Parquet.
                       Defaults to data/raw/nba_api/ relative to project root.
        """
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            from src.utils.paths import RAW_DIR
            self.cache_dir = RAW_DIR / "nba_api"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _request_with_retry(self, fetch_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a fetch function with rate limiting and exponential backoff.

        Args:
            fetch_fn: Callable that performs the actual API request.
            *args, **kwargs: Arguments passed to fetch_fn.

        Returns:
            The result of fetch_fn.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                time.sleep(self.REQUEST_DELAY)
                result = fetch_fn(*args, **kwargs)
                return result
            except Exception as e:
                wait = min(self.BACKOFF_FACTOR ** (attempt + 1), self.MAX_BACKOFF)
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %ds.",
                    attempt + 1,
                    self.MAX_RETRIES,
                    str(e),
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Failed after {self.MAX_RETRIES} retries for {fetch_fn.__name__}"
        )

    def _get_cache_path(self, name: str, season: str | None = None) -> Path:
        """Build a cache file path.

        Args:
            name: Base name for the cache file (e.g., "games", "boxscore_0022400001").
            season: Optional season string to include in the filename.

        Returns:
            Path to the Parquet cache file.
        """
        if season:
            return self.cache_dir / f"{name}_{season.replace('-', '_')}.parquet"
        return self.cache_dir / f"{name}.parquet"

    def _load_from_cache(self, cache_path: Path) -> pd.DataFrame | None:
        """Load a DataFrame from cache if it exists.

        Returns:
            DataFrame if cache exists, None otherwise.
        """
        if cache_path.exists():
            logger.info("Loading from cache: %s", cache_path)
            return pd.read_parquet(cache_path)
        return None

    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path) -> None:
        """Save a DataFrame to cache as Parquet."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info("Cached %d rows to %s", len(df), cache_path)

    # ---- DataProvider interface methods (implemented in Phase 1) ----

    def fetch_games(self, season: str) -> pd.DataFrame:
        """Fetch all games for a season. Implemented in Phase 1."""
        raise NotImplementedError("fetch_games will be implemented in Phase 1")

    def fetch_team_game_logs(self, season: str) -> pd.DataFrame:
        """Fetch team game logs for a season. Implemented in Phase 1."""
        raise NotImplementedError("fetch_team_game_logs will be implemented in Phase 1")

    def fetch_box_scores(self, game_id: str) -> pd.DataFrame:
        """Fetch box score for a game. Implemented in Phase 1."""
        raise NotImplementedError("fetch_box_scores will be implemented in Phase 1")

    def fetch_player_info(self, season: str) -> pd.DataFrame:
        """Fetch season roster info for all NBA teams."""
        cache_path = self._get_cache_path("player_info", season)
        cached = self._load_from_cache(cache_path)
        if cached is not None:
            return cached

        rows: list[pd.DataFrame] = []
        for team in nba_teams.get_teams():
            team_id = int(team["id"])
            team_abbr = str(team["abbreviation"]).upper()
            roster = self._request_with_retry(
                commonteamroster.CommonTeamRoster,
                team_id=team_id,
                season=season,
            ).get_data_frames()[0]
            if roster.empty:
                continue

            roster = roster.rename(
                columns={
                    "PLAYER_ID": "player_id",
                    "PLAYER": "player_name",
                    "POSITION": "position",
                    "HEIGHT": "height",
                    "WEIGHT": "weight",
                    "BIRTH_DATE": "birth_date",
                    "EXP": "experience",
                    "HOW_ACQUIRED": "how_acquired",
                }
            )
            roster["team_abbr"] = team_abbr
            roster["season"] = season
            rows.append(
                roster[
                    [
                        "season",
                        "player_id",
                        "player_name",
                        "team_abbr",
                        "position",
                        "height",
                        "weight",
                        "birth_date",
                        "experience",
                        "how_acquired",
                    ]
                ].copy()
            )

        result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if not result.empty:
            result["player_id"] = result["player_id"].astype(int)
        self._save_to_cache(result, cache_path)
        return result
