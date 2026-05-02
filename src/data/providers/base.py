"""Abstract base class for all NBA data sources.

All data fetching in this project goes through a DataProvider implementation.
This abstraction allows swapping data sources (e.g., nba_api → Basketball Reference)
without touching feature engineering or model code.
"""

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    """Abstract base for all NBA data sources.

    Implementations must handle:
    - Rate limiting (see NbaApiProvider for reference)
    - Response validation
    - Caching of raw responses as Parquet snapshots
    """

    @abstractmethod
    def fetch_games(self, season: str) -> pd.DataFrame:
        """Fetch all games for a season.

        Args:
            season: Season string, e.g. "2024-25".

        Returns:
            DataFrame with columns:
                game_id, date, season, home_team_abbr, away_team_abbr,
                home_score, away_score, arena, start_time_utc
        """
        ...

    @abstractmethod
    def fetch_team_game_logs(self, season: str) -> pd.DataFrame:
        """Fetch team-level game logs for a season.

        Args:
            season: Season string, e.g. "2024-25".

        Returns:
            DataFrame with one row per team per game, including:
                game_id, date, team_abbr, opponent_abbr, is_home, won,
                points_for, points_against, and advanced stats.
        """
        ...

    @abstractmethod
    def fetch_box_scores(self, game_id: str) -> pd.DataFrame:
        """Fetch player-level box score for a single game.

        Args:
            game_id: NBA game ID string, e.g. "0022400001".

        Returns:
            DataFrame with one row per player, including:
                player_id, player_name, team_abbr, minutes, points,
                rebounds, assists, and other box score stats.
        """
        ...

    @abstractmethod
    def fetch_player_info(self, season: str) -> pd.DataFrame:
        """Fetch player roster and info for a season.

        Args:
            season: Season string, e.g. "2024-25".

        Returns:
            DataFrame with columns:
                player_id, player_name, team_abbr, position, height,
                weight, birth_date, experience
        """
        ...
