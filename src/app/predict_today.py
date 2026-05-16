"""Script to fetch today's NBA games and run the prediction pipeline.

Usage:
    python -m src.app.predict_today
    python -m src.app.predict_today --date 2026-05-16
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import scoreboardv2

from src.models.predict import PredictionPipeline
from src.utils.logging import setup_logging
from src.utils.paths import DATA_DIR, PROCESSED_DIR, PROJECT_ROOT, PREDICTIONS_DIR

logger = logging.getLogger(__name__)

def fetch_schedule(date_str: str) -> pd.DataFrame:
    """Fetch the schedule for a given date."""
    logger.info("Fetching schedule for %s...", date_str)
    
    # We use scoreboardv2 for backward compatibility despite line score warning
    sb = scoreboardv2.ScoreboardV2(game_date=date_str)
    df = sb.get_data_frames()[0]
    
    if len(df) == 0:
        logger.info("No games scheduled for %s.", date_str)
        return pd.DataFrame()
        
    # Load team mappings
    with open(DATA_DIR / "mappings" / "team_to_idx.json") as f:
        team_mapping = json.load(f)
        
    games = []
    for _, row in df.iterrows():
        home_id = str(row["HOME_TEAM_ID"])
        away_id = str(row["VISITOR_TEAM_ID"])
        
        # Format season: "2025" -> "2025-26"
        season_year = int(row["SEASON"])
        season_str = f"{season_year}-{(season_year + 1) % 100:02d}"
        
        if home_id in team_mapping and away_id in team_mapping:
            games.append({
                "game_id": str(row["GAME_ID"]),
                "date": date_str,
                "season": season_str,
                "home_team_idx": team_mapping[home_id],
                "away_team_idx": team_mapping[away_id],
            })
            
    logger.info("Found %d scheduled games.", len(games))
    return pd.DataFrame(games)


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Predict today's NBA games.")
    parser.add_argument("--date", type=str, default=datetime.today().strftime("%Y-%m-%d"),
                        help="Date to predict for in YYYY-MM-DD format.")
    args = parser.parse_args()
    
    date_str = args.date
    
    # 1. Fetch Schedule
    target_games = fetch_schedule(date_str)
    if target_games.empty:
        return
        
    # 2. Load Historical Data
    logger.info("Loading historical games and team logs...")
    hist_games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    hist_logs = pd.read_parquet(PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet")
    
    # Note: If the pipeline is run daily, hist_games might already include some games from today 
    # if fetch_boxscores was run early. We should filter out any games from >= date_str 
    # to avoid leakage (so we only use past context).
    hist_games = hist_games[hist_games["date"] < date_str].copy()
    hist_logs = hist_logs[hist_logs["date"] < date_str].copy()
    
    # 3. Initialize Pipeline and Predict
    pipeline = PredictionPipeline()
    results = pipeline.predict_games(target_games, hist_games, hist_logs)
    
    # 4. Save JSON Output
    output = {
        "date": date_str,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": "ensemble_v1",
        "predictions": results
    }
    
    out_dir = PREDICTIONS_DIR / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{date_str}.json"
    
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
        
    logger.info("Saved daily predictions to %s", out_file)
    
    # Print preview
    for res in results:
        logger.info(
            "Game %s | Home Win Prob: %.1f%% | Confidence: %s",
            res["game_id"],
            res["home_win_probability"] * 100,
            res["confidence_bucket"]
        )

if __name__ == "__main__":
    main()
