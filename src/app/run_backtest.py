"""Script to run historical backtesting of the prediction pipeline.

Usage:
    python -m src.app.run_backtest --start-date 2024-01-01 --end-date 2024-01-07
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from src.models.predict import PredictionPipeline
from src.utils.logging import setup_logging
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, PREDICTIONS_DIR

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Run historical backtest.")
    parser.add_argument("--start-date", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()
    
    start_date = args.start_date
    end_date = args.end_date
    
    logger.info("Loading full historical dataset...")
    full_games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    full_logs = pd.read_parquet(PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet")
    
    # Filter games in date range
    mask = (full_games["date"] >= start_date) & (full_games["date"] <= end_date)
    test_games = full_games[mask].copy()
    
    if test_games.empty:
        logger.info("No games found in the specified date range.")
        return
        
    logger.info("Found %d games to backtest.", len(test_games))
    
    pipeline = PredictionPipeline()
    all_results = []
    
    # We predict day by day to simulate reality and prevent leakage
    dates = sorted(test_games["date"].unique())
    
    for current_date in dates:
        logger.info("Backtesting date: %s", current_date)
        
        # Target games for this day
        daily_target = test_games[test_games["date"] == current_date]
        
        # Historical context up to yesterday
        hist_games = full_games[full_games["date"] < current_date].copy()
        hist_logs = full_logs[full_logs["date"] < current_date].copy()
        
        # We need to strip out "home_win" from daily_target so pipeline doesn't see it
        # But we need to keep it to evaluate later
        actuals = daily_target[["game_id", "home_win"]].copy()
        daily_target_clean = daily_target.drop(columns=["home_win"])
        
        # Predict
        results = pipeline.predict_games(daily_target_clean, hist_games, hist_logs)
        
        # Attach actuals to results for evaluation
        actuals_dict = dict(zip(actuals["game_id"], actuals["home_win"]))
        for res in results:
            res["actual_home_win"] = int(actuals_dict[res["game_id"]])
            all_results.append(res)
            
    # Evaluate
    y_true = [res["actual_home_win"] for res in all_results]
    y_prob = [res["home_win_probability"] for res in all_results]
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_prob)
    
    logger.info("Backtest Complete!")
    logger.info("Accuracy: %.3f", acc)
    logger.info("Log Loss: %.4f", ll)
    
    # Save Report
    report = {
        "start_date": start_date,
        "end_date": end_date,
        "total_games": len(all_results),
        "accuracy": float(acc),
        "log_loss": float(ll),
        "predictions": all_results
    }
    
    out_dir = PREDICTIONS_DIR / "historical_backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"backtest_{start_date}_to_{end_date}.json"
    
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info("Saved backtest report to %s", out_file)

if __name__ == "__main__":
    main()
