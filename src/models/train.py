"""Training loop for the matchup fusion model.

Handles:
  - Loading sequence data, context, injury, and news features
  - Time-based train/val/test splits
  - Training with AdamW, early stopping, checkpointing
  - Ablation study across feature stream combinations
  - Evaluation on validation and test sets

Usage:
    python -m src.models.train
    python -m src.models.train --ablation    # run full ablation study
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.features.injury_features import INJURY_FEATURE_COLS
from src.features.news_features import NEWS_FEATURE_COLS
from src.features.sequence_builder import (
    build_context_features,
    build_team_sequences,
)
from src.models.matchup_fusion_model import MatchupFusionModel
from src.utils.logging import setup_logging
from src.utils.paths import CONFIGS_DIR, MODELS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

NEURAL_DIR = MODELS_DIR / "neural"


def _load_config() -> dict:
    """Load full model config."""
    with open(CONFIGS_DIR / "model_config.yaml") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int = 42) -> None:
    """Set reproducibility seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device(config: dict) -> torch.device:
    """Determine device from config."""
    device_str = config.get("device", "auto")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _split_indices_by_season(
    game_seasons: np.ndarray,
    split_config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split game indices by season."""
    train_end = split_config["train_end"]
    val_season = split_config["validation"]
    test_seasons = split_config["test"]

    def season_start_year(s: str) -> int:
        return int(s.split("-")[0])

    train_end_year = season_start_year(train_end)

    train_idx = np.array([
        i for i, s in enumerate(game_seasons)
        if season_start_year(s) <= train_end_year
    ])
    val_idx = np.array([
        i for i, s in enumerate(game_seasons) if s == val_season
    ])
    test_idx = np.array([
        i for i, s in enumerate(game_seasons) if s in test_seasons
    ])

    return train_idx, val_idx, test_idx


def load_injury_news_features(
    games: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load injury and news feature arrays aligned with games.

    Returns:
        (home_injury, away_injury, home_news, away_news, news_available)
        Each has shape [n_games, n_features].
    """
    games_sorted = games.sort_values("date").reset_index(drop=True)

    # Injury features (exclude game_id, team_idx, injury_data_available from model input)
    injury_cols = [c for c in INJURY_FEATURE_COLS if c != "injury_data_available"]
    injury_path = PROCESSED_DIR / "injury_features" / "injury_features.parquet"
    if injury_path.exists():
        inj_df = pd.read_parquet(injury_path)
        home_inj_list = []
        away_inj_list = []
        for _, game in games_sorted.iterrows():
            gid = game["game_id"]
            h_idx = int(game["home_team_idx"])
            a_idx = int(game["away_team_idx"])
            h_row = inj_df[(inj_df["game_id"] == gid) & (inj_df["team_idx"] == h_idx)]
            a_row = inj_df[(inj_df["game_id"] == gid) & (inj_df["team_idx"] == a_idx)]
            if len(h_row) > 0:
                home_inj_list.append(h_row[injury_cols].values[0])
            else:
                home_inj_list.append(np.zeros(len(injury_cols)))
            if len(a_row) > 0:
                away_inj_list.append(a_row[injury_cols].values[0])
            else:
                away_inj_list.append(np.zeros(len(injury_cols)))
        home_injury = np.array(home_inj_list, dtype=np.float32)
        away_injury = np.array(away_inj_list, dtype=np.float32)
    else:
        n = len(games_sorted)
        home_injury = np.zeros((n, len(injury_cols)), dtype=np.float32)
        away_injury = np.zeros((n, len(injury_cols)), dtype=np.float32)

    # News features (exclude game_id, team_idx from model input; keep news_available separate)
    news_cols = [c for c in NEWS_FEATURE_COLS if c != "news_available"]
    news_path = PROCESSED_DIR / "news_features" / "news_features.parquet"
    if news_path.exists():
        news_df = pd.read_parquet(news_path)
        home_news_list = []
        away_news_list = []
        news_avail_list = []
        for _, game in games_sorted.iterrows():
            gid = game["game_id"]
            h_idx = int(game["home_team_idx"])
            a_idx = int(game["away_team_idx"])
            h_row = news_df[(news_df["game_id"] == gid) & (news_df["team_idx"] == h_idx)]
            a_row = news_df[(news_df["game_id"] == gid) & (news_df["team_idx"] == a_idx)]
            if len(h_row) > 0:
                home_news_list.append(h_row[news_cols].values[0])
                # Use max of home/away news_available
                na_val = float(h_row["news_available"].values[0])
            else:
                home_news_list.append(np.zeros(len(news_cols)))
                na_val = 0.0
            if len(a_row) > 0:
                away_news_list.append(a_row[news_cols].values[0])
                na_val = max(na_val, float(a_row["news_available"].values[0]))
            else:
                away_news_list.append(np.zeros(len(news_cols)))
            news_avail_list.append(na_val)
        home_news = np.array(home_news_list, dtype=np.float32)
        away_news = np.array(away_news_list, dtype=np.float32)
        news_available = np.array(news_avail_list, dtype=np.float32).reshape(-1, 1)
    else:
        n = len(games_sorted)
        home_news = np.zeros((n, len(news_cols)), dtype=np.float32)
        away_news = np.zeros((n, len(news_cols)), dtype=np.float32)
        news_available = np.zeros((n, 1), dtype=np.float32)

    return home_injury, away_injury, home_news, away_news, news_available


def create_dataloader(
    sequences: dict,
    context: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    home_injury: np.ndarray | None = None,
    away_injury: np.ndarray | None = None,
    home_news: np.ndarray | None = None,
    away_news: np.ndarray | None = None,
    news_available: np.ndarray | None = None,
) -> DataLoader:
    """Create a DataLoader for a split."""
    tensors = [
        torch.from_numpy(sequences["home_sequences"][indices]),
        torch.from_numpy(sequences["away_sequences"][indices]),
        torch.from_numpy(sequences["home_masks"][indices]),
        torch.from_numpy(sequences["away_masks"][indices]),
        torch.from_numpy(context[indices]),
        torch.from_numpy(sequences["targets"][indices]).unsqueeze(1),
    ]

    if home_injury is not None:
        tensors.append(torch.from_numpy(home_injury[indices]))
        tensors.append(torch.from_numpy(away_injury[indices]))
    if home_news is not None:
        tensors.append(torch.from_numpy(home_news[indices]))
        tensors.append(torch.from_numpy(away_news[indices]))
        tensors.append(torch.from_numpy(news_available[indices]))

    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: MatchupFusionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    has_injury: bool = False,
    has_news: bool = False,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        # Unpack base tensors
        home_seq = batch[0].to(device)
        away_seq = batch[1].to(device)
        home_mask = batch[2].to(device)
        away_mask = batch[3].to(device)
        context = batch[4].to(device)
        target = batch[5].to(device)

        # Optional injury/news tensors
        kwargs = {}
        idx = 6
        if has_injury:
            kwargs["home_injury"] = batch[idx].to(device)
            kwargs["away_injury"] = batch[idx + 1].to(device)
            idx += 2
        if has_news:
            kwargs["home_news"] = batch[idx].to(device)
            kwargs["away_news"] = batch[idx + 1].to(device)
            kwargs["news_available"] = batch[idx + 2].to(device)

        optimizer.zero_grad()
        pred = model(home_seq, away_seq, context, home_mask, away_mask, **kwargs)
        loss = criterion(pred, target)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: MatchupFusionModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    has_injury: bool = False,
    has_news: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model, return (loss, predictions, actuals)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        home_seq = batch[0].to(device)
        away_seq = batch[1].to(device)
        home_mask = batch[2].to(device)
        away_mask = batch[3].to(device)
        context = batch[4].to(device)
        target = batch[5].to(device)

        kwargs = {}
        idx = 6
        if has_injury:
            kwargs["home_injury"] = batch[idx].to(device)
            kwargs["away_injury"] = batch[idx + 1].to(device)
            idx += 2
        if has_news:
            kwargs["home_news"] = batch[idx].to(device)
            kwargs["away_news"] = batch[idx + 1].to(device)
            kwargs["news_available"] = batch[idx + 2].to(device)

        pred = model(home_seq, away_seq, context, home_mask, away_mask, **kwargs)
        loss = criterion(pred, target)
        total_loss += loss.item()
        n_batches += 1

        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    preds = np.concatenate(all_preds, axis=0).squeeze()
    targets = np.concatenate(all_targets, axis=0).squeeze()
    return avg_loss, preds, targets


def train_model(
    sequences: dict,
    context: np.ndarray,
    game_seasons: np.ndarray,
    config: dict,
    feature_streams: list[str] | None = None,
    home_injury: np.ndarray | None = None,
    away_injury: np.ndarray | None = None,
    home_news: np.ndarray | None = None,
    away_news: np.ndarray | None = None,
    news_available: np.ndarray | None = None,
    save_prefix: str = "",
) -> tuple[MatchupFusionModel, dict]:
    """Full training pipeline with early stopping.

    Returns:
        (best_model, training_history)
    """
    _set_seed(config["random_seed"])
    device = _get_device(config)
    logger.info("Training on device: %s", device)

    neural_config = config["neural"]
    split_config = config["splits"]

    if feature_streams is None:
        feature_streams = ["performance", "context"]

    has_injury = "injury" in feature_streams and home_injury is not None
    has_news = "news" in feature_streams and home_news is not None

    # Split data
    train_idx, val_idx, test_idx = _split_indices_by_season(
        game_seasons, split_config,
    )
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_idx), len(val_idx), len(test_idx),
    )

    # Create dataloaders
    batch_size = neural_config["batch_size"]

    dl_kwargs = {}
    if has_injury:
        dl_kwargs.update(home_injury=home_injury, away_injury=away_injury)
    if has_news:
        dl_kwargs.update(
            home_news=home_news,
            away_news=away_news,
            news_available=news_available,
        )

    train_loader = create_dataloader(
        sequences, context, train_idx, batch_size, shuffle=True, **dl_kwargs
    )
    val_loader = create_dataloader(
        sequences, context, val_idx, batch_size, shuffle=False, **dl_kwargs
    )
    test_loader = create_dataloader(
        sequences, context, test_idx, batch_size, shuffle=False, **dl_kwargs
    )

    # Build model
    game_feature_dim = sequences["home_sequences"].shape[2]
    context_feature_dim = context.shape[1]
    injury_feature_dim = home_injury.shape[1] if has_injury else 0
    news_feature_dim = home_news.shape[1] if has_news else 0

    model = MatchupFusionModel(
        game_feature_dim=game_feature_dim,
        context_feature_dim=context_feature_dim,
        injury_feature_dim=injury_feature_dim,
        news_feature_dim=news_feature_dim,
        hidden_dim=neural_config["hidden_dim"],
        num_gru_layers=neural_config["num_layers"],
        dropout=neural_config["dropout"],
        feature_streams=feature_streams,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d (streams: %s)", n_params, feature_streams)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=neural_config["learning_rate"],
        weight_decay=neural_config["weight_decay"],
    )
    criterion = nn.BCELoss()

    # Training loop
    max_epochs = neural_config["max_epochs"]
    patience = neural_config["patience"]
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    model_filename = f"{save_prefix}best_model.pt" if save_prefix else "best_model.pt"
    best_model_path = NEURAL_DIR / model_filename

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            has_injury=has_injury, has_news=has_news,
        )
        val_loss, val_preds, val_targets = evaluate(
            model, val_loader, criterion, device,
            has_injury=has_injury, has_news=has_news,
        )
        val_acc = ((val_preds >= 0.5) == val_targets).mean()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d — train_loss=%.4f  val_loss=%.4f  val_acc=%.3f",
                epoch, train_loss, val_loss, val_acc,
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping at epoch %d (best val_loss=%.4f at epoch %d)",
                    epoch, best_val_loss, epoch - patience,
                )
                break

    # Load best model and evaluate on test
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device,
        has_injury=has_injury, has_news=has_news,
    )
    test_acc = ((test_preds >= 0.5) == test_targets).mean()

    logger.info(
        "Test results — loss=%.4f  accuracy=%.3f",
        test_loss, test_acc,
    )

    # Save test predictions
    test_results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "best_val_loss": float(best_val_loss),
        "best_epoch": max(1, len(history["val_loss"]) - patience),
        "total_epochs": len(history["val_loss"]),
        "n_params": n_params,
        "feature_streams": feature_streams,
    }

    # Save predictions
    prefix = save_prefix if save_prefix else ""
    test_preds_df = pd.DataFrame({
        "game_id": sequences["game_ids"][test_idx],
        "pred_home_win": test_preds,
        "actual_home_win": test_targets,
    })
    test_preds_df.to_parquet(NEURAL_DIR / f"{prefix}neural_predictions_test.parquet", index=False)

    val_loss_final, val_preds_final, val_targets_final = evaluate(
        model, val_loader, criterion, device,
        has_injury=has_injury, has_news=has_news,
    )
    val_preds_df = pd.DataFrame({
        "game_id": sequences["game_ids"][val_idx],
        "pred_home_win": val_preds_final,
        "actual_home_win": val_targets_final,
    })
    val_preds_df.to_parquet(NEURAL_DIR / f"{prefix}neural_predictions_val.parquet", index=False)

    # Save training history and results
    with open(NEURAL_DIR / f"{prefix}training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(NEURAL_DIR / f"{prefix}test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    logger.info("Model and results saved to %s (prefix: %s)", NEURAL_DIR, prefix or "none")
    return model, test_results


def run_ablation(
    sequences: dict,
    context: np.ndarray,
    game_seasons: np.ndarray,
    config: dict,
    home_injury: np.ndarray,
    away_injury: np.ndarray,
    home_news: np.ndarray,
    away_news: np.ndarray,
    news_available: np.ndarray,
) -> dict:
    """Run ablation study across feature stream combinations.

    Trains 4 variants:
        A: performance + context (baseline GRU)
        B: performance + context + injury
        C: performance + context + news
        D: performance + context + injury + news (full fusion)

    Returns:
        Dict of variant name -> test results.
    """
    variants = {
        "A_perf_context": ["performance", "context"],
        "B_plus_injury": ["performance", "context", "injury"],
        "C_plus_news": ["performance", "context", "news"],
        "D_full_fusion": ["performance", "context", "injury", "news"],
    }

    ablation_results = {}

    for variant_name, streams in variants.items():
        logger.info("=" * 60)
        logger.info("ABLATION: Training variant %s with streams %s", variant_name, streams)
        logger.info("=" * 60)

        inj_args = {}
        if "injury" in streams:
            inj_args = dict(home_injury=home_injury, away_injury=away_injury)

        news_args = {}
        if "news" in streams:
            news_args = dict(
                home_news=home_news,
                away_news=away_news,
                news_available=news_available,
            )

        _, results = train_model(
            sequences=sequences,
            context=context,
            game_seasons=game_seasons,
            config=config,
            feature_streams=streams,
            save_prefix=f"ablation_{variant_name}_",
            **inj_args,
            **news_args,
        )

        ablation_results[variant_name] = results

    # Save ablation summary
    with open(NEURAL_DIR / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 60)
    for name, res in ablation_results.items():
        logger.info(
            "  %s: accuracy=%.3f  log_loss=%.4f  streams=%s",
            name, res["test_accuracy"], res["test_loss"], res["feature_streams"],
        )

    return ablation_results


def main():
    """Build sequences and train model."""
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true", help="Run full ablation study")
    args, _ = parser.parse_known_args()

    config = _load_config()

    # Load data
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    team_logs = pd.read_parquet(PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet")
    matchup = pd.read_parquet(PROCESSED_DIR / "matchup_rows" / "matchup_dataset.parquet")

    # Build sequences
    seq_config = config["sequence"]
    logger.info("Building sequences (seq_len=%d)...", seq_config["length"])
    sequences = build_team_sequences(
        team_logs, games, seq_len=seq_config["length"],
    )

    # Build context features
    context = build_context_features(matchup, games)

    # Load injury and news features
    logger.info("Loading injury and news features...")
    home_injury, away_injury, home_news, away_news, news_available = (
        load_injury_news_features(games)
    )

    # Get game seasons for splitting
    games_sorted = games.sort_values("date")
    game_seasons = games_sorted["season"].values

    if args.ablation:
        # Run full ablation study
        logger.info("Starting ablation study...")
        run_ablation(
            sequences, context, game_seasons, config,
            home_injury, away_injury, home_news, away_news, news_available,
        )
    else:
        # Train full fusion model
        logger.info("Starting full fusion training...")
        model, results = train_model(
            sequences=sequences,
            context=context,
            game_seasons=game_seasons,
            config=config,
            feature_streams=["performance", "context", "injury", "news"],
            home_injury=home_injury,
            away_injury=away_injury,
            home_news=home_news,
            away_news=away_news,
            news_available=news_available,
        )

        logger.info("Training complete!")
        logger.info(
            "Test accuracy: %.3f, Test loss: %.4f",
            results["test_accuracy"],
            results["test_loss"],
        )


if __name__ == "__main__":
    main()
