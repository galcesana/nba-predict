"""Training loop for the matchup fusion model.

Handles:
  - Loading sequence data and context features
  - Time-based train/val/test splits
  - Training with AdamW, early stopping, checkpointing
  - Evaluation on validation and test sets

Usage:
    python -m src.models.train
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.features.sequence_builder import (
    SEQUENCE_FEATURES,
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


def create_dataloader(
    sequences: dict,
    context: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for a split."""
    dataset = TensorDataset(
        torch.from_numpy(sequences["home_sequences"][indices]),
        torch.from_numpy(sequences["away_sequences"][indices]),
        torch.from_numpy(sequences["home_masks"][indices]),
        torch.from_numpy(sequences["away_masks"][indices]),
        torch.from_numpy(context[indices]),
        torch.from_numpy(sequences["targets"][indices]).unsqueeze(1),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: MatchupFusionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for home_seq, away_seq, home_mask, away_mask, context, target in dataloader:
        home_seq = home_seq.to(device)
        away_seq = away_seq.to(device)
        home_mask = home_mask.to(device)
        away_mask = away_mask.to(device)
        context = context.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(home_seq, away_seq, context, home_mask, away_mask)
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
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model, return (loss, predictions, actuals)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    for home_seq, away_seq, home_mask, away_mask, context, target in dataloader:
        home_seq = home_seq.to(device)
        away_seq = away_seq.to(device)
        home_mask = home_mask.to(device)
        away_mask = away_mask.to(device)
        context = context.to(device)
        target = target.to(device)

        pred = model(home_seq, away_seq, context, home_mask, away_mask)
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
    train_loader = create_dataloader(sequences, context, train_idx, batch_size, shuffle=True)
    val_loader = create_dataloader(sequences, context, val_idx, batch_size, shuffle=False)
    test_loader = create_dataloader(sequences, context, test_idx, batch_size, shuffle=False)

    # Build model
    game_feature_dim = sequences["home_sequences"].shape[2]
    context_feature_dim = context.shape[1]

    model = MatchupFusionModel(
        game_feature_dim=game_feature_dim,
        context_feature_dim=context_feature_dim,
        hidden_dim=neural_config["hidden_dim"],
        num_gru_layers=neural_config["num_layers"],
        dropout=neural_config["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d", n_params)

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
    best_model_path = NEURAL_DIR / "best_model.pt"

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        val_acc = ((val_preds >= 0.5) == val_targets).mean()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))

        if epoch % 5 == 0 or epoch == 1:
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
    test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
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
        "best_epoch": len(history["val_loss"]) - patience,
        "total_epochs": len(history["val_loss"]),
        "n_params": n_params,
    }

    # Save predictions
    test_preds_df = pd.DataFrame({
        "game_id": sequences["game_ids"][test_idx],
        "pred_home_win": test_preds,
        "actual_home_win": test_targets,
    })
    test_preds_df.to_parquet(NEURAL_DIR / "neural_predictions_test.parquet", index=False)

    val_preds_df = pd.DataFrame({
        "game_id": sequences["game_ids"][val_idx],
        "pred_home_win": val_preds,
        "actual_home_win": val_targets,
    })
    val_preds_df.to_parquet(NEURAL_DIR / "neural_predictions_val.parquet", index=False)

    # Save training history and results
    with open(NEURAL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(NEURAL_DIR / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    logger.info("Model and results saved to %s", NEURAL_DIR)
    return model, test_results


def main():
    """Build sequences and train model."""
    setup_logging()

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

    # Get game seasons for splitting
    games_sorted = games.sort_values("date")
    game_seasons = games_sorted["season"].values

    # Train
    logger.info("Starting training...")
    model, results = train_model(sequences, context, game_seasons, config)

    logger.info("Training complete!")
    logger.info("Test accuracy: %.3f, Test loss: %.4f", results["test_accuracy"], results["test_loss"])


if __name__ == "__main__":
    main()
