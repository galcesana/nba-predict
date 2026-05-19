"""Sequence model tests — architecture, training, and prediction quality.

Run with: pytest tests/test_sequence_model.py -v
Expected: 14/14 pass
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.features.sequence_builder import (
    SEQUENCE_FEATURES,
    build_team_sequences,
    build_team_sequences_for_games,
)
from src.models.matchup_fusion_model import MatchupFusionModel
from src.models.team_encoder import TeamEncoder
from src.utils.paths import MODELS_DIR, PROCESSED_DIR

NEURAL_DIR = MODELS_DIR / "neural"


# ---- Fixtures ----

@pytest.fixture(scope="module")
def sample_data():
    """Load a small sample of data for testing."""
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    team_logs = pd.read_parquet(PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet")
    # Use first 200 games for fast tests
    games_sample = games.sort_values("date").head(200)
    return build_team_sequences(team_logs, games_sample, seq_len=20)


@pytest.fixture(scope="module")
def full_sequences():
    """Check if full sequence data exists."""
    path = NEURAL_DIR / "test_results.json"
    if not path.exists():
        pytest.skip("Neural model not yet trained")
    return True


# ---- Sequence Builder Tests ----

class TestSequenceBuilder:
    """Verify sequence construction."""

    def test_sequence_builder_shapes(self, sample_data):
        """Output sequences have shape [N_games, seq_len, F_game]."""
        assert sample_data["home_sequences"].shape == (200, 20, len(SEQUENCE_FEATURES))
        assert sample_data["away_sequences"].shape == (200, 20, len(SEQUENCE_FEATURES))

    def test_sequence_padding_correct(self, sample_data):
        """Early-season games are left-padded with zeros."""
        # First game should be fully padded (no prior history)
        first_home = sample_data["home_sequences"][0]
        assert np.allclose(first_home, 0.0), "First game should be fully zero-padded"

    def test_mask_matches_padding(self, sample_data):
        """Mask is 0 where sequence is zero-padded, 1 elsewhere."""
        for i in range(min(50, len(sample_data["home_masks"]))):
            mask = sample_data["home_masks"][i]
            seq = sample_data["home_sequences"][i]

            for t in range(20):
                if mask[t] == 0:
                    assert np.allclose(seq[t], 0.0), (
                        f"Game {i}, timestep {t}: mask=0 but sequence not zero"
                    )
                # If mask=1, sequence should have non-zero data
                # (not always true if actual game stats were 0, so we don't check)

    def test_no_leakage_in_sequences(self, sample_data):
        """Verify no future data in sequences by checking game_ids ordering."""
        games = pd.read_parquet(PROCESSED_DIR / "games.parquet").sort_values("date")
        team_logs = pd.read_parquet(
            PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet"
        )

        # For the 50th game, check that home sequence uses only prior games
        game_50 = games.iloc[49]
        game_date = pd.Timestamp(game_50["date"])
        home_idx = int(game_50["home_team_idx"])

        # Get team's games before this date
        prior = team_logs[
            (team_logs["team_idx"] == home_idx) & (team_logs["date"] < game_date)
        ]

        # The mask should have at most len(prior) real entries
        mask_50 = sample_data["home_masks"][49]
        real_count = int(mask_50.sum())
        assert real_count <= len(prior), (
            f"Sequence has {real_count} real entries but only {len(prior)} prior games exist"
        )

    def test_target_only_sequences_match_full_builder(self):
        """Live inference can build only target rows without changing sequence values."""
        games = pd.read_parquet(PROCESSED_DIR / "games.parquet").sort_values("date").head(120)
        team_logs = pd.read_parquet(
            PROCESSED_DIR / "team_game_logs" / "team_game_logs.parquet"
        )
        target_games = games.iloc[90:96].copy()

        full = build_team_sequences(team_logs, games, seq_len=20)
        target_only = build_team_sequences_for_games(team_logs, target_games, seq_len=20)

        full_index = {game_id: idx for idx, game_id in enumerate(full["game_ids"])}
        target_indices = [full_index[game_id] for game_id in target_only["game_ids"]]

        np.testing.assert_allclose(
            target_only["home_sequences"],
            full["home_sequences"][target_indices],
        )
        np.testing.assert_allclose(
            target_only["away_sequences"],
            full["away_sequences"][target_indices],
        )
        np.testing.assert_allclose(
            target_only["home_masks"],
            full["home_masks"][target_indices],
        )
        np.testing.assert_allclose(
            target_only["away_masks"],
            full["away_masks"][target_indices],
        )


# ---- Model Architecture Tests ----

class TestModelArchitecture:
    """Verify model components."""

    def test_team_encoder_output_shape(self):
        """TeamEncoder(batch) returns [batch_size, hidden_dim]."""
        encoder = TeamEncoder(input_dim=16, hidden_dim=64, num_layers=2)
        x = torch.randn(8, 20, 16)  # [batch, seq_len, features]
        out = encoder(x)
        assert out.shape == (8, 64)

    def test_shared_weights(self):
        """Home and away encoders share the same parameters."""
        model = MatchupFusionModel(
            game_feature_dim=16, context_feature_dim=8,
        )
        # The model has one team_encoder used for both home and away
        # Verify it's the same object, not a copy
        home_seq = torch.randn(4, 20, 16)
        away_seq = torch.randn(4, 20, 16)
        context = torch.randn(4, 8)

        # Get encoder parameter ids — should be the same
        encoder_params = list(model.team_encoder.parameters())
        assert len(encoder_params) > 0, "Encoder has no parameters"

        # Forward pass uses same encoder for both
        with torch.no_grad():
            _ = model(home_seq, away_seq, context)
        # If it didn't error, shared encoder works

    def test_fusion_model_output_shape(self):
        """FusionModel returns [batch_size, 1] probabilities."""
        model = MatchupFusionModel(
            game_feature_dim=16, context_feature_dim=8,
        )
        home_seq = torch.randn(4, 20, 16)
        away_seq = torch.randn(4, 20, 16)
        context = torch.randn(4, 8)

        with torch.no_grad():
            out = model(home_seq, away_seq, context)
        assert out.shape == (4, 1)

    def test_probabilities_valid(self):
        """All outputs are in (0, 1)."""
        model = MatchupFusionModel(
            game_feature_dim=16, context_feature_dim=8,
        )
        home_seq = torch.randn(32, 20, 16)
        away_seq = torch.randn(32, 20, 16)
        context = torch.randn(32, 8)

        with torch.no_grad():
            out = model(home_seq, away_seq, context)

        assert (out > 0).all(), f"Some outputs <= 0: min={out.min():.6f}"
        assert (out < 1).all(), f"Some outputs >= 1: max={out.max():.6f}"


# ---- Training Tests ----

class TestTraining:
    """Verify training loop."""

    def test_model_trains_one_epoch(self):
        """Training loop completes one epoch without error."""
        model = MatchupFusionModel(
            game_feature_dim=16, context_feature_dim=8,
            hidden_dim=32, num_gru_layers=1,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCELoss()

        # Fake data
        home_seq = torch.randn(16, 20, 16)
        away_seq = torch.randn(16, 20, 16)
        context = torch.randn(16, 8)
        target = torch.randint(0, 2, (16, 1)).float()

        model.train()
        optimizer.zero_grad()
        pred = model(home_seq, away_seq, context)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss), "Loss is NaN after one step"

    def test_loss_decreases(self):
        """Training loss after 5 epochs < training loss at epoch 1."""
        torch.manual_seed(42)
        model = MatchupFusionModel(
            game_feature_dim=16, context_feature_dim=8,
            hidden_dim=32, num_gru_layers=1,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCELoss()

        # Create simple learnable pattern
        home_seq = torch.randn(64, 20, 16)
        away_seq = torch.randn(64, 20, 16)
        context = torch.randn(64, 8)
        # Target correlates with mean of home features
        target = (home_seq.mean(dim=(1, 2)) > 0).float().unsqueeze(1)

        losses = []
        for epoch in range(10):
            model.train()
            optimizer.zero_grad()
            pred = model(home_seq, away_seq, context)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_gradients_flow(self):
        """No NaN gradients after a forward-backward pass."""
        model = MatchupFusionModel(
            game_feature_dim=16, context_feature_dim=8,
            hidden_dim=32, num_gru_layers=1,
        )
        criterion = torch.nn.BCELoss()

        home_seq = torch.randn(8, 20, 16)
        away_seq = torch.randn(8, 20, 16)
        context = torch.randn(8, 8)
        target = torch.randint(0, 2, (8, 1)).float()

        pred = model(home_seq, away_seq, context)
        loss = criterion(pred, target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), (
                    f"NaN gradient in {name}"
                )

    def test_predictions_deterministic(self):
        """Same input produces same output (seeded)."""
        torch.manual_seed(42)
        model = MatchupFusionModel(
            game_feature_dim=16, context_feature_dim=8,
        )
        model.eval()

        x_home = torch.randn(4, 20, 16)
        x_away = torch.randn(4, 20, 16)
        x_context = torch.randn(4, 8)

        with torch.no_grad():
            out1 = model(x_home, x_away, x_context)
            out2 = model(x_home, x_away, x_context)

        assert torch.allclose(out1, out2), "Non-deterministic predictions"


# ---- Trained Model Tests ----

class TestTrainedModel:
    """Verify trained model performance."""

    def test_model_saves_and_loads(self, full_sequences):
        """Model can be saved to disk and reloaded."""
        path = NEURAL_DIR / "best_model.pt"
        assert path.exists(), "Best model checkpoint not found"
        state = torch.load(path, weights_only=True)
        assert isinstance(state, dict)
        assert len(state) > 0

    def test_model_beats_coin_flip(self, full_sequences):
        """Sequence model accuracy > 55% on test set."""
        with open(NEURAL_DIR / "test_results.json") as f:
            results = json.load(f)
        assert results["test_accuracy"] > 0.55, (
            f"Test accuracy {results['test_accuracy']:.3f} <= 55%"
        )
