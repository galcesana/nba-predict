"""Matchup fusion model — combines team encodings + context to predict P(home_win).

Architecture:
    home_state ──┐
    away_state ──┤
    home - away ─┤── concat → FusionMLP → sigmoid → P(home_win)
    home * away ─┤
    context ─────┘
"""

import torch
import torch.nn as nn

from src.models.team_encoder import TeamEncoder


class ContextMLP(nn.Module):
    """Encodes schedule/rest context features."""

    def __init__(self, input_dim: int, output_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MatchupFusionModel(nn.Module):
    """Full matchup prediction model.

    Combines:
      - Shared GRU encoder for home/away team sequences
      - Context MLP for schedule features
      - Fusion head with concat + difference + element-wise product

    Output: P(home_win) ∈ (0, 1)
    """

    def __init__(
        self,
        game_feature_dim: int,
        context_feature_dim: int,
        hidden_dim: int = 64,
        num_gru_layers: int = 2,
        context_output_dim: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared team encoder (same weights for home and away)
        self.team_encoder = TeamEncoder(
            input_dim=game_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gru_layers,
            dropout=dropout,
        )

        # Context encoder
        self.context_mlp = ContextMLP(
            input_dim=context_feature_dim,
            output_dim=context_output_dim,
            dropout=dropout,
        )

        # Fusion head
        # Input: home_state + away_state + diff + product + context
        # Dims:  hidden_dim + hidden_dim + hidden_dim + hidden_dim + context_output_dim
        fusion_input_dim = 4 * hidden_dim + context_output_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        home_seq: torch.Tensor,
        away_seq: torch.Tensor,
        context: torch.Tensor,
        home_mask: torch.Tensor | None = None,
        away_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict P(home_win).

        Args:
            home_seq: [batch, seq_len, game_features] — home team sequence
            away_seq: [batch, seq_len, game_features] — away team sequence
            context: [batch, context_features] — schedule context
            home_mask: [batch, seq_len] — home padding mask
            away_mask: [batch, seq_len] — away padding mask

        Returns:
            [batch, 1] — predicted P(home_win)
        """
        # Encode both teams through SHARED encoder
        home_state = self.team_encoder(home_seq, home_mask)  # [batch, hidden]
        away_state = self.team_encoder(away_seq, away_mask)  # [batch, hidden]

        # Encode context
        context_state = self.context_mlp(context)  # [batch, context_dim]

        # Matchup features
        diff = home_state - away_state
        product = home_state * away_state

        # Concatenate all signals
        fused = torch.cat([
            home_state, away_state, diff, product, context_state,
        ], dim=1)

        # Predict
        logit = self.fusion_head(fused)
        return torch.sigmoid(logit)
