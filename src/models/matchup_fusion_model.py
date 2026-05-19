"""Matchup fusion model — combines all encoder streams to predict P(home_win).

Full 4-stream architecture:
    home_state ──────┐
    away_state ──────┤
    home - away ─────┤
    home * away ─────┤
    home_injury ─────┤
    away_injury ─────┤── concat → FusionMLP → sigmoid → P(home_win)
    injury_diff ─────┤
    home_news ───────┤
    away_news ───────┤
    news_diff ───────┤
    context ─────────┤
    news_available ──┘

Supports ablation via feature_streams config:
    "performance"  — GRU team sequences
    "injury"       — injury feature encoder
    "news"         — news/sentiment feature encoder
    "context"      — schedule/rest context
"""

import torch
import torch.nn as nn

from src.models.injury_encoder import InjuryEncoder
from src.models.news_encoder import NewsEncoder
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
      - Shared injury MLP encoder
      - Shared news MLP encoder
      - Context MLP for schedule features
      - Fusion head with concat + difference + element-wise product

    Output: P(home_win) ∈ (0, 1)
    """

    def __init__(
        self,
        game_feature_dim: int,
        context_feature_dim: int,
        injury_feature_dim: int = 0,
        news_feature_dim: int = 0,
        hidden_dim: int = 64,
        num_gru_layers: int = 2,
        context_output_dim: int = 16,
        injury_output_dim: int = 16,
        news_output_dim: int = 16,
        dropout: float = 0.3,
        feature_streams: list[str] | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Default: all streams
        if feature_streams is None:
            feature_streams = ["performance", "context", "injury", "news"]
        self.feature_streams = feature_streams

        # Performance encoder (always present)
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

        # Injury encoder (optional)
        self.injury_encoder = None
        if "injury" in feature_streams and injury_feature_dim > 0:
            self.injury_encoder = InjuryEncoder(
                input_dim=injury_feature_dim,
                output_dim=injury_output_dim,
                dropout=dropout,
            )

        # News encoder (optional)
        self.news_encoder = None
        if "news" in feature_streams and news_feature_dim > 0:
            self.news_encoder = NewsEncoder(
                input_dim=news_feature_dim,
                output_dim=news_output_dim,
                dropout=dropout,
            )

        # Compute fusion input dim
        # Performance: home + away + diff + product = 4 * hidden_dim
        fusion_input_dim = 4 * hidden_dim

        # Context
        if "context" in feature_streams:
            fusion_input_dim += context_output_dim

        # Injury: home + away + diff = 3 * injury_output_dim
        if self.injury_encoder is not None:
            fusion_input_dim += 3 * injury_output_dim

        # News: home + away + diff = 3 * news_output_dim + news_available flag
        if self.news_encoder is not None:
            fusion_input_dim += 3 * news_output_dim + 1  # +1 for news_available flag

        # Fusion head
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
        home_injury: torch.Tensor | None = None,
        away_injury: torch.Tensor | None = None,
        home_news: torch.Tensor | None = None,
        away_news: torch.Tensor | None = None,
        news_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict P(home_win).

        Args:
            home_seq: [batch, seq_len, game_features] — home team sequence
            away_seq: [batch, seq_len, game_features] — away team sequence
            context: [batch, context_features] — schedule context
            home_mask: [batch, seq_len] — home padding mask
            away_mask: [batch, seq_len] — away padding mask
            home_injury: [batch, injury_features] — home injury vector
            away_injury: [batch, injury_features] — away injury vector
            home_news: [batch, news_features] — home news vector
            away_news: [batch, news_features] — away news vector
            news_available: [batch, 1] — news availability flag

        Returns:
            [batch, 1] — predicted P(home_win)
        """
        # Encode both teams through SHARED encoder
        home_state = self.team_encoder(home_seq, home_mask)
        away_state = self.team_encoder(away_seq, away_mask)

        # Performance matchup features
        diff = home_state - away_state
        product = home_state * away_state

        parts = [home_state, away_state, diff, product]

        # Context
        if "context" in self.feature_streams:
            context_state = self.context_mlp(context)
            parts.append(context_state)

        # Injury stream
        if self.injury_encoder is not None and home_injury is not None:
            home_inj_state = self.injury_encoder(home_injury)
            away_inj_state = self.injury_encoder(away_injury)
            inj_diff = home_inj_state - away_inj_state
            parts.extend([home_inj_state, away_inj_state, inj_diff])

        # News stream
        if self.news_encoder is not None and home_news is not None:
            home_news_state = self.news_encoder(home_news)
            away_news_state = self.news_encoder(away_news)
            news_diff = home_news_state - away_news_state
            parts.extend([home_news_state, away_news_state, news_diff])
            if news_available is not None:
                parts.append(news_available)

        # Concatenate all signals
        fused = torch.cat(parts, dim=1)

        # Predict
        logit = self.fusion_head(fused)
        return torch.sigmoid(logit)
