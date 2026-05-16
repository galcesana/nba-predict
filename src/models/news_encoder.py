"""Shared news MLP encoder.

Encodes a team's news/sentiment feature vector into a fixed-size hidden state.
Shared between home and away teams (same weights).

Handles the news_available=0 case gracefully — zero input produces
a valid (but uninformative) hidden state.

Architecture:
    Input: [batch, n_news_features]
    → Linear(n_features, hidden_dim) → ReLU → Dropout
    → Linear(hidden_dim, output_dim) → ReLU
    Output: [batch, output_dim]
"""

import torch
import torch.nn as nn


class NewsEncoder(nn.Module):
    """MLP encoder for news/sentiment feature vectors."""

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 32,
        output_dim: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode news/sentiment features.

        Args:
            x: [batch, n_news_features]

        Returns:
            [batch, output_dim]
        """
        return self.net(x)
