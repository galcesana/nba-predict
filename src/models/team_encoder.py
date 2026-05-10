"""GRU-based team sequence encoder.

Encodes a team's last N games into a fixed-size vector.
Shared between home and away teams (same weights).

Architecture:
    Input: [batch, seq_len, n_features]
    → GRU(hidden_dim, num_layers, dropout)
    → Last hidden state
    → Output: [batch, hidden_dim]
"""

import torch
import torch.nn as nn


class TeamEncoder(nn.Module):
    """GRU encoder for a team's recent game sequence.

    Processes a sequence of per-game feature vectors and returns
    a fixed-size representation of the team's recent form.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # GRU encoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a sequence of game features.

        Args:
            x: [batch, seq_len, input_dim] — game feature sequences.
            mask: [batch, seq_len] — 1 for real games, 0 for padding.

        Returns:
            [batch, hidden_dim] — team state representation.
        """
        # Project input features to hidden dim
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Pack sequences if mask provided (for efficient GRU processing)
        if mask is not None:
            lengths = mask.sum(dim=1).long().clamp(min=1)  # Ensure at least 1
            # Sort by length (descending) for pack_padded_sequence
            sorted_lengths, sort_idx = lengths.sort(descending=True)
            x_sorted = x[sort_idx]

            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True,
            )
            _, hidden = self.gru(packed)

            # Unsort
            _, unsort_idx = sort_idx.sort()
            hidden = hidden[:, unsort_idx, :]
        else:
            _, hidden = self.gru(x)

        # Take the last layer's hidden state
        last_hidden = hidden[-1]  # [batch, hidden_dim]

        # Output projection
        out = self.output_proj(last_hidden)
        return out
