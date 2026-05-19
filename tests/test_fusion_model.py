"""Tests for the full Phase 7 Fusion Model.

Run with: pytest tests/test_fusion_model.py -v
"""

import pytest
import torch

from src.models.injury_encoder import InjuryEncoder
from src.models.matchup_fusion_model import MatchupFusionModel
from src.models.news_encoder import NewsEncoder


@pytest.fixture(scope="module")
def fusion_model() -> MatchupFusionModel:
    return MatchupFusionModel(
        game_feature_dim=16,
        context_feature_dim=8,
        injury_feature_dim=6,
        news_feature_dim=6,
        hidden_dim=32,
        num_gru_layers=1,
        feature_streams=["performance", "context", "injury", "news"],
    )


class TestMatchupFusionModel:
    def test_model_initialization(self, fusion_model):
        """Model initializes correctly with all streams."""
        assert fusion_model is not None
        assert isinstance(fusion_model.injury_encoder, InjuryEncoder)
        assert isinstance(fusion_model.news_encoder, NewsEncoder)

    def test_forward_pass_shapes(self, fusion_model):
        """Forward pass handles all streams and produces correct shape."""
        batch_size = 4
        seq_len = 20

        # Create dummy tensors
        home_seq = torch.randn(batch_size, seq_len, 16)
        away_seq = torch.randn(batch_size, seq_len, 16)
        context = torch.randn(batch_size, 8)

        home_mask = torch.ones(batch_size, seq_len)
        away_mask = torch.ones(batch_size, seq_len)

        home_injury = torch.randn(batch_size, 6)
        away_injury = torch.randn(batch_size, 6)

        home_news = torch.randn(batch_size, 6)
        away_news = torch.randn(batch_size, 6)
        news_available = torch.ones(batch_size, 1)

        pred = fusion_model(
            home_seq=home_seq,
            away_seq=away_seq,
            context=context,
            home_mask=home_mask,
            away_mask=away_mask,
            home_injury=home_injury,
            away_injury=away_injury,
            home_news=home_news,
            away_news=away_news,
            news_available=news_available,
        )

        assert pred.shape == (batch_size, 1)
        assert (pred >= 0).all() and (pred <= 1).all()

    def test_ablation_initialization(self):
        """Model can be initialized without optional streams."""
        ablation = MatchupFusionModel(
            game_feature_dim=16,
            context_feature_dim=8,
            hidden_dim=32,
            feature_streams=["performance", "context"],
        )
        assert ablation.injury_encoder is None
        assert ablation.news_encoder is None

        batch_size = 2
        seq_len = 5
        home_seq = torch.randn(batch_size, seq_len, 16)
        away_seq = torch.randn(batch_size, seq_len, 16)
        context = torch.randn(batch_size, 8)

        pred = ablation(home_seq, away_seq, context)
        assert pred.shape == (batch_size, 1)
