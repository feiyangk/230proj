#!/usr/bin/env python3
"""
PAN-NAN Fusion model inspired by the fusion_source_paper.

PAN (Price Attention Network):
    - Reuses the decoder-only transformer stack with FinCast backbones to encode
      ticker close series plus ticker-level synthetic indicators (volume, SMAs, etc.).

NAN (Non-price Attention Network):
    - Processes GDELT-derived sentiment features via a ConvLSTM-like stack
      (approximated with temporal 1-D conv + Transformer encoder + BiGRU).

Fusion:
    - Concatenates PAN and NAN predictions before passing through a small MLP
      that outputs the final multi-horizon forecast.

The goal is to stay close to the current decoder_transformer training pipeline so
that train/eval scripts can import FusionDecoderTransformer just like the
existing DecoderTransformerWithFinCast.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

project_root = Path(__file__).parent.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

decoder_train_mod = import_module("scripts.training.decoder_transformer.decoder_transformer_train")
fincast_mod = import_module("scripts.training.decoder_transformer.fincast_extension")

DecoderOnlyTransformerAR = decoder_train_mod.DecoderOnlyTransformerAR
PositionalEncoding = decoder_train_mod.PositionalEncoding
FINCAST_AVAILABLE = fincast_mod.FINCAST_AVAILABLE
FinCastBackbone = fincast_mod.FinCastBackbone


def _load_feature_names(metadata_path: Path = Path("data/processed/metadata.yaml")) -> List[str]:
    """Load ordered feature names from metadata if available."""
    if metadata_path.exists():
        with metadata_path.open("r") as f:
            metadata = yaml.safe_load(f)
            if metadata and "features" in metadata:
                return metadata["features"]
    return []


def _index_by_name(features: Sequence[str], names: Iterable[str]) -> List[int]:
    """Return indices of features whose name matches any entry in names."""
    lookup = {name: i for i, name in enumerate(features)}
    return [lookup[name] for name in names if name in lookup]


class SentimentFeatureBranch(nn.Module):
    """
    Simplified ConvLSTM + attention + BiGRU pipeline for GDELT sentiment.

    Inputs:  x_gdelt -> [batch, lookback, num_gdelt_features]
    Outputs: predictions -> [batch, num_horizons]
    """

    def __init__(
        self,
        lookback: int,
        num_gdelt_features: int,
        num_horizons: int,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=num_gdelt_features,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(lookback) 
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=lookback)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x shape [B, T, F]
        x = x.transpose(1, 2)  # -> [B, F, T]
        x = self.conv(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  # -> [B, T, hidden]

        x = self.proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)

        gru_out, _ = self.bigru(x)
        context = gru_out[:, -1, :]  # last step summary
        return self.head(context)


class PANDecoderTransformer(nn.Module):
    """Wrapper that turns decoder transformer + FinCast into the PAN branch."""

    def __init__(
        self,
        config: Dict,
        fincast_config: Optional[Dict] = None,
        decoder_class=DecoderOnlyTransformerAR,
    ) -> None:
        super().__init__()

        if not FINCAST_AVAILABLE:
            raise ImportError("FinCast not available; cannot build PANDecoderTransformer.")

        self.config = config
        self.lookback = config["data"].get("lookback", config["data"].get("lookback_window"))

        self.feature_names = _load_feature_names()
        if not self.feature_names:
            raise ValueError(
                "Feature metadata not found. Run data pipeline to generate data/processed/metadata.yaml."
            )

        self.price_indices = self._find_price_indices()
        if not self.price_indices:
            raise ValueError("No close_* features detected for FinCast processing.")

        gdelt_names = self._gdelt_feature_names()
        self.gdelt_indices = _index_by_name(self.feature_names, gdelt_names)

        self.synthetic_indices = [
            idx
            for idx in range(len(self.feature_names))
            if idx not in self.price_indices and idx not in self.gdelt_indices
        ]

        fincast_config = fincast_config or {}
        fincast_output_dim = fincast_config.get("output_dim", 128)

        self.fincast = FinCastBackbone(
            d_model=fincast_output_dim,
            n_heads=16,
            n_layers=50,
            d_ff=128,
            dropout=fincast_config.get("dropout", 0.1),
            max_len=fincast_config.get("max_len", 512),
            freeze=fincast_config.get("freeze", True),
            pretrained_path=fincast_config.get("pretrained_path"),
        )

        num_synth_features = len(self.synthetic_indices)
        augmented_features = num_synth_features + len(self.price_indices) * fincast_output_dim
        self.feature_norm = nn.LayerNorm(augmented_features)

        self.decoder_transformer = decoder_class(
            config=config,
            num_features=augmented_features,
        )

        self.register_buffer("price_idx", torch.tensor(self.price_indices, dtype=torch.long))
        self.register_buffer("synth_idx", torch.tensor(self.synthetic_indices, dtype=torch.long))
        self.register_buffer("gdelt_idx", torch.tensor(self.gdelt_indices, dtype=torch.long))

    def _find_price_indices(self) -> List[int]:
        return [
            i
            for i, feat in enumerate(self.feature_names)
            if feat.startswith("close_") and not feat.endswith("_returns")
        ]

    def _gdelt_feature_names(self) -> List[str]:
        gdelt_cfg = self.config.get("gdelt", {})
        features = gdelt_cfg.get("features", [])
        if gdelt_cfg.get("include_lags"):
            features.extend([f"sentiment_lag_{lag}" for lag in gdelt_cfg.get("lag_periods", [])])
        return list(dict.fromkeys(features))  # dedupe preserving order

    def forward(
        self,
        x: torch.Tensor,
        y_future: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        price_series = torch.index_select(x, dim=2, index=self.price_idx)
        synth_features = torch.index_select(x, dim=2, index=self.synth_idx)
        gdelt_features = torch.index_select(x, dim=2, index=self.gdelt_idx)

        augmented = self._augment_with_fincast(price_series, synth_features)
        pan_pred = self.decoder_transformer(
            augmented,
            y_future=y_future,
            teacher_forcing=teacher_forcing,
        )
        return pan_pred, gdelt_features

    def _augment_with_fincast(self, price_series: torch.Tensor, synth_features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_prices = price_series.shape

        prices_flat = price_series.permute(0, 2, 1).contiguous().view(-1, seq_len)
        fincast_hidden = self.fincast(prices_flat)

        if fincast_hidden.size(1) != seq_len:
            if fincast_hidden.size(1) > seq_len:
                fincast_hidden = fincast_hidden[:, -seq_len:, :]
            else:
                pad_len = seq_len - fincast_hidden.size(1)
                padding = torch.zeros(fincast_hidden.size(0), pad_len, fincast_hidden.size(2), device=fincast_hidden.device)
                fincast_hidden = torch.cat([padding, fincast_hidden], dim=1)

        hidden_dim = fincast_hidden.size(2)
        fincast_hidden = fincast_hidden.view(batch_size, num_prices, seq_len, hidden_dim)
        fincast_hidden = fincast_hidden.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

        augmented = torch.cat([synth_features, fincast_hidden], dim=-1)
        return self.feature_norm(augmented)


class FusionDecoderTransformer(nn.Module):
    """
    Full PAN + NAN fusion module.

    Forward input: same as DecoderOnlyTransformerAR → [batch, lookback, num_features]
    Returns: [batch, num_horizons]
    """

    def __init__(
        self,
        config: Dict,
        num_features: Optional[int] = None,
        fincast_config: Optional[Dict] = None,
        decoder_transformer_class=None,
        fusion_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        data_cfg = config["data"]
        self.lookback = data_cfg.get("lookback", data_cfg.get("lookback_window"))
        if self.lookback is None:
            raise ValueError("Config must include data.lookback or data.lookback_window.")

        self.prediction_horizons = config["data"]["prediction_horizons"]
        self.num_horizons = len(self.prediction_horizons)

        fusion_cfg = fusion_config or config.get("fusion", {})
        hidden_dim = fusion_cfg.get("sentiment_hidden_dim", 64)
        transformer_layers = fusion_cfg.get("sentiment_layers", 2)
        transformer_heads = fusion_cfg.get("sentiment_heads", 4)
        dropout = fusion_cfg.get("dropout", 0.1)

        decoder_class = decoder_transformer_class or DecoderOnlyTransformerAR

        self.pan_branch = PANDecoderTransformer(
            config=config,
            fincast_config=fincast_config or config.get("fincast", {}),
            decoder_class=decoder_class,
        )

        num_gdelt_features = len(self.pan_branch.gdelt_indices)
        if num_gdelt_features == 0:
            raise ValueError("No GDELT features detected; cannot build NAN branch.")

        self.nan_branch = SentimentFeatureBranch(
            lookback=self.lookback,
            num_gdelt_features=num_gdelt_features,
            num_horizons=self.num_horizons,
            hidden_dim=hidden_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=dropout,
        )

        fusion_input_dim = self.num_horizons * 2
        fusion_hidden = fusion_cfg.get("fusion_hidden_dim", 128)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, self.num_horizons),
        )

    def forward(
        self,
        x: torch.Tensor,
        y_future: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
    ) -> torch.Tensor:
        pan_pred, gdelt_features = self.pan_branch(
            x,
            y_future=y_future,
            teacher_forcing=teacher_forcing,
        )
        nan_pred = self.nan_branch(gdelt_features)

        fused = torch.cat([pan_pred, nan_pred], dim=-1)
        return self.fusion_head(fused)


__all__ = [
    "FusionDecoderTransformer",
    "PANDecoderTransformer",
    "SentimentFeatureBranch",
]

