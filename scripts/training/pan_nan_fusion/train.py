#!/usr/bin/env python3
"""
Training entrypoint for the PAN-NAN fusion model.

This script mirrors the decoder_transformer training workflow but swaps the
model implementation with FusionDecoderTransformer. Configuration is provided
via YAML (defaults to configs/model_pan_nan_fusion.yaml).
"""

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.training.decoder_transformer import decoder_transformer_train as decoder_train  # noqa: E402
from scripts.training.pan_nan_fusion.model import (
    FINCAST_AVAILABLE,
    FusionDecoderTransformer,
)


def parse_horizons(horizons_str: str) -> list:
    """
    Parse horizons from comma-separated string.
    
    Args:
        horizons_str: Comma-separated string of integers (e.g., "4,8,16")
    
    Returns:
        List of integers
    """
    try:
        horizons = [int(h.strip()) for h in horizons_str.split(',')]
        if not horizons:
            raise ValueError("At least one horizon must be specified")
        if any(h <= 0 for h in horizons):
            raise ValueError("All horizons must be positive integers")
        return horizons
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid horizons format: {e}. Expected comma-separated integers (e.g., '4,8,16')")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the PAN-NAN fusion decoder transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_pan_nan_fusion.yaml",
        help="Path to fusion model config YAML",
    )
    parser.add_argument(
        "--download-fincast",
        action="store_true",
        help="Automatically download FinCast checkpoints from Hugging Face if missing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (sets PyTorch, NumPy, and Python random seeds)",
    )
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=None,
        help="Override prediction horizons from config. Comma-separated list of integers (e.g., '4,8,16')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config. Useful for managing GPU memory (e.g., 32, 16, 8)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    # Override prediction horizons if provided via command line
    if args.horizons is not None:
        if "data" not in config:
            config["data"] = {}
        config["data"]["prediction_horizons"] = args.horizons
        print(f"\n📊 Overriding prediction horizons: {args.horizons}")
    
    # Override batch size if provided via command line
    if args.batch_size is not None:
        if "training" not in config:
            config["training"] = {}
        original_batch_size = config["training"].get("batch_size", "N/A")
        config["training"]["batch_size"] = args.batch_size
        print(f"\n📦 Overriding batch size: {original_batch_size} → {args.batch_size}")

    if not config.get("fincast", {}).get("enabled", False):
        raise ValueError(
            "PAN-NAN fusion requires fincast.enabled: true in the config "
            "because PAN depends on FinCast embeddings."
        )

    if not config.get("fusion", {}).get("enabled", False):
        raise ValueError(
            "Fusion block missing or disabled. Set fusion.enabled: true in the config."
        )

    if args.download_fincast:
        _ensure_fincast_checkpoint(config)

    # Swap decoder transformer FinCast implementation with fusion wrapper.
    decoder_train.DecoderTransformerWithFinCast = FusionDecoderTransformer
    decoder_train.FINCAST_AVAILABLE = FINCAST_AVAILABLE

    # Write updated config to temporary file if any overrides were provided
    if args.horizons is not None or args.batch_size is not None:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_config:
            yaml.dump(config, tmp_config)
            tmp_config_path = tmp_config.name
        overrides = []
        if args.horizons is not None:
            overrides.append("horizons")
        if args.batch_size is not None:
            overrides.append("batch_size")
        print(f"📝 Using temporary config with overridden {', '.join(overrides)}: {tmp_config_path}")
        config_path_to_use = tmp_config_path
    else:
        config_path_to_use = str(config_path)

    # Kick off the standard decoder training loop using the provided config.
    decoder_train.train(config_path=config_path_to_use, seed=args.seed)


def _ensure_fincast_checkpoint(config: dict) -> None:
    fincast_cfg = config.get("fincast", {})
    checkpoint_path = Path(fincast_cfg.get("checkpoint_path", "external/fincast/checkpoints/v1.pth"))

    if checkpoint_path.exists():
        print(f"✅ FinCast checkpoint already present at {checkpoint_path}")
        return

    print(f"📥 FinCast checkpoint not found at {checkpoint_path}. Downloading from Hugging Face...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to auto-download FinCast checkpoints. "
            "Install it with `pip install huggingface_hub` or download manually."
        ) from exc

    repo_id = fincast_cfg.get("huggingface_repo", "Vincent05R/FinCast")
    checkpoint_dir = checkpoint_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(repo_id=repo_id, local_dir=str(checkpoint_dir))

    if checkpoint_path.exists():
        print(f"✅ Download complete: {checkpoint_path}")
    else:
        raise FileNotFoundError(
            f"Downloaded repo {repo_id}, but expected checkpoint {checkpoint_path} was not found.\n"
            f"Please verify the repo contents or update fincast.checkpoint_path."
        )


if __name__ == "__main__":
    main()

