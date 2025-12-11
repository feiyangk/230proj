#!/usr/bin/env python3
"""Minimal TFT feature verification stub (logging removed)."""

import os
import sys
import argparse


def print_header(text: str) -> None:
    """Placeholder header function."""
    pass


def verify_features(raw_file: str = 'data/raw/tft_features.parquet',
                   config_path: str = 'configs/model_tft_config.yaml') -> bool:
    """Return True if the raw file exists."""
    return os.path.exists(raw_file)


def main() -> None:
    parser = argparse.ArgumentParser(description='Verify TFT features (stub).')
    parser.add_argument('--raw-file', default='data/raw/tft_features.parquet', help='Raw features file to verify')
    parser.add_argument('--config', default='configs/model_tft_config.yaml', help='Config file path')
    args = parser.parse_args()
    _ = verify_features(raw_file=args.raw_file, config_path=args.config)


if __name__ == '__main__':
    main()
