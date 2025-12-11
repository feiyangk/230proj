#!/usr/bin/env python3
"""
Generic Model Architecture Tester

Tests PyTorch model with dummy data forward pass.
Works with any model type (TFT, LSTM, Transformer, etc.)

Usage:
    # Test with saved checkpoint
    python scripts/03_training/test_architectures.py --checkpoint models/tft/tft_best.pt
    
    # Test with custom dimensions
    python scripts/03_training/test_architectures.py \
        --checkpoint models/tft/tft_best.pt \
        --batch-size 8 --seq-len 192 --n-features 7
"""

import sys
import os
from pathlib import Path
import argparse
import torch
import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_forward_pass(checkpoint_path, batch_size, seq_len, n_features):
    """
    Test model forward pass with dummy data.
    
    Note: This is a placeholder since we don't have the actual model class yet.
    Once you implement your TFT model, you can:
    1. Load the model class
    2. Instantiate it with config from checkpoint
    3. Load the state dict
    4. Run forward pass
    """
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get config
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    for key, value in model_config.items():
    
    # Create dummy input
    
    dummy_input = torch.randn(batch_size, seq_len, n_features)
    
    
    # Show expected output shape
    prediction_horizons = config.get('data', {}).get('prediction_horizons', [1])
    expected_output_shape = (batch_size, len(prediction_horizons))
    
    


def compare_architectures(checkpoints):
    """Compare multiple model checkpoints."""
    
    results = []
    
    for ckpt_path in checkpoints:
        if not Path(ckpt_path).exists():
            continue
        
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # Count parameters
        total_params = 0
        if 'model_state_dict' in checkpoint:
            for tensor in checkpoint['model_state_dict'].values():
                total_params += tensor.numel()
        
        # Get metrics
        val_loss = checkpoint.get('val_loss', float('inf'))
        val_mae = checkpoint.get('val_mae', float('inf'))
        
        results.append({
            'path': ckpt_path,
            'params': total_params,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'epoch': checkpoint.get('epoch', 'N/A')
        })
    
    # Print comparison table
    
    for result in results:
        model_name = Path(result['path']).stem
    
    # Find best model
    if results:
        best = min(results, key=lambda x: x['val_loss'])
    


def main():
    parser = argparse.ArgumentParser(
        description='Test PyTorch model architecture with forward pass',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single model
  python scripts/03_training/test_architectures.py --checkpoint models/tft/tft_best.pt
  
  # Test with custom dimensions
  python scripts/03_training/test_architectures.py \
      --checkpoint models/tft/tft_best.pt \
      --batch-size 8 --seq-len 192 --n-features 7
  
  # Compare multiple models
  python scripts/03_training/test_architectures.py \
      --compare models/tft/tft_best.pt models/lstm/lstm_best.pt
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--compare',
        type=str,
        nargs='+',
        help='Compare multiple checkpoints'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Test batch size (default: 4)'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=192,
        help='Sequence length (default: 192)'
    )
    parser.add_argument(
        '--n-features',
        type=int,
        default=7,
        help='Number of features (default: 7)'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple models
        compare_architectures(args.compare)
    elif args.checkpoint:
        # Test single model
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            sys.exit(1)
        
        test_forward_pass(checkpoint_path, args.batch_size, args.seq_len, args.n_features)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
