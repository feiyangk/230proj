#!/usr/bin/env python3
"""
Generic Model Inspector

Inspects PyTorch model architecture and parameters.
Works with any model type (TFT, LSTM, Transformer, etc.)

Usage:
    # Inspect saved model checkpoint
    python scripts/03_training/inspect_model.py --checkpoint models/tft/tft_best.pt
    
    # Inspect model from config (no checkpoint)
    python scripts/03_training/inspect_model.py --model-type tft --config configs/model_tft_config.yaml
"""

import sys
import os
from pathlib import Path
import argparse
import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def count_parameters(model):
    """Count total, trainable, and non-trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def inspect_checkpoint(checkpoint_path):
    """Inspect a saved model checkpoint."""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Print checkpoint contents
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            num_params = len(checkpoint[key])
        elif key == 'config':
        elif key == 'scalers':
            num_scalers = len(checkpoint[key]) if isinstance(checkpoint[key], dict) else 1
        else:
    
    
    # Analyze model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        
        # Count parameters by layer type
        layer_types = {}
        total_params = 0
        
        for name, tensor in state_dict.items():
            layer_type = name.split('.')[0] if '.' in name else name
            params = tensor.numel()
            total_params += params
            
            if layer_type not in layer_types:
                layer_types[layer_type] = {'count': 0, 'params': 0}
            layer_types[layer_type]['count'] += 1
            layer_types[layer_type]['params'] += params
        
        for layer_type, info in sorted(layer_types.items()):
        
    
    # Show config if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        if 'model' in config:
            for key, value in config['model'].items():
    


def inspect_from_config(model_type, config_path):
    """Inspect model architecture from config (without loading checkpoint)."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    
    if 'model' in config:
        for key, value in config['model'].items():
    


def main():
    parser = argparse.ArgumentParser(
        description='Inspect PyTorch model architecture and parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect saved checkpoint
  python scripts/03_training/inspect_model.py --checkpoint models/tft/tft_best.pt
  
  # Inspect from config (no checkpoint needed)
  python scripts/03_training/inspect_model.py --model-type tft --config configs/model_tft_config.yaml
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='tft',
        help='Model type (tft, lstm, etc.) - used if no checkpoint provided'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_tft_config.yaml',
        help='Path to model config YAML'
    )
    
    args = parser.parse_args()
    
    if args.checkpoint:
        # Inspect saved checkpoint
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            sys.exit(1)
        
        inspect_checkpoint(checkpoint_path)
    else:
        # Inspect from config
        config_path = Path(args.config)
        if not config_path.exists():
            sys.exit(1)
        
        inspect_from_config(args.model_type, config_path)


if __name__ == '__main__':
    main()
