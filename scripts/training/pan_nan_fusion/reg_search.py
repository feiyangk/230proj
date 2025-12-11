#!/usr/bin/env python3
"""
Hyperparameter search over regularization parameters for PAN-NAN fusion model.

Searches over:
- model.dropout: Main transformer dropout
- training.weight_decay: L2 regularization
- training.gradient_clip_norm: Gradient clipping threshold
- fusion.dropout: Fusion block dropout (optional)

Options:
1. Grid search over predefined regularization combinations
2. Random search over ranges
"""

import argparse
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import itertools
import random

def encode_reg_name(model_dropout, weight_decay, clip_norm, fusion_dropout=None):
    """Encode regularization parameters into a run name."""
    if fusion_dropout is not None:
        return f"reg_md{model_dropout}_wd{weight_decay}_gc{clip_norm}_fd{fusion_dropout}"
    else:
        return f"reg_md{model_dropout}_wd{weight_decay}_gc{clip_norm}"

def run_training(config_path, run_name, epochs=50, seed=42, extra_args=None):
    """Run training with a specific config."""
    cmd = [
        "python", "scripts/training/pan_nan_fusion/train.py",
        "--config", config_path,
        "--seed", str(seed),
        "--run-name", run_name,
    ]
    
    # Add extra arguments if provided
    if extra_args:
        cmd.extend(extra_args)
    
    # Add epochs if specified
    if epochs:
        cmd.extend(["--epochs", str(epochs)])
    
    # Run with real-time output
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Print output in real-time
    if result.stdout:
    
    return result

def grid_search(
    config_path,
    model_dropout_values,
    weight_decay_values,
    clip_norm_values,
    fusion_dropout_values=None,
    epochs=50,
    seed=42,
    extra_args=None
):
    """Grid search over regularization combinations."""
    
    # Generate all combinations
    if fusion_dropout_values is not None:
        combinations = list(itertools.product(
            model_dropout_values,
            weight_decay_values,
            clip_norm_values,
            fusion_dropout_values
        ))
    else:
        combinations = list(itertools.product(
            model_dropout_values,
            weight_decay_values,
            clip_norm_values
        ))
    
    results = []
    
    if fusion_dropout_values:
    if extra_args:
    
    for i, combo in enumerate(combinations):
        if fusion_dropout_values is not None:
            model_dropout, weight_decay, clip_norm, fusion_dropout = combo
        else:
            model_dropout, weight_decay, clip_norm = combo
            fusion_dropout = None
        
        run_name = encode_reg_name(model_dropout, weight_decay, clip_norm, fusion_dropout)
        if fusion_dropout is not None:
        else:
        
        # Modify config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Store original values
        original_model_dropout = config['model']['dropout']
        original_weight_decay = config['training']['weight_decay']
        original_clip_norm = config['training']['gradient_clip_norm']
        original_fusion_dropout = None
        if fusion_dropout is not None and 'fusion' in config:
            original_fusion_dropout = config['fusion'].get('dropout')
        
        # Update config
        config['model']['dropout'] = model_dropout
        config['training']['weight_decay'] = weight_decay
        config['training']['gradient_clip_norm'] = clip_norm
        if fusion_dropout is not None and 'fusion' in config:
            config['fusion']['dropout'] = fusion_dropout
        
        # Save temporary config
        temp_config = config_path.replace('.yaml', f'_temp_reg_{run_name}.yaml')
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        try:
            
            result = run_training(temp_config, run_name, epochs, seed, extra_args)
            
            success = result.returncode == 0
            
            result_entry = {
                'model_dropout': model_dropout,
                'weight_decay': weight_decay,
                'clip_norm': clip_norm,
                'run_name': run_name,
                'success': success,
            }
            if fusion_dropout is not None:
                result_entry['fusion_dropout'] = fusion_dropout
            
            results.append(result_entry)
            
            
        except Exception as e:
            result_entry = {
                'model_dropout': model_dropout,
                'weight_decay': weight_decay,
                'clip_norm': clip_norm,
                'run_name': run_name,
                'success': False,
                'error': str(e)
            }
            if fusion_dropout is not None:
                result_entry['fusion_dropout'] = fusion_dropout
            results.append(result_entry)
        finally:
            # Clean up temp config
            Path(temp_config).unlink(missing_ok=True)
    
    return results

def random_search(
    config_path,
    model_dropout_range,
    weight_decay_range,
    clip_norm_range,
    fusion_dropout_range=None,
    num_trials=10,
    epochs=50,
    seed=42,
    extra_args=None
):
    """Random search over regularization ranges."""
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    results = []
    
    if fusion_dropout_range:
    if extra_args:
    
    for i in range(num_trials):
        # Sample regularization values
        model_dropout = random.uniform(model_dropout_range[0], model_dropout_range[1])
        weight_decay = random.uniform(weight_decay_range[0], weight_decay_range[1])
        clip_norm = random.uniform(clip_norm_range[0], clip_norm_range[1])
        fusion_dropout = None
        if fusion_dropout_range:
            fusion_dropout = random.uniform(fusion_dropout_range[0], fusion_dropout_range[1])
        
        run_name = encode_reg_name(model_dropout, weight_decay, clip_norm, fusion_dropout)
        if fusion_dropout is not None:
        else:
        
        # Modify config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['model']['dropout'] = model_dropout
        config['training']['weight_decay'] = weight_decay
        config['training']['gradient_clip_norm'] = clip_norm
        if fusion_dropout is not None and 'fusion' in config:
            config['fusion']['dropout'] = fusion_dropout
        
        # Save temporary config
        temp_config = config_path.replace('.yaml', f'_temp_reg_{run_name}.yaml')
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        try:
            
            result = run_training(temp_config, run_name, epochs, seed, extra_args)
            
            success = result.returncode == 0
            
            result_entry = {
                'model_dropout': model_dropout,
                'weight_decay': weight_decay,
                'clip_norm': clip_norm,
                'run_name': run_name,
                'success': success,
            }
            if fusion_dropout is not None:
                result_entry['fusion_dropout'] = fusion_dropout
            
            results.append(result_entry)
            
            
        except Exception as e:
            result_entry = {
                'model_dropout': model_dropout,
                'weight_decay': weight_decay,
                'clip_norm': clip_norm,
                'run_name': run_name,
                'success': False,
                'error': str(e)
            }
            if fusion_dropout is not None:
                result_entry['fusion_dropout'] = fusion_dropout
            results.append(result_entry)
        finally:
            # Clean up temp config
            Path(temp_config).unlink(missing_ok=True)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Regularization search for PAN-NAN fusion',
        allow_abbrev=False
    )
    parser.add_argument('--config', type=str, default='configs/model_pan_nan_fusion.yaml',
                       help='Config file path')
    parser.add_argument('--method', type=str, choices=['grid', 'random'],
                       default='grid', help='Search method')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Epochs per trial')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Grid search options
    parser.add_argument('--model-dropout', type=float, nargs='+',
                       help='model.dropout values for grid search (e.g., 0.1 0.2 0.3)')
    parser.add_argument('--weight-decay', type=float, nargs='+',
                       help='training.weight_decay values (e.g., 0.01 0.02 0.05)')
    parser.add_argument('--clip-norm', type=float, nargs='+',
                       help='training.gradient_clip_norm values (e.g., 0.5 1.0 2.0)')
    parser.add_argument('--fusion-dropout', type=float, nargs='+',
                       help='fusion.dropout values (optional, e.g., 0.05 0.1 0.15)')
    
    # Random search options
    parser.add_argument('--num-trials', type=int, default=10,
                       help='Number of trials for random search')
    parser.add_argument('--model-dropout-range', type=float, nargs=2,
                       help='model.dropout range [min, max] for random search')
    parser.add_argument('--weight-decay-range', type=float, nargs=2,
                       help='training.weight_decay range [min, max]')
    parser.add_argument('--clip-norm-range', type=float, nargs=2,
                       help='training.gradient_clip_norm range [min, max]')
    parser.add_argument('--fusion-dropout-range', type=float, nargs=2,
                       help='fusion.dropout range [min, max]')
    
    # Parse known args, leave unknown args for pass-through
    args, extra_args = parser.parse_known_args()
    
    # Print extra args that will be passed through
    if extra_args:
    
    if args.method == 'grid':
        # Default values if not specified
        model_dropout_values = args.model_dropout or [0.1, 0.2, 0.3]
        weight_decay_values = args.weight_decay or [0.01, 0.02, 0.05]
        clip_norm_values = args.clip_norm or [0.5, 1.0, 2.0]
        fusion_dropout_values = args.fusion_dropout if args.fusion_dropout else None
        
        results = grid_search(
            args.config,
            model_dropout_values,
            weight_decay_values,
            clip_norm_values,
            fusion_dropout_values=fusion_dropout_values,
            epochs=args.epochs,
            seed=args.seed,
            extra_args=extra_args
        )
    
    elif args.method == 'random':
        # Default ranges if not specified
        model_dropout_range = args.model_dropout_range or [0.1, 0.3]
        weight_decay_range = args.weight_decay_range or [0.01, 0.05]
        clip_norm_range = args.clip_norm_range or [0.5, 2.0]
        fusion_dropout_range = args.fusion_dropout_range if args.fusion_dropout_range else None
        
        results = random_search(
            args.config,
            model_dropout_range,
            weight_decay_range,
            clip_norm_range,
            fusion_dropout_range=fusion_dropout_range,
            num_trials=args.num_trials,
            epochs=args.epochs,
            seed=args.seed,
            extra_args=extra_args
        )
    
    # Save results
    results_file = f"reg_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    

if __name__ == '__main__':
    main()



