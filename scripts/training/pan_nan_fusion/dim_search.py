#!/usr/bin/env python3
"""
Hyperparameter search over hidden dimensions for PAN-NAN fusion model.

Searches over:
- model.d_model: Main transformer embedding dimension
- model.d_ff: Feedforward dimension (can be tied to d_model or independent)
- fusion.sentiment_hidden_dim: NAN branch hidden dimension
- fusion.fusion_hidden_dim: Fusion head hidden dimension

Options:
1. Grid search over predefined dimension combinations
2. Random search over ranges
3. Independent search (search each dimension separately)
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

def encode_dim_name(d_model, d_ff, sent_hidden, fusion_hidden):
    """Encode dimension parameters into a run name."""
    # Format: dim_dM_dF_sH_fH
    return f"dim_{d_model}_{d_ff}_{sent_hidden}_{fusion_hidden}"

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
    d_model_values,
    d_ff_values,
    sentiment_hidden_values,
    fusion_hidden_values,
    tie_d_ff_to_d_model=False,
    epochs=50,
    seed=42,
    extra_args=None
):
    """Grid search over dimension combinations."""
    
    # Generate all combinations
    if tie_d_ff_to_d_model:
        # d_ff is automatically 4x d_model
        combinations = list(itertools.product(
            d_model_values,
            sentiment_hidden_values,
            fusion_hidden_values
        ))
        # Add d_ff = 4 * d_model for each combination
        combinations = [(d_m, 4 * d_m, s_h, f_h) for d_m, s_h, f_h in combinations]
    else:
        combinations = list(itertools.product(
            d_model_values,
            d_ff_values,
            sentiment_hidden_values,
            fusion_hidden_values
        ))
    
    results = []
    
    if tie_d_ff_to_d_model:
    else:
    if extra_args:
    
    for i, (d_model, d_ff, sent_hidden, fusion_hidden) in enumerate(combinations):
        run_name = encode_dim_name(d_model, d_ff, sent_hidden, fusion_hidden)
        
        # Modify config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Store original values
        original_d_model = config['model']['d_model']
        original_d_ff = config['model']['d_ff']
        original_sent_hidden = config['fusion']['sentiment_hidden_dim']
        original_fusion_hidden = config['fusion']['fusion_hidden_dim']
        
        # Update config
        config['model']['d_model'] = d_model
        config['model']['d_ff'] = d_ff
        config['fusion']['sentiment_hidden_dim'] = sent_hidden
        config['fusion']['fusion_hidden_dim'] = fusion_hidden
        
        # Save temporary config
        temp_config = config_path.replace('.yaml', f'_temp_dim_{run_name}.yaml')
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        try:
            
            result = run_training(temp_config, run_name, epochs, seed, extra_args)
            
            success = result.returncode == 0
            
            results.append({
                'd_model': d_model,
                'd_ff': d_ff,
                'sentiment_hidden_dim': sent_hidden,
                'fusion_hidden_dim': fusion_hidden,
                'run_name': run_name,
                'success': success,
            })
            
            
        except Exception as e:
            results.append({
                'd_model': d_model,
                'd_ff': d_ff,
                'sentiment_hidden_dim': sent_hidden,
                'fusion_hidden_dim': fusion_hidden,
                'run_name': run_name,
                'success': False,
                'error': str(e)
            })
        finally:
            # Clean up temp config
            Path(temp_config).unlink(missing_ok=True)
    
    return results

def random_search(
    config_path,
    d_model_range,
    d_ff_range,
    sentiment_hidden_range,
    fusion_hidden_range,
    tie_d_ff_to_d_model=False,
    num_trials=10,
    epochs=50,
    seed=42,
    extra_args=None
):
    """Random search over dimension ranges."""
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    results = []
    
    if tie_d_ff_to_d_model:
    else:
    if extra_args:
    
    for i in range(num_trials):
        # Sample dimensions
        d_model = random.choice(d_model_range)
        if tie_d_ff_to_d_model:
            d_ff = 4 * d_model
        else:
            d_ff = random.choice(d_ff_range)
        sent_hidden = random.choice(sentiment_hidden_range)
        fusion_hidden = random.choice(fusion_hidden_range)
        
        run_name = encode_dim_name(d_model, d_ff, sent_hidden, fusion_hidden)
        
        # Modify config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['model']['d_model'] = d_model
        config['model']['d_ff'] = d_ff
        config['fusion']['sentiment_hidden_dim'] = sent_hidden
        config['fusion']['fusion_hidden_dim'] = fusion_hidden
        
        # Save temporary config
        temp_config = config_path.replace('.yaml', f'_temp_dim_{run_name}.yaml')
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        try:
            
            result = run_training(temp_config, run_name, epochs, seed, extra_args)
            
            success = result.returncode == 0
            
            results.append({
                'd_model': d_model,
                'd_ff': d_ff,
                'sentiment_hidden_dim': sent_hidden,
                'fusion_hidden_dim': fusion_hidden,
                'run_name': run_name,
                'success': success,
            })
            
            
        except Exception as e:
            results.append({
                'd_model': d_model,
                'd_ff': d_ff,
                'sentiment_hidden_dim': sent_hidden,
                'fusion_hidden_dim': fusion_hidden,
                'run_name': run_name,
                'success': False,
                'error': str(e)
            })
        finally:
            # Clean up temp config
            Path(temp_config).unlink(missing_ok=True)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Dimension search for PAN-NAN fusion',
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
    parser.add_argument('--tie-d-ff', action='store_true',
                       help='Tie d_ff to 4x d_model (default: search independently)')
    
    # Grid search options
    parser.add_argument('--d-model', type=int, nargs='+',
                       help='d_model values for grid search (e.g., 64 96 128)')
    parser.add_argument('--d-ff', type=int, nargs='+',
                       help='d_ff values for grid search (e.g., 256 384 512). Ignored if --tie-d-ff')
    parser.add_argument('--sentiment-hidden', type=int, nargs='+',
                       help='sentiment_hidden_dim values (e.g., 32 64 128)')
    parser.add_argument('--fusion-hidden', type=int, nargs='+',
                       help='fusion_hidden_dim values (e.g., 64 128 256)')
    
    # Random search options
    parser.add_argument('--num-trials', type=int, default=10,
                       help='Number of trials for random search')
    parser.add_argument('--d-model-range', type=int, nargs=2,
                       help='d_model range [min, max] for random search')
    parser.add_argument('--d-ff-range', type=int, nargs=2,
                       help='d_ff range [min, max] for random search. Ignored if --tie-d-ff')
    parser.add_argument('--sentiment-hidden-range', type=int, nargs=2,
                       help='sentiment_hidden_dim range [min, max]')
    parser.add_argument('--fusion-hidden-range', type=int, nargs=2,
                       help='fusion_hidden_dim range [min, max]')
    
    # Parse known args, leave unknown args for pass-through
    args, extra_args = parser.parse_known_args()
    
    # Print extra args that will be passed through
    if extra_args:
    
    if args.method == 'grid':
        # Default values if not specified
        d_model_values = args.d_model or [64, 96, 128]
        d_ff_values = args.d_ff or [256, 384, 512]
        sentiment_hidden_values = args.sentiment_hidden or [32, 64, 128]
        fusion_hidden_values = args.fusion_hidden or [64, 128, 256]
        
        results = grid_search(
            args.config,
            d_model_values,
            d_ff_values,
            sentiment_hidden_values,
            fusion_hidden_values,
            tie_d_ff_to_d_model=args.tie_d_ff,
            epochs=args.epochs,
            seed=args.seed,
            extra_args=extra_args
        )
    
    elif args.method == 'random':
        # Default ranges if not specified
        d_model_range = args.d_model_range or [64, 128]
        d_ff_range = args.d_ff_range or [256, 512]
        sentiment_hidden_range = args.sentiment_hidden_range or [32, 128]
        fusion_hidden_range = args.fusion_hidden_range or [64, 256]
        
        # Convert ranges to lists of possible values (step by 16 for efficiency)
        d_model_range = list(range(d_model_range[0], d_model_range[1] + 1, 16))
        if not args.tie_d_ff:
            d_ff_range = list(range(d_ff_range[0], d_ff_range[1] + 1, 32))
        sentiment_hidden_range = list(range(sentiment_hidden_range[0], sentiment_hidden_range[1] + 1, 16))
        fusion_hidden_range = list(range(fusion_hidden_range[0], fusion_hidden_range[1] + 1, 32))
        
        results = random_search(
            args.config,
            d_model_range,
            d_ff_range,
            sentiment_hidden_range,
            fusion_hidden_range,
            tie_d_ff_to_d_model=args.tie_d_ff,
            num_trials=args.num_trials,
            epochs=args.epochs,
            seed=args.seed,
            extra_args=extra_args
        )
    
    # Save results
    results_file = f"dim_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    

if __name__ == '__main__':
    main()

