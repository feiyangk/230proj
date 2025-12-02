#!/usr/bin/env python3
"""
Learning rate search for PAN-NAN fusion model.

Options:
1. Grid search over predefined learning rates
2. Random search over a range
3. Learning rate finder (train for a few epochs, find optimal range)
"""

import argparse
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np

def run_training(config_path, learning_rate, run_name, epochs=50, seed=42, extra_args=None):
    """Run training with a specific learning rate."""
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
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def extract_metrics(log_file):
    """Extract best validation loss from training log."""
    best_val_loss = float('inf')
    best_epoch = 0
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "Val   Loss:" in line:
                    # Parse: "Val   Loss: 0.1234, MAE: 0.0567, RMSE: 0.0789"
                    parts = line.split("Val   Loss:")[1].split(",")[0].strip()
                    try:
                        val_loss = float(parts)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                    except:
                        pass
    except:
        pass
    
    return best_val_loss, best_epoch

def grid_search(config_path, learning_rates, epochs=50, seed=42, extra_args=None):
    """Grid search over learning rates."""
    results = []
    
    print("="*80)
    print("Learning Rate Grid Search")
    print("="*80)
    print(f"Testing {len(learning_rates)} learning rates")
    print(f"Learning rates: {learning_rates}")
    print(f"Epochs per run: {epochs}")
    if extra_args:
        print(f"Extra args: {' '.join(extra_args)}")
    print("="*80)
    
    for i, lr in enumerate(learning_rates):
        run_name = f"lr_search_{lr:.2e}".replace(".", "_").replace("-", "m")
        print(f"\n[{i+1}/{len(learning_rates)}] Testing LR: {lr:.2e}")
        print(f"Run name: {run_name}")
        
        # Modify config temporarily
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        original_lr = config['training']['learning_rate']
        config['training']['learning_rate'] = lr
        
        # Save temporary config
        temp_config = config_path.replace('.yaml', f'_temp_lr_{lr:.2e}.yaml')
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        try:
            # Run training
            result = run_training(temp_config, lr, run_name, epochs, seed, extra_args)
            
            # Extract metrics (you'll need to parse from logs or checkpoints)
            # For now, we'll just track if it completed
            success = result.returncode == 0
            
            results.append({
                'learning_rate': lr,
                'run_name': run_name,
                'success': success,
                'log_file': f"training_{run_name}.log"
            })
            
            print(f"  {'✅' if success else '❌'} Completed")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'learning_rate': lr,
                'run_name': run_name,
                'success': False,
                'error': str(e)
            })
        finally:
            # Clean up temp config
            Path(temp_config).unlink(missing_ok=True)
    
    return results

def learning_rate_finder(config_path, min_lr=1e-6, max_lr=1e-2, num_trials=10, epochs=20, seed=42, extra_args=None):
    """Learning rate finder - exponential range search."""
    # Generate learning rates on log scale
    learning_rates = np.logspace(np.log10(min_lr), np.log10(max_lr), num_trials)
    
    print("="*80)
    print("Learning Rate Finder (Exponential Search)")
    print("="*80)
    print(f"Range: {min_lr:.2e} to {max_lr:.2e}")
    print(f"Trials: {num_trials}")
    print(f"Epochs per trial: {epochs}")
    if extra_args:
        print(f"Extra args: {' '.join(extra_args)}")
    print("="*80)
    
    return grid_search(config_path, learning_rates.tolist(), epochs=epochs, seed=seed, extra_args=extra_args)

def main():
    parser = argparse.ArgumentParser(
        description='Learning rate search for PAN-NAN fusion',
        # Allow unknown arguments to be passed through
        allow_abbrev=False
    )
    parser.add_argument('--config', type=str, default='configs/model_pan_nan_fusion.yaml',
                       help='Config file path')
    parser.add_argument('--method', type=str, choices=['grid', 'finder', 'custom'],
                       default='grid', help='Search method')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Epochs per learning rate trial')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Grid search options
    parser.add_argument('--lrs', type=str, nargs='+',
                       help='Custom learning rates for grid search (e.g., 1e-5 5e-5 1e-4)')
    
    # Finder options
    parser.add_argument('--min-lr', type=float, default=1e-6,
                       help='Minimum learning rate for finder')
    parser.add_argument('--max-lr', type=float, default=1e-2,
                       help='Maximum learning rate for finder')
    parser.add_argument('--num-trials', type=int, default=10,
                       help='Number of trials for finder')
    
    # Parse known args, leave unknown args for pass-through
    args, extra_args = parser.parse_known_args()
    
    # Print extra args that will be passed through
    if extra_args:
        print(f"\n📋 Extra arguments to pass through: {' '.join(extra_args)}")
        print("   (e.g., --no-fincast, --fincast, --download-fincast, --horizons, etc.)\n")
    
    if args.method == 'grid':
        if args.lrs:
            learning_rates = [float(lr) for lr in args.lrs]
        else:
            # Default grid for PAN-NAN with FinCast
            # Remember: actual LR = base_lr * lr_scale (0.2)
            # So 5e-5 becomes 1e-5 effective
            learning_rates = [
                1e-6,   # Very small
                5e-6,   # Small
                1e-5,   # Current effective (5e-5 * 0.2)
                5e-5,   # Medium
                1e-4,   # Large
                5e-4,   # Very large
            ]
        
        results = grid_search(args.config, learning_rates, args.epochs, args.seed, extra_args)
    
    elif args.method == 'finder':
        results = learning_rate_finder(
            args.config, args.min_lr, args.max_lr, args.num_trials, args.epochs, args.seed, extra_args
        )
    
    # Save results
    results_file = f"lr_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("Search Complete!")
    print("="*80)
    print(f"Results saved to: {results_file}")
    print("\nNext steps:")
    print("1. Check training logs for each run")
    print("2. Compare validation losses")
    print("3. Choose LR with best validation performance")
    print("="*80)

if __name__ == '__main__':
    main()

