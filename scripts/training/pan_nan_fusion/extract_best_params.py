#!/usr/bin/env python3
"""
Extract best hyperparameters from TensorBoard logs or checkpoints.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False


def extract_metrics_from_tensorboard(log_dir):
    """Extract final validation metrics from TensorBoard log directory."""
    if not TB_AVAILABLE:
        return None
    
    try:
        ea = EventAccumulator(str(log_dir))
        ea.Reload()
        
        # Get scalar metrics
        scalar_tags = ea.Tags()['scalars']
        
        metrics = {}
        
        # Extract final validation loss
        if 'Loss/val' in scalar_tags:
            val_loss_events = ea.Scalars('Loss/val')
            if val_loss_events:
                metrics['best_val_loss'] = min(e.value for e in val_loss_events)
                metrics['final_val_loss'] = val_loss_events[-1].value
        
        # Extract final validation MAE
        if 'Metrics/MAE' in scalar_tags:
            mae_events = ea.Scalars('Metrics/MAE')
            if mae_events:
                metrics['best_mae'] = min(e.value for e in mae_events)
                metrics['final_mae'] = mae_events[-1].value
        
        # Extract final validation RMSE
        if 'Metrics/RMSE' in scalar_tags:
            rmse_events = ea.Scalars('Metrics/RMSE')
            if rmse_events:
                metrics['best_rmse'] = min(e.value for e in rmse_events)
                metrics['final_rmse'] = rmse_events[-1].value
        
        return metrics
    except Exception as e:
        return None


def parse_run_name(run_name):
    """Parse hyperparameters from run name."""
    # Format: reg_md0.1_wd0.02_gc0.5_fd0.05
    parts = run_name.replace('reg_', '').split('_')
    params = {}
    for part in parts:
        if part.startswith('md'):
            params['model_dropout'] = float(part[2:])
        elif part.startswith('wd'):
            params['weight_decay'] = float(part[2:])
        elif part.startswith('gc'):
            params['clip_norm'] = float(part[2:])
        elif part.startswith('fd'):
            params['fusion_dropout'] = float(part[2:])
    return params


def find_best_from_tensorboard(log_base_dir, results_json=None):
    """Find best hyperparameters from TensorBoard logs."""
    log_dir = Path(log_base_dir)
    
    if not log_dir.exists():
        return None
    
    # Find all reg_* run directories
    reg_runs = [d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith('reg_')]
    
    if not reg_runs:
        return None
    
    
    all_results = []
    
    for run_dir in reg_runs:
        run_name = run_dir.name
        params = parse_run_name(run_name)
        metrics = extract_metrics_from_tensorboard(run_dir)
        
        if metrics:
            result = {**params, **metrics, 'run_name': run_name}
            all_results.append(result)
        else:
    
    if not all_results:
        return None
    
    # Find best by validation loss
    best_by_loss = min(all_results, key=lambda x: x.get('best_val_loss', float('inf')))
    
    # Find best by MAE
    best_by_mae = min(all_results, key=lambda x: x.get('best_mae', float('inf')))
    
    return {
        'all_results': all_results,
        'best_by_val_loss': best_by_loss,
        'best_by_mae': best_by_mae
    }


def main():
    parser = argparse.ArgumentParser(description='Find best hyperparameters from search results')
    parser.add_argument('--tensorboard-dir', type=str,
                       default='logs/tensorboard/pan-nan_ar',
                       help='TensorBoard log directory')
    parser.add_argument('--results-json', type=str, default=None,
                       help='Optional: JSON results file to cross-reference')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for best results')
    
    args = parser.parse_args()
    
    
    results = find_best_from_tensorboard(args.tensorboard_dir, args.results_json)
    
    if not results:
        return
    
    
    best_loss = results['best_by_val_loss']
    
    best_mae = results['best_by_mae']
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()



