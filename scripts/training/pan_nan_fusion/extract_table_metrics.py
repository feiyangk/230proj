#!/usr/bin/env python3
"""
Extract metrics from TensorBoard logs for table generation.
Extracts validation and test metrics from a specific TensorBoard run.
"""

import argparse
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    exit(1)


def extract_all_metrics(log_dir):
    """Extract all metrics from TensorBoard log directory."""
    if not TB_AVAILABLE:
        return None
    
    log_path = Path(log_dir)
    if not log_path.exists():
        return None
    
    try:
        ea = EventAccumulator(str(log_path))
        ea.Reload()
        
        metrics = {}
        
        # Get all scalar tags
        scalar_tags = ea.Tags().get('scalars', [])
        
        # Extract validation metrics (best values)
        if 'Loss/val' in scalar_tags:
            val_loss_events = ea.Scalars('Loss/val')
            if val_loss_events:
                metrics['best_val_loss'] = min(e.value for e in val_loss_events)
                metrics['final_val_loss'] = val_loss_events[-1].value
        
        if 'Metrics/MAE' in scalar_tags:
            mae_events = ea.Scalars('Metrics/MAE')
            if mae_events:
                metrics['best_val_mae'] = min(e.value for e in mae_events)
                metrics['final_val_mae'] = mae_events[-1].value
        
        if 'Metrics/RMSE' in scalar_tags:
            rmse_events = ea.Scalars('Metrics/RMSE')
            if rmse_events:
                metrics['best_val_rmse'] = min(e.value for e in rmse_events)
        
        if 'Metrics/DirectionalAccuracy' in scalar_tags:
            dir_acc_events = ea.Scalars('Metrics/DirectionalAccuracy')
            if dir_acc_events:
                # Best directional accuracy (validation)
                metrics['best_val_dir_acc'] = max(e.value for e in dir_acc_events)
                metrics['final_val_dir_acc'] = dir_acc_events[-1].value
        
        # Extract per-horizon validation metrics
        per_horizon_tags = [tag for tag in scalar_tags if 'Metrics/PerHorizon' in tag]
        for tag in per_horizon_tags:
            events = ea.Scalars(tag)
            if events:
                # Extract horizon name (e.g., "H7_MAE" from "Metrics/PerHorizon/H7_MAE")
                horizon_name = tag.split('/')[-1]
                metrics[f'val_{horizon_name.lower()}'] = min(e.value for e in events)
        
        # Extract test metrics from hyperparameters
        hparams = ea.Tags().get('hparams', [])
        if hparams:
            # Try to get hparam metrics
            try:
                # HParams are stored differently - need to check if there's a way to extract them
                # For now, we'll look for scalar tags that might contain test metrics
                pass
            except:
                pass
        
        # Check for test metrics in scalar tags (if they were logged)
        test_tags = [tag for tag in scalar_tags if 'test' in tag.lower()]
        for tag in test_tags:
            events = ea.Scalars(tag)
            if events:
                tag_name = tag.replace('/', '_').lower()
                metrics[tag_name] = events[-1].value  # Use final value
        
        # Also check hparams metrics if available
        # TensorBoard stores hparams differently - we need to parse the event file
        # For now, let's try to extract from the last event if available
        
        return metrics
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def extract_hparam_metrics(log_dir):
    """Extract metrics from hparams section (test metrics are often here)."""
    log_path = Path(log_dir)
    metrics = {}
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(log_path))
        ea.Reload()
        
        # Check if hparams exist
        if 'hparams' in ea.Tags():
            try:
                hparam_data = ea.HParams()
                if hparam_data:
                    # HParams structure: {'hparam_infos': {...}, 'sessionGroups': [...]}
                    # Metrics are in sessionGroups[0]['metrics']
                    if 'sessionGroups' in hparam_data and len(hparam_data['sessionGroups']) > 0:
                        session = hparam_data['sessionGroups'][0]
                        if 'metrics' in session:
                            for metric_name, metric_info in session['metrics'].items():
                                # metric_info is a dict with 'value' key
                                if isinstance(metric_info, dict) and 'value' in metric_info:
                                    value = metric_info['value']
                                    if isinstance(value, (int, float)):
                                        # Remove 'hparam/' prefix if present
                                        clean_name = metric_name.replace('hparam/', '')
                                        metrics[clean_name] = value
            except Exception as e:
                # Try alternative method: parse event files directly
                import struct
                from tensorboard.compat.proto import event_pb2
                from tensorboard.util import tensor_util
                
                event_files = sorted(log_path.glob("events.out.tfevents.*"))
                for event_file in event_files:
                    try:
                        with open(event_file, 'rb') as f:
                            # Read events (simplified - TensorBoard format is complex)
                            # Look for Summary protos with hparam tags
                            pass
                    except:
                        pass
    except Exception as e:
        pass
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Extract metrics from TensorBoard logs for table')
    parser.add_argument('--log-dir', type=str, required=True,
                       help='TensorBoard log directory (e.g., logs/tensorboard/pan-nan_ar/20251205_073310)')
    parser.add_argument('--format', type=str, choices=['table', 'json', 'latex'], default='table',
                       help='Output format')
    
    args = parser.parse_args()
    
    
    # Extract metrics
    metrics = extract_all_metrics(args.log_dir)
    
    if not metrics:
        return
    
    # Also try to extract hparam metrics
    hparam_metrics = extract_hparam_metrics(args.log_dir)
    if hparam_metrics:
        metrics.update(hparam_metrics)
    
    # Print metrics in table format
    
    # Validation metrics
    
    # Test metrics (from hparams or scalars)
    test_loss = metrics.get('hparam_test_loss') or metrics.get('test_loss') or metrics.get('Loss/test')
    test_mae = metrics.get('hparam_test_mae') or metrics.get('test_mae') or metrics.get('Metrics/test_mae')
    test_rmse = metrics.get('hparam_test_rmse') or metrics.get('test_rmse') or metrics.get('Metrics/test_rmse')
    test_dir_acc = metrics.get('hparam_test_dir_acc') or metrics.get('test_dir_acc') or metrics.get('Metrics/test_directionalaccuracy')
    
    
    # Per-horizon metrics
    for horizon in ['H7', 'H14', 'H28']:
        h_mae = metrics.get(f'val_{horizon.lower()}_mae') or metrics.get(f'{horizon.lower()}_mae')
    
    # Generate LaTeX table row
    if args.format == 'latex':
        val_loss = metrics.get('best_val_loss', 0)
        val_mae = metrics.get('best_val_mae', 0)
        val_dir_acc = metrics.get('best_val_dir_acc', 0)
        
        # Format test metrics safely
        if test_loss:
        else:
        if test_mae:
        else:
        if test_rmse:
        else:
        if test_dir_acc:
        else:
        
        for horizon in ['H7', 'H14', 'H28']:
            h_mae = metrics.get(f'val_{horizon.lower()}_mae') or metrics.get(f'{horizon.lower()}_mae')
            if h_mae:
            else:


if __name__ == '__main__':
    main()

