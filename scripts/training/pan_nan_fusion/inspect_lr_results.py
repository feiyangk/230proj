#!/usr/bin/env python3
"""
Inspect learning rate search results.

Usage:
    python scripts/training/pan_nan_fusion/inspect_lr_results.py lr_search_results_20251202_030427.json
"""

import json
import sys
import yaml
from pathlib import Path
import re

def extract_val_loss_from_log(log_file):
    """Extract best validation loss from log file."""
    # Try multiple possible locations
    possible_paths = [
        log_file,
        Path(log_file),
        Path(".") / log_file,
        Path("nohup.out"),
    ]
    
    log_path = None
    for path in possible_paths:
        if Path(path).exists():
            log_path = path
            break
    
    if log_path is None:
        return None, f"Log file not found: {log_file}"
    
    best_val_loss = float('inf')
    best_epoch = 0
    epoch = 0
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Track epoch number
                if "Epoch" in line and "/" in line:
                    match = re.search(r'Epoch (\d+)/', line)
                    if match:
                        epoch = int(match.group(1))
                
                # Extract validation loss
                if "Val   Loss:" in line:
                    # Format: "Val   Loss: 0.1234, MAE: 0.0567, RMSE: 0.0789"
                    parts = line.split("Val   Loss:")[1].split(",")[0].strip()
                    try:
                        val_loss = float(parts)
                        if val_loss < best_val_loss and not (val_loss != val_loss):  # Check for NaN
                            best_val_loss = val_loss
                            best_epoch = epoch
                    except:
                        pass
    except Exception as e:
        return None, str(e)
    
    if best_val_loss == float('inf'):
        return None, "No validation loss found"
    
    return best_val_loss, best_epoch

def inspect_results(results_file):
    """Inspect and display learning rate search results."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    
    # Try to extract validation losses from logs
    enriched_results = []
    for result in results:
        lr = result['learning_rate']
        run_name = result.get('run_name', f"lr_search_{lr:.2e}")
        log_file = result.get('log_file', f"training_{run_name}.log")
        
        # Also try to find TensorBoard logs
        val_loss, error = extract_val_loss_from_log(log_file)
        
        # If log file not found, try to extract from TensorBoard or checkpoints
        if val_loss is None:
            # Try to find in TensorBoard logs
            tb_dir = Path("logs/tensorboard")
            if tb_dir.exists():
                # Look for run directories matching this LR
                for run_dir in tb_dir.rglob(f"*{run_name}*"):
                    events_file = list(run_dir.glob("events.out.tfevents.*"))
                    if events_file:
                        # TensorBoard parsing would go here, but for now just note it exists
                        error = f"TensorBoard log found at {run_dir}, but parsing not implemented"
        
        enriched = result.copy()
        enriched['val_loss'] = val_loss
        enriched['val_loss_error'] = error
        enriched_results.append(enriched)
    
    # Sort by validation loss (best first)
    valid_results = [r for r in enriched_results if r['val_loss'] is not None]
    valid_results.sort(key=lambda x: x['val_loss'])
    
    invalid_results = [r for r in enriched_results if r['val_loss'] is None]
    
    # Display results
    if valid_results:
        
        for i, result in enumerate(valid_results):
            lr = result['learning_rate']
            val_loss = result['val_loss']
            epoch = result.get('best_epoch', 'N/A')
            status = "✅" if result.get('success', False) else "❌"
            run_name = result.get('run_name', 'N/A')
            
            marker = "🏆" if i == 0 else "  "
        
    
    if invalid_results:
        for result in invalid_results:
            lr = result['learning_rate']
            error = result.get('val_loss_error', 'Unknown error')
    
    # Summary statistics
    if valid_results:
        losses = [r['val_loss'] for r in valid_results]
        
        # Find LR with best loss
        best_lr = valid_results[0]['learning_rate']
    
    # Check TensorBoard logs

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        sys.exit(1)
    
    inspect_results(results_file)

if __name__ == '__main__':
    main()

