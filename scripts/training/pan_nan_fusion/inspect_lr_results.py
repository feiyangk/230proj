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
    if not Path(log_file).exists():
        return None, None
    
    best_val_loss = float('inf')
    best_epoch = 0
    epoch = 0
    
    try:
        with open(log_file, 'r') as f:
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
    
    print("="*80)
    print("Learning Rate Search Results")
    print("="*80)
    print(f"Total trials: {len(results)}\n")
    
    # Try to extract validation losses from logs
    enriched_results = []
    for result in results:
        lr = result['learning_rate']
        run_name = result.get('run_name', f"lr_search_{lr:.2e}")
        log_file = result.get('log_file', f"training_{run_name}.log")
        
        val_loss, error = extract_val_loss_from_log(log_file)
        
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
        print("📊 Results (sorted by validation loss, best first):")
        print("-" * 80)
        print(f"{'LR':<12} {'Val Loss':<12} {'Epoch':<8} {'Status':<10} {'Run Name'}")
        print("-" * 80)
        
        for i, result in enumerate(valid_results):
            lr = result['learning_rate']
            val_loss = result['val_loss']
            epoch = result.get('best_epoch', 'N/A')
            status = "✅" if result.get('success', False) else "❌"
            run_name = result.get('run_name', 'N/A')
            
            marker = "🏆" if i == 0 else "  "
            print(f"{marker} {lr:<11.2e} {val_loss:<12.6f} {str(epoch):<8} {status:<10} {run_name}")
        
        print("\n" + "="*80)
        print("Best Learning Rate:")
        print(f"  LR: {valid_results[0]['learning_rate']:.2e}")
        print(f"  Validation Loss: {valid_results[0]['val_loss']:.6f}")
        print(f"  Run Name: {valid_results[0].get('run_name', 'N/A')}")
        print("="*80)
    
    if invalid_results:
        print(f"\n⚠️  {len(invalid_results)} trial(s) with missing/invalid results:")
        for result in invalid_results:
            lr = result['learning_rate']
            error = result.get('val_loss_error', 'Unknown error')
            print(f"  LR {lr:.2e}: {error}")
    
    # Summary statistics
    if valid_results:
        losses = [r['val_loss'] for r in valid_results]
        print(f"\n📈 Summary:")
        print(f"  Best LR: {valid_results[0]['learning_rate']:.2e} (val_loss: {min(losses):.6f})")
        print(f"  Worst LR: {valid_results[-1]['learning_rate']:.2e} (val_loss: {max(losses):.6f})")
        print(f"  Range: {max(losses) - min(losses):.6f}")
        
        # Find LR with best loss
        best_lr = valid_results[0]['learning_rate']
        print(f"\n💡 Recommendation: Use LR = {best_lr:.2e}")
        print(f"   Consider fine-tuning around this value (±2x range)")
    
    # Check TensorBoard logs
    print(f"\n📊 To visualize in TensorBoard:")
    print(f"   tensorboard --logdir logs/tensorboard --port 6006")
    print(f"   Look for runs starting with 'lr_search_'")

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_lr_results.py <results_json_file>")
        print("\nExample:")
        print("  python scripts/training/pan_nan_fusion/inspect_lr_results.py lr_search_results_20251202_030427.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        sys.exit(1)
    
    inspect_results(results_file)

if __name__ == '__main__':
    main()

