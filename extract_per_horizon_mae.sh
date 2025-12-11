#!/bin/bash
# Extract best per-horizon MAE from PAN-NAN-No-Fincast TensorBoard logs

LOG_BASE="logs/tensorboard/pan-nan_ar"
BEST_RUN="reg_md0.1_wd0.07_gc0.5_fd0.15"

echo "=========================================="
echo "Extracting Best Per-Horizon MAE"
echo "=========================================="
echo ""

python3 << 'EOF'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import glob

log_base = "logs/tensorboard/pan-nan_ar"
best_run = "reg_md0.1_wd0.07_gc0.5_fd0.15"

# Check the best run first
log_dir = Path(f"{log_base}/{best_run}")

if not log_dir.exists():
    log_dir = None

if log_dir and log_dir.exists():
    try:
        ea = EventAccumulator(str(log_dir))
        ea.Reload()
        
        scalar_tags = ea.Tags().get('scalars', [])
        
        # Find all per-horizon MAE tags
        per_horizon_tags = [tag for tag in scalar_tags if 'PerHorizon' in tag and 'MAE' in tag]
        
        if per_horizon_tags:
            
            results = {}
            for tag in sorted(per_horizon_tags):
                events = ea.Scalars(tag)
                if events:
                    best_mae = min(e.value for e in events)
                    best_epoch = min(e.step for e in events if e.value == best_mae)
                    
                    # Extract horizon name (e.g., "H7_MAE" from "Metrics/PerHorizon/H7_MAE")
                    horizon_name = tag.split('/')[-1]
                    results[horizon_name] = (best_mae, best_epoch)
            
            # Print in order: H7, H14, H28
            for horizon in ['H7_MAE', 'H14_MAE', 'H28_MAE']:
                if horizon in results:
                    mae, epoch = results[horizon]
                else:
                    # Try alternative naming
                    alt_name = None
                    for key in results.keys():
                        if horizon.split('_')[0] in key:
                            alt_name = key
                            break
                    if alt_name:
                        mae, epoch = results[alt_name]
                    else:
            
        else:
    except Exception as e:
        import traceback
        traceback.print_exc()
else:
    # Search all runs
    
    best_results = {}
    best_run_name = None
    
    for run_dir in sorted(glob.glob(f"{log_base}/*")):
        run_name = Path(run_dir).name
        if not Path(run_dir).is_dir():
            continue
        
        try:
            ea = EventAccumulator(run_dir)
            ea.Reload()
            
            scalar_tags = ea.Tags().get('scalars', [])
            per_horizon_tags = [tag for tag in scalar_tags if 'PerHorizon' in tag and 'MAE' in tag]
            
            if per_horizon_tags:
                run_results = {}
                for tag in per_horizon_tags:
                    events = ea.Scalars(tag)
                    if events:
                        best_mae = min(e.value for e in events)
                        horizon_name = tag.split('/')[-1]
                        run_results[horizon_name] = best_mae
                
                # Check if this run has better results
                if not best_results or all(
                    run_results.get(h, float('inf')) < best_results.get(h, float('inf'))
                    for h in ['H7_MAE', 'H14_MAE', 'H28_MAE']
                    if h in run_results
                ):
                    best_results = run_results
                    best_run_name = run_name
        except:
            pass
    
    if best_results:
        for horizon in ['H7_MAE', 'H14_MAE', 'H28_MAE']:
            if horizon in best_results:
    else:
EOF

echo ""
echo "=========================================="
echo "To check a specific run:"
echo "  python3 scripts/training/pan_nan_fusion/extract_table_metrics.py --log-dir $LOG_BASE/<run_name>"
echo "=========================================="



