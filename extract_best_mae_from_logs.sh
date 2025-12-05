#!/bin/bash
# Extract best validation MAE from PAN-NAN-No-Fincast TensorBoard logs

LOG_BASE="logs/tensorboard/pan-nan_ar"

echo "=========================================="
echo "Extracting Best Validation MAE"
echo "=========================================="
echo ""

# Check the best regularization run first (based on config comments)
BEST_RUN="reg_md0.1_wd0.07_gc0.5_fd0.15"
echo "Checking best regularization run: $BEST_RUN"
echo ""

python3 << EOF
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

log_dir = Path("$LOG_BASE/$BEST_RUN")

if log_dir.exists():
    try:
        ea = EventAccumulator(str(log_dir))
        ea.Reload()
        
        if 'Metrics/MAE' in ea.Tags()['scalars']:
            events = ea.Scalars('Metrics/MAE')
            best_mae = min(e.value for e in events)
            best_epoch = min(e.step for e in events if e.value == best_mae)
            
            # Get related metrics
            best_val_loss = None
            if 'Loss/val' in ea.Tags()['scalars']:
                val_loss = ea.Scalars('Loss/val')
                best_val_loss = min(e.value for e in val_loss)
            
            best_dir_acc = None
            if 'Metrics/DirectionalAccuracy' in ea.Tags()['scalars']:
                dir_acc = ea.Scalars('Metrics/DirectionalAccuracy')
                best_dir_acc = max(e.value for e in dir_acc)
            
            print(f"✅ Best Validation MAE: {best_mae:.6f} (epoch {best_epoch})")
            if best_val_loss:
                print(f"   Best Validation Loss: {best_val_loss:.6f}")
            if best_dir_acc:
                print(f"   Best Directional Accuracy: {best_dir_acc * 100:.2f}%")
            print(f"   Run: $BEST_RUN")
        else:
            print("⚠️  Metrics/MAE not found in this run")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print(f"⚠️  Directory not found: {log_dir}")
    print("")
    print("Checking all runs to find best MAE...")
    print("")
    
    # Check all runs and find the one with best MAE
    import glob
    best_overall_mae = float('inf')
    best_run_name = None
    best_epoch = None
    
    for run_dir in glob.glob("$LOG_BASE/*"):
        run_name = Path(run_dir).name
        if not Path(run_dir).is_dir():
            continue
        
        try:
            ea = EventAccumulator(run_dir)
            ea.Reload()
            
            if 'Metrics/MAE' in ea.Tags()['scalars']:
                events = ea.Scalars('Metrics/MAE')
                run_best_mae = min(e.value for e in events)
                run_best_epoch = min(e.step for e in events if e.value == run_best_mae)
                
                if run_best_mae < best_overall_mae:
                    best_overall_mae = run_best_mae
                    best_run_name = run_name
                    best_epoch = run_best_epoch
        except:
            pass
    
    if best_run_name:
        print(f"✅ Best Validation MAE across all runs: {best_overall_mae:.6f} (epoch {best_epoch})")
        print(f"   Best run: {best_run_name}")
    else:
        print("❌ Could not find any valid runs")
EOF

echo ""
echo "=========================================="
echo "To check a specific run:"
echo "  python3 scripts/training/pan_nan_fusion/extract_table_metrics.py --log-dir $LOG_BASE/<run_name>"
echo "=========================================="

