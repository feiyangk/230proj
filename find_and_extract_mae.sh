#!/bin/bash
# Find TensorBoard logs and extract best validation MAE for PAN-NAN-No-Fincast

echo "=========================================="
echo "Finding TensorBoard log directories..."
echo "=========================================="
echo ""

# First, find all TensorBoard log directories
echo "Searching for TensorBoard event files..."
find logs/tensorboard -name "events.out.tfevents.*" -type f 2>/dev/null | head -10 | while read event_file; do
    log_dir=$(dirname "$event_file")
    echo "Found: $log_dir"
done

echo ""
echo "=========================================="
echo "Checking each log directory for metrics..."
echo "=========================================="
echo ""

# Find all directories that might contain TensorBoard logs
for log_dir in $(find logs/tensorboard -type d 2>/dev/null | grep -v "__pycache__" | sort); do
    # Check if this directory contains event files
    if ls "$log_dir"/events.out.tfevents.* 1> /dev/null 2>&1; then
        # Skip fincast runs
        if echo "$log_dir" | grep -qi "fincast"; then
            continue
        fi
        
        echo "Processing: $log_dir"
        
        python3 << EOF
import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: TensorBoard not available")
    sys.exit(1)

log_path = Path("$log_dir")
if not log_path.exists():
    sys.exit(1)

try:
    ea = EventAccumulator(str(log_path))
    ea.Reload()
    
    scalar_tags = ea.Tags().get('scalars', [])
    
    if 'Metrics/MAE' in scalar_tags:
        mae_events = ea.Scalars('Metrics/MAE')
        if mae_events:
            best_mae = min(e.value for e in mae_events)
            best_epoch = min(e.step for e in mae_events if e.value == best_mae)
            
            # Also get other metrics for context
            best_val_loss = None
            if 'Loss/val' in scalar_tags:
                val_loss_events = ea.Scalars('Loss/val')
                if val_loss_events:
                    best_val_loss = min(e.value for e in val_loss_events)
            
            best_dir_acc = None
            if 'Metrics/DirectionalAccuracy' in scalar_tags:
                dir_acc_events = ea.Scalars('Metrics/DirectionalAccuracy')
                if dir_acc_events:
                    best_dir_acc = max(e.value for e in dir_acc_events)
            
            print(f"  ✅ Best Validation MAE: {best_mae:.6f} (epoch {best_epoch})")
            if best_val_loss:
                print(f"     Best Validation Loss: {best_val_loss:.6f}")
            if best_dir_acc:
                print(f"     Best Directional Accuracy: {best_dir_acc * 100:.2f}%")
            print(f"     Directory: $log_dir")
            print("")
except Exception as e:
    # Silently skip directories that can't be read
    pass
EOF
    fi
done

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="

