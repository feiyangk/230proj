#!/bin/bash
# Extract best directional accuracy from TensorBoard logs for PAN-NAN-With-Fincast

# Find TensorBoard log directories for PAN-NAN with FinCast
# Adjust the path pattern based on your actual log directory structure
LOG_BASE_DIR="logs/tensorboard"

echo "Searching for PAN-NAN-With-Fincast TensorBoard logs..."
echo ""

# Find all TensorBoard log directories
# Adjust pattern based on your naming convention (e.g., pan_nan_fincast, pan-nan-fincast, etc.)
for log_dir in $(find "$LOG_BASE_DIR" -type d -name "*fincast*" -o -name "*pan*nan*fincast*" 2>/dev/null); do
    echo "=========================================="
    echo "Checking: $log_dir"
    echo "=========================================="
    
    # Use Python to extract metrics
    python3 << EOF
import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit(1)

log_path = Path("$log_dir")
if not log_path.exists():
    sys.exit(1)

try:
    ea = EventAccumulator(str(log_path))
    ea.Reload()
    
    scalar_tags = ea.Tags().get('scalars', [])
    
    if 'Metrics/DirectionalAccuracy' in scalar_tags:
        dir_acc_events = ea.Scalars('Metrics/DirectionalAccuracy')
        if dir_acc_events:
            best_dir_acc = max(e.value for e in dir_acc_events)
            final_dir_acc = dir_acc_events[-1].value
            best_epoch = max(e.step for e in dir_acc_events if e.value == best_dir_acc)
            
        else:
    else:
        
except Exception as e:
    import traceback
    traceback.print_exc()
EOF

    echo ""
done

echo ""
echo "To extract metrics from a specific log directory, run:"
echo "  python3 scripts/training/pan_nan_fusion/extract_table_metrics.py --log-dir <path_to_log_dir>"



