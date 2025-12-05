#!/bin/bash
# Extract best validation MAE from TensorBoard logs for PAN-NAN-No-Fincast

LOG_BASE_DIR="logs/tensorboard"

echo "=========================================="
echo "Searching for PAN-NAN-No-Fincast TensorBoard logs..."
echo "=========================================="
echo ""

# Find TensorBoard log directories (excluding fincast runs)
# Look for pan_nan or decoder_ar runs that don't have fincast in the name
for log_dir in $(find "$LOG_BASE_DIR" -type d \( -name "*pan*nan*" -o -name "*decoder_ar*" \) ! -name "*fincast*" 2>/dev/null | sort); do
    # Skip if it's a parent directory
    if [ ! -f "$log_dir/events.out.tfevents"* ] 2>/dev/null; then
        continue
    fi
    
    echo "Checking: $log_dir"
    
    # Use Python to extract best validation MAE
    python3 << EOF
import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: TensorBoard not available. Install with: pip install tensorboard")
    sys.exit(1)

log_path = Path("$log_dir")
if not log_path.exists():
    print(f"Log directory not found: $log_dir")
    sys.exit(1)

try:
    ea = EventAccumulator(str(log_path))
    ea.Reload()
    
    scalar_tags = ea.Tags().get('scalars', [])
    
    results = {}
    
    # Extract best validation MAE
    if 'Metrics/MAE' in scalar_tags:
        mae_events = ea.Scalars('Metrics/MAE')
        if mae_events:
            best_mae = min(e.value for e in mae_events)
            best_epoch = min(e.step for e in mae_events if e.value == best_mae)
            final_mae = mae_events[-1].value
            results['best_val_mae'] = best_mae
            results['best_epoch'] = best_epoch
            results['final_mae'] = final_mae
            results['total_epochs'] = len(mae_events)
    
    # Also get validation loss for context
    if 'Loss/val' in scalar_tags:
        val_loss_events = ea.Scalars('Loss/val')
        if val_loss_events:
            best_val_loss = min(e.value for e in val_loss_events)
            results['best_val_loss'] = best_val_loss
    
    # Get directional accuracy for context
    if 'Metrics/DirectionalAccuracy' in scalar_tags:
        dir_acc_events = ea.Scalars('Metrics/DirectionalAccuracy')
        if dir_acc_events:
            best_dir_acc = max(e.value for e in dir_acc_events)
            results['best_dir_acc'] = best_dir_acc
    
    if results:
        print(f"✅ Found metrics:")
        if 'best_val_mae' in results:
            print(f"   Best Validation MAE: {results['best_val_mae']:.6f} (epoch {results['best_epoch']})")
            print(f"   Final Validation MAE: {results['final_mae']:.6f}")
        if 'best_val_loss' in results:
            print(f"   Best Validation Loss: {results['best_val_loss']:.6f}")
        if 'best_dir_acc' in results:
            print(f"   Best Directional Accuracy: {results['best_dir_acc'] * 100:.2f}%")
        if 'total_epochs' in results:
            print(f"   Total epochs: {results['total_epochs']}")
        print(f"   Log directory: $log_dir")
        print("")
    else:
        print("⚠️  No validation MAE metrics found")
        print(f"   Available tags: {scalar_tags[:5]}...")
        print("")
        
except Exception as e:
    print(f"❌ Error reading TensorBoard log: {e}")
    import traceback
    traceback.print_exc()
    print("")
EOF

done

echo ""
echo "=========================================="
echo "Summary: Best Validation MAE for PAN-NAN-No-Fincast"
echo "=========================================="
echo ""
echo "To extract from a specific log directory:"
echo "  python3 scripts/training/pan_nan_fusion/extract_table_metrics.py --log-dir <path>"
echo ""

