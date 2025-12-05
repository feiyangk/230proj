#!/bin/bash
# Extract test metrics from training logs or checkpoint

cd ~/230proj

echo "Looking for test metrics in training logs..."
echo ""

# Check nohup.out
if [ -f nohup.out ]; then
    echo "=== Checking nohup.out ==="
    grep -A 15 "Test Set Results" nohup.out | tail -20
    echo ""
fi

# Check all log files
echo "=== Checking log files ==="
find . -name "*.log" -type f | while read logfile; do
    if grep -q "Test Set Results" "$logfile"; then
        echo "Found in: $logfile"
        grep -A 15 "Test Set Results" "$logfile" | tail -20
        echo ""
    fi
done

# Check checkpoint file
echo "=== Checking checkpoint file ==="
CHECKPOINT="models/decoder_transformer/decoder_transformer_best_ar.pt"
if [ -f "$CHECKPOINT" ]; then
    python3 << 'PYEOF'
import torch
checkpoint = torch.load('models/decoder_transformer/decoder_transformer_best_ar.pt', map_location='cpu')
print("Checkpoint keys:", list(checkpoint.keys()))
if 'test_loss' in checkpoint:
    print(f"\nTest Loss: {checkpoint['test_loss']}")
if 'test_mae' in checkpoint:
    print(f"Test MAE: {checkpoint['test_mae']}")
if 'test_rmse' in checkpoint:
    print(f"Test RMSE: {checkpoint['test_rmse']}")
if 'test_dir_acc' in checkpoint:
    print(f"Test Dir Acc: {checkpoint['test_dir_acc']}")
PYEOF
else
    echo "Checkpoint not found at $CHECKPOINT"
fi

