#!/bin/bash
# Training script for PAN-NAN without FinCast

cd /home/ubuntu/230proj

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run training with nohup
nohup python scripts/training/pan_nan_fusion/train.py \
  --config configs/model_pan_nan_fusion.yaml \
  --no-fincast \
  --seed 42 \
  --epochs 100 \
  --batch-size 64 \
  > pan_nan_no_fincast_training.log 2>&1 &

echo "Training started with PID $!"
echo "Logs: pan_nan_no_fincast_training.log"
echo "Monitor with: tail -f pan_nan_no_fincast_training.log"

