#!/bin/bash
# Dimension search script for PAN-NAN Fusion with FinCast enabled
# Runs grid search over dimension hyperparameters in the background

# Configuration
CONFIG="configs/model_pan_nan_fusion.yaml"
LOG_FILE="dim_search_fincast_$(date +%Y%m%d_%H%M%S).log"
EPOCHS=50
SEED=42

# Dimension search ranges (matching the hyperparameter table)
D_MODEL_VALUES=(96 128)
D_FF_VALUES=(384 512)
SENTIMENT_HIDDEN_VALUES=(64 128)
FUSION_HIDDEN_VALUES=(128 256)

echo "=================================================================================="
echo "Starting Dimension Search with FinCast"
echo "=================================================================================="
echo "Config: $CONFIG"
echo "Log file: $LOG_FILE"
echo "Epochs per trial: $EPOCHS"
echo "Seed: $SEED"
echo ""
echo "Search ranges:"
echo "  d_model: ${D_MODEL_VALUES[@]}"
echo "  d_ff: ${D_FF_VALUES[@]}"
echo "  sentiment_hidden_dim: ${SENTIMENT_HIDDEN_VALUES[@]}"
echo "  fusion_hidden_dim: ${FUSION_HIDDEN_VALUES[@]}"
echo ""
echo "Total combinations: $((${#D_MODEL_VALUES[@]} * ${#D_FF_VALUES[@]} * ${#SENTIMENT_HIDDEN_VALUES[@]} * ${#FUSION_HIDDEN_VALUES[@]}))"
echo "=================================================================================="
echo ""

# Run dimension search with FinCast enabled (default behavior, no --no-fincast flag)
nohup python scripts/training/pan_nan_fusion/dim_search.py \
  --config "$CONFIG" \
  --method grid \
  --d-model "${D_MODEL_VALUES[@]}" \
  --d-ff "${D_FF_VALUES[@]}" \
  --sentiment-hidden "${SENTIMENT_HIDDEN_VALUES[@]}" \
  --fusion-hidden "${FUSION_HIDDEN_VALUES[@]}" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  > "$LOG_FILE" 2>&1 &

# Get the process ID
PID=$!

echo "✅ Dimension search started in background"
echo "   Process ID: $PID"
echo "   Log file: $LOG_FILE"
echo ""
echo "To monitor progress:"
echo "   tail -f $LOG_FILE"
echo ""
echo "To check if still running:"
echo "   ps aux | grep $PID"
echo "   or"
echo "   ps aux | grep dim_search"
echo ""
echo "To stop the search:"
echo "   kill $PID"
echo ""

# Save PID to file for easy reference
echo "$PID" > "${LOG_FILE%.log}.pid"
echo "PID saved to: ${LOG_FILE%.log}.pid"

