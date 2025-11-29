#!/usr/bin/env python3
"""
TensorBoard utility functions for consistent logging across training scripts.
"""

import os
import yaml
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def initialize_tensorboard_writer(config, model_name, eval_suffix):
    """
    Initialize TensorBoard writer with proper directory structure.
    
    Args:
        config: Configuration dictionary
        model_name: Name of the model (e.g., 'decoder', 'tft')
        eval_suffix: Suffix for evaluation mode (e.g., '_tf', '_ar')
    
    Returns:
        SummaryWriter instance or None if TensorBoard is disabled
    """
    if not config.get('logging', {}).get('tensorboard', False):
        return None
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Check for Vertex AI TensorBoard directory (set automatically by Vertex AI)
        tensorboard_log_dir = os.getenv('AIP_TENSORBOARD_LOG_DIR')
        
        if tensorboard_log_dir:
            # Vertex AI managed TensorBoard - logs auto-sync
            log_dir = tensorboard_log_dir
            writer = SummaryWriter(str(log_dir))
            print(f"\n📊 TensorBoard (Vertex AI): {log_dir}")
            print(f"   Logs will auto-sync to TensorBoard instance")
        else:
            # Local or manual TensorBoard - use local paths
            base_log_dir = Path(config.get('logging', {}).get('log_dir', 'logs/tensorboard'))
            log_dir = base_log_dir / f"{model_name}{eval_suffix}" / timestamp
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(log_dir))
            print(f"\n📊 TensorBoard logs → {log_dir}")
            print(f"   View with: tensorboard --logdir {base_log_dir}")
        
        return writer
    except Exception as e:
        print(f"\n⚠️  TensorBoard not available: {type(e).__name__}")
        print(f"   Error details: {str(e)}")
        print("   Training will continue without TensorBoard logging")
        return None


def log_experiment_metadata(
    writer, dataset_version, start_date, end_date, horizons_config,
    lookback, num_features, num_horizons,
    train_samples, val_samples, test_samples,
    model_name, config, total_params, trainable_params,
    additional_model_info=None
):
    """
    Log comprehensive experiment metadata to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        dataset_version: Dataset version string
        start_date: Start date of dataset
        end_date: End date of dataset
        horizons_config: List of prediction horizons
        lookback: Lookback window size
        num_features: Number of input features
        num_horizons: Number of prediction horizons
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        model_name: Name of the model
        config: Full configuration dictionary
        total_params: Total number of model parameters
        trainable_params: Number of trainable parameters
        additional_model_info: Optional dict with additional model information
    """
    if writer is None:
        return
    
    # Log text summary
    summary_text = f"""
# Experiment Metadata

## Dataset
- Version: {dataset_version or 'N/A'}
- Date Range: {start_date} to {end_date}
- Lookback Window: {lookback} timesteps
- Features: {num_features}
- Prediction Horizons: {horizons_config}

## Data Splits
- Training: {train_samples:,} samples
- Validation: {val_samples:,} samples
- Test: {test_samples:,} samples

## Model
- Name: {model_name}
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
"""
    
    if additional_model_info:
        summary_text += "\n## Additional Info\n"
        for key, value in additional_model_info.items():
            summary_text += f"- {key}: {value}\n"
    
    # Log as text
    writer.add_text('Experiment/Metadata', summary_text)
    
    # Log config as YAML
    try:
        config_yaml = yaml.dump(config, default_flow_style=False)
        writer.add_text('Experiment/Config', f"```yaml\n{config_yaml}\n```")
    except Exception:
        pass


def log_epoch_metrics(
    writer, epoch, train_loss, val_loss, mae, rmse, dir_acc,
    per_horizon_metrics, learning_rate, avg_unclipped, avg_clipped
):
    """
    Log epoch metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        mae: Mean absolute error
        rmse: Root mean squared error
        dir_acc: Directional accuracy
        per_horizon_metrics: Dictionary of per-horizon metrics (e.g., {'H4_MAE': 0.5, 'H8_MAE': 0.6})
        learning_rate: Current learning rate
        avg_unclipped: Average unclipped gradient norm
        avg_clipped: Average clipped gradient norm
    """
    if writer is None:
        return
    
    # Loss metrics
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    
    # Performance metrics
    writer.add_scalar('Metrics/MAE', mae, epoch)
    writer.add_scalar('Metrics/RMSE', rmse, epoch)
    writer.add_scalar('Metrics/DirectionalAccuracy', dir_acc, epoch)
    
    # Per-horizon metrics
    if per_horizon_metrics:
        for metric_name, metric_value in per_horizon_metrics.items():
            if 'MAE' in metric_name:
                writer.add_scalar(f'Metrics/PerHorizon/{metric_name}', metric_value, epoch)
            elif 'RMSE' in metric_name:
                writer.add_scalar(f'Metrics/PerHorizon/{metric_name}', metric_value, epoch)
    
    # Gradient metrics
    writer.add_scalar('Gradients/Unclipped_Avg', avg_unclipped, epoch)
    writer.add_scalar('Gradients/Clipped_Avg', avg_clipped, epoch)
    
    # Learning rate
    writer.add_scalar('LR', learning_rate, epoch)
    
    writer.flush()


def log_gradients_and_weights(writer, model, epoch):
    """
    Log gradient and weight histograms for all model parameters.
    
    Args:
        writer: TensorBoard SummaryWriter
        model: PyTorch model
        epoch: Current epoch number
    """
    if writer is None:
        return
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Log gradient histogram
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        # Log weight histogram
        writer.add_histogram(f'Weights/{name}', param.data, epoch)


def log_hyperparameters(writer, hparams, metrics):
    """
    Log hyperparameters and final metrics to TensorBoard HParams dashboard.
    
    Args:
        writer: TensorBoard SummaryWriter
        hparams: Dictionary of hyperparameters
        metrics: Dictionary of final metrics (should have 'hparam/' prefix)
    """
    if writer is None:
        return
    
    try:
        # Log hyperparameters using add_hparams
        writer.add_hparams(hparams, metrics)
    except Exception as e:
        # Fallback: log as scalars if add_hparams fails
        print(f"⚠️  Could not log hyperparameters: {e}")
        for key, value in hparams.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f'HParams/{key}', value, 0)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, 0)

