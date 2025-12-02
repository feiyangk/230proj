#!/usr/bin/env python3
"""
TensorBoard utility functions for consistent logging across training scripts.
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional


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
        
        # Check for managed TensorBoard directories
        tensorboard_log_dir = os.getenv('AIP_TENSORBOARD_LOG_DIR')
        sm_tensorboard_base = os.getenv('SM_OUTPUT_DATA_DIR')
        
        if tensorboard_log_dir:
            # Vertex AI managed TensorBoard - logs auto-sync
            log_dir = tensorboard_log_dir
            writer = SummaryWriter(str(log_dir))
            print(f"\n📊 TensorBoard (Vertex AI): {log_dir}")
            print(f"   Logs will auto-sync to TensorBoard instance")
        elif sm_tensorboard_base:
            base_log_dir = Path(sm_tensorboard_base) / 'tensorboard'
            log_dir = base_log_dir / f"{model_name}{eval_suffix}" / timestamp
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(log_dir))
            print(f"\n📊 TensorBoard logs (SageMaker) → {log_dir}")
            print(f"   These logs will be available in the SageMaker trial artifacts")
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


def log_attention_heatmaps(writer, model, val_loader, device, epoch, num_samples=2):
    """
    Extract and log attention weight heatmaps to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        model: PyTorch model
        val_loader: Validation DataLoader
        device: Device to run on
        epoch: Current epoch number
        num_samples: Number of samples to visualize
    """
    if writer is None:
        return
    
    try:
        model.eval()
        
        # Get the base model if it's wrapped (e.g., FinCast wrapper or PAN-NAN fusion)
        base_model = model
        preprocess_input = None
        
        if hasattr(model, 'decoder_transformer'):
            # Standard decoder transformer or FinCast wrapper
            base_model = model.decoder_transformer
        elif hasattr(model, 'pan_branch') and hasattr(model.pan_branch, 'decoder_transformer'):
            # PAN-NAN fusion model - need to preprocess input through PAN branch
            base_model = model.pan_branch.decoder_transformer
            def preprocess_input(x_sample):
                # Extract price, synth features like PAN branch does
                price_series = torch.index_select(x_sample, dim=2, index=model.pan_branch.price_idx)
                synth_features = torch.index_select(x_sample, dim=2, index=model.pan_branch.synth_idx)
                # Use the PAN branch's augmentation method
                augmented = model.pan_branch._augment_with_fincast(price_series, synth_features)
                return augmented
        
        # Check if model has transformer encoder
        if not hasattr(base_model, 'transformer_encoder'):
            print(f"  ⚠️  Base model does not have transformer_encoder")
            return
        
        # Get a few samples from validation set
        sample_count = 0
        attention_weights_found = False
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                if sample_count >= num_samples:
                    break
                
                X_batch = X_batch.to(device)
                batch_size = X_batch.shape[0]
                
                # Process each sample in batch
                for i in range(min(batch_size, num_samples - sample_count)):
                    x_sample = X_batch[i:i+1]  # [1, seq_len, features]
                    
                    # Preprocess input if needed (for PAN-NAN fusion)
                    if preprocess_input is not None:
                        x_processed = preprocess_input(x_sample)
                    else:
                        x_processed = x_sample
                    
                    # Extract attention weights
                    attention_weights = _extract_attention_from_model(base_model, x_processed, device)
                    
                    def log_heatmap(attn_tensor, layer_name, sample_idx, head_label="mean"):
                        attn_map = attn_tensor[0].cpu().numpy()
                        if attn_map.size == 0 or np.isnan(attn_map).all():
                            return
                        
                        # Percentile-based normalization to highlight subtle differences
                        # Use 1st and 99th percentiles to clip outliers and show distribution
                        attn_min_raw = attn_map.min()
                        attn_max_raw = attn_map.max()
                        attn_p1 = np.percentile(attn_map, 1)  # 1st percentile
                        attn_p99 = np.percentile(attn_map, 99)  # 99th percentile
                        attn_median = np.median(attn_map)
                        attn_std = attn_map.std()
                        
                        # Normalize using percentile range
                        attn_range = attn_p99 - attn_p1
                        if attn_range < 1e-8:
                            # If range is too small, use median-centered normalization with std
                            if attn_std < 1e-8:
                                attn_map_norm = np.ones_like(attn_map) * 0.5
                            else:
                                # Use std-based normalization centered at median
                                attn_map_norm = (attn_map - attn_median) / (3 * attn_std) + 0.5
                                attn_map_norm = np.clip(attn_map_norm, 0, 1)
                        else:
                            # Percentile-based normalization: clip to [p1, p99], then normalize to [0, 1]
                            attn_map_clipped = np.clip(attn_map, attn_p1, attn_p99)
                            attn_map_norm = (attn_map_clipped - attn_p1) / attn_range
                        
                        fig, ax = plt.subplots(figsize=(10, 10))
                        im = ax.imshow(attn_map_norm, cmap='viridis', aspect='auto', origin='upper', vmin=0, vmax=1)
                        title = f'Attention Weights - {layer_name} ({head_label}) - Sample {sample_idx+1}\n'
                        title += f'Raw: [{attn_min_raw:.4f}, {attn_max_raw:.4f}] | '
                        title += f'P1-P99: [{attn_p1:.4f}, {attn_p99:.4f}] | '
                        title += f'Median: {attn_median:.4f}, Std: {attn_std:.4f}'
                        ax.set_title(title)
                        ax.set_xlabel('Key Position (attended to)')
                        ax.set_ylabel('Query Position (attending from)')
                        plt.colorbar(im, ax=ax, label='Attention Weight')
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        buf.seek(0)
                        img = Image.open(buf)
                        img_array = np.array(img)
                        plt.close(fig)
                        if img_array.ndim == 3:
                            img_array = np.transpose(img_array, (2, 0, 1))
                        tag = f'Attention/{layer_name}/{head_label}/sample_{sample_idx}'
                        writer.add_image(tag, img_array, epoch, dataformats='CHW')

                    if attention_weights:
                        attention_weights_found = True
                        for layer_name, attn_weights in attention_weights.items():
                            if attn_weights.dim() == 4:  # [batch, heads, seq, seq]
                                # Log each head separately
                                num_heads = attn_weights.shape[1]
                                for head_idx in range(num_heads):
                                    log_heatmap(
                                        attn_weights[:, head_idx, :, :],
                                        layer_name,
                                        sample_count,
                                        head_label=f'head_{head_idx}'
                                    )
                                # Also log the mean over heads for convenience
                                attn_mean = attn_weights.mean(dim=1, keepdim=False)
                                log_heatmap(attn_mean, layer_name, sample_count, head_label='mean')
                            elif attn_weights.dim() == 3:  # [batch, seq, seq]
                                log_heatmap(attn_weights, layer_name, sample_count, head_label='mean')
                    
                    sample_count += 1
                    if sample_count >= num_samples:
                        break
                
                if sample_count >= num_samples:
                    break
        
        if attention_weights_found:
            print(f"  ✅ Logged attention heatmaps")
        else:
            print(f"  ⚠️  No attention weights extracted")
    
    except Exception as e:
        print(f"  ⚠️  Could not log attention heatmaps: {e}")
        import traceback
        traceback.print_exc()


def _extract_attention_from_model(model, x_sample, device):
    """
    Helper function to extract attention weights by manually calling attention layers.
    """
    attention_weights = {}
    
    # Check if model has transformer encoder
    if not hasattr(model, 'transformer_encoder'):
        return attention_weights
    
    encoder = model.transformer_encoder
    seq_len = x_sample.shape[1]
    
    # Prepare input through model's preprocessing
    if hasattr(model, 'input_projection'):
        x = model.input_projection(x_sample)
        if hasattr(model, 'pos_encoder'):
            x = model.pos_encoder(x)
        if hasattr(model, 'dropout_layer'):
            x = model.dropout_layer(x)
    else:
        x = x_sample
    
    # Get mask
    mask = None
    if hasattr(model, 'causal_mask'):
        mask = model.causal_mask[:seq_len, :seq_len]
    
    # Extract attention from each layer by manually calling attention
    x_temp = x
    for layer_idx, layer in enumerate(encoder.layers):
        if hasattr(layer, 'self_attn'):
            try:
                # Apply layer norm if present (pre-norm architecture)
                if hasattr(layer, 'norm1'):
                    x_norm = layer.norm1(x_temp)
                else:
                    x_norm = x_temp
                
                # Call attention directly with need_weights=True
                # MultiheadAttention returns (attn_output, attn_output_weights)
                _, attn_weights = layer.self_attn(
                    x_norm, x_norm, x_norm,
                    need_weights=True,
                    attn_mask=mask if mask is not None else None
                )
                
                if attn_weights is not None:
                    attention_weights[f'layer_{layer_idx}'] = attn_weights
                
                # Continue through layer to get input for next layer
                # This is needed to get the correct input for subsequent layers
                x_temp = layer(x_temp, src_mask=mask)
                
            except Exception as e:
                # If direct call fails, skip this layer
                continue
    
    return attention_weights

