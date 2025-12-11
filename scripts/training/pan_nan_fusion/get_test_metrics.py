#!/usr/bin/env python3
"""
Re-evaluate model on test set to get test metrics.
Loads checkpoint and runs evaluation.
"""

import sys
import torch
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from scripts.training.decoder_transformer.decoder_transformer_train import evaluate
from scripts.training.pan_nan_fusion.model import FusionDecoderTransformer

def main():
    config_path = Path("configs/model_pan_nan_fusion.yaml")
    checkpoint_path = Path("models/decoder_transformer/decoder_transformer_best_ar.pt")
    data_dir = Path("data/processed")
    
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load test data
    test_X_np = np.load(data_dir / 'X_test.npy', allow_pickle=True)
    test_y_np = np.load(data_dir / 'y_test.npy', allow_pickle=True)
    
    # Handle object arrays
    if test_X_np.dtype == object:
        test_X_np = test_X_np.item() if test_X_np.shape == () else np.array(test_X_np.tolist())
    if test_y_np.dtype == object:
        test_y_np = test_y_np.item() if test_y_np.shape == () else np.array(test_y_np.tolist())
    
    # Convert to tensors
    test_X = torch.FloatTensor(test_X_np.astype(np.float32))
    test_y = torch.FloatTensor(test_y_np.astype(np.float32))
    
    # Create DataLoader
    batch_size = config['training']['batch_size']
    test_dataset = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model (PAN-NAN fusion)
    model = FusionDecoderTransformer(
        config=config,
        num_features=None,  # Will be inferred
        fincast_config=config.get('fincast', {})
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get evaluation mode from config
    use_teacher_forcing_eval = config['training'].get('eval_teacher_forcing', True)
    eval_mode = "Teacher Forcing" if use_teacher_forcing_eval else "Autoregressive"
    
    # Get horizons
    horizons = config['data']['prediction_horizons']
    
    # Evaluate on test set
    criterion = torch.nn.MSELoss()
    
    test_loss, test_mae, test_rmse, test_dir_acc, test_per_horizon = evaluate(
        model, test_loader, criterion, device,
        use_teacher_forcing=use_teacher_forcing_eval,
        log_mode=True,
        horizons=horizons
    )
    
    # Print results
    
    if test_per_horizon:
        for key, value in sorted(test_per_horizon.items()):
            if 'MAE' in key:
    
    # Print LaTeX format
    
    if test_per_horizon:
        for horizon in ['H7', 'H14', 'H28']:
            h_key = f"{horizon}_MAE"
            h_mae = test_per_horizon.get(h_key)
            if h_mae:
            else:
    

if __name__ == '__main__':
    main()

