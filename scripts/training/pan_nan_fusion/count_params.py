#!/usr/bin/env python3
"""
Count trainable parameters in PAN-NAN fusion model with and without FinCast.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import torch
from scripts.training.pan_nan_fusion.model import FusionDecoderTransformer, PANDecoderTransformer

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def count_by_module(model, prefix=""):
    """Count parameters by module."""
    results = {}
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        results[full_name] = {"total": total, "trainable": trainable}
        
        # Recursively count submodules
        if len(list(module.children())) > 0:
            sub_results = count_by_module(module, full_name)
            results.update(sub_results)
    
    return results

def main():
    config_path = "configs/model_pan_nan_fusion.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Estimate num_features (will be calculated from data)
    # For estimation, use a typical value
    num_features = 118  # Typical value from your data
    
    
    # WITH FinCast (frozen)
    
    fincast_config = config.get("fincast", {})
    fincast_config["freeze"] = True  # Ensure frozen
    
    model_with_fincast = FusionDecoderTransformer(
        config=config,
        num_features=num_features,
        fincast_config=fincast_config,
        fusion_config=config.get("fusion", {})
    )
    
    total_with, trainable_with = count_parameters(model_with_fincast)
    
    
    # Breakdown by component
    breakdown = count_by_module(model_with_fincast)
    
    # PAN branch
    pan_total = sum(v["total"] for k, v in breakdown.items() if "pan_branch" in k)
    pan_trainable = sum(v["trainable"] for k, v in breakdown.items() if "pan_branch" in k)
    
    # FinCast within PAN
    fincast_total = sum(v["total"] for k, v in breakdown.items() if "fincast" in k.lower())
    fincast_trainable = sum(v["trainable"] for k, v in breakdown.items() if "fincast" in k.lower())
    
    # Decoder transformer within PAN
    decoder_total = sum(v["total"] for k, v in breakdown.items() if "decoder_transformer" in k and "pan_branch" in k)
    decoder_trainable = sum(v["trainable"] for k, v in breakdown.items() if "decoder_transformer" in k and "pan_branch" in k)
    
    # NAN branch
    nan_total = sum(v["total"] for k, v in breakdown.items() if "nan_branch" in k)
    nan_trainable = sum(v["trainable"] for k, v in breakdown.items() if "nan_branch" in k)
    
    # Fusion head
    fusion_total = sum(v["total"] for k, v in breakdown.items() if "fusion_head" in k)
    fusion_trainable = sum(v["trainable"] for k, v in breakdown.items() if "fusion_head" in k)
    
    # WITHOUT FinCast (not recommended, but let's see)
    
    # FinCast is huge - estimate
    # FinCast FFM: ~50 layers, d_model=1280, n_heads=16
    # Rough estimate: ~100-200M parameters
    fincast_size_estimate = fincast_total
    

if __name__ == "__main__":
    main()

