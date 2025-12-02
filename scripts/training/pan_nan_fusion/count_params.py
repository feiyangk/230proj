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
    
    print("="*80)
    print("PAN-NAN Fusion Model Parameter Count")
    print("="*80)
    
    # WITH FinCast (frozen)
    print("\n1. WITH FinCast (frozen backbone):")
    print("-" * 80)
    
    fincast_config = config.get("fincast", {})
    fincast_config["freeze"] = True  # Ensure frozen
    
    model_with_fincast = FusionDecoderTransformer(
        config=config,
        num_features=num_features,
        fincast_config=fincast_config,
        fusion_config=config.get("fusion", {})
    )
    
    total_with, trainable_with = count_parameters(model_with_fincast)
    
    print(f"   Total parameters:     {total_with:,}")
    print(f"   Trainable parameters: {trainable_with:,}")
    print(f"   Frozen parameters:   {total_with - trainable_with:,}")
    
    # Breakdown by component
    print("\n   Breakdown:")
    breakdown = count_by_module(model_with_fincast)
    
    # PAN branch
    pan_total = sum(v["total"] for k, v in breakdown.items() if "pan_branch" in k)
    pan_trainable = sum(v["trainable"] for k, v in breakdown.items() if "pan_branch" in k)
    print(f"     PAN branch:         {pan_total:,} total, {pan_trainable:,} trainable")
    
    # FinCast within PAN
    fincast_total = sum(v["total"] for k, v in breakdown.items() if "fincast" in k.lower())
    fincast_trainable = sum(v["trainable"] for k, v in breakdown.items() if "fincast" in k.lower())
    print(f"       └─ FinCast:       {fincast_total:,} total, {fincast_trainable:,} trainable (frozen)")
    
    # Decoder transformer within PAN
    decoder_total = sum(v["total"] for k, v in breakdown.items() if "decoder_transformer" in k and "pan_branch" in k)
    decoder_trainable = sum(v["trainable"] for k, v in breakdown.items() if "decoder_transformer" in k and "pan_branch" in k)
    print(f"       └─ Decoder:       {decoder_total:,} total, {decoder_trainable:,} trainable")
    
    # NAN branch
    nan_total = sum(v["total"] for k, v in breakdown.items() if "nan_branch" in k)
    nan_trainable = sum(v["trainable"] for k, v in breakdown.items() if "nan_branch" in k)
    print(f"     NAN branch:         {nan_total:,} total, {nan_trainable:,} trainable")
    
    # Fusion head
    fusion_total = sum(v["total"] for k, v in breakdown.items() if "fusion_head" in k)
    fusion_trainable = sum(v["trainable"] for k, v in breakdown.items() if "fusion_head" in k)
    print(f"     Fusion head:        {fusion_total:,} total, {fusion_trainable:,} trainable")
    
    # WITHOUT FinCast (not recommended, but let's see)
    print("\n2. WITHOUT FinCast (PAN-NAN requires FinCast, this is for comparison only):")
    print("-" * 80)
    print("   ⚠️  Note: PAN-NAN fusion requires FinCast. Without it, the model")
    print("      architecture would be different (no PAN branch augmentation).")
    print("      This comparison shows the impact of FinCast parameters.")
    
    # FinCast is huge - estimate
    # FinCast FFM: ~50 layers, d_model=1280, n_heads=16
    # Rough estimate: ~100-200M parameters
    fincast_size_estimate = fincast_total
    print(f"\n   Estimated FinCast size: {fincast_size_estimate:,} parameters")
    print(f"   Without FinCast, trainable would be: ~{trainable_with + fincast_size_estimate:,}")
    print(f"   (But model wouldn't work without FinCast in PAN-NAN)")
    
    print("\n" + "="*80)
    print(f"Summary:")
    print(f"  With FinCast (frozen):  {trainable_with:,} trainable parameters")
    print(f"  FinCast contribution:    {fincast_size_estimate:,} parameters (frozen)")
    print(f"  Total model size:       {total_with:,} parameters")
    print("="*80)

if __name__ == "__main__":
    main()

