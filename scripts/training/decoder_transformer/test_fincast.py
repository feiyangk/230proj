import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path.cwd()))

# Import using importlib to avoid path issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fincast_extension",
    "fincast_extension.py"
)
fincast_ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fincast_ext)

FinCastBackbone = fincast_ext.FinCastBackbone
FINCAST_AVAILABLE = fincast_ext.FINCAST_AVAILABLE


if FINCAST_AVAILABLE:
    # Create test data
    test_data = torch.randn(2, 60)  # 2 samples, 60 timesteps
    
    fincast = FinCastBackbone(d_model=128, freeze=True)
    
    embeddings = fincast(test_data)
    
else:
