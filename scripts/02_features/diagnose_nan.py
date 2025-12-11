#!/usr/bin/env python3
"""Diagnose NaN values in processed features."""

import pandas as pd
import numpy as np


for split in ['train', 'val', 'test']:
    
    df = pd.read_parquet(f'data/processed/{split}.parquet')
    feature_cols = [c for c in df.columns if c not in ['date', 'target_1M', 'target_3M', 'target_6M']]
    
    total_features = len(feature_cols)
    total_samples = len(df)
    feature_nans = df[feature_cols].isna().sum().sum()
    
    
    if feature_nans > 0:
        # Show worst features
        nan_counts = df[feature_cols].isna().sum()
        worst = nan_counts[nan_counts > 0].sort_values(ascending=False).head(20)
        
        for feat, count in worst.items():
            pct = (count / total_samples) * 100
        
        # Check if NaN at start
        first_50_nans = df[feature_cols].iloc[:50].isna().sum().sum()
        
        # Group by type
        ma_features = [c for c in feature_cols if '_sma_' in c or '_ma_dev_' in c]
        if ma_features:
            ma_nans = df[ma_features].isna().sum().sum()
        
        return_features = [c for c in feature_cols if '_return_' in c]
        if return_features:
            ret_nans = df[return_features].isna().sum().sum()

