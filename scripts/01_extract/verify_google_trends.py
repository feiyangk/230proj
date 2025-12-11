#!/usr/bin/env python
"""Verify extracted Google Trends data quality."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def verify_google_trends(file_path: str = 'data/raw/google_trends.parquet') -> bool:
    """
    Verify Google Trends data quality.
    
    Args:
        file_path: Path to Google Trends parquet file
        
    Returns:
        True if all checks pass, False otherwise
    """
    
    checks_passed = 0
    checks_failed = 0
    warnings = 0
    
    # Check 1: File exists
    file_path = Path(file_path)
    if not file_path.exists():
        return False
    checks_passed += 1
    
    # Check 2: Load data
    try:
        df = pd.read_parquet(file_path)
        checks_passed += 1
    except Exception as e:
        return False
    
    # Check 3: Required columns
    required_cols = ['date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        checks_failed += 1
    else:
        checks_passed += 1
    
    # Get keyword columns (all except 'date')
    keyword_cols = [col for col in df.columns if col != 'date']
    
    # Check 4: Date column
    try:
        df['date'] = pd.to_datetime(df['date'])
        checks_passed += 1
    except Exception as e:
        checks_failed += 1
    
    # Check 5: Date continuity
    df_sorted = df.sort_values('date')
    date_diffs = df_sorted['date'].diff().dt.days
    gaps = date_diffs[date_diffs > 1]
    if len(gaps) > 0:
        warnings += 1
    else:
        checks_passed += 1
    
    # Check 6: Value ranges (Google Trends is 0-100)
    all_in_range = True
    for col in keyword_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val < 0 or max_val > 100:
                all_in_range = False
            else:
    
    if all_in_range:
        checks_passed += 1
    else:
        checks_failed += 1
    
    # Check 7: Missing values
    missing_counts = df[keyword_cols].isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        for col in keyword_cols:
            if missing_counts[col] > 0:
                pct = (missing_counts[col] / len(df)) * 100
        warnings += 1
    else:
        checks_passed += 1
    
    # Check 8: Data statistics
    for col in keyword_cols:
        if col in df.columns:
            stats = df[col].describe()
            
            # Check for constant values
            if stats['std'] < 0.1:
                warnings += 1
    
    checks_passed += 1
    
    # Summary
    if checks_failed > 0:
    if warnings > 0:
    
    success = checks_failed == 0
    if success:
    else:
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description='Verify Google Trends data quality',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--file',
        default='data/raw/google_trends.parquet',
        help='Path to Google Trends parquet file'
    )
    
    args = parser.parse_args()
    
    success = verify_google_trends(args.file)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
