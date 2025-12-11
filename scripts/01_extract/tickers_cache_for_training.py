"""
Phase 1: Extract stock data from BigQuery and save as Parquet.

This script extracts OHLCV data for specified tickers and creates
a clean, wide-format dataset for feature engineering.

Usage:
    # Use config file
    python scripts/01_extract/extract_tickers.py
    
    # Override tickers
    python scripts/01_extract/extract_tickers.py --tickers AAPL MSFT GOOGL
    
    # Override date range
    python scripts/01_extract/extract_tickers.py --start-date 2020-01-01
    
    # Add to existing tickers
    python scripts/01_extract/extract_tickers.py --add-tickers TSLA NVDA
    
    # Remove specific tickers
    python scripts/01_extract/extract_tickers.py --remove-tickers VIX
    
    # Force refresh (re-download even if file exists)
    python scripts/01_extract/extract_tickers.py --force
"""

import os
import sys
import argparse
import yaml
from datetime import datetime, date
from typing import List, Optional

import pandas as pd
from google.cloud import bigquery

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to path (two levels up from scripts/01_extract/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))




def load_config(config_path: str = 'configs/tickers.yaml') -> dict:
    """Load ticker configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, config_path: str = 'configs/tickers.yaml'):
    """Save ticker configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def build_query(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str],
    table_name: str,
    columns: Optional[List[str]] = None
) -> str:
    """
    Build BigQuery SQL query to extract ticker data.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD) or None for today
        table_name: Full BigQuery table name (project.dataset.table)
        columns: Columns to select (None = all)
    
    Returns:
        SQL query string
    """
    # Use all columns if not specified
    if columns is None:
        col_list = '*'
    else:
        col_list = ', '.join(columns)
    
    # Format ticker list for SQL IN clause
    ticker_list = ', '.join([f"'{t}'" for t in tickers])
    
    # Build date filter
    if end_date is None:
        end_date = date.today().strftime('%Y-%m-%d')
    
    query = f"""
    SELECT 
        {col_list}
    FROM 
        `{table_name}`
    WHERE 
        ticker IN ({ticker_list})
        AND date >= '{start_date}'
        AND date <= '{end_date}'
    ORDER BY 
        date ASC, ticker ASC
    """
    
    return query


def extract_data(
    config: dict,
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force: bool = False
) -> pd.DataFrame:
    """
    Extract data from BigQuery.
    
    Args:
        config: Configuration dictionary
        tickers: Override tickers from config
        start_date: Override start date
        end_date: Override end date
        force: Force re-download even if file exists
    
    Returns:
        DataFrame with OHLCV data
    """
    # Use config values if not overridden
    tickers = tickers or config['tickers']
    start_date = start_date or config['date_range']['start_date']
    end_date = end_date or config['date_range']['end_date']
    
    # Check if output already exists
    output_path = config['output']['raw_data_path']
    if os.path.exists(output_path) and not force:
        response = input("Load existing file? (y/n): ")
        if response.lower() == 'y':
            return pd.read_parquet(output_path)
    
    # Build BigQuery table name
    bq_config = config['bigquery']
    table_name = f"{bq_config['project_id']}.{bq_config['dataset_id']}.{bq_config['table_name']}"
    
    # Build query
    query = build_query(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        table_name=table_name,
        columns=config.get('columns')
    )
    
    
    # Show query (first 500 chars)
    
    # Execute query
    client = bigquery.Client(project=project_id)
    
    try:
        df = client.query(query).to_dataframe()
    except Exception as e:
        raise
    
    return df


def pivot_to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long format (ticker, date, value) to wide format.
    
    Input (long):
        date       | ticker | open | high | low | close | volume
        2023-01-01 | AAPL   | 130  | 132  | 129 | 131   | 100M
        2023-01-01 | MSFT   | 240  | 242  | 239 | 241   | 50M
        2023-01-02 | AAPL   | 131  | 133  | 130 | 132   | 95M
    
    Output (wide):
        date       | AAPL_open | AAPL_high | AAPL_low | AAPL_close | AAPL_volume | MSFT_open | ...
        2023-01-01 | 130       | 132       | 129      | 131        | 100M        | 240       | ...
        2023-01-02 | 131       | 133       | 130      | 132        | 95M         | 242       | ...
    """
    
    # Get value columns (everything except ticker and date)
    value_cols = [col for col in df.columns if col not in ['ticker', 'date', 'timestamp', 'ingested_at']]
    
    # Pivot each value column
    wide_dfs = []
    for col in value_cols:
        pivot = df.pivot(index='date', columns='ticker', values=col)
        # Rename columns: AAPL -> AAPL_close
        pivot.columns = [f"{ticker}_{col}" for ticker in pivot.columns]
        wide_dfs.append(pivot)
    
    # Combine all pivoted columns
    wide_df = pd.concat(wide_dfs, axis=1)
    wide_df.reset_index(inplace=True)
    
    
    return wide_df


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing data (holidays, gaps).
    
    For stock data:
    - Keep only actual trading days (no weekends/holidays)
    - Fill minor gaps with forward-fill
    """
    
    # Convert date to datetime if not already
    df['date'] = pd.to_datetime(df['date'])
    
    # Check calendar coverage
    date_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
    missing_dates = len(date_range) - len(df)
    
    # Forward-fill any minor gaps within trading days
    # This handles rare cases where a ticker might be missing a single day
    ticker_cols = [col for col in df.columns if col != 'date']
    df[ticker_cols] = df[ticker_cols].ffill()
    
    # Count remaining missing values
    missing_summary = df.isnull().sum()
    if missing_summary.sum() > 0:
        for col, count in missing_summary[missing_summary > 0].items():
            pass
    # Drop rows where ALL ticker columns are null (shouldn't happen)
    df = df.dropna(subset=ticker_cols, how='all')
    
    
    return df


def save_to_parquet(df: pd.DataFrame, output_path: str, compression: str = 'snappy'):
    """Save DataFrame to Parquet format."""
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to Parquet
    df.to_parquet(output_path, compression=compression, index=False)
    
    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    # Show summary


def check_data_availability(config: dict, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Check data availability for all tickers.
    
    Queries BigQuery to determine:
    - Start date
    - End date  
    - Number of trading days available
    - Data gaps
    
    Args:
        config: Configuration dictionary
        tickers: Override tickers from config
    
    Returns:
        DataFrame with availability summary per ticker
    """
    tickers = tickers or config['tickers']
    
    # Build BigQuery table name from environment variables
    project_id = os.environ.get('GCP_PROJECT_ID')
    dataset_id = os.environ.get('BQ_DATASET', 'raw_dataset')
    table_name_only = os.environ.get('BQ_TABLE', 'raw_ohlcv')
    table_name = f"{project_id}.{dataset_id}.{table_name_only}"
    
    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable not set")
    
    
    # Format ticker list for SQL
    ticker_list = ', '.join([f"'{t}'" for t in tickers])
    
    # Query to get availability per ticker
    query = f"""
    SELECT 
        ticker,
        MIN(date) as start_date,
        MAX(date) as end_date,
        COUNT(DISTINCT date) as trading_days
    FROM 
        `{table_name}`
    WHERE 
        ticker IN ({ticker_list})
    GROUP BY 
        ticker
    ORDER BY 
        ticker
    """
    
    # Execute query
    client = bigquery.Client(project=bq_config['project_id'])
    
    try:
        df = client.query(query).to_dataframe()
    except Exception as e:
        raise
    
    # Check for missing tickers
    found_tickers = set(df['ticker'].tolist())
    missing_tickers = set(tickers) - found_tickers
    
    if missing_tickers:
    
    # Print summary table
    
    for _, row in df.iterrows():
        ticker = row['ticker']
        start = row['start_date'].strftime('%Y-%m-%d')
        end = row['end_date'].strftime('%Y-%m-%d')
        days = int(row['trading_days'])
        
    
    
    # Summary statistics
    if len(df) > 0:
        
        # Identify tickers with significantly fewer days than average
        avg_days = df['trading_days'].mean()
        threshold = avg_days * 0.9  # Less than 90% of average
        limited_data = df[df['trading_days'] < threshold]
        if len(limited_data) > 0:
            for _, row in limited_data.iterrows():
    
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Extract stock data from BigQuery (Phase 1)')
    
    # Ticker management
    parser.add_argument('--tickers', nargs='+', help='Override tickers from config file')
    parser.add_argument('--add-tickers', nargs='+', help='Add tickers to config list')
    parser.add_argument('--remove-tickers', nargs='+', help='Remove tickers from config list')
    parser.add_argument('--replace-tickers', nargs='+', help='Replace all tickers in config')
    
    # Date range
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    # Config and output
    parser.add_argument('--config', default='configs/tickers.yaml', help='Path to config file')
    parser.add_argument('--output', help='Override output path')
    parser.add_argument('--force', action='store_true', help='Force re-download even if file exists')
    parser.add_argument('--save-config', action='store_true', help='Save updated ticker list to config')
    
    # Data availability check
    parser.add_argument('--check-availability', action='store_true', help='Check data availability for all tickers (no extraction)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Manage ticker list
    tickers = config['tickers'].copy()
    
    if args.replace_tickers:
        tickers = args.replace_tickers
        if args.save_config:
            config['tickers'] = tickers
            save_config(config, args.config)
    
    if args.add_tickers:
        new_tickers = [t for t in args.add_tickers if t not in tickers]
        tickers.extend(new_tickers)
        if args.save_config:
            config['tickers'] = tickers
            save_config(config, args.config)
    
    if args.remove_tickers:
        tickers = [t for t in tickers if t not in args.remove_tickers]
        if args.save_config:
            config['tickers'] = tickers
            save_config(config, args.config)
    
    if args.tickers:
        tickers = args.tickers
    
    # Check availability mode
    if args.check_availability:
        check_data_availability(config, tickers)
        return
    
    # Override output path if specified
    if args.output:
        config['output']['raw_data_path'] = args.output
    
    # Extract data
    df = extract_data(
        config=config,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        force=args.force
    )
    
    # Transform to wide format
    df_wide = pivot_to_wide_format(df)
    
    # Handle missing data
    df_clean = handle_missing_data(df_wide)
    
    # Save to Parquet
    save_to_parquet(
        df_clean,
        config['output']['raw_data_path'],
        config['output']['compression']
    )
    


if __name__ == '__main__':
    main()
