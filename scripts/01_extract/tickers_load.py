"""Master orchestration script for loading and validating ticker data.

Runs the complete pipeline:
1. Load raw OHLCV from Polygon.io (with validation)
2. Compute synthetic indicators (with validation)
3. Verify and export combined data

Reads configuration from tickers.yaml with override capability.
Cleans up and stops on validation failure at any step.
"""
import os
import sys
import argparse
import subprocess
import yaml
from datetime import datetime
from pathlib import Path
from google.cloud import bigquery

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
DATASET_ID = os.environ.get('BQ_DATASET', 'raw_dataset')
RAW_TABLE = os.environ.get('BQ_TABLE', 'raw_ohlcv')
SYNTHETIC_TABLE = os.environ.get('BQ_SYNTHETIC_TABLE', 'synthetic_indicators')

# Default indicators to compute
DEFAULT_INDICATORS = [
    'sma_10', 'sma_20', 'sma_50', 'sma_200',
    'volatility_10', 'volatility_20',
    'bb_upper_20', 'bb_lower_20',
    'atr_14'
]

def print_header(text: str):
    """Print formatted section header."""

def load_config(config_path: str = 'configs/tickers.yaml') -> dict:
    """Load ticker configuration from YAML file."""
    if not os.path.exists(config_path):
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def validate_ticker(ticker: str, config: dict) -> bool:
    """Check if ticker exists in config."""
    tickers = config.get('tickers', [])
    if ticker not in tickers:
        return False
    return True

def clean_ticker_data(ticker: str, frequency: str, table: str, client: bigquery.Client):
    """Delete all data for a specific ticker and frequency from a table."""
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table}"
    
    query = f"""
    DELETE FROM `{table_id}`
    WHERE ticker = '{ticker}' AND frequency = '{frequency}'
    """
    
    try:
        client.query(query).result()
    except Exception as e:

def run_command(cmd: list, step_name: str) -> bool:
    """Run a shell command and return success status."""
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        return False

def validate_raw_data(ticker: str, frequency: str, start: str, end: str, 
                      client: bigquery.Client) -> tuple[bool, str]:
    """Validate raw OHLCV data exists and has reasonable coverage."""
    
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{RAW_TABLE}"
    
    query = f"""
    SELECT 
        COUNT(*) as row_count,
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(DISTINCT date) as unique_days
    FROM `{table_id}`
    WHERE ticker = '{ticker}' 
        AND frequency = '{frequency}'
        AND date >= '{start}'
        AND date <= '{end}'
    """
    
    try:
        df = client.query(query).to_dataframe()
        
        if df.empty or df['row_count'].iloc[0] == 0:
            return False, "No data found"
        
        row_count = df['row_count'].iloc[0]
        min_date = df['min_date'].iloc[0]
        max_date = df['max_date'].iloc[0]
        unique_days = df['unique_days'].iloc[0]
        
        
        # Basic validation: should have some data
        if row_count < 100:
            return False, f"Insufficient data (only {row_count} rows)"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Query error: {e}"

def validate_synthetic_data(ticker: str, frequency: str, start: str, end: str,
                           indicators: list, client: bigquery.Client) -> tuple[bool, str]:
    """Validate synthetic indicator data exists."""
    
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{SYNTHETIC_TABLE}"
    
    # Check each indicator
    for ind in indicators:
        query = f"""
        SELECT COUNT(*) as count
        FROM `{table_id}`
        WHERE ticker = '{ticker}'
            AND frequency = '{frequency}'
            AND date >= '{start}'
            AND date <= '{end}'
            AND {ind} IS NOT NULL
        """
        
        try:
            df = client.query(query).to_dataframe()
            count = df['count'].iloc[0]
            
            if count == 0:
                return False, f"No data for indicator: {ind}"
            
            
        except Exception as e:
            return False, f"Query error for {ind}: {e}"
    
    return True, "Valid"

def main():
    parser = argparse.ArgumentParser(
        description='Master script to load and validate ticker data pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Load all tickers from config
  python scripts/01_extract/tickers_load.py --frequency daily
  
  # Load specific ticker
  python scripts/01_extract/tickers_load.py --ticker SPY --frequency daily
  
  # Override dates and indicators
  python scripts/01_extract/tickers_load.py \\
    --ticker SPY \\
    --frequency daily \\
    --start 2024-01-01 \\
    --end 2024-01-31 \\
    --indicators sma_50 sma_200 volatility_20
  
  # Skip polygon load (data already exists)
  python scripts/01_extract/tickers_load.py \\
    --ticker SPY \\
    --frequency daily \\
    --skip-polygon
  
  # Clean and reload all tickers
  python scripts/01_extract/tickers_load.py \\
    --frequency daily \\
    --clean
  
  # Top-up mode (incremental load)
  python scripts/01_extract/tickers_load.py \\
    --frequency daily \\
    --top-up
        """
    )
    
    parser.add_argument('--ticker', type=str,
                       help='Ticker symbol (e.g., SPY). If not specified, processes all tickers from config')
    parser.add_argument('--frequency', type=str, required=True,
                       choices=['5m', '15m', '1h', 'hourly', '1d', 'daily'],
                       help='Data frequency')
    parser.add_argument('--start', type=str,
                       help='Start date (YYYY-MM-DD), defaults to config')
    parser.add_argument('--end', type=str,
                       help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--indicators', nargs='+',
                       help='Indicators to compute (defaults to config)')
    parser.add_argument('--config', type=str, default='configs/tickers.yaml',
                       help='Path to ticker config file (default: configs/tickers.yaml)')
    parser.add_argument('--skip-polygon', action='store_true',
                       help='Skip polygon data load (assume data exists)')
    parser.add_argument('--skip-synthetic', action='store_true',
                       help='Skip synthetic indicators (assume they exist)')
    parser.add_argument('--skip-verify', action='store_true',
                       help='Skip verification step')
    parser.add_argument('--clean', action='store_true',
                       help='Clean existing data before loading')
    parser.add_argument('--reload', action='store_true',
                       help='Reload existing data (same as --clean but matches underlying script flags)')
    parser.add_argument('--top-up', action='store_true',
                       help='Incremental mode: auto-detect latest data and fill gaps to end date')
    parser.add_argument('--export-path', type=str,
                       help='Custom path for verification export files')
    
    args = parser.parse_args()
    
    # Check PROJECT_ID
    if not PROJECT_ID:
        sys.exit(1)
    
    # Initialize BigQuery client
    client = bigquery.Client(project=PROJECT_ID)
    
    # Load config
    print_header("Loading Configuration")
    config = load_config(args.config)
    
    # Determine tickers to process
    if args.ticker:
        tickers = [args.ticker]
        validate_ticker(args.ticker, config)
    else:
        tickers = config.get('tickers', [])
        if not tickers:
            sys.exit(1)
    
    # Determine dates
    start_date = args.start or config['date_range']['start_date']
    end_date = args.end or config['date_range'].get('end_date') or datetime.now().strftime('%Y-%m-%d')
    
    # Determine indicators (priority: CLI args > config > defaults)
    indicators = args.indicators or config.get('indicators', DEFAULT_INDICATORS)
    
    if args.top_up:
    else:
    
    # Clean if requested
    if args.clean:
        print_header("Cleaning Existing Data")
        for ticker in tickers:
            clean_ticker_data(ticker, args.frequency, RAW_TABLE, client)
            clean_ticker_data(ticker, args.frequency, SYNTHETIC_TABLE, client)
    
    # ========================================
    # STEP 1: Load Polygon Data
    # ========================================
    if not args.skip_polygon:
        print_header("Step 1: Loading Raw OHLCV from Polygon.io")
        
        cmd = [
            'python', 'scripts/01_extract/tickers_load_polygon.py',
            '--frequency', args.frequency,
            '--start', start_date,
            '--end', end_date
        ]
        
        # Add tickers
        if args.ticker:
            cmd.extend(['--tickers', args.ticker])
        # else: script will load from config
        
        if args.reload or args.clean:
            cmd.append('--reload')
        
        if args.top_up:
            cmd.append('--top-up')
        
        if not run_command(cmd, "Polygon Data Load"):
            if args.ticker:
                clean_ticker_data(args.ticker, args.frequency, RAW_TABLE, client)
            sys.exit(1)
        
        # Quick check: did we actually insert any data?
        if args.ticker:
            try:
                check_query = f"""
                SELECT COUNT(*) as count
                FROM `{PROJECT_ID}.{DATASET_ID}.{RAW_TABLE}`
                WHERE ticker = '{args.ticker}'
                  AND frequency = '{args.frequency}'
                  AND date >= '{start_date}'
                  AND date <= '{end_date}'
                """
                result = client.query(check_query).to_dataframe()
                row_count = result['count'].iloc[0]
                
                if row_count == 0:
                    sys.exit(1)
            except Exception as e:
        
        # Validate raw data (only if single ticker)
        if args.ticker:
            valid, message = validate_raw_data(args.ticker, args.frequency, start_date, end_date, client)
            if not valid:
                clean_ticker_data(args.ticker, args.frequency, RAW_TABLE, client)
                sys.exit(1)
    else:
    
    # ========================================
    # STEP 2: Load Synthetic Indicators
    # ========================================
    if not args.skip_synthetic:
        print_header("Step 2: Computing Synthetic Indicators")
        
        cmd = [
            'python', 'scripts/01_extract/tickers_load_synthetic.py',
            '--frequency', args.frequency,
            '--start', start_date,
            '--end', end_date
        ]
        
        # Add tickers
        if args.ticker:
            cmd.extend(['--tickers', args.ticker])
        # else: script will load from config
        
        if args.reload or args.clean:
            cmd.append('--reload')
        
        # Add indicators if specified
        if args.indicators:
            cmd.append('--indicators')
            cmd.extend(indicators)
        # else: script will load from config
        
        if args.top_up:
            cmd.append('--top-up')
        
        if not run_command(cmd, "Synthetic Indicators Load"):
            if args.ticker:
                clean_ticker_data(args.ticker, args.frequency, SYNTHETIC_TABLE, client)
            sys.exit(1)
        
        # Validate synthetic data (only if single ticker)
        if args.ticker:
            valid, message = validate_synthetic_data(
                args.ticker, args.frequency, start_date, end_date, indicators, client
            )
            if not valid:
                clean_ticker_data(args.ticker, args.frequency, SYNTHETIC_TABLE, client)
                sys.exit(1)
    else:
    
    # ========================================
    # STEP 3: Verify and Export
    # ========================================
    if not args.skip_verify:
        # Only verify if single ticker (verification script requires single ticker)
        if args.ticker:
            print_header("Step 3: Verification and Export")
            
            # For top-up mode, only verify recent data (last 30 days)
            # to avoid failing on ancient warmup period issues
            if args.top_up:
                from datetime import timedelta
                verify_start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                verify_end = end_date
            else:
                verify_start = start_date
                verify_end = end_date
            
            cmd = [
                'python', 'scripts/01_extract/tickers_verify_synthetic.py',
                '--ticker', args.ticker,
                '--frequency', args.frequency,
                '--exclude-weekends'
            ]
            
            if not run_command(cmd, "Data Verification"):
                sys.exit(1)
        else:
    else:
    
    # ========================================
    # SUCCESS
    # ========================================
    print_header("Pipeline Completed Successfully")
    if args.ticker:
    else:

if __name__ == '__main__':
    main()
