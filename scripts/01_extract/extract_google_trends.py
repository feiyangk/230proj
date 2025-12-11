#!/usr/bin/env python
"""Extract Google Trends data."""

import argparse
import random
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from pytrends.request import TrendReq


def extract_trends(
    keywords: list,
    start_date: str,
    end_date: str,
    geo: str = 'US',
    output_path: str = 'data/raw/google_trends.parquet',
    throttle_seconds: float = 5.0,
    max_retries: int = 5,
    force: bool = False
) -> pd.DataFrame:
    """
    Extract Google Trends data for specified keywords.
    
    Args:
        keywords: List of search terms (e.g., ['spy', 'spy price'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        geo: Geographic region (default: 'US')
        output_path: Output parquet file path
        throttle_seconds: Sleep time between API requests (default: 5.0)
        max_retries: Maximum number of retries for failed requests (default: 5)
        force: Force re-download even if file exists
        
    Returns:
        DataFrame with trends data
    """
    # Check if file exists
    output_file = Path(output_path)
    if output_file.exists() and not force:
        return pd.read_parquet(output_path)
    
    if force:
    
    # Initialize pytrends
    pytrends = TrendReq(hl='en-US', tz=360)
    
    # Parse dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Google Trends API limits:
    # - Daily data: max 270 days per request
    # - Need to split into chunks if range > 270 days
    max_days = 269  # Use 269 to be safe
    
    # Calculate total number of chunks for progress tracking
    total_days = (end - start).days
    total_chunks = (total_days // max_days) + (1 if total_days % max_days > 0 else 1)
    
    
    all_data = []
    current_start = start
    chunk_num = 0
    start_time = time.time()
    
    while current_start < end:
        chunk_num += 1
        current_end = min(current_start + timedelta(days=max_days), end)
        
        timeframe = f"{current_start.strftime('%Y-%m-%d')} {current_end.strftime('%Y-%m-%d')}"
        elapsed = time.time() - start_time
        
        
        # Retry logic with exponential backoff
        retry_count = 0
        success = False
        
        while retry_count <= max_retries and not success:
            try:
                # Build payload
                pytrends.build_payload(
                    kw_list=keywords,
                    cat=0,
                    timeframe=timeframe,
                    geo=geo,
                    gprop=''
                )
                
                # Get interest over time
                df = pytrends.interest_over_time()
                
                if df is not None and not df.empty:
                    # Remove 'isPartial' column if present
                    if 'isPartial' in df.columns:
                        df = df.drop('isPartial', axis=1)
                    
                    all_data.append(df)
                else:
                
                success = True
                
                # Throttle to avoid rate limiting (with jitter)
                jitter = random.uniform(0, throttle_seconds * 0.3)
                wait = throttle_seconds + jitter
                time.sleep(wait)
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error (429)
                if '429' in error_msg:
                    retry_count += 1
                    
                    if retry_count <= max_retries:
                        # Exponential backoff with jitter
                        wait_time = min(throttle_seconds * (2 ** retry_count), 120)  # Cap at 120s
                        jitter = random.uniform(0, wait_time * 0.3)
                        total_wait = wait_time + jitter
                        
                        
                        # Show countdown for long waits
                        if total_wait > 10:
                            # Sleep in 4 equal chunks, showing progress 3 times
                            chunk_size = total_wait / 4
                            for i in range(3):
                                time.sleep(chunk_size)
                                elapsed = chunk_size * (i + 1)
                                remaining = total_wait - elapsed
                            # Sleep the final chunk
                            time.sleep(chunk_size)
                        else:
                            time.sleep(total_wait)
                    else:
                else:
                    # Non-rate-limit error - log and move on
                    break
        
        # Move to next chunk
        current_start = current_end + timedelta(days=1)
    
    if not all_data:
        return None
    
    # Combine all chunks
    total_elapsed = time.time() - start_time
    combined = pd.concat(all_data, axis=0)
    
    # Remove duplicates (overlap between chunks)
    combined = combined[~combined.index.duplicated(keep='first')]
    
    # Sort by date
    combined = combined.sort_index()
    
    # Reset index to make date a column
    combined = combined.reset_index()
    combined.rename(columns={'date': 'date'}, inplace=True)
    
    
    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    combined.to_parquet(output_path, index=False)
    
    return combined


def load_config(config_path: str = 'configs/google_trends.yaml') -> dict:
    """Load Google Trends configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract Google Trends data for SPY indicators',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract using config file (configs/google_trends.yaml)
  python scripts/01_extract/extract_google_trends.py
  
  # Use custom config file
  python scripts/01_extract/extract_google_trends.py --config configs/my_trends.yaml
  
  # Override keywords from command line
  python scripts/01_extract/extract_google_trends.py --keywords "spy" "spy price" "inflation"
  
  # Custom date range
  python scripts/01_extract/extract_google_trends.py --start-date 2015-10-28 --end-date 2025-10-24
  
  # Different region
  python scripts/01_extract/extract_google_trends.py --geo UK
        """
    )
    
    parser.add_argument(
        '--config',
        default='configs/google_trends.yaml',
        help='Path to config file (default: configs/google_trends.yaml)'
    )
    parser.add_argument(
        '--keywords',
        nargs='+',
        default=None,
        help='Search keywords (overrides config file)'
    )
    parser.add_argument(
        '--start-date',
        default=None,
        help='Start date YYYY-MM-DD (overrides config file)'
    )
    parser.add_argument(
        '--end-date',
        default=None,
        help='End date YYYY-MM-DD (overrides config file)'
    )
    parser.add_argument(
        '--geo',
        default=None,
        help='Geographic region (overrides config file)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output file path (overrides config file)'
    )
    parser.add_argument(
        '--throttle',
        type=float,
        default=None,
        help='Seconds to wait between API requests (overrides config file)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=None,
        help='Maximum number of retries for failed requests (overrides config file)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if file exists'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: extract only last 30 days of data'
    )
    
    args = parser.parse_args()
    
    # Load config file
    config = load_config(args.config)
    
    # Use config values as defaults, command-line args override
    if config:
        keywords = args.keywords if args.keywords else config.get('keywords', ['spy', 'spy price'])
        start_date = args.start_date if args.start_date else config.get('date_range', {}).get('start_date', '2015-10-28')
        end_date = args.end_date if args.end_date else config.get('date_range', {}).get('end_date')
        geo = args.geo if args.geo else config.get('region', {}).get('geo', 'US')
        output_path = args.output if args.output else config.get('output', {}).get('path', 'data/raw/google_trends.parquet')
        throttle = args.throttle if args.throttle else config.get('api_settings', {}).get('throttle_seconds', 5.0)
        max_retries = args.max_retries if args.max_retries else config.get('api_settings', {}).get('max_retries', 5)
    else:
        # Fallback to hardcoded defaults if no config
        keywords = args.keywords if args.keywords else ['spy', 'spy price']
        start_date = args.start_date if args.start_date else '2015-10-28'
        end_date = args.end_date
        geo = args.geo if args.geo else 'US'
        output_path = args.output if args.output else 'data/raw/google_trends.parquet'
        throttle = args.throttle if args.throttle else 5.0
        max_retries = args.max_retries if args.max_retries else 5
    
    # Set end_date to today if None
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Override dates if test mode
    if args.test:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    try:
        extract_trends(
            keywords=keywords,
            start_date=start_date,
            end_date=end_date,
            geo=geo,
            output_path=output_path,
            throttle_seconds=throttle,
            max_retries=max_retries,
            force=args.force
        )
        
    except Exception as e:
        raise


if __name__ == '__main__':
    main()
