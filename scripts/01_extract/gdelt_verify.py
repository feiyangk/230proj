"""Verify GDELT sentiment data quality and completeness.

This script:
1. Checks data completeness (missing intervals)
2. Validates sentiment values (tone, polarity ranges)
3. Checks for duplicate timestamps
4. Exports sample data for manual inspection to temp/ directory
5. Provides summary statistics
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import bigquery
from typing import Dict, List, Tuple
from pathlib import Path
import yaml

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
DATASET_ID = os.environ.get('BQ_DATASET', 'raw_dataset')
GDELT_TABLE = os.environ.get('BQ_GDELT_TABLE', 'gdelt_sentiment')


def load_config(config_path: str = 'configs/gdelt.yaml') -> dict:
    """Load GDELT configuration from YAML file."""
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
DEFAULT_FREQUENCY = os.environ.get('GDELT_FREQUENCY', '15m')  # GDELT native frequency


def print_header(text: str):
    """Print formatted header."""


def check_completeness(client: bigquery.Client, start_date: str, end_date: str, frequency: str = '15m', topic_group_id: str = None) -> Dict:
    """Check for missing intervals."""
    print_header("Data Completeness Check")
    
    # Map frequency to BigQuery interval
    freq_map = {
        '15m': 'INTERVAL 15 MINUTE',
        '1h': 'INTERVAL 1 HOUR',
        '4h': 'INTERVAL 4 HOUR',
        '1d': 'INTERVAL 1 DAY',
        '1w': 'INTERVAL 7 DAY'
    }
    
    interval_expr = freq_map.get(frequency, 'INTERVAL 15 MINUTE')
    
    # For daily frequency, generate date-based timestamps at midnight
    if frequency == '1d':
        timestamp_generation = f"""
        TIMESTAMP(date_val) AS expected_timestamp
      FROM
        UNNEST(GENERATE_DATE_ARRAY(
          DATE('{start_date}'),
          DATE('{end_date}')
        )) AS date_val
        """
    else:
        timestamp_generation = f"""
        ts AS expected_timestamp
      FROM
        UNNEST(GENERATE_TIMESTAMP_ARRAY(
          TIMESTAMP('{start_date}'),
          TIMESTAMP('{end_date} 23:59:59'),
          {interval_expr}
        )) AS ts
        """
    
    query = f"""
    WITH expected_intervals AS (
      SELECT
        {timestamp_generation}
    ),
    actual_data AS (
      SELECT timestamp, num_articles, num_sources
      FROM `{PROJECT_ID}.{DATASET_ID}.{GDELT_TABLE}`
      WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
        AND frequency = '{frequency}'
        {f"AND topic_group_id = '{topic_group_id}'" if topic_group_id else ""}
    )
    SELECT
      COUNT(DISTINCT e.expected_timestamp) as expected_intervals,
      COUNT(DISTINCT a.timestamp) as actual_intervals,
      COUNT(DISTINCT e.expected_timestamp) - COUNT(DISTINCT a.timestamp) as missing_intervals
    FROM expected_intervals e
    LEFT JOIN actual_data a ON e.expected_timestamp = a.timestamp
    """
    
    result = client.query(query).to_dataframe().iloc[0]
    
    
    completeness_pct = (result['actual_intervals'] / result['expected_intervals']) * 100 if result['expected_intervals'] > 0 else 0
    
    if result['missing_intervals'] == 0:
        return {'complete': True, 'stats': result}
    elif completeness_pct >= 95:
        return {'complete': True, 'stats': result}
    else:
        return {'complete': False, 'stats': result}


def show_missing_intervals(client: bigquery.Client, start_date: str, end_date: str, frequency: str = '15m', topic_group_id: str = None, limit: int = 50):
    """Show which specific intervals are missing."""
    print_header("Missing Intervals Details")
    
    query = f"""
    WITH expected_intervals AS (
      SELECT
        TIMESTAMP_TRUNC(ts, HOUR) + 
        INTERVAL DIV(EXTRACT(MINUTE FROM ts), 15) * 15 MINUTE AS expected_timestamp
      FROM
        UNNEST(GENERATE_TIMESTAMP_ARRAY(
          TIMESTAMP('{start_date}'),
          TIMESTAMP('{end_date} 23:59:59'),
          INTERVAL 15 MINUTE
        )) AS ts
    ),
    actual_data AS (
      SELECT timestamp
      FROM `{PROJECT_ID}.{DATASET_ID}.{GDELT_TABLE}`
      WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
        AND frequency = '{frequency}'
        {f"AND topic_group_id = '{topic_group_id}'" if topic_group_id else ""}
    )
    SELECT
      e.expected_timestamp as missing_timestamp,
      FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', e.expected_timestamp) as formatted_time,
      EXTRACT(DAYOFWEEK FROM e.expected_timestamp) as day_of_week,
      EXTRACT(HOUR FROM e.expected_timestamp) as hour
    FROM expected_intervals e
    LEFT JOIN actual_data a ON e.expected_timestamp = a.timestamp
    WHERE a.timestamp IS NULL
    ORDER BY e.expected_timestamp
    LIMIT {limit}
    """
    
    missing = client.query(query).to_dataframe()
    
    if missing.empty:
        return
    
    
    # Show by day
    day_names = ['', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    
    for _, row in missing.iterrows():
        day_name = day_names[int(row['day_of_week'])]
    
    # Show summary by hour
    if len(missing) > 0:
        hour_counts = missing['hour'].value_counts().sort_index()
        for hour, count in hour_counts.items():
            pass

def check_duplicates(client: bigquery.Client, start_date: str, end_date: str, frequency: str = '15m', topic_group_id: str = None) -> Dict:
    """Check for duplicate timestamps."""
    print_header("Duplicate Timestamp Check")
    
    query = f"""
    SELECT
      timestamp,
      frequency,
      topic_group_id,
      COUNT(*) as count
    FROM `{PROJECT_ID}.{DATASET_ID}.{GDELT_TABLE}`
    WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
      AND frequency = '{frequency}'
      {f"AND topic_group_id = '{topic_group_id}'" if topic_group_id else ""}
    GROUP BY timestamp, frequency, topic_group_id
    HAVING COUNT(*) > 1
    ORDER BY count DESC, timestamp DESC
    LIMIT 10
    """
    
    duplicates = client.query(query).to_dataframe()
    
    if duplicates.empty:
        return {'has_duplicates': False, 'count': 0}
    else:
        return {'has_duplicates': True, 'count': len(duplicates), 'samples': duplicates}


def validate_sentiment_ranges(client: bigquery.Client, start_date: str, end_date: str, frequency: str = '15m', topic_group_id: str = None) -> Dict:
    """Validate sentiment values are within expected ranges."""
    print_header("Sentiment Value Validation")
    
    query = f"""
    SELECT
      COUNT(*) as total_intervals,
      
      -- Tone should be between -10 and +10
      COUNTIF(weighted_avg_tone < -10 OR weighted_avg_tone > 10) as invalid_tone,
      MIN(weighted_avg_tone) as min_tone,
      MAX(weighted_avg_tone) as max_tone,
      AVG(weighted_avg_tone) as avg_tone,
      
      -- Polarity should be non-negative
      COUNTIF(weighted_avg_polarity < 0) as invalid_polarity,
      MIN(weighted_avg_polarity) as min_polarity,
      MAX(weighted_avg_polarity) as max_polarity,
      AVG(weighted_avg_polarity) as avg_polarity,
      
      -- Counts should be positive
      COUNTIF(num_articles <= 0) as zero_articles,
      COUNTIF(num_sources <= 0) as zero_sources,
      MIN(num_articles) as min_articles,
      MAX(num_articles) as max_articles,
      AVG(num_articles) as avg_articles,
      AVG(num_sources) as avg_sources
      
    FROM `{PROJECT_ID}.{DATASET_ID}.{GDELT_TABLE}`
    WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
      AND frequency = '{frequency}'
      {f"AND topic_group_id = '{topic_group_id}'" if topic_group_id else ""}
    """
    
    result = client.query(query).to_dataframe().iloc[0]
    
    
    
    
    
    issues = result['invalid_tone'] + result['invalid_polarity'] + result['zero_articles']
    
    if issues == 0:
        return {'valid': True, 'stats': result}
    else:
        return {'valid': False, 'stats': result}


def show_extreme_dates(client: bigquery.Client, start_date: str, end_date: str, frequency: str = '15m', topic_group_id: str = None):
    """Show 10 most positive and 10 most negative dates."""
    print_header("Extreme Sentiment Dates")
    
    # Most negative dates
    negative_query = f"""
    SELECT
      DATE(timestamp) as date,
      AVG(weighted_avg_tone) as avg_tone,
      SUM(num_articles) as total_articles,
      COUNT(*) as intervals
    FROM `{PROJECT_ID}.{DATASET_ID}.{GDELT_TABLE}`
    WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
      AND frequency = '{frequency}'
      {f"AND topic_group_id = '{topic_group_id}'" if topic_group_id else ""}
    GROUP BY date
    ORDER BY avg_tone ASC
    LIMIT 10
    """
    
    # Most positive dates
    positive_query = f"""
    SELECT
      DATE(timestamp) as date,
      AVG(weighted_avg_tone) as avg_tone,
      SUM(num_articles) as total_articles,
      COUNT(*) as intervals
    FROM `{PROJECT_ID}.{DATASET_ID}.{GDELT_TABLE}`
    WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
      AND frequency = '{frequency}'
      {f"AND topic_group_id = '{topic_group_id}'" if topic_group_id else ""}
    GROUP BY date
    ORDER BY avg_tone DESC
    LIMIT 10
    """
    
    negative_df = client.query(negative_query).to_dataframe()
    positive_df = client.query(positive_query).to_dataframe()
    
    if not negative_df.empty:
    
    if not positive_df.empty:


def show_random_sample(client: bigquery.Client, start_date: str, end_date: str, frequency: str = '15m', topic_group_id: str = None, num_rows: int = 10):
    """Show random sample rows from the data."""
    print_header("Random Sample Data")
    
    query = f"""
    SELECT
      timestamp,
      frequency,
      topic_group_id,
      weighted_avg_tone,
      weighted_avg_positive,
      weighted_avg_negative,
      weighted_avg_polarity,
      num_articles,
      num_sources,
      min_tone,
      max_tone
    FROM `{PROJECT_ID}.{DATASET_ID}.{GDELT_TABLE}`
    WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
      AND frequency = '{frequency}'
      {f"AND topic_group_id = '{topic_group_id}'" if topic_group_id else ""}
    ORDER BY RAND()
    LIMIT {num_rows}
    """
    
    sample = client.query(query).to_dataframe()
    
    if sample.empty:
        return
    
    # Format for better display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


def show_invalid_records(client: bigquery.Client, start_date: str, end_date: str, frequency: str = '15m', topic_group_id: str = None):
    """Show records with invalid sentiment values."""
    print_header("Invalid Sentiment Records")
    
    query = f"""
    SELECT
      timestamp,
      topic_group_id,
      weighted_avg_tone,
      weighted_avg_polarity,
      num_articles,
      num_sources
    FROM `{PROJECT_ID}.{DATASET_ID}.{GDELT_TABLE}`
    WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
      AND frequency = '{frequency}'
      {f"AND topic_group_id = '{topic_group_id}'" if topic_group_id else ""}
      AND (
        weighted_avg_tone < -10 OR weighted_avg_tone > 10
        OR weighted_avg_polarity < 0
      )
    ORDER BY ABS(weighted_avg_tone) DESC
    LIMIT 20
    """
    
    invalid = client.query(query).to_dataframe()
    
    if invalid.empty:
        return
    


def fetch_sample_data(client: bigquery.Client, start_date: str, end_date: str, frequency: str = '15m', topic_group_id: str = None, limit: int = 1000) -> pd.DataFrame:
    """Fetch sample data for inspection."""
    query = f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET_ID}.{GDELT_TABLE}`
    WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
      AND frequency = '{frequency}'
      {f"AND topic_group_id = '{topic_group_id}'" if topic_group_id else ""}
    ORDER BY timestamp DESC
    LIMIT {limit}
    """
    
    return client.query(query).to_dataframe()


def get_summary_stats(client: bigquery.Client, start_date: str, end_date: str, frequency: str = '15m', topic_group_id: str = None) -> pd.DataFrame:
    """Get daily summary statistics."""
    query = f"""
    SELECT
      DATE(timestamp) as date,
      COUNT(*) as intervals,
      SUM(num_articles) as total_articles,
      SUM(num_sources) as total_sources,
      AVG(weighted_avg_tone) as avg_tone,
      MIN(weighted_avg_tone) as min_tone,
      MAX(weighted_avg_tone) as max_tone,
      AVG(num_articles) as avg_articles_per_interval
    FROM `{PROJECT_ID}.{DATASET_ID}.{GDELT_TABLE}`
    WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
      AND frequency = '{frequency}'
      {f"AND topic_group_id = '{topic_group_id}'" if topic_group_id else ""}
    GROUP BY date
    ORDER BY date
    """
    
    return client.query(query).to_dataframe()


def export_sample_data(df: pd.DataFrame, output_file: str, start_date: str, end_date: str):
    """Export sample data to parquet and CSV files."""
    print_header("Exporting Sample Data")
    
    # Create temp directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
    parquet_size = Path(output_file).stat().st_size / 1024 / 1024
    
    # Also save as CSV
    csv_file = output_file.replace('.parquet', '.csv')
    df.to_csv(csv_file, index=False)
    csv_size = Path(csv_file).stat().st_size / 1024 / 1024
    


def main():
    parser = argparse.ArgumentParser(
        description='Verify GDELT sentiment data quality and completeness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Use dates from config (configs/gdelt.yaml)
  python gdelt_verify.py --frequency 1d
  
  # Full verification for specific date range
  python gdelt_verify.py --start 2025-11-11 --end 2025-11-18 --frequency 1d
  
  # Export sample data to temp/
  python gdelt_verify.py --start 2025-11-11 --end 2025-11-18 --export
  
  # Quick completeness check only
  python gdelt_verify.py --frequency 1d --completeness-only
  
  # Show daily statistics
  python gdelt_verify.py --frequency 1d --show-daily-stats
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/gdelt.yaml',
        help='Path to GDELT config file (default: configs/gdelt.yaml)'
    )
    parser.add_argument(
        '--topic-group',
        type=str,
        required=True,
        help='Topic group to verify (e.g., inflation_prices, fed_policy). Required.'
    )
    parser.add_argument(
        '--start',
        '--start-date',
        dest='start_date',
        type=str,
        required=False,
        help='Start date (YYYY-MM-DD), defaults to config'
    )
    parser.add_argument(
        '--end',
        '--end-date',
        dest='end_date',
        type=str,
        required=False,
        help='End date (YYYY-MM-DD), defaults to config'
    )
    parser.add_argument(
        '--frequency',
        type=str,
        default='15m',
        help='Data frequency (default: 15m - GDELT native interval)'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export sample data to parquet file in temp/ directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: temp/gdelt_sentiment_{start}_{end}.parquet)'
    )
    parser.add_argument(
        '--completeness-only',
        action='store_true',
        help='Only check data completeness'
    )
    parser.add_argument(
        '--show-missing',
        action='store_true',
        help='Show which specific timestamps are missing'
    )
    parser.add_argument(
        '--show-daily-stats',
        action='store_true',
        help='Show daily summary statistics'
    )
    
    args = parser.parse_args()
    
    # Validate PROJECT_ID
    if not PROJECT_ID:
        sys.exit(1)
    
    # Load config for default dates and topic group validation
    config = load_config(args.config)
    
    # Validate topic group
    topic_group_id = args.topic_group
    if config and 'topic_groups' in config:
        if topic_group_id not in config['topic_groups']:
            available = ', '.join(config['topic_groups'].keys())
            sys.exit(1)
        else:
            group_config = config['topic_groups'][topic_group_id]
            description = group_config.get('description', '')
    
    # Determine dates (CLI args override config)
    start_date = args.start_date
    end_date = args.end_date
    
    if not start_date or not end_date:
        if 'date_range' in config:
            start_date = start_date or config['date_range'].get('start_date')
            end_date = end_date or config['date_range'].get('end_date')
        else:
            sys.exit(1)
    
    # Validate dates
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if end < start:
            sys.exit(1)
    except ValueError:
        sys.exit(1)
    
    # Update args with resolved dates
    args.start_date = start_date
    args.end_date = end_date
    
    # Initialize BigQuery client
    client = bigquery.Client(project=PROJECT_ID)
    
    # Quick data check - show what's in the table
    check_query = f"""
    SELECT 
        COUNT(*) as row_count,
        MIN(timestamp) as min_ts,
        MAX(timestamp) as max_ts,
        STRING_AGG(DISTINCT frequency ORDER BY frequency) as frequencies
    FROM `{PROJECT_ID}.{DATASET_ID}.{GDELT_TABLE}`
    WHERE frequency = '{args.frequency}'
      AND topic_group_id = '{topic_group_id}'
      AND DATE(timestamp) BETWEEN '{args.start_date}' AND '{args.end_date}'
    """
    try:
        result = client.query(check_query).to_dataframe().iloc[0]
        if result['row_count'] > 0:
        else:
            all_freq_query = f"""
            SELECT frequency, COUNT(*) as count
            FROM `{PROJECT_ID}.{DATASET_ID}.{GDELT_TABLE}`
            GROUP BY frequency
            ORDER BY frequency
            """
            all_freqs = client.query(all_freq_query).to_dataframe()
            if not all_freqs.empty:
                for _, row in all_freqs.iterrows():
            else:
    except Exception as e:
    
    print_header("GDELT Sentiment Data Verification")
    
    # Run checks
    completeness_result = check_completeness(client, args.start_date, args.end_date, args.frequency, topic_group_id)
    
    # Show missing intervals if requested or if there are many missing
    if args.show_missing or (not completeness_result['complete'] and completeness_result['stats']['missing_intervals'] > 0):
        show_missing_intervals(client, args.start_date, args.end_date, args.frequency, topic_group_id)
    
    if args.completeness_only:
        sys.exit(0 if completeness_result['complete'] else 1)
    
    duplicates_result = check_duplicates(client, args.start_date, args.end_date, args.frequency, topic_group_id)
    validation_result = validate_sentiment_ranges(client, args.start_date, args.end_date, args.frequency, topic_group_id)
    
    # Show invalid records if detected
    if not validation_result['valid']:
        show_invalid_records(client, args.start_date, args.end_date, args.frequency, topic_group_id)
    
    # Show extreme dates
    show_extreme_dates(client, args.start_date, args.end_date, args.frequency, topic_group_id)
    
    # Show random sample of data
    show_random_sample(client, args.start_date, args.end_date, args.frequency, topic_group_id, num_rows=10)
    
    # Show daily stats if requested
    if args.show_daily_stats:
        print_header("Daily Summary Statistics")
        stats = get_summary_stats(client, args.start_date, args.end_date, args.frequency, topic_group_id)
    
    # Export if requested
    if args.export:
        if not args.output:
            os.makedirs('temp', exist_ok=True)
            args.output = f"temp/gdelt_sentiment_{topic_group_id}_{args.frequency}_{args.start_date}_{args.end_date}.parquet"
        
        sample_df = fetch_sample_data(client, args.start_date, args.end_date, args.frequency, topic_group_id)
        export_sample_data(sample_df, args.output, args.start_date, args.end_date)
    
    # Final summary
    print_header("Verification Summary")
    
    all_passed = (
        completeness_result['complete'] and
        not duplicates_result['has_duplicates'] and
        validation_result['valid']
    )
    
    if all_passed:
        sys.exit(0)
    else:
        if not completeness_result['complete']:
        if duplicates_result['has_duplicates']:
        if not validation_result['valid']:
        sys.exit(1)


if __name__ == '__main__':
    main()
