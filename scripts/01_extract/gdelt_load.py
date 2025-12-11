"""Load GDELT GKG sentiment data from BigQuery and aggregate to 15-minute intervals.

Queries Google's public GDELT BigQuery dataset, filters for relevant topics/themes,
aggregates sentiment to 15-minute intervals, and saves to project BigQuery table.

GDELT GKG (Global Knowledge Graph) contains:
- Tone: Overall sentiment (-10 to +10, negative = bad news)
- Polarity: Strength of sentiment  
- Themes: Topics mentioned in articles
- Updated every 15 minutes

Public Dataset: `gdelt-bq.gdeltv2.gkg`
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
from google.cloud import bigquery
import pandas as pd
from pathlib import Path
import yaml
from typing import List, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
DATASET_ID = os.environ.get('BQ_DATASET', 'raw_dataset')
TABLE_NAME = os.environ.get('BQ_GDELT_TABLE', 'gdelt_sentiment')
STAGING_TABLE_NAME = os.environ.get('BQ_GDELT_STAGING_TABLE', 'gdelt_sentiment_staging')

# GDELT public BigQuery dataset
GDELT_PROJECT = 'gdelt-bq'
GDELT_DATASET = 'gdeltv2'
GDELT_TABLE = 'gkg_partitioned'  # Use partitioned table for 28x cost reduction

# Default topics to track
DEFAULT_TOPICS = [
    'FED',  # Federal Reserve
    'FEDERAL_RESERVE',
    'INTEREST_RATE',
    'INFLATION',
    'RECESSION',
]


def build_gdelt_query(
    start_date: str,
    end_date: str,
    topics: List[str]
) -> str:
    """
    Build BigQuery SQL to fetch and aggregate GDELT GKG sentiment data.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)  
        topics: List of topics/themes to filter for
    
    Returns:
        SQL query string
    """
    # Convert topics to UPPER and underscore format (GDELT format)
    topic_patterns = [f"'%{topic.upper().replace(' ', '_')}%'" for topic in topics]
    topic_conditions = ' OR '.join([f"V2Themes LIKE {pattern}" for pattern in topic_patterns])
    
    query = f"""
    WITH filtered_articles AS (
        SELECT
            -- Parse timestamp and round to 15-minute intervals
            TIMESTAMP(PARSE_DATETIME('%Y%m%d%H%M%S', CAST(DATE AS STRING))) AS article_timestamp,
            TIMESTAMP_TRUNC(
                TIMESTAMP(PARSE_DATETIME('%Y%m%d%H%M%S', CAST(DATE AS STRING))),
                HOUR
            ) + INTERVAL DIV(EXTRACT(MINUTE FROM TIMESTAMP(PARSE_DATETIME('%Y%m%d%H%M%S', CAST(DATE AS STRING)))), 15) * 15 MINUTE AS interval_timestamp,
            
            -- Parse V2Tone: tone,positive,negative,polarity,activity,self_ref,word_count
            CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS tone,
            CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(1)] AS FLOAT64) AS positive_score,
            CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(2)] AS FLOAT64) AS negative_score,
            CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(3)] AS FLOAT64) AS polarity,
            CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(6)] AS INT64) AS word_count,
            
            V2Themes,
            SourceCommonName
            
        FROM `{GDELT_PROJECT}.{GDELT_DATASET}.{GDELT_TABLE}`
        
        WHERE
            -- Partition filter (reduces cost by 28x - only scans relevant days)
            _PARTITIONTIME >= TIMESTAMP("{start_date}")
            AND _PARTITIONTIME <= TIMESTAMP("{end_date}")
            
            -- Date range filter (DATE is INT64 in format YYYYMMDDHHMMSS)
            AND DATE >= {start_date.replace("-", "")}000000
            AND DATE <= {end_date.replace("-", "")}235959
            
            -- Topic filter (must mention at least one topic)
            AND ({topic_conditions})
            
            -- US-related articles only (mentions US locations)
            AND V2Locations LIKE '%#US#%'
            
            -- Valid tone data
            AND V2Tone IS NOT NULL
            AND LENGTH(V2Tone) > 0
    ),
    
    aggregated AS (
        SELECT
            interval_timestamp AS timestamp,
            
            -- Weighted averages (weight by word count)
            SUM(tone * COALESCE(word_count, 1)) / SUM(COALESCE(word_count, 1)) AS weighted_avg_tone,
            SUM(positive_score * COALESCE(word_count, 1)) / SUM(COALESCE(word_count, 1)) AS weighted_avg_positive,
            SUM(negative_score * COALESCE(word_count, 1)) / SUM(COALESCE(word_count, 1)) AS weighted_avg_negative,
            SUM(polarity * COALESCE(word_count, 1)) / SUM(COALESCE(word_count, 1)) AS weighted_avg_polarity,
            
            -- Counts
            COUNT(*) AS num_articles,
            COUNT(DISTINCT SourceCommonName) AS num_sources,
            SUM(COALESCE(word_count, 0)) AS total_word_count,
            
            -- Min/max for quality checks
            MIN(tone) AS min_tone,
            MAX(tone) AS max_tone
            
        FROM filtered_articles
        
        GROUP BY interval_timestamp
        ORDER BY interval_timestamp
    )
    
    SELECT * FROM aggregated
    """
    
    return query


def resample_to_frequency(
    df: pd.DataFrame,
    target_frequency: str,
    method: str = 'mean'
) -> pd.DataFrame:
    """Resample 15-minute GDELT data to target frequency.
    
    Args:
        df: DataFrame with 15-minute intervals
        target_frequency: Target frequency ('15m', '1h', '4h', '1d', '1w')
        method: Aggregation method ('mean', 'median', 'first', 'last')
    
    Returns:
        Resampled DataFrame
    """
    if target_frequency == '15m':
        return df  # Already at native frequency
    
    
    # Set timestamp as index for resampling
    df_resampled = df.set_index('timestamp')
    
    # Define aggregation rules
    agg_rules = {
        'weighted_avg_tone': method,
        'weighted_avg_positive': method,
        'weighted_avg_negative': method,
        'weighted_avg_polarity': method,
        'num_articles': 'sum',
        'num_sources': 'sum',
        'total_word_count': 'sum',
        'min_tone': 'min',
        'max_tone': 'max'
    }
    
    # Resample
    df_resampled = df_resampled.resample(target_frequency).agg(agg_rules)
    
    # Reset index
    df_resampled = df_resampled.reset_index()
    
    # Remove rows with no data
    df_resampled = df_resampled[df_resampled['num_articles'] > 0]
    
    
    return df_resampled


def save_query_metadata(
    client: bigquery.Client,
    start_date: str,
    end_date: str,
    topics: List[str],
    num_records: int,
    num_articles: int,
    topic_group_id: str,
    config_file: Optional[str] = None,
    frequency: str = "15m",
    notes: Optional[str] = None
) -> str:
    """Save metadata about this GDELT query for data lineage.
    
    Args:
        client: BigQuery client
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        topics: List of topics queried
        num_records: Number of records loaded
        num_articles: Total articles processed
        topic_group_id: Human-readable ID for topic group (e.g., 'inflation_prices', 'fed_policy')
        config_file: Path to config file used
        frequency: Data frequency
        notes: Optional notes
    
    Returns:
        query_id: Unique ID for this query
    """
    from datetime import datetime
    
    # Generate query ID
    query_id = f"gdelt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Metadata table
    metadata_table = f"{PROJECT_ID}.{DATASET_ID}.gdelt_query_metadata"
    
    # Prepare row
    row = {
        "query_id": query_id,
        "topic_group_id": topic_group_id,
        "load_timestamp": datetime.now().isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "topics": topics,
        "config_file": config_file,
        "num_records": num_records,
        "num_articles": num_articles,
        "frequency": frequency,
        "notes": notes
    }
    
    # Insert
    try:
        errors = client.insert_rows_json(metadata_table, [row])
        
        if errors:
            raise RuntimeError(f"Failed to insert metadata rows: {errors}")
        else:
            return query_id
    except Exception as e:
        raise


def clear_topic_group_frequency_data(client: bigquery.Client, table_id: str, topic_group_id: str, frequency: str, project_id: str, dataset_id: str) -> None:
    """Delete all data for a specific topic group and frequency."""
    
    # Delete from main sentiment table
    delete_query = f"""
    DELETE FROM `{table_id}`
    WHERE topic_group_id = '{topic_group_id}'
      AND frequency = '{frequency}'
    """
    
    try:
        job = client.query(delete_query)
        job.result()
    except Exception as e:
    
    # Also delete metadata entries for this topic group and frequency
    metadata_table_id = f"{project_id}.{dataset_id}.gdelt_query_metadata"
    delete_metadata_query = f"""
    DELETE FROM `{metadata_table_id}`
    WHERE topic_group_id = '{topic_group_id}'
      AND frequency = '{frequency}'
    """
    
    try:
        job = client.query(delete_metadata_query)
        job.result()
    except Exception as e:


def fetch_gdelt_sentiment(
    start_date: str,
    end_date: str,
    topics: List[str] = None,
    project_id: str = None
) -> pd.DataFrame:
    """
    Fetch and aggregate GDELT sentiment data from BigQuery.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        topics: List of topics to filter for (default: DEFAULT_TOPICS)
        project_id: GCP project ID (for billing)
    
    Returns:
        DataFrame with aggregated sentiment by 15-minute intervals
    """
    if topics is None:
        topics = DEFAULT_TOPICS
    
    if project_id is None:
        project_id = PROJECT_ID
    
    
    # Build query
    query = build_gdelt_query(start_date, end_date, topics)
    
    # Execute query
    client = bigquery.Client(project=project_id)
    
    try:
        df = client.query(query).to_dataframe()
        
        if df.empty:
            return df
        
        
        return df
        
    except Exception as e:
        raise


def check_existing_data(
    client: bigquery.Client,
    table_id: str,
    topic_group_id: str,
    frequency: str,
    start_date: str,
    end_date: str
) -> bool:
    """Check if data already exists for the given parameters.
    
    Args:
        client: BigQuery client
        table_id: Full table ID
        topic_group_id: Topic group ID
        frequency: Data frequency
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        True if data exists, False otherwise
    """
    query = f"""
    SELECT COUNT(*) as count
    FROM `{table_id}`
    WHERE topic_group_id = '{topic_group_id}'
      AND frequency = '{frequency}'
      AND DATE(timestamp) >= '{start_date}'
      AND DATE(timestamp) <= '{end_date}'
    """
    
    try:
        result = client.query(query).to_dataframe()
        count = result['count'].iloc[0] if not result.empty else 0
        return count > 0
    except Exception as e:
        # Table might not exist yet
        return False


def save_to_bigquery(
    df: pd.DataFrame,
    project_id: str,
    dataset_id: str,
    topic_group_id: str,
    dry_run: bool = False,
    auto_merge: bool = True
) -> None:
    """
    Save sentiment data to BigQuery using staging + MERGE pattern.
    
    Args:
        df: Sentiment DataFrame
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        topic_group_id: Topic group ID to tag records with
        dry_run: If True, skip actual save
        auto_merge: If True, automatically merge staging to main
    """
    if df.empty:
        return
    
    if dry_run:
        return
    
    client = bigquery.Client(project=project_id)
    
    # Table references
    staging_table_id = f"{project_id}.{dataset_id}.{STAGING_TABLE_NAME}"
    main_table_id = f"{project_id}.{dataset_id}.{TABLE_NAME}"
    
    # Schema
    schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("frequency", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("topic_group_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("weighted_avg_tone", "FLOAT"),
        bigquery.SchemaField("weighted_avg_positive", "FLOAT"),
        bigquery.SchemaField("weighted_avg_negative", "FLOAT"),
        bigquery.SchemaField("weighted_avg_polarity", "FLOAT"),
        bigquery.SchemaField("num_articles", "INTEGER"),
        bigquery.SchemaField("num_sources", "INTEGER"),
        bigquery.SchemaField("total_word_count", "INTEGER"),
        bigquery.SchemaField("min_tone", "FLOAT"),
        bigquery.SchemaField("max_tone", "FLOAT"),
        bigquery.SchemaField("ingestion_timestamp", "TIMESTAMP"),
    ]
    
    # Add derived columns for consistency with tickers table
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    # Only set frequency if not already set (resampling sets it in main())
    if 'frequency' not in df.columns:
        df['frequency'] = '15m'  # GDELT native 15-minute interval
    df['topic_group_id'] = topic_group_id  # Tag with topic group
    df['ingestion_timestamp'] = pd.Timestamp.now(tz='UTC')
    
    
    # Load to staging table (overwrite)
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_TRUNCATE",
    )
    
    job = client.load_table_from_dataframe(df, staging_table_id, job_config=job_config)
    job.result()  # Wait for completion
    
    
    # Create main table if it doesn't exist
    try:
        client.get_table(main_table_id)
    except:
        table = bigquery.Table(main_table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="date"
        )
        table.clustering_fields = ["frequency", "topic_group_id"]
        client.create_table(table)
    
    # MERGE staging into main table
    merge_query = f"""
    MERGE `{main_table_id}` T
    USING `{staging_table_id}` S
    ON T.timestamp = S.timestamp AND T.frequency = S.frequency AND T.topic_group_id = S.topic_group_id
    WHEN MATCHED THEN
        UPDATE SET
            date = S.date,
            weighted_avg_tone = S.weighted_avg_tone,
            weighted_avg_positive = S.weighted_avg_positive,
            weighted_avg_negative = S.weighted_avg_negative,
            weighted_avg_polarity = S.weighted_avg_polarity,
            num_articles = S.num_articles,
            num_sources = S.num_sources,
            total_word_count = S.total_word_count,
            min_tone = S.min_tone,
            max_tone = S.max_tone,
            ingestion_timestamp = S.ingestion_timestamp
    WHEN NOT MATCHED THEN
        INSERT (
            timestamp,
            date,
            frequency,
            topic_group_id,
            weighted_avg_tone,
            weighted_avg_positive,
            weighted_avg_negative,
            weighted_avg_polarity,
            num_articles,
            num_sources,
            total_word_count,
            min_tone,
            max_tone,
            ingestion_timestamp
        )
        VALUES (
            timestamp,
            date,
            frequency,
            topic_group_id,
            weighted_avg_tone,
            weighted_avg_positive,
            weighted_avg_negative,
            weighted_avg_polarity,
            num_articles,
            num_sources,
            total_word_count,
            min_tone,
            max_tone,
            ingestion_timestamp
        )
    """
    
    merge_job = client.query(merge_query)
    merge_job.result()
    
    if auto_merge:
        merge_job = client.query(merge_query)
        merge_job.result()
    else:


def load_config(config_path: str) -> dict:
    """Load GDELT config from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def flush_existing_data(
    client: bigquery.Client,
    table_id: str,
    start_date: str,
    end_date: str
) -> None:
    """Delete existing data for date range."""
    
    delete_query = f"""
    DELETE FROM `{table_id}`
    WHERE DATE(timestamp) >= '{start_date}'
      AND DATE(timestamp) <= '{end_date}'
    """
    
    try:
        job = client.query(delete_query)
        job.result()
    except Exception as e:


def purge_all_data(client: bigquery.Client, table_id: str) -> None:
    """Delete ALL data from table."""
    
    response = input("   ⚠️  Delete ALL sentiment data? [y/N]: ")
    if response.lower() != 'y':
        return
    
    try:
        client.query(f"TRUNCATE TABLE `{table_id}`").result()
    except Exception as e:


def get_latest_timestamp(client: bigquery.Client, table_id: str) -> Optional[str]:
    """Get latest timestamp from table."""
    query = f"SELECT MAX(DATE(timestamp)) as max_date FROM `{table_id}`"
    
    try:
        result = client.query(query).to_dataframe()
        if not result.empty and result['max_date'].iloc[0] is not None:
            return result['max_date'].iloc[0].strftime('%Y-%m-%d')
    except Exception as e:
    
    return None



def main():
    parser = argparse.ArgumentParser(
        description='Fetch GDELT GKG sentiment data from BigQuery and aggregate to 15-minute intervals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Fetch using config
  python gdelt_load.py --topic-group inflation_prices --config configs/gdelt.yaml
  
  # Fetch specific date range
  python gdelt_load.py --topic-group monetary_policy --start-date 2025-11-11 --end-date 2025-11-18
  
  # Reload existing data
  python gdelt_load.py --topic-group inflation_prices --start-date 2025-11-11 --end-date 2025-11-18 --reload
  
  # Top-up with new data
  python gdelt_load.py --topic-group inflation_prices --config configs/gdelt.yaml --top-up
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/gdelt.yaml',
        help='Path to GDELT YAML config file'
    )
    parser.add_argument(
        '--topic-group',
        type=str,
        required=True,
        help='Topic group to load (e.g., inflation_prices, fed_policy). Required.'
    )
    parser.add_argument(
        '--start-date',
        '--start',
        dest='start_date',
        type=str,
        required=False,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        '--end',
        dest='end_date',
        type=str,
        required=False,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--topics',
        nargs='+',
        default=None,
        help=f'Topics/themes to track (default from config)'
    )
    parser.add_argument(
        '--top-up',
        action='store_true',
        help='Auto-detect latest and fetch new data only'
    )
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='Skip auto-merge staging to main'
    )
    parser.add_argument(
        '--skip-bigquery',
        action='store_true',
        help='Skip BigQuery upload'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Clear all existing data for this topic group and frequency before loading'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be saved without actually saving'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = None
    topics = args.topics
    start_date = args.start_date
    end_date = args.end_date
    topic_group_id = args.topic_group
    
    if args.config:
        try:
            config = load_config(args.config)
            
            # Validate topic group exists in config
            if 'topic_groups' in config:
                if topic_group_id not in config['topic_groups']:
                    available = ', '.join(config['topic_groups'].keys())
                    sys.exit(1)
                
                # Get topics from selected group
                group_config = config['topic_groups'][topic_group_id]
                if not topics and 'topics' in group_config:
                    topics = group_config['topics']
                    description = group_config.get('description', '')
            else:
            
            if not start_date and 'date_range' in config:
                start_date = config['date_range'].get('start_date')
            
            if not end_date and 'date_range' in config:
                end_date_cfg = config['date_range'].get('end_date')
                end_date = datetime.now().strftime('%Y-%m-%d') if end_date_cfg is None else end_date_cfg
            
            # Read frequency from nested config (aggregation.frequency)
            if 'aggregation' in config and 'frequency' in config['aggregation']:
                frequency = config['aggregation']['frequency']
            
        except FileNotFoundError:
            if args.config != 'configs/gdelt.yaml':
    
    if not topics:
        sys.exit(1)
    
    # Set default frequency if not set
    if 'frequency' not in locals():
        frequency = '1d'  # Default to native GDELT frequency
    
    # Handle modes
    if args.top_up:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = '2016-01-01'
    else:
        if not start_date or not end_date:
            sys.exit(1)
    
    # Validate dates
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if end < start and not args.top_up:
            sys.exit(1)
            
        # Warn if date range is very large (expensive query)
        days = (end - start).days
        if days > 90:
            response = input("   Continue? [y/N]: ")
            if response.lower() != 'y':
                sys.exit(0)
                
    except ValueError:
        sys.exit(1)
    
    # Validate project ID
    if not PROJECT_ID and not args.skip_bigquery:
        sys.exit(1)
    
    # Handle top-up
    if args.top_up:
        if not PROJECT_ID:
            sys.exit(1)
        client = bigquery.Client(project=PROJECT_ID)
        latest = get_latest_timestamp(client, f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}")
        if latest:
            start_date = latest
        else:
    
    # Check if data already exists (unless reload is specified)
    if not args.reload and not args.skip_bigquery:
        client = bigquery.Client(project=PROJECT_ID)
        main_table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"
        
        
        if check_existing_data(client, main_table_id, topic_group_id, frequency, start_date, end_date):
            sys.exit(0)
        else:
    
    # Fetch sentiment data (15m intervals from BigQuery)
    df = fetch_gdelt_sentiment(
        start_date=start_date,
        end_date=end_date,
        topics=topics,
        project_id=PROJECT_ID
    )
    
    if df.empty:
        sys.exit(0)
    
    # Resample to target frequency if needed
    if frequency != '15m':
        df = resample_to_frequency(df, frequency, method='mean')
        # Update frequency column to match resampled frequency
        df['frequency'] = frequency
        # Recalculate date column after resampling
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # Save to BigQuery (unless skipped)
    if not args.skip_bigquery:
        # Handle --reload: clear all data for this topic group and frequency first
        if args.reload and not args.dry_run:
            client = bigquery.Client(project=PROJECT_ID)
            main_table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"
            clear_topic_group_frequency_data(client, main_table_id, topic_group_id, frequency, PROJECT_ID, DATASET_ID)
        
        save_to_bigquery(
            df,
            PROJECT_ID,
            DATASET_ID,
            topic_group_id=topic_group_id,
            dry_run=args.dry_run,
            auto_merge=not args.no_merge
        )
        
        # Save query metadata for data lineage
        if not args.dry_run:
            client = bigquery.Client(project=PROJECT_ID)
            num_articles = int(df['num_articles'].sum()) if 'num_articles' in df.columns else 0
            save_query_metadata(
                client=client,
                start_date=start_date,
                end_date=end_date,
                topics=topics,
                num_records=len(df),
                num_articles=num_articles,
                topic_group_id=topic_group_id,
                config_file=args.config if args.config != 'configs/gdelt.yaml' else 'configs/gdelt.yaml',
                frequency=frequency,
                notes=f"Loaded via gdelt_load.py {'(top-up)' if args.top_up else '(full load)'}"
            )
    


if __name__ == '__main__':
    main()
