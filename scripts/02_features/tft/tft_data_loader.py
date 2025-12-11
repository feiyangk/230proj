#!/usr/bin/env python3
"""
TFT Data Loader

Fetches  OHLCV and synthetic indicators from BigQuery,
prepares time series data for TFT multi-horizon quantile forecasting.
"""

import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader


class MultiTickerDataLoader:
    """Load and prepare multi-ticker data from BigQuery for TFT training."""
    
    def __init__(self, config_path: str = 'configs/model_tft_config.yaml', export_temp: bool = False):
        """Initialize data loader with configuration.
        
        Args:
            config_path: Path to config YAML file
            export_temp: If True, export raw data to temp/ directory (for debugging)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        bigquery_cfg = self.config.get('bigquery', {})
        self.project_id = os.getenv('GCP_PROJECT_ID', bigquery_cfg.get('project_id'))
        if not self.project_id:
            raise ValueError(
                "Missing GCP project. Set GCP_PROJECT_ID env var or bigquery.project_id in the config."
            )
        
        self.client = bigquery.Client(project=self.project_id)
        
        # Extract config
        self.tickers = self.config['data']['tickers']['symbols']
        self.frequency = self.config['data']['tickers']['frequency']
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
        self.lookback = self.config['data']['lookback_window']
        self.horizons = self.config['data']['prediction_horizons']
        
        self.raw_features = self.config['data']['tickers']['raw_features']
        self.synthetic_features = self.config['data']['tickers']['synthetic_features']
        self.target_col = self.config['data']['target']
        
        # GDELT config
        self.use_gdelt = self.config['data']['gdelt'].get('enabled', False)
        if self.use_gdelt:
            self.gdelt_frequency = self.config['data']['gdelt'].get('frequency', self.frequency)  # Use GDELT-specific frequency
            self.gdelt_topic_groups = self.config['data']['gdelt'].get('topic_groups', ['inflation_prices'])  # Topic groups to load
            self.gdelt_features = self.config['data']['gdelt']['features']
            self.gdelt_normalize_counts = self.config['data']['gdelt'].get('normalize_counts', True)
            self.gdelt_include_lags = self.config['data']['gdelt'].get('include_lags', True)
            self.gdelt_lag_periods = self.config['data']['gdelt'].get('lag_periods', [1, 4, 16])
        
        # Normalization (default to standard scaling if not specified)
        self.normalize = self.config['data'].get('normalize', True)
        self.norm_method = self.config['data'].get('normalization_method', 'standard')
        self.scalers = {}
        
        # Weekend filtering
        self.skip_weekends = self.config['data'].get('skip_weekends', False)
        
        # Forward filling config
        self.forward_fill_config = self.config['data'].get('forward_fill', {})
        self.forward_fill_enabled = self.forward_fill_config.get('enabled', True)
        self.forward_fill_log_stats = self.forward_fill_config.get('log_stats', True)
        self.forward_fill_max_limit = self.forward_fill_config.get('max_fill_limit', 5)
        
        # Export settings
        self.export_temp = export_temp
        
    def fetch_ticker_data(self) -> pd.DataFrame:
        """Fetch combined raw OHLCV and synthetic indicators from BigQuery for all tickers."""
        dataset_id = self.config['bigquery']['dataset_id']
        raw_table = self.config['bigquery']['ticker']['raw_table']
        synthetic_table = self.config['bigquery']['ticker']['synthetic_table']
        
        
        # Build feature lists for SQL
        raw_cols = ', '.join([f'r.{col}' for col in self.raw_features])
        synthetic_cols = ', '.join([f's.{col}' for col in self.synthetic_features])
        
        # Build ticker list for SQL IN clause
        ticker_list = "', '".join(self.tickers)
        
        query = f"""
        SELECT 
            r.ticker,
            r.timestamp,
            r.date,
            {raw_cols},
            {synthetic_cols}
        FROM `{self.project_id}.{dataset_id}.{raw_table}` r
        INNER JOIN `{self.project_id}.{dataset_id}.{synthetic_table}` s
            ON r.ticker = s.ticker
            AND r.timestamp = s.timestamp
            AND r.frequency = s.frequency
        WHERE r.ticker IN ('{ticker_list}')
            AND r.frequency = '{self.frequency}'
            AND DATE(r.timestamp) BETWEEN '{self.start_date}' AND '{self.end_date}'
        ORDER BY r.ticker, r.timestamp
        """
        
        df = self.client.query(query).to_dataframe()
        for ticker in self.tickers:
            ticker_rows = len(df[df['ticker'] == ticker])
        
        return df
    
    def fetch_gdelt_data(self) -> pd.DataFrame:
        """Fetch GDELT sentiment data from BigQuery for specified topic groups."""
        if not self.use_gdelt:
            return None
        
        dataset_id = self.config['bigquery']['dataset_id']
        gdelt_table = self.config['bigquery']['gdelt']['table']
        
        
        # Build feature list for SQL
        gdelt_cols = ', '.join([f'g.{col}' for col in self.gdelt_features])
        
        # Build topic group filter
        topic_group_list = "', '".join(self.gdelt_topic_groups)
        
        query = f"""
        SELECT 
            g.timestamp,
            g.topic_group_id,
            {gdelt_cols}
        FROM `{self.project_id}.{dataset_id}.{gdelt_table}` g
        WHERE g.frequency = '{self.gdelt_frequency}'
            AND g.topic_group_id IN ('{topic_group_list}')
            AND DATE(g.timestamp) BETWEEN '{self.start_date}' AND '{self.end_date}'
        ORDER BY g.timestamp, g.topic_group_id
        """
        
        df = self.client.query(query).to_dataframe()
        
        if len(df) == 0:
            return None
        
        
        # Show breakdown by topic group
        for topic_group in self.gdelt_topic_groups:
            count = len(df[df['topic_group_id'] == topic_group])
        
        # If multiple topic groups, aggregate them (average sentiment across groups)
        if len(self.gdelt_topic_groups) > 1:
            # Group by timestamp and average the sentiment features
            agg_dict = {col: 'mean' for col in self.gdelt_features}
            df = df.groupby('timestamp').agg(agg_dict).reset_index()
        else:
            # Single topic group - just drop the topic_group_id column
            df = df.drop(columns=['topic_group_id'])
        
        return df
    
    def fetch_agriculture_basket(self) -> pd.DataFrame:
        """Fetch agriculture basket (WEAT, SOYB, RJA) for target computation."""
        dataset_id = self.config['bigquery']['dataset_id']
        raw_table = self.config['bigquery']['ticker']['raw_table']
        
        agriculture_tickers = ['WEAT', 'SOYB', 'RJA']
        ticker_list = "', '".join(agriculture_tickers)
        
        
        query = f"""
        SELECT 
            timestamp,
            ticker,
            close
        FROM `{self.project_id}.{dataset_id}.{raw_table}`
        WHERE ticker IN ('{ticker_list}')
            AND frequency = '{self.frequency}'
            AND DATE(timestamp) BETWEEN '{self.start_date}' AND '{self.end_date}'
        ORDER BY timestamp, ticker
        """
        
        df = self.client.query(query).to_dataframe()
        
        if len(df) == 0:
            return None
        
        # Pivot to get one column per ticker
        df_pivot = df.pivot(index='timestamp', columns='ticker', values='close')
        
        # Compute average close price across available tickers (skip NaN)
        # This ensures we average only available tickers if some are missing
        df_pivot['agriculture_basket_close'] = df_pivot[agriculture_tickers].mean(axis=1, skipna=True)
        
        # Count how many tickers contributed to each average
        df_pivot['num_tickers_available'] = df_pivot[agriculture_tickers].notna().sum(axis=1)
        
        # Keep only the average column
        result = df_pivot[['agriculture_basket_close']].reset_index()
        
        for ticker in agriculture_tickers:
            if ticker in df_pivot.columns:
                count = df_pivot[ticker].notna().sum()
        
        # Show statistics on ticker availability
        ticker_counts = df_pivot['num_tickers_available'].value_counts().sort_index()
        for num_tickers, count in ticker_counts.items():
        
        return result
    
    def compute_basket_target(self, ticker_df: pd.DataFrame, basket_df: pd.DataFrame) -> pd.DataFrame:
        """Join agriculture basket close prices to ticker data for target computation."""
        if basket_df is None:
            return ticker_df
        
        
        # Left join to preserve all ticker timestamps
        df = ticker_df.merge(basket_df, on='timestamp', how='left')
        
        # Forward fill missing basket values
        missing_before = df['agriculture_basket_close'].isnull().sum()
        if missing_before > 0:
            df = self.forward_fill_with_stats(df, ['agriculture_basket_close'], context="Agriculture basket")
        
        
        return df
    
    def filter_weekends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove weekend data (Saturday/Sunday) if configured."""
        if not self.skip_weekends:
            return df
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        initial_rows = len(df)
        # Filter out weekends (dayofweek: 5=Saturday, 6=Sunday)
        df = df[df['timestamp'].dt.dayofweek < 5].copy()
        removed_rows = initial_rows - len(df)
        
        if removed_rows > 0:
        
        return df
    
    def forward_fill_with_stats(self, df: pd.DataFrame, columns: List[str], context: str = "") -> pd.DataFrame:
        
        Args:
            df: DataFrame to fill
            columns: List of column names to forward fill
        
        Returns:
            DataFrame with forward filled values
        """
        if not self.forward_fill_enabled:
            return df
        
        df = df.copy()
        fill_stats = {}
        total_filled = 0
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Count missing before
            missing_before = df[col].isnull().sum()
            if missing_before == 0:
                continue
            
            # Forward fill with limit
            df[col] = df[col].ffill(limit=self.forward_fill_max_limit)
            
            # Count missing after
            missing_after = df[col].isnull().sum()
            filled_count = missing_before - missing_after
            total_filled += filled_count
            
            fill_stats[col] = {
                'missing_before': missing_before,
                'filled': filled_count,
                'still_missing': missing_after,
                'pct_filled': (filled_count / missing_before * 100) if missing_before > 0 else 0
            }
        
        # Log statistics if enabled
        if self.forward_fill_log_stats and fill_stats:
            
            for col, stats in fill_stats.items():
                      f"{stats['still_missing']:<15,} {stats['pct_filled']:<10.1f}%")
            
            # Warn about columns that still have missing values
            still_missing_cols = [col for col, stats in fill_stats.items() if stats['still_missing'] > 0]
            if still_missing_cols:
                for col in still_missing_cols:
        
        return df
    
    def join_gdelt_features(self, ticker_df: pd.DataFrame, gdelt_df: pd.DataFrame) -> pd.DataFrame:
        """Join GDELT features with ticker data."""
        if gdelt_df is None or not self.use_gdelt:
            return ticker_df
        
        
        # Left join to preserve all ticker timestamps
        df = ticker_df.merge(gdelt_df, on='timestamp', how='left')
        
        # Forward fill missing GDELT values (for gaps in sentiment data)
        df = self.forward_fill_with_stats(df, self.gdelt_features, context="GDELT features")
        
        # Normalize article/source counts if configured
        if self.gdelt_normalize_counts:
            if 'num_articles' in df.columns:
                df['num_articles'] = np.log1p(df['num_articles'])  # Log transform
            if 'num_sources' in df.columns:
                df['num_sources'] = np.log1p(df['num_sources'])  # Log transform
        
        # Add lagged sentiment features if configured
        if self.gdelt_include_lags and 'weighted_avg_tone' in df.columns:
            lag_cols = []
            for lag in self.gdelt_lag_periods:
                col_name = f'sentiment_lag_{lag}'
                df[col_name] = df['weighted_avg_tone'].shift(lag)
                lag_cols.append(col_name)
            
            # Forward fill NaN from initial lags
            df = self.forward_fill_with_stats(df, lag_cols, context="GDELT lagged features")
        
        
        return df
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch and combine all data sources."""
        # Fetch ticker data (OHLCV + indicators)
        ticker_df = self.fetch_ticker_data()
        
        # Fetch and join GDELT data if enabled
        if self.use_gdelt:
            gdelt_df = self.fetch_gdelt_data()
            df = self.join_gdelt_features(ticker_df, gdelt_df)
        else:
            df = ticker_df
        
        # Fetch and join agriculture basket for target computation
        basket_df = self.fetch_agriculture_basket()
        df = self.compute_basket_target(df, basket_df)
        
        # Filter weekends if configured (after joining all data)
        df = self.filter_weekends(df)
        
        # Check for remaining missing values
        missing = df.isnull().sum()
        if missing.any():
            missing_cols = missing[missing > 0].index.tolist()
            for col in missing_cols[:10]:  # Show first 10
            if len(missing_cols) > 10:
            
            # Forward fill remaining missing values with stats
            df = self.forward_fill_with_stats(df, missing_cols, context="Remaining features")
            
            # Backward fill for any remaining (at start of series)
            still_missing = df.isnull().sum()
            if still_missing.any():
                still_missing_cols = still_missing[still_missing > 0].index.tolist()
                df = df.bfill()
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features (future-known)."""
        df = df.copy()
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time features (these are future-known)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # Saturday=5, Sunday=6
        
        # Cyclical encoding (helps model learn periodicity)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        
        return df
    
    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize features using standard or minmax scaling."""
        if not self.normalize:
            return df
        
        df = df.copy()
        
        # Get all features from config (time-varying known + unknown)
        time_varying_known = self.config['model'].get('time_varying_known', [])
        time_varying_unknown = self.config['model']['time_varying_unknown']
        all_features = list(time_varying_known) + list(time_varying_unknown)
        
        # Time features should NOT be normalized (they're already in good ranges)
        time_features_to_skip = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                                  'month_sin', 'month_cos', 'is_weekend',
                                  'hour', 'day_of_week', 'month', 'day_of_month']
        
        for col in all_features:
            if col not in df.columns:
                continue
            
            # Skip time features (already in good ranges)
            if col in time_features_to_skip:
                continue
            
            if fit:
                # Fit scaler on training data
                if self.norm_method == 'standard':
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()
                
                df[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            else:
                # Use existing scaler (for validation/test)
                if col in self.scalers:
                    df[col] = self.scalers[col].transform(df[[col]])
        
        return df
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Create lookback sequences and multi-horizon targets.
        Groups by timestamp to ensure each unique date appears only once.
        
        Returns:
            X: [num_samples, lookback, num_features]
            y: [num_samples, num_horizons]
            timestamps: [num_samples]
            grouped: Pivoted dataframe (one row per date with ticker-specific columns)
        """
        # Combine all features: time-varying known + unknown + time features
        time_varying_known = self.config['model'].get('time_varying_known', [])
        time_varying_unknown = self.config['model']['time_varying_unknown']
        
        # Time features (if they exist in the dataframe)
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_weekend']
        available_time_features = [f for f in time_features if f in df.columns]
        
        # Combine all features: time_varying_known first, then time_varying_unknown
        all_features = list(time_varying_known) + list(time_varying_unknown)
        
        # Note: Don't filter here - pivoting will create ticker-specific columns
        
        # Pivot data: one row per timestamp with ticker-specific columns
        
        # Identify ticker-specific vs shared features
        ticker_features = ['close', 'volume', 'sma_50', 'sma_200']
        gdelt_features = ['weighted_avg_tone', 'weighted_avg_polarity', 'num_articles', 
                         'num_sources', 'sentiment_lag_1', 'sentiment_lag_7', 'sentiment_lag_30']
        time_features = ['month_sin', 'month_cos', 'is_weekend']
        
        # Get unique tickers (sorted for consistency)
        if 'ticker' in df.columns:
            tickers = sorted(df['ticker'].unique())
            
            # Start with timestamps
            timestamps = sorted(df['timestamp'].unique())
            grouped = pd.DataFrame({'timestamp': timestamps})
            
            # Pivot ticker-specific features
            for ticker in tickers:
                ticker_data = df[df['ticker'] == ticker].set_index('timestamp')
                for feat in ticker_features:
                    if feat in df.columns:
                        col_name = f'{feat}_{ticker}'
                        grouped[col_name] = grouped['timestamp'].map(ticker_data[feat])
            
            # Add shared features (GDELT and time) - take first value for each timestamp
            for feat in gdelt_features + time_features:
                if feat in df.columns:
                    feat_values = df.groupby('timestamp')[feat].first()
                    grouped[feat] = grouped['timestamp'].map(feat_values)
            
            # Add agriculture basket
            if 'agriculture_basket_close' in df.columns:
                basket_values = df.groupby('timestamp')['agriculture_basket_close'].first()
                grouped['agriculture_basket_close'] = grouped['timestamp'].map(basket_values)
            
            # Update all_features to include ticker-specific columns
            new_features = []
            for feat in all_features:
                if feat in ticker_features:
                    # Replace with ticker-specific versions
                    new_features.extend([f'{feat}_{ticker}' for ticker in tickers])
                else:
                    # Keep GDELT and time features as-is
                    new_features.append(feat)
            all_features = new_features
            
            # Save final features for metadata export
            self.final_features = all_features
            
        else:
            # No ticker column, just group by timestamp (shouldn't happen)
            grouped = df.groupby('timestamp').first().reset_index()
        
        # Store final feature list for metadata
        self.final_features = all_features
        
        
        feature_data = grouped[all_features].values
        
        # Get target prices for computing true multi-horizon returns
        # Use agriculture basket if available, otherwise use ticker close price
        if 'agriculture_basket_close' in grouped.columns:
            target_prices = grouped['agriculture_basket_close'].values
        else:
            target_prices = grouped['close'].values
        
        timestamps = grouped['timestamp'].values
        
        max_horizon = max(self.horizons)
        
        X, y, ts = [], [], []
        
        # Create sequences (use grouped data length)
        for i in range(self.lookback, len(grouped) - max_horizon):
            # Input: lookback window of features
            X.append(feature_data[i - self.lookback:i])
            
            # Target: TRUE k-period forward returns from current price
            # Formula: (price[t+k] - price[t]) / price[t]
            current_price = target_prices[i]
            targets = [(target_prices[i + h] - current_price) / current_price for h in self.horizons]
            y.append(targets)
            
            # Timestamp of prediction point
            ts.append(timestamps[i])
        
        return np.array(X), np.array(y), np.array(ts), grouped
    
    def split_data(self, X: np.ndarray, y: np.ndarray, ts: np.ndarray
                   ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Split data into train/val/test sets."""
        n_samples = len(X)
        
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        splits = {
            'train': (X[:train_end], y[:train_end], ts[:train_end]),
            'val': (X[train_end:val_end], y[train_end:val_end], ts[train_end:val_end]),
            'test': (X[val_end:], y[val_end:], ts[val_end:])
        }
        
        for split_name, (x, _, _) in splits.items():
        
        return splits
    
    def save_processed_data(self, splits: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]], 
                           df_raw: pd.DataFrame):
        """Save train/val/test splits as numpy arrays (.npy) in data/processed."""
        import shutil
        import pickle
        from pathlib import Path
        
        output_dir = Path('data/processed')
        
        # Clear existing data
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get feature list from config (same as create_sequences)
        # Use final_features if available (after pivoting), otherwise use config
        if hasattr(self, 'final_features'):
            all_features = self.final_features
            time_varying_known = self.config['model'].get('time_varying_known', [])
            time_varying_unknown = [f for f in all_features if f not in time_varying_known]
        else:
            time_varying_known = self.config['model'].get('time_varying_known', [])
            time_varying_unknown = self.config['model']['time_varying_unknown']
            all_features = list(time_varying_known) + list(time_varying_unknown)
        
        
        # Save each split as separate X, y, ts arrays
        for split_name, (X, y, ts) in splits.items():
            n_samples, lookback, n_features = X.shape
            n_horizons = y.shape[1] if len(y.shape) > 1 else 1
            
            
            # Save arrays (memory efficient - no copies)
            X_file = output_dir / f'X_{split_name}.npy'
            y_file = output_dir / f'y_{split_name}.npy'
            ts_file = output_dir / f'ts_{split_name}.npy'
            
            np.save(X_file, X)
            
            np.save(y_file, y)
            
            np.save(ts_file, ts)
            
            # Calculate file sizes
            X_size_mb = X_file.stat().st_size / 1024 / 1024
            y_size_mb = y_file.stat().st_size / 1024 / 1024
            ts_size_mb = ts_file.stat().st_size / 1024 / 1024
            total_split_mb = X_size_mb + y_size_mb + ts_size_mb
            
        
        # Save scalers for inverse transform
        scalers_file = output_dir / 'scalers.pkl'
        with open(scalers_file, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        scalers_size_kb = scalers_file.stat().st_size / 1024
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'tickers': self.tickers,  # List of ticker symbols
            'frequency': self.frequency,
            'lookback_window': self.lookback,
            'prediction_horizons': self.horizons,
            'raw_features': self.raw_features,
            'synthetic_features': self.synthetic_features,
            'target': self.target_col,
            'normalization': self.norm_method if self.normalize else 'none',
            'train_samples': len(splits['train'][0]),
            'val_samples': len(splits['val'][0]),
            'test_samples': len(splits['test'][0]),
            'total_features': len(all_features),
            'features': all_features,  # Actual feature list (after pivoting)
            'time_varying_known': time_varying_known,
            'time_varying_unknown': time_varying_unknown,
            'array_shapes': {
                'X': f"[samples, {self.lookback}, {len(all_features)}]",
                'y': f"[samples, {len(self.horizons)}]",
                'ts': '[samples]'
            }
        }
        
        metadata_file = output_dir / 'metadata.yaml'
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        metadata_size_kb = metadata_file.stat().st_size / 1024
        
        
        # Save feature names
        feature_file = output_dir / 'feature_names.txt'
        with open(feature_file, 'w') as f:
            f.write("TIME-VARYING KNOWN FEATURES:\n")
            for feat in time_varying_known:
                f.write(f"  - {feat}\n")
            f.write("\nTIME-VARYING UNKNOWN FEATURES:\n")
            for feat in time_varying_unknown:
                f.write(f"  - {feat}\n")
            f.write("\n# Usage Example:\n")
            f.write("# import numpy as np\n")
            f.write("# X_train = np.load('data/processed/X_train.npy')\n")
            f.write("# y_train = np.load('data/processed/y_train.npy')\n")
            f.write("# ts_train = np.load('data/processed/ts_train.npy')\n")
        
        feature_file_size_kb = feature_file.stat().st_size / 1024
        
        
        # Calculate total directory size
        total_size_mb = sum(f.stat().st_size for f in output_dir.glob('*')) / 1024 / 1024
        
    
    def _export_raw_validation(self, df: pd.DataFrame):
        """Export raw validation data to data/raw (always called)."""
        
        raw_dir = Path('data/raw')
        raw_dir.mkdir(parents=True, exist_ok=True)
        self._export_raw_with_targets(df, raw_dir)
    
    def _export_raw_with_targets(self, df_raw: pd.DataFrame, output_dir: Path):
        """Export raw (non-normalized) pivoted data with computed target labels.
        
        df_raw is the grouped/pivoted dataframe with one row per date and ticker-specific columns.
        """
        df_raw = df_raw.copy()
        
        # Add target labels (TRUE k-period forward returns)
        # Use agriculture basket if available, otherwise use first ticker close
        if 'agriculture_basket_close' in df_raw.columns:
            target_price_col = 'agriculture_basket_close'
        else:
            # Find first ticker close column
            close_cols = [c for c in df_raw.columns if c.startswith('close_')]
            target_price_col = close_cols[0] if close_cols else 'close'
        
        # Formula: (price[t+k] - price[t]) / price[t]
        for horizon in self.horizons:
            col_name = f'target_{horizon}periods_ahead'
            df_raw[col_name] = (df_raw[target_price_col].shift(-horizon) - df_raw[target_price_col]) / df_raw[target_price_col]
        
        # Add helpful time columns if not present
        if 'hour' not in df_raw.columns:
            df_raw['hour'] = pd.to_datetime(df_raw['timestamp']).dt.hour
        if 'day_of_week' not in df_raw.columns:
            df_raw['day_of_week'] = pd.to_datetime(df_raw['timestamp']).dt.day_name()
        
        # Reorder columns for readability: timestamp, time info, then all features, then targets
        time_cols = ['timestamp', 'hour', 'day_of_week']
        target_cols = [c for c in df_raw.columns if c.startswith('target_')]
        feature_cols = [c for c in df_raw.columns if c not in time_cols and c not in target_cols]
        df_raw = df_raw[time_cols + feature_cols + target_cols]
        
        # Export to CSV and Parquet (clean naming)
        csv_file = output_dir / 'tft_features.csv'
        parquet_file = output_dir / 'tft_features.parquet'
        
        df_raw.to_csv(csv_file, index=False, float_format='%.8f')  # 8 decimals for small indicator values
        df_raw.to_parquet(parquet_file, index=False)
        
        
        # After pivoting, close columns are ticker-specific (e.g., close_SPY, close_QQQ)
        close_cols = [col for col in df_raw.columns if col.startswith('close_')]
        if close_cols:
            # Show first ticker's close price as example
            first_close_col = close_cols[0]
        
        if len(target_cols) > 0:
            for tc in target_cols:
                val = df_raw[tc].iloc[0]
                if pd.notna(val):
    
    def _export_normalized_data(self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray, 
                                ts: np.ndarray, base_name: str):
        """Export normalized data and sequence summaries (original behavior)."""
        
        # 1. Export normalized dataframe
        normalized_file = f"{base_name}_normalized.parquet"
        df.to_parquet(normalized_file, index=False)
        
        
        # 2. Export sequence summary
        summary_file = f"{base_name}_sequences_summary.csv"
        
        summary_data = []
        for i in [0, len(X)//2, len(X)-1]:  # First, middle, last
            summary_data.append({
                'sequence_id': i,
                'timestamp': pd.Timestamp(ts[i]),
                'target_1h': y[i, 0] if len(y.shape) > 1 else y[i],
                'target_2h': y[i, 1] if len(y.shape) > 1 and y.shape[1] > 1 else None,
                'target_4h': y[i, 2] if len(y.shape) > 1 and y.shape[1] > 2 else None,
                'input_shape': str(X[i].shape),
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        
    
    def prepare_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Full data preparation pipeline."""
        # Fetch data
        df = self.fetch_data()
        
        # Add time features (before normalization)
        df = self.add_time_features(df)
        
        # Create sequences from RAW data (pivoting happens inside)
        X_raw, y_raw, ts, df_raw = self.create_sequences(df)
        
        # Split data FIRST (before normalization)
        n_samples = len(X_raw)
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train_raw = X_raw[:train_end]
        X_val_raw = X_raw[train_end:val_end]
        X_test_raw = X_raw[val_end:]
        
        y_train = y_raw[:train_end]
        y_val = y_raw[train_end:val_end]
        y_test = y_raw[val_end:]
        
        ts_train = ts[:train_end]
        ts_val = ts[train_end:val_end]
        ts_test = ts[val_end:]
        
        
        # Normalize ONLY on training data (both features and targets)
        X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm = self._normalize_splits(
            X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test
        )
        
        splits = {
            'train': (X_train_norm, y_train_norm, ts_train),
            'val': (X_val_norm, y_val_norm, ts_val),
            'test': (X_test_norm, y_test_norm, ts_test)
        }
        
        # Always export raw validation data to data/raw
        self._export_raw_validation(df_raw)
        
        # Export debug data to temp (optional)
        if self.export_temp:
            # Note: Exporting normalized version
            df_norm = self.normalize_data(df, fit=True)
            self._export_normalized_data(df_norm, X_raw, y_raw, ts, f"temp/tft_data_{self.ticker.replace(':', '_')}_{self.frequency}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Save processed parquet files for inspection (always)
        self.save_processed_data(splits, df_raw)
        
        return splits
    
    def _normalize_splits(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Normalize train/val/test splits. Fit ONLY on training data."""
        from sklearn.preprocessing import StandardScaler
        
        # CRITICAL: Use final_features (after pivoting) instead of config!
        # After pivoting, features like 'close' become 'close_SPY', 'close_QQQ', etc.
        if not hasattr(self, 'final_features'):
            raise ValueError("final_features not set! create_sequences() must be called first.")
        
        all_features = self.final_features
        
        # Time features should NOT be normalized (already in good ranges: -1 to 1)
        time_features_to_skip = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                                  'month_sin', 'month_cos', 'is_weekend']
        
        # Get indices of features to normalize
        features_to_normalize = []
        for i, feat in enumerate(all_features):
            if feat not in time_features_to_skip:
                features_to_normalize.append(i)
        
        skipped_features = [f for f in all_features if f in time_features_to_skip]
        if skipped_features:
        else:
        
        # Normalize each feature across the sequence dimension
        X_train_norm = X_train.copy()
        X_val_norm = X_val.copy()
        X_test_norm = X_test.copy()
        
        import time
        n_features = len(features_to_normalize)
        start_time = time.time()
        
        for i, feat_idx in enumerate(features_to_normalize, 1):
            feat_name = all_features[feat_idx]
            
            # Reshape to (n_samples * lookback, 1) for fitting
            train_values = X_train[:, :, feat_idx].reshape(-1, 1)
            
            # Fit scaler on training data only
            scaler = StandardScaler()
            scaler.fit(train_values)
            
            # Transform all splits
            X_train_norm[:, :, feat_idx] = scaler.transform(
                X_train[:, :, feat_idx].reshape(-1, 1)
            ).reshape(X_train.shape[0], X_train.shape[1])
            
            X_val_norm[:, :, feat_idx] = scaler.transform(
                X_val[:, :, feat_idx].reshape(-1, 1)
            ).reshape(X_val.shape[0], X_val.shape[1])
            
            X_test_norm[:, :, feat_idx] = scaler.transform(
                X_test[:, :, feat_idx].reshape(-1, 1)
            ).reshape(X_test.shape[0], X_test.shape[1])
            
            # Store scaler
            self.scalers[feat_name] = scaler
            
            # Progress update (log every feature)
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (n_features - i) if i < n_features else 0
            pct = (i / n_features) * 100
            
            # Shorten feature name if too long (e.g., ticker-specific features)
            display_name = feat_name if len(feat_name) <= 30 else feat_name[:27] + '...'
        
        total_time = time.time() - start_time
        
        # Normalize targets (y) - FIT on actual multi-horizon target distribution
        # NOTE: Do NOT use the 1-period 'returns' scaler! Multi-horizon returns have different variance.
        
        # Show raw statistics BEFORE normalization (diagnostic)
        
        # Always fit a NEW scaler specifically for targets
        # (Multi-horizon returns have different statistics than 1-period returns)
        target_scaler = StandardScaler()
        # Fit on training targets (all horizons combined)
        target_scaler.fit(y_train.reshape(-1, 1))
        self.scalers['target'] = target_scaler  # Store under 'target' key, not self.target_col
        
        
        # Transform targets for all splits (flatten, transform, reshape)
        # y shape: (n_samples, n_horizons) -> flatten -> transform -> reshape back
        y_train_norm = target_scaler.transform(y_train.flatten().reshape(-1, 1)).reshape(y_train.shape)
        y_val_norm = target_scaler.transform(y_val.flatten().reshape(-1, 1)).reshape(y_val.shape)
        y_test_norm = target_scaler.transform(y_test.flatten().reshape(-1, 1)).reshape(y_test.shape)
        
        
        # CRITICAL: Check for NaN/Inf after normalization
        
        try:
            # Ensure numeric dtype
            X_train_arr = np.asarray(X_train_norm, dtype=np.float64)
            X_val_arr = np.asarray(X_val_norm, dtype=np.float64)
            y_train_arr = np.asarray(y_train_norm, dtype=np.float64)
            y_val_arr = np.asarray(y_val_norm, dtype=np.float64)
            
            # Check X (features)
            train_nan = np.isnan(X_train_arr).sum()
            train_inf = np.isinf(X_train_arr).sum()
            val_nan = np.isnan(X_val_arr).sum()
            val_inf = np.isinf(X_val_arr).sum()
            
            
            # Check y (targets)
            y_train_nan = np.isnan(y_train_arr).sum()
            y_train_inf = np.isinf(y_train_arr).sum()
            y_val_nan = np.isnan(y_val_arr).sum()
            y_val_inf = np.isinf(y_val_arr).sum()
            
            
            # If any NaN/Inf found, identify which features
            if train_nan > 0 or train_inf > 0:
            nan_features = []
            inf_features = []
            for i, feat_name in enumerate(all_features):
                feat_data = X_train_arr[:, :, i]
                if np.isnan(feat_data).any():
                    nan_count = np.isnan(feat_data).sum()
                    nan_features.append(f"{feat_name} ({nan_count:,} NaN)")
                if np.isinf(feat_data).any():
                    inf_count = np.isinf(feat_data).sum()
                    inf_features.append(f"{feat_name} ({inf_count:,} Inf)")
            
            if nan_features:
                for feat in nan_features[:10]:  # Show first 10
                if len(nan_features) > 10:
            
            if inf_features:
                for feat in inf_features[:10]:  # Show first 10
                if len(inf_features) > 10:
            
                raise ValueError(
                    f"Data contains NaN ({train_nan:,}) or Inf ({train_inf:,}) after normalization. "
                    "This will cause model training to fail. Check the features listed above."
                )
            
            
        except (TypeError, ValueError) as e:
            if 'isnan' in str(e) or 'dtype' in str(e):
            else:
                raise
        
        return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm


class TFTDataset(Dataset):
    """PyTorch Dataset for TFT training."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, timestamps: np.ndarray):
        """
        Args:
            X: [num_samples, lookback, num_features]
            y: [num_samples, num_horizons]
            timestamps: [num_samples]
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.timestamps = timestamps
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_data_loaders(config_path: str = 'configs/model_tft_config.yaml',
                        export_temp: bool = False,
                        force_refresh: bool = False
                        ) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders for train/val/test.
    
    Args:
        config_path: Path to config YAML
        export_temp: If True, export raw data to temp/ directory (for debugging)
        force_refresh: If True, always fetch from BigQuery (ignore cached numpy arrays)
    """
    from pathlib import Path
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if pre-processed numpy arrays exist (unless force_refresh is True)
    processed_dir = Path('data/processed')
    # Use new naming convention: X_train.npy (not train_X.npy)
    train_X_file = processed_dir / 'X_train.npy'
    train_y_file = processed_dir / 'y_train.npy'
    train_ts_file = processed_dir / 'ts_train.npy'
    
    use_cached = (train_X_file.exists() and train_y_file.exists() and train_ts_file.exists()) and not force_refresh
    
    if use_cached:
        # Load pre-processed data from numpy arrays
        splits = {}
        for split_name in ['train', 'val', 'test']:
            # Use new naming convention: X_train.npy
            X = np.load(processed_dir / f'X_{split_name}.npy', allow_pickle=True)
            y = np.load(processed_dir / f'y_{split_name}.npy', allow_pickle=True)
            ts = np.load(processed_dir / f'ts_{split_name}.npy', allow_pickle=True)  # Timestamps are datetime objects
            
            # Ensure arrays are float type (not object)
            X = X.astype(np.float32)
            y = y.astype(np.float32)
            
            splits[split_name] = (X, y, ts)
        
        # Load scalers
        scalers_file = processed_dir / 'scalers.pkl'
        if scalers_file.exists():
            import pickle
            with open(scalers_file, 'rb') as f:
                scalers = pickle.load(f)
        else:
            scalers = None
    else:
        # Prepare data from BigQuery (original behavior)
        loader = MultiTickerDataLoader(config_path, export_temp=export_temp)
        splits = loader.prepare_data()

        # Ensure arrays are float type (not object) - convert in-place
        for split_name in splits:
            X, y, ts = splits[split_name]
            splits[split_name] = (X.astype(np.float32), y.astype(np.float32), ts)

        # Note: Arrays already saved by prepare_data() with new naming convention
        # No need to save again

        # Load scalers (already saved by prepare_data)
        scalers_file = processed_dir / 'scalers.pkl'
        if scalers_file.exists():
            import pickle
            with open(scalers_file, 'rb') as f:
                scalers = pickle.load(f)
        else:
            scalers = loader.scalers if hasattr(loader, 'scalers') else None
    
    # Create datasets
    datasets = {
        split_name: TFTDataset(X, y, ts)
        for split_name, (X, y, ts) in splits.items()
    }
    
    # Create dataloaders (use sensible defaults if config is missing sections)
    batch_size = config.get('training', {}).get('batch_size', 64)
    hardware_cfg = config.get('hardware', {})
    num_workers = hardware_cfg.get('num_workers', 0)
    pin_memory = hardware_cfg.get('pin_memory', False)
    
    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }
    
    return dataloaders, scalers


if __name__ == '__main__':
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='TFT Data Loader - Fetch and process multi-ticker data for training')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_tft_config.yaml',
        help='Path to model config YAML (controls tickers, dates, BigQuery tables)',
    )
    parser.add_argument(
        '--force-refresh', '--reload',
        action='store_true',
        dest='force_refresh',
        help='Force reload from BigQuery (ignore cached .npy files)',
    )
    parser.add_argument(
        '--export-temp',
        action='store_true',
        help='Export debug data to temp/ directory',
    )
    args = parser.parse_args()
    
    # Test data loading
    
    if args.force_refresh:
        pass
    
    loaders, scalers = create_data_loaders(
        config_path=args.config,
        export_temp=args.export_temp,
        force_refresh=args.force_refresh
    )
    
    
    # Test batch
    for batch_X, batch_y in loaders['train']:
        break
    
    
