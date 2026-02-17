import pandas as pd
import numpy as np
import os

class DataLoader:
    def load_and_process(self, sentiment_path, trades_path):
        """
        Loads raw data, standardizes timestamps to daily granularity,
        and aggregates trade data to account-level metrics.
        """
        # File Existence Check
        if not os.path.exists(sentiment_path):
            raise FileNotFoundError(f"Sentiment file not found at: {sentiment_path}")
        if not os.path.exists(trades_path):
            raise FileNotFoundError(f"Trades file not found at: {trades_path}")

        try:
            # 1. Load Data
            df_sent = pd.read_csv(sentiment_path)
            df_trades = pd.read_csv(trades_path)

            # 2. Date Standardization
            # Ensure 'Date' column exists or find closest match
            if 'Date' in df_sent.columns:
                df_sent['date_dt'] = pd.to_datetime(df_sent['Date'], errors='coerce')
            else:
                raise KeyError("Sentiment CSV missing 'Date' column")

            # Handle trade timestamps (assuming 'time' column exists)
            if 'time' in df_trades.columns:
                # Convert logic handles both unix and string formats
                df_trades['time'] = pd.to_datetime(df_trades['time'], errors='coerce')
                df_trades['date_dt'] = df_trades['time'].dt.floor('D')
            else:
                raise KeyError("Trades CSV missing 'time' column")

            # Drop rows where date conversion failed
            df_sent.dropna(subset=['date_dt'], inplace=True)
            df_trades.dropna(subset=['date_dt'], inplace=True)

            # 3. Aggregate Trades (Daily Metrics per Account)
            # We aggregate first to avoid row explosion during merge
            daily_metrics = df_trades.groupby(['date_dt', 'account']).agg({
                'closedPnL': 'sum',
                'leverage': 'mean',
                'size': 'sum',        # Total Volume
                'side': 'count'       # Trade Frequency
            }).reset_index()

            # Calculate Win Rate separately
            df_trades['is_win'] = (df_trades['closedPnL'] > 0).astype(int)
            win_rates = df_trades.groupby(['date_dt', 'account'])['is_win'].mean().reset_index()

            # Merge Aggregations
            df_features = pd.merge(daily_metrics, win_rates, on=['date_dt', 'account'])

            # 4. Merge with Market Sentiment
            # Left join ensures we keep trade data even if sentiment is missing for a day
            df_final = pd.merge(df_features, df_sent[['date_dt', 'value', 'value_classification']], 
                                on='date_dt', how='left')

            # Handle missing sentiment values (Forward Fill)
            df_final['value'] = df_final['value'].ffill()
            df_final['value_classification'] = df_final['value_classification'].ffill()
            
            # Final cleanup
            df_final.dropna(subset=['value'], inplace=True)
            
            return df_final

        except Exception as e:
            raise RuntimeError(f"Data processing failed: {str(e)}")
