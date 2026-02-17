import pandas as pd
import numpy as np
import io
import os
class DataLoader:
    """
    HEAVY ARMOR DATA ENGINE (Pandas Edition)
    Optimized for stability over raw speed. 
    Guaranteed to parse your specific date formats: '02-12-2024 22:50' and '2018-02-01'.
    """

    def _clean_numeric(self, df, col_name):
        """Forces a column to be numeric, turning errors (strings) into 0."""
        if col_name in df.columns:
            # Remove '$', ',' and whitespace
            df[col_name] = df[col_name].astype(str).str.replace(r'[$,]', '', regex=True)
            # Coerce to number (errors become NaN)
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
        return df

    def _parse_dates_pandas(self, series):
        """
        The 'Magic' Date Parser. 
        Uses Pandas flexible parsing which handles "02-12-2024" (Day First) correctly.
        """
        # 1. Try ISO / Standard / Mixed format (Slow but powerful)
        # dayfirst=True is CRITICAL for your '02-12-2024' format
        return pd.to_datetime(series, dayfirst=True, errors='coerce')

    def load_and_process(self, sentiment_source, trades_source):
        try:
            print("--- STARTING HEAVY ARMOR PIPELINE ---")
            
            # ---------------------------------------------------------
            # 1. INGEST SENTIMENT (Fear & Greed)
            # ---------------------------------------------------------
            # Handle both file path and Streamlit UploadedFile
            if hasattr(sentiment_source, 'getvalue'):
                df_sent = pd.read_csv(sentiment_source)
            else:
                df_sent = pd.read_csv(sentiment_source)

            # NORMALIZE COLUMNS (Sentiment)
            # Map whatever user has to -> [date_dt, value, value_classification]
            sent_cols = {c.lower(): c for c in df_sent.columns}
            
            # Find Date
            if 'date' in sent_cols: df_sent.rename(columns={sent_cols['date']: 'date_dt'}, inplace=True)
            elif 'timestamp' in sent_cols: df_sent.rename(columns={sent_cols['timestamp']: 'date_dt'}, inplace=True)
            
            # Find Value
            if 'value' in sent_cols: df_sent.rename(columns={sent_cols['value']: 'value'}, inplace=True)
            elif 'fng_value' in sent_cols: df_sent.rename(columns={sent_cols['fng_value']: 'value'}, inplace=True)
            
            # Find Class
            if 'classification' in sent_cols: df_sent.rename(columns={sent_cols['classification']: 'value_classification'}, inplace=True)
            elif 'label' in sent_cols: df_sent.rename(columns={sent_cols['label']: 'value_classification'}, inplace=True)

            # PARSE DATES (Sentiment)
            df_sent['date_dt'] = self._parse_dates_pandas(df_sent['date_dt'])
            # Drop invalid dates
            df_sent = df_sent.dropna(subset=['date_dt'])
            
            # Select only needed
            df_sent = df_sent[['date_dt', 'value', 'value_classification']].copy()

            # ---------------------------------------------------------
            # 2. INGEST TRADES (Historical Data)
            # ---------------------------------------------------------
            if hasattr(trades_source, 'getvalue'):
                df_trades = pd.read_csv(trades_source)
            else:
                df_trades = pd.read_csv(trades_source)

            # NORMALIZE COLUMNS (Trades)
            trade_cols = {c.lower(): c for c in df_trades.columns}
            
            # Mapping dictionary {Internal: [Possible External]}
            mapping = {
                'time_str': ['timestamp ist', 'date', 'time'], # Your specific column
                'account': ['account', 'user id'],
                'closedPnL': ['closed pnl', 'pnl', 'realized pnl'],
                'size': ['size usd', 'size', 'volume'],
                'leverage': ['leverage', 'lev'],
                'side': ['side', 'direction']
            }

            for internal, aliases in mapping.items():
                for alias in aliases:
                    if alias in trade_cols:
                        df_trades.rename(columns={trade_cols[alias]: internal}, inplace=True)
                        break
            
            # SELF HEALING: Add missing columns
            if 'leverage' not in df_trades.columns: df_trades['leverage'] = 1.0
            if 'size' not in df_trades.columns: df_trades['size'] = 0.0

            # CLEAN NUMERICS
            df_trades = self._clean_numeric(df_trades, 'closedPnL')
            df_trades = self._clean_numeric(df_trades, 'leverage')
            df_trades = self._clean_numeric(df_trades, 'size')

            # PARSE DATES (Trades)
            # We specifically look for 'time_str' (Timestamp IST) first
            if 'time_str' in df_trades.columns:
                df_trades['date_dt'] = self._parse_dates_pandas(df_trades['time_str'])
            elif 'timestamp' in df_trades.columns: 
                # Fallback to Epoch if string missing
                df_trades['date_dt'] = pd.to_datetime(df_trades['timestamp'], unit='ms', errors='coerce')
            else:
                raise ValueError("Could not find 'Timestamp IST' or 'Timestamp' column in Trades.")

            # Drop rows where date failed
            df_trades = df_trades.dropna(subset=['date_dt'])
            
            # Normalize to Day (remove time)
            df_trades['date_dt'] = df_trades['date_dt'].dt.normalize()

            # ---------------------------------------------------------
            # 3. AGGREGATE & MERGE
            # ---------------------------------------------------------
            
            # Group Trades by Day + Account
            df_metrics = df_trades.groupby(['date_dt', 'account']).agg({
                'closedPnL': 'sum',
                'leverage': 'mean',
                'size': 'sum',
                'side': 'count'
            }).reset_index()

            # Win Rate (Helper)
            df_trades['is_win'] = (df_trades['closedPnL'] > 0).astype(int)
            win_rates = df_trades.groupby(['date_dt', 'account'])['is_win'].mean().reset_index()
            
            # Merge Metrics
            df_final = pd.merge(df_metrics, win_rates, on=['date_dt', 'account'])

            # Merge with Sentiment
            # Use 'left' join to keep trades even if sentiment missing
            df_final = pd.merge(df_final, df_sent, on='date_dt', how='left')

            # Fill Missing Sentiment (Forward Fill)
            df_final['value'] = df_final['value'].ffill().bfill()
            df_final['value_classification'] = df_final['value_classification'].ffill().bfill()
            
            print("--- PIPELINE SUCCESS ---")
            return df_final

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"DATA LOADER ERROR: {str(e)}")
