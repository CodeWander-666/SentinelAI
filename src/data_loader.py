import pandas as pd
import numpy as np
import warnings

# Suppress annoying warnings for clean logs
warnings.simplefilter(action='ignore', category=FutureWarning)

class DataLoader:
    """
    FINAL PRODUCTION ENGINE
    Strategy:
    1. Prioritize 'Epoch' timestamps (Scientific Notation) for 100% precision.
    2. Fallback to 'Day-First' string parsing for Indian formats.
    3. Left Join to preserve ALL trades even if Sentiment is missing.
    """

    def _clean_numeric(self, df, col, default=0.0):
        """Forces column to float, handles currency symbols."""
        if col in df.columns:
            # Convert to string, strip symbols
            s = df[col].astype(str).str.replace(r'[$,\s]', '', regex=True)
            # Coerce to number
            df[col] = pd.to_numeric(s, errors='coerce').fillna(default)
        else:
            df[col] = default
        return df

    def load_and_process(self, sentiment_source, trades_source):
        try:
            print("--- [Loader] Starting Ingestion ---")
            
            # ==========================================
            # 1. LOAD TRADES (The Priority Dataset)
            # ==========================================
            df_t = pd.read_csv(trades_source)
            
            # NORMALIZE HEADERS (Trim & Lowercase)
            df_t.columns = [c.strip() for c in df_t.columns]
            
            # MAP COLUMNS (Your specific file headers)
            # We look for 'Timestamp' (Epoch) specifically as it's safer
            col_map = {
                'Timestamp IST': 'time_str', 
                'Timestamp': 'time_epoch', # The scientific notation column
                'Account': 'account',
                'Closed PnL': 'closedPnL',
                'Size USD': 'size',
                'Side': 'side',
                'Leverage': 'leverage'
            }
            df_t.rename(columns=col_map, inplace=True)
            
            # PARSE DATES (The Fix)
            # Priority 1: Use Epoch (Scientific Notation) if available
            if 'time_epoch' in df_t.columns:
                print("--- [Loader] Using Epoch Timestamp for Precision ---")
                # Coerce to float first to handle scientific notation
                epoch_series = pd.to_numeric(df_t['time_epoch'], errors='coerce')
                # Convert (Your data is likely Milliseconds 1.73E12)
                df_t['date_dt'] = pd.to_datetime(epoch_series, unit='ms', errors='coerce')
            
            # Priority 2: Use String (Timestamp IST)
            if 'date_dt' not in df_t.columns or df_t['date_dt'].isnull().all():
                if 'time_str' in df_t.columns:
                    print("--- [Loader] Fallback to String Parsing (Day First) ---")
                    df_t['date_dt'] = pd.to_datetime(df_t['time_str'], dayfirst=True, errors='coerce')
            
            # Drop bad dates
            initial_count = len(df_t)
            df_t = df_t.dropna(subset=['date_dt'])
            print(f"--- [Loader] Trades: Loaded {len(df_t)}/{initial_count} rows ---")

            # Clean Metrics
            df_t = self._clean_numeric(df_t, 'closedPnL', 0.0)
            df_t = self._clean_numeric(df_t, 'leverage', 1.0)
            df_t = self._clean_numeric(df_t, 'size', 0.0)
            
            # Normalize to Midnight (for merging)
            df_t['date_dt'] = df_t['date_dt'].dt.normalize()

            # AGGREGATE TRADES (Daily Summary)
            df_agg = df_t.groupby(['date_dt', 'account']).agg({
                'closedPnL': 'sum',
                'leverage': 'mean',
                'size': 'sum',
                'side': 'count'
            }).reset_index()
            
            # Add Win Rate
            df_t['is_win'] = (df_t['closedPnL'] > 0).astype(int)
            win_rates = df_t.groupby(['date_dt', 'account'])['is_win'].mean().reset_index()
            df_agg = pd.merge(df_agg, win_rates, on=['date_dt', 'account'])

            # ==========================================
            # 2. LOAD SENTIMENT
            # ==========================================
            df_s = pd.read_csv(sentiment_source)
            df_s.columns = [c.strip().lower() for c in df_s.columns]
            
            # Map headers
            s_map = {'date': 'date_dt', 'timestamp': 'date_dt', 'value': 'value', 'classification': 'value_classification'}
            df_s.rename(columns=s_map, inplace=True)
            
            # Parse Dates (Standard YYYY-MM-DD in your file)
            df_s['date_dt'] = pd.to_datetime(df_s['date_dt'], errors='coerce')
            df_s = df_s.dropna(subset=['date_dt'])
            
            # Filter columns
            df_s = df_s[['date_dt', 'value', 'value_classification']]

            # ==========================================
            # 3. MERGE (LEFT JOIN)
            # ==========================================
            # We use LEFT join so we NEVER lose trade data even if Sentiment is missing
            df_final = pd.merge(df_agg, df_s, on='date_dt', how='left')
            
            # Fill missing sentiment
            df_final['value'] = df_final['value'].ffill().bfill()
            df_final['value_classification'] = df_final['value_classification'].ffill().bfill().fillna('Neutral')

            print(f"--- [Loader] Final Dataset: {len(df_final)} rows ---")
            return df_final

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise RuntimeError(f"Data Pipeline Error: {e}")
            
