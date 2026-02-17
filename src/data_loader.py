import pandas as pd
import numpy as np
import logging

# Configure robust logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    GOLD MASTER DATA ENGINE
    Strategy: Strict Type Enforcement & Aggressive Deduplication.
    """

    def _to_float(self, series):
        """Forces data to float, stripping symbols like '$' or ','."""
        return pd.to_numeric(series.astype(str).str.replace(r'[$,\s]', '', regex=True), errors='coerce').fillna(0.0)

    def load_and_process(self, sentiment_source, trades_source):
        try:
            logger.info("--- [ETL] Starting Data Pipeline ---")

            # ======================================================
            # PHASE 1: TRADES INGESTION (historical_data.csv)
            # ======================================================
            df_t = pd.read_csv(trades_source)
            df_t.columns = [c.strip() for c in df_t.columns] # Clean headers

            # 1. Map Columns (Specific to your file)
            # We map 'Timestamp' (Epoch) as priority, 'Timestamp IST' as fallback
            col_map = {
                'Timestamp': 'time_epoch',
                'Timestamp IST': 'time_str',
                'Account': 'account',
                'Closed PnL': 'pnl',
                'Size USD': 'size',
                'Side': 'side',
                'Leverage': 'leverage' # Might be missing, handled below
            }
            df_t.rename(columns=col_map, inplace=True)

            # 2. Date Parsing (The "Nuclear" Fix)
            # Priority: Scientific Notation Epoch (1.73E+12) -> Exact precision
            if 'time_epoch' in df_t.columns:
                # Coerce to numeric first to handle scientific notation strings
                epoch_vals = pd.to_numeric(df_t['time_epoch'], errors='coerce')
                df_t['date_dt'] = pd.to_datetime(epoch_vals, unit='ms', errors='coerce')
            
            # Fallback: String Parsing (02-12-2024)
            if 'date_dt' not in df_t.columns or df_t['date_dt'].isnull().all():
                logger.warning("Epoch failed, falling back to String Parsing")
                df_t['date_dt'] = pd.to_datetime(df_t['time_str'], dayfirst=True, errors='coerce')

            # Drop bad dates immediately
            df_t = df_t.dropna(subset=['date_dt'])
            
            # Normalize to Midnight (Critical for Merging)
            df_t['date_dt'] = df_t['date_dt'].dt.normalize()

            # 3. Numeric Cleaning & Self-Healing
            if 'leverage' not in df_t.columns:
                df_t['leverage'] = 1.0 # Default to Spot
            
            df_t['pnl'] = self._to_float(df_t['pnl'])
            df_t['size'] = self._to_float(df_t['size'])
            df_t['leverage'] = self._to_float(df_t['leverage'])
            df_t['is_win'] = (df_t['pnl'] > 0).astype(int)

            # 4. Aggregation (Prevents Duplicate Key Errors)
            # Collapse 1M trades into ~1000 daily summaries
            df_daily = df_t.groupby(['date_dt', 'account']).agg({
                'pnl': 'sum',
                'leverage': 'mean',
                'size': 'sum',
                'is_win': 'mean',
                'side': 'count'
            }).reset_index()

            # ======================================================
            # PHASE 2: SENTIMENT INGESTION (fear_greed_index.csv)
            # ======================================================
            df_s = pd.read_csv(sentiment_source)
            df_s.columns = [c.strip().lower() for c in df_s.columns]
            
            # Map headers
            s_map = {'date': 'date_dt', 'value': 'fng_val', 'classification': 'regime'}
            df_s.rename(columns=s_map, inplace=True)
            
            # Clean Dates
            df_s['date_dt'] = pd.to_datetime(df_s['date_dt'], errors='coerce')
            df_s = df_s.dropna(subset=['date_dt']).drop_duplicates('date_dt')

            # ======================================================
            # PHASE 3: FINAL MERGE & SAFETY
            # ======================================================
            # Left Join: Keep trade data even if sentiment is missing
            df_final = pd.merge(df_daily, df_s[['date_dt', 'fng_val', 'regime']], on='date_dt', how='left')

            # Impute missing sentiment
            df_final['fng_val'] = df_final['fng_val'].ffill().bfill().fillna(50)
            df_final['regime'] = df_final['regime'].ffill().bfill().fillna('Neutral')

            # Rename for App Compatibility
            df_final.rename(columns={
                'pnl': 'closedPnL', 
                'regime': 'value_classification', 
                'fng_val': 'value'
            }, inplace=True)

            # CRITICAL: ARROW CRASH FIX
            # Convert Date to String to prevent serialization errors in Streamlit
            df_final['date_str'] = df_final['date_dt'].dt.strftime('%Y-%m-%d')
            # Drop the complex datetime object to be safe, or keep it if needed for sorting
            # We keep 'date_dt' for sorting but ensuring it's clean
            
            logger.info(f"--- [ETL] Success. Rows: {len(df_final)} ---")
            return df_final

        except Exception as e:
            logger.error(f"FATAL PIPELINE ERROR: {str(e)}")
            raise RuntimeError(f"Data Processing Failed: {str(e)}")
