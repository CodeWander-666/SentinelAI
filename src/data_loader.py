import pandas as pd
import logging
from src.cleaner import AutomatedCleaner, StructuralFixer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def load_and_process(self, sentiment_source, trades_source):
        try:
            cleaner = AutomatedCleaner()
            
            # --- 1. LOAD & MAP TRADES ---
            df_t = pd.read_csv(trades_source)
            df_t.columns = [c.strip().lower() for c in df_t.columns]
            
            # Map raw headers to internal schema
            t_map = {
                'timestamp ist': 'date_dt', 'timestamp': 'date_dt',
                'account': 'account', 'closed pnl': 'closedPnL',
                'size usd': 'size', 'leverage': 'leverage', 'side': 'side'
            }
            # Fuzzy rename
            df_t.rename(columns=lambda x: t_map.get(x, x), inplace=True)
            
            # Apply Specialized Cleaning Classes
            df_t = StructuralFixer.standardize_date(df_t, 'date_dt')
            df_t = cleaner.clean_financial_data(df_t)
            
            # Ensure Leverage Exists (Validation)
            if 'leverage' not in df_t.columns: df_t['leverage'] = 1.0

            # --- 2. LOAD & MAP SENTIMENT ---
            df_s = pd.read_csv(sentiment_source)
            df_s.columns = [c.strip().lower() for c in df_s.columns]
            
            s_map = {'date': 'date_dt', 'value': 'fng_val', 'classification': 'regime'}
            df_s.rename(columns=lambda x: s_map.get(x, x), inplace=True)
            
            df_s = StructuralFixer.standardize_date(df_s, 'date_dt')
            # Deduplicate Sentiment (Keep latest per day)
            df_s = df_s.sort_values('date_dt').drop_duplicates('date_dt', keep='last')

            # --- 3. MERGE ---
            # Aggregate trades to daily to match sentiment frequency
            df_daily = df_t.groupby(['date_dt', 'account']).agg({
                'closedPnL': 'sum',
                'leverage': 'mean',
                'size': 'sum',
                'side': 'count'
            }).reset_index()
            
            df_final = pd.merge(df_daily, df_s, on='date_dt', how='left')
            
            # Final Imputation
            df_final['fng_val'] = df_final['fng_val'].ffill().bfill().fillna(50)
            df_final['regime'] = df_final['regime'].ffill().bfill().fillna("Neutral")
            
            # Final Rename
            df_final.rename(columns={'regime': 'value_classification', 'fng_val': 'value'}, inplace=True)
            
            # Type Casting for Arrow Stability
            df_final['is_win'] = (df_final['closedPnL'] > 0).astype(int)
            
            return df_final

        except Exception as e:
            raise RuntimeError(f"Pipeline Execution Failed: {str(e)}")
