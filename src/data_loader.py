import pandas as pd
import numpy as np
from src.cleaner import DataCleaner

class DataLoader:
    def load_and_process(self, sentiment_source, trades_source, tracker):
        try:
            cleaner = DataCleaner(tracker)
            
            # --- PHASE 1: TRADES (0-40%) ---
            tracker.log("Reading Trades CSV (Large File)...", 5)
            df_t = pd.read_csv(trades_source)
            
            tracker.log("Normalizing Trade Headers...", 15)
            df_t.columns = [c.strip().lower() for c in df_t.columns]
            
            t_map = {
                'timestamp ist': 'date_dt', 'timestamp': 'date_dt',
                'account': 'account', 'closed pnl': 'closedPnL',
                'size usd': 'size', 'leverage': 'leverage', 'side': 'side'
            }
            df_t.rename(columns=lambda x: t_map.get(x, x), inplace=True)
            
            # Validate
            if 'date_dt' not in df_t.columns: raise ValueError("Missing 'Timestamp' in Trades")
            if 'leverage' not in df_t.columns: df_t['leverage'] = 1.0

            # Run Cleaning Suite on Trades
            df_t = cleaner.clean_financial_data(df_t)
            
            # Aggregate Daily (Optimization)
            tracker.log("Aggregating Daily Metrics...", 40)
            df_t['is_win'] = (df_t['closedPnL'] > 0).astype(int)
            df_daily = df_t.groupby(['date_dt', 'account']).agg({
                'closedPnL': 'sum', 'leverage': 'mean', 'size': 'sum', 
                'is_win': 'mean', 'side': 'count'
            }).reset_index()

            # --- PHASE 2: SENTIMENT (40-60%) ---
            tracker.log("Reading Sentiment Data...", 45)
            df_s = pd.read_csv(sentiment_source)
            df_s.columns = [c.strip().lower() for c in df_s.columns]
            
            s_map = {'date': 'date_dt', 'value': 'fng_val', 'classification': 'regime'}
            df_s.rename(columns=lambda x: s_map.get(x, x), inplace=True)
            
            # Basic clean for sentiment
            df_s['date_dt'] = pd.to_datetime(df_s['date_dt'], errors='coerce')
            df_s = df_s.dropna(subset=['date_dt']).drop_duplicates('date_dt')

            # --- PHASE 3: MERGE & FINALIZE (60-100%) ---
            tracker.log("Merging Datasets...", 60)
            df_final = pd.merge(df_daily, df_s, on='date_dt', how='left')
            
            tracker.log("Imputing Gaps...", 80)
            df_final['fng_val'] = df_final['fng_val'].ffill().bfill().fillna(50)
            df_final['regime'] = df_final['regime'].ffill().bfill().fillna("Neutral")
            
            df_final.rename(columns={'regime': 'value_classification', 'fng_val': 'value'}, inplace=True)
            
            tracker.log("Pipeline Complete. Rendering Dashboard.", 100)
            return df_final

        except Exception as e:
            tracker.log(f"CRITICAL ERROR: {str(e)}", 0)
            raise e
