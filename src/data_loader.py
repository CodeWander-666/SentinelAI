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
            # 1. Clean Headers
            df_t.columns = [c.strip().lower() for c in df_t.columns]
            
            # 2. COLLISION AVOIDANCE (The Fix)
            # We check which columns exist and pick the BEST one to be 'date_dt'
            # Priority: 'timestamp' (Epoch) > 'timestamp ist' (String)
            
            if 'timestamp' in df_t.columns:
                tracker.log("Detected Precision Timestamp (Epoch). Using primary.", 18)
                df_t.rename(columns={'timestamp': 'date_dt'}, inplace=True)
                # Drop conflicting column if it exists
                if 'timestamp ist' in df_t.columns:
                    df_t.drop(columns=['timestamp ist'], inplace=True)
            elif 'timestamp ist' in df_t.columns:
                tracker.log("Detected Standard Timestamp. Using secondary.", 18)
                df_t.rename(columns={'timestamp ist': 'date_dt'}, inplace=True)
            
            # 3. Rename remaining columns safely
            t_map = {
                'account': 'account', 
                'closed pnl': 'closedPnL',
                'size usd': 'size', 
                'leverage': 'leverage', 
                'side': 'side'
            }
            # Only rename if the column actually exists (avoid key errors)
            df_t.rename(columns=lambda x: t_map.get(x, x), inplace=True)
            
            # Validate Critical Columns
            if 'date_dt' not in df_t.columns: 
                raise ValueError(f"CRITICAL: No timestamp column found. Headers: {list(df_t.columns)}")
            
            # Self-Heal Leverage
            if 'leverage' not in df_t.columns: 
                tracker.log("Missing Leverage column. Injecting defaults.", 20)
                df_t['leverage'] = 1.0

            # Run Cleaning Suite
            df_t = cleaner.clean_financial_data(df_t)
            
            # Aggregate Daily
            tracker.log("Aggregating Daily Metrics...", 40)
            df_t['is_win'] = (df_t['closedPnL'] > 0).astype(int)
            
            # Group by safely
            df_daily = df_t.groupby(['date_dt', 'account']).agg({
                'closedPnL': 'sum', 
                'leverage': 'mean', 
                'size': 'sum', 
                'is_win': 'mean', 
                'side': 'count'
            }).reset_index()

            # --- PHASE 2: SENTIMENT (40-60%) ---
            tracker.log("Reading Sentiment Data...", 45)
            df_s = pd.read_csv(sentiment_source)
            df_s.columns = [c.strip().lower() for c in df_s.columns]
            
            # Collision Check for Sentiment too
            if 'date' in df_s.columns:
                df_s.rename(columns={'date': 'date_dt'}, inplace=True)
            elif 'timestamp' in df_s.columns:
                df_s.rename(columns={'timestamp': 'date_dt'}, inplace=True)
                
            s_map = {'value': 'fng_val', 'classification': 'regime'}
            df_s.rename(columns=lambda x: s_map.get(x, x), inplace=True)
            
            # Structural Clean
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
            # Re-raise to stop execution flow
            raise e
