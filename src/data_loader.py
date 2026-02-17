import pandas as pd
import numpy as np
import logging
from src.cleaner import DataCleaner

class DataLoader:
    def _find_column(self, df, candidates):
        """
        Fuzzy searches for a column. Returns the first match found.
        """
        # 1. Exact Match Check
        for cand in candidates:
            if cand in df.columns:
                return cand
        
        # 2. Substring Match Check (e.g. 'date (utc)' contains 'date')
        for col in df.columns:
            for cand in candidates:
                if cand in col:
                    return col
        return None

    def load_and_process(self, sentiment_source, trades_source, tracker):
        try:
            cleaner = DataCleaner(tracker)
            
            # =================================================
            # PHASE 1: TRADES (Historical Data)
            # =================================================
            tracker.log("Reading Trades CSV...", 5)
            df_t = pd.read_csv(trades_source)
            
            # Normalization
            df_t.columns = [str(c).strip().lower() for c in df_t.columns]
            
            # Smart Date Mapping
            # Priority: 'timestamp' (often Epoch) > 'timestamp ist' (String)
            t_col = self._find_column(df_t, ['timestamp', 'time', 'date'])
            
            if t_col:
                tracker.log(f"Trades: Found Timestamp column '{t_col}'", 10)
                df_t.rename(columns={t_col: 'date_dt'}, inplace=True)
                
                # Cleanup: If we renamed one, ensure no other conflicting 'date_dt' exists
                # (Rare edge case where multiple date cols exist)
                df_t = df_t.loc[:, ~df_t.columns.duplicated()]
            else:
                # DIAGNOSIS TRIGGER
                cols_found = list(df_t.columns)
                tracker.log(f"DIAGNOSIS: Trades Columns Found -> {cols_found}", 0)
                raise ValueError("Trades CSV missing Date/Time column.")

            # Map remaining columns
            t_map = {
                'account': 'account', 'closed pnl': 'closedPnL',
                'size usd': 'size', 'leverage': 'leverage', 'side': 'side'
            }
            df_t.rename(columns=lambda x: t_map.get(x, x), inplace=True)
            
            # Clean
            if 'leverage' not in df_t.columns: df_t['leverage'] = 1.0
            df_t = cleaner.clean_financial_data(df_t)
            
            # Aggregate
            tracker.log("Aggregating Daily Metrics...", 40)
            df_t['is_win'] = (df_t['closedPnL'] > 0).astype(int)
            df_daily = df_t.groupby(['date_dt', 'account']).agg({
                'closedPnL': 'sum', 'leverage': 'mean', 'size': 'sum', 
                'is_win': 'mean', 'side': 'count'
            }).reset_index()

            # =================================================
            # PHASE 2: SENTIMENT (The Crash Zone)
            # =================================================
            tracker.log("Reading Sentiment Data...", 45)
            df_s = pd.read_csv(sentiment_source)
            df_s.columns = [str(c).strip().lower() for c in df_s.columns]
            
            # 1. FIND DATE COLUMN (The Fix)
            # Search for 'date', 'timestamp', 'day', 'time', 'dt'
            s_date_col = self._find_column(df_s, ['date', 'timestamp', 'time', 'day', 'dt'])
            
            if s_date_col:
                tracker.log(f"Sentiment: Found Date column '{s_date_col}'", 48)
                df_s.rename(columns={s_date_col: 'date_dt'}, inplace=True)
            else:
                # DIAGNOSIS TRIGGER: Show user exactly what is wrong
                cols_found = list(df_s.columns)
                tracker.log(f"DIAGNOSIS: Sentiment Columns Found -> {cols_found}", 48)
                raise ValueError(f"Could not identify Date column in Sentiment file. Found: {cols_found}")

            # 2. Map Value/Class
            s_map = {'value': 'fng_val', 'classification': 'regime'}
            df_s.rename(columns=lambda x: s_map.get(x, x), inplace=True)
            
            # 3. Clean
            df_s['date_dt'] = pd.to_datetime(df_s['date_dt'], errors='coerce')
            df_s = df_s.dropna(subset=['date_dt']).drop_duplicates('date_dt')

            # =================================================
            # PHASE 3: MERGE
            # =================================================
            tracker.log("Merging & Finalizing...", 60)
            df_final = pd.merge(df_daily, df_s, on='date_dt', how='left')
            
            df_final['fng_val'] = df_final['fng_val'].ffill().bfill().fillna(50)
            df_final['regime'] = df_final['regime'].ffill().bfill().fillna("Neutral")
            df_final.rename(columns={'regime': 'value_classification', 'fng_val': 'value'}, inplace=True)
            
            tracker.log("Analysis Complete.", 100)
            return df_final

        except Exception as e:
            tracker.log(f"CRITICAL ERROR: {str(e)}", 0)
            raise e
