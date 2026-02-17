import pandas as pd
import numpy as np
from src.cleaner import DataCleaner

class DataLoader:
    def _clean_header(self, c):
        return str(c).strip().lower().replace(" ", "_")

    def load_and_process(self, sentiment_source, trades_source, tracker):
        try:
            cleaner = DataCleaner(tracker)
            
            # --- LOAD TRADES ---
            tracker.log("Ingesting Trades...", 10)
            try: df_t = pd.read_csv(trades_source, sep=None, engine='python')
            except: df_t = pd.read_csv(trades_source)
            
            df_t.columns = [self._clean_header(c) for c in df_t.columns]
            
            # Timestamp Logic
            if 'timestamp_ist' in df_t.columns:
                df_t['date_dt'] = pd.to_datetime(df_t['timestamp_ist'], dayfirst=True, errors='coerce')
            elif 'timestamp' in df_t.columns:
                df_t['date_dt'] = pd.to_datetime(pd.to_numeric(df_t['timestamp'], errors='coerce'), unit='ms', errors='coerce')
            else:
                # Fallback for 'date' or 'time'
                for c in df_t.columns:
                    if 'date' in c or 'time' in c:
                        df_t['date_dt'] = pd.to_datetime(df_t[c], errors='coerce')
                        break
            
            df_t = df_t.dropna(subset=['date_dt'])
            df_t['date_dt'] = df_t['date_dt'].dt.normalize()

            # Mapping
            col_map = {
                'account': ['account', 'user'], 'closedPnL': ['closed_pnl', 'pnl'],
                'size': ['size_usd', 'size'], 'leverage': ['leverage', 'lev'],
                'side': ['side', 'direction']
            }
            for target, aliases in col_map.items():
                for alias in aliases:
                    if alias in df_t.columns and target not in df_t.columns:
                        df_t.rename(columns={alias: target}, inplace=True)

            # Inject Leverage if missing
            if 'leverage' not in df_t.columns: df_t['leverage'] = 1.0
            
            df_t = cleaner.clean_financial_data(df_t)

            # --- PART A: CALCULATE METRICS ---
            tracker.log("Calculating Daily Metrics (Part A)...", 40)
            df_t['is_win'] = (df_t['closedPnL'] > 0).astype(int)
            df_t['is_long'] = (df_t['side'].astype(str).str.lower().str.contains('buy')).astype(int)
            
            # Aggregation
            df_daily = df_t.groupby(['date_dt', 'account']).agg({
                'closedPnL': 'sum',
                'leverage': 'mean',
                'size': 'sum',
                'is_win': 'mean',
                'side': 'count', # Trade Freq
                'is_long': 'mean' # Ratio close to 1 is Long, 0 is Short
            }).reset_index()
            
            df_daily.rename(columns={'side': 'trade_count', 'is_long': 'long_ratio'}, inplace=True)

            # --- LOAD SENTIMENT ---
            tracker.log("Ingesting Sentiment...", 60)
            try: df_s = pd.read_csv(sentiment_source, sep=None, engine='python')
            except: df_s = pd.read_csv(sentiment_source)
            
            df_s.columns = [self._clean_header(c) for c in df_s.columns]
            
            # Find Date
            if 'date' in df_s.columns: df_s.rename(columns={'date': 'date_dt'}, inplace=True)
            elif 'timestamp' in df_s.columns: df_s.rename(columns={'timestamp': 'date_dt'}, inplace=True)
            
            # Find Value
            if 'value' in df_s.columns: df_s.rename(columns={'value': 'fng_val'}, inplace=True)
            if 'classification' in df_s.columns: df_s.rename(columns={'classification': 'regime'}, inplace=True)

            df_s['date_dt'] = pd.to_datetime(df_s['date_dt'], errors='coerce')
            df_s = df_s.dropna(subset=['date_dt']).drop_duplicates('date_dt', keep='last')

            # --- MERGE ---
            tracker.log("Merging & Aligning...", 80)
            df_final = pd.merge(df_daily, df_s, on='date_dt', how='left')
            
            # Impute
            df_final['fng_val'] = df_final['fng_val'].ffill().bfill().fillna(50)
            df_final['regime'] = df_final['regime'].ffill().bfill().fillna('Neutral')
            
            df_final.rename(columns={'regime': 'value_classification', 'fng_val': 'value'}, inplace=True)
            
            tracker.log("Ready.", 100)
            return df_final

        except Exception as e:
            tracker.log(f"CRASH: {str(e)}", 0)
            raise e
