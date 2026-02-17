import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class DataLoader:
    """
    DECOUPLED DATA ENGINE (V8.0)
    Strictly handles Sentiment vs Trades separately to prevent schema leakage.
    Self-heals missing 'leverage' and handles scientific notation (1.73E+12).
    """

    def _to_num(self, series):
        """Cleans strings like '$1,200.50' into floats."""
        return pd.to_numeric(series.astype(str).str.replace(r'[$,\s]', '', regex=True), errors='coerce').fillna(0.0)

    def load_and_process(self, sentiment_source, trades_source):
        try:
            # --- 1. PROCESS TRADES (historical_data.csv) ---
            df_t = pd.read_csv(trades_source)
            df_t.columns = [c.strip() for c in df_t.columns] # Remove accidental spaces
            
            # Map specific Hyperliquid headers to internal names
            t_map = {
                'Timestamp IST': 'time_str',
                'Timestamp': 'time_epoch',
                'Account': 'account',
                'Closed PnL': 'pnl',
                'Size USD': 'size',
                'Side': 'side',
                'Leverage': 'leverage'
            }
            df_t.rename(columns=t_map, inplace=True)

            # Date Parsing: Use Epoch (1.73E12) first for 100% accuracy
            if 'time_epoch' in df_t.columns:
                df_t['date_dt'] = pd.to_datetime(pd.to_numeric(df_t['time_epoch'], errors='coerce'), unit='ms', errors='coerce')
            
            # Fallback to IST String (02-12-2024)
            if 'date_dt' not in df_t.columns or df_t['date_dt'].isnull().all():
                df_t['date_dt'] = pd.to_datetime(df_t['time_str'], dayfirst=True, errors='coerce')

            df_t = df_t.dropna(subset=['date_dt'])
            df_t['date_dt'] = df_t['date_dt'].dt.normalize()

            # Self-Healing: If 'leverage' is missing from the CSV, create it
            if 'leverage' not in df_t.columns:
                df_t['leverage'] = 1.0
            
            # Clean numeric data
            df_t['pnl'] = self._to_num(df_t['pnl'])
            df_t['size'] = self._to_num(df_t['size'])
            df_t['leverage'] = self._to_num(df_t['leverage'])
            df_t['is_win'] = (df_t['pnl'] > 0).astype(int)

            # Grouping to prevent "Duplicate Key" merge errors
            df_daily_trades = df_t.groupby(['date_dt', 'account']).agg({
                'pnl': 'sum',
                'leverage': 'mean',
                'size': 'sum',
                'is_win': 'mean',
                'side': 'count'
            }).reset_index()

            # --- 2. PROCESS SENTIMENT (fear_greed_index.csv) ---
            df_s = pd.read_csv(sentiment_source)
            df_s.columns = [c.strip().lower() for c in df_s.columns]
            
            s_map = {'date': 'date_dt', 'value': 'fng_val', 'classification': 'regime'}
            df_s.rename(columns=s_map, inplace=True)
            
            df_s['date_dt'] = pd.to_datetime(df_s['date_dt'], errors='coerce')
            df_s = df_s.dropna(subset=['date_dt']).drop_duplicates('date_dt')
            df_s = df_s[['date_dt', 'fng_val', 'regime']]

            # --- 3. FINAL MERGE ---
            df_final = pd.merge(df_daily_trades, df_s, on='date_dt', how='left')

            # Impute missing values
            df_final['fng_val'] = df_final['fng_val'].ffill().bfill()
            df_final['regime'] = df_final['regime'].ffill().bfill().fillna('Neutral')

            # Rename for Dashboard compatibility
            df_final.rename(columns={'pnl': 'closedPnL', 'regime': 'value_classification', 'fng_val': 'value'}, inplace=True)
            
            return df_final

        except Exception as e:
            raise RuntimeError(f"Industrial Ingestion Failed: {str(e)}")
