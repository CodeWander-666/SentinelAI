import pandas as pd
import numpy as np
import logging
from src.cleaner import DataCleaner

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    AUTO-FIXING DATA ENGINE
    This loader treats the incoming CSV as 'hostile' and forcibly cleans it 
    before the app tries to use it.
    """

    def _clean_header(self, col):
        """Turns 'Timestamp IST ' -> 'timestamp_ist'"""
        return str(col).strip().lower().replace(" ", "_")

    def load_and_process(self, sentiment_source, trades_source, tracker):
        try:
            cleaner = DataCleaner(tracker)
            
            # =========================================================
            # PHASE 1: TRADES (Deep Clean Mode)
            # =========================================================
            tracker.log("Ingesting Trades (Deep Clean Mode)...", 10)
            
            # 1. Load with flexible separator
            try:
                df_t = pd.read_csv(trades_source, sep=None, engine='python')
            except:
                df_t = pd.read_csv(trades_source)

            # 2. Normalize Headers immediately
            df_t.columns = [self._clean_header(c) for c in df_t.columns]
            
            # 3. FIX TIMESTAMP (The Critical Error Source)
            # We explicitly look for 'timestamp_ist' because we know your file has it
            if 'timestamp_ist' in df_t.columns:
                tracker.log("Found 'Timestamp IST'. Parsing...", 15)
                df_t['date_dt'] = pd.to_datetime(df_t['timestamp_ist'], dayfirst=True, errors='coerce')
            elif 'timestamp' in df_t.columns:
                # Handle Scientific Notation 1.73E+12
                tracker.log("Found 'Timestamp'. Converting Scientific Notation...", 15)
                # Force to float first, then to datetime
                df_t['date_dt'] = pd.to_datetime(pd.to_numeric(df_t['timestamp'], errors='coerce'), unit='ms', errors='coerce')
            else:
                # Last ditch search
                found = False
                for c in df_t.columns:
                    if 'date' in c or 'time' in c:
                        df_t['date_dt'] = pd.to_datetime(df_t[c], errors='coerce')
                        found = True
                        break
                if not found: raise ValueError("CRITICAL: No timestamp found in trades.")

            # Drop invalid dates
            df_t = df_t.dropna(subset=['date_dt'])
            df_t['date_dt'] = df_t['date_dt'].dt.normalize()

            # 4. FIX MISSING LEVERAGE
            # Your file DOES NOT have leverage. We must inject it or the dashboard crashes.
            if 'leverage' not in df_t.columns:
                tracker.log("⚠️ No Leverage found. Defaulting to 1.0", 20)
                df_t['leverage'] = 1.0
            
            # 5. Map Columns (Flexible Mapping)
            # We map specific dirty names to our clean internal names
            col_map = {
                'account': ['account', 'user', 'wallet'],
                'closedPnL': ['closed_pnl', 'pnl', 'realized_pnl', 'profit'],
                'size': ['size_usd', 'size', 'volume', 'amount'],
                'side': ['side', 'direction', 'type']
            }
            
            for target, aliases in col_map.items():
                if target in df_t.columns: continue # Already exists
                for alias in aliases:
                    if alias in df_t.columns:
                        df_t.rename(columns={alias: target}, inplace=True)
                        break

            # 6. Clean Numerics (Remove '$' and ',')
            for col in ['closedPnL', 'size', 'leverage']:
                if col in df_t.columns:
                    df_t[col] = pd.to_numeric(
                        df_t[col].astype(str).str.replace(r'[$,]', '', regex=True),
                        errors='coerce'
                    ).fillna(0.0)

            # AGGREGATE
            tracker.log("Aggregating Daily Data...", 40)
            df_t['is_win'] = (df_t['closedPnL'] > 0).astype(int)
            df_daily = df_t.groupby(['date_dt', 'account']).agg({
                'closedPnL': 'sum', 'leverage': 'mean', 'size': 'sum',
                'is_win': 'mean', 'side': 'count'
            }).reset_index()

            # =========================================================
            # PHASE 2: SENTIMENT (Auto-Detect Fix)
            # =========================================================
            tracker.log("Ingesting Sentiment...", 50)
            
            # 1. Load with flexible separator (Handles Tab vs Comma)
            try:
                df_s = pd.read_csv(sentiment_source, sep=None, engine='python')
            except:
                df_s = pd.read_csv(sentiment_source)
                
            df_s.columns = [self._clean_header(c) for c in df_s.columns]

            # 2. Find Date
            if 'date' in df_s.columns:
                df_s.rename(columns={'date': 'date_dt'}, inplace=True)
            elif 'timestamp' in df_s.columns:
                df_s.rename(columns={'timestamp': 'date_dt'}, inplace=True)
            else:
                raise ValueError(f"No Date in Sentiment. Found: {list(df_s.columns)}")

            # 3. Find Value/Class
            if 'value' in df_s.columns: df_s.rename(columns={'value': 'fng_val'}, inplace=True)
            if 'classification' in df_s.columns: df_s.rename(columns={'classification': 'regime'}, inplace=True)

            # 4. Clean Date
            df_s['date_dt'] = pd.to_datetime(df_s['date_dt'], errors='coerce')
            df_s = df_s.dropna(subset=['date_dt']).drop_duplicates('date_dt', keep='last')

            # =========================================================
            # PHASE 3: MERGE
            # =========================================================
            tracker.log("Merging...", 80)
            
            # Force types to match
            df_daily['date_dt'] = pd.to_datetime(df_daily['date_dt'])
            df_s['date_dt'] = pd.to_datetime(df_s['date_dt'])
            
            df_final = pd.merge(df_daily, df_s, on='date_dt', how='left')
            
            # Fill NAs
            df_final['fng_val'] = df_final['fng_val'].ffill().bfill().fillna(50)
            df_final['regime'] = df_final['regime'].ffill().bfill().fillna('Neutral')
            
            df_final.rename(columns={'regime': 'value_classification', 'fng_val': 'value'}, inplace=True)
            
            tracker.log("Success.", 100)
            return df_final

        except Exception as e:
            tracker.log(f"FATAL: {str(e)}", 0)
            raise e
