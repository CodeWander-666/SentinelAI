import pandas as pd
import numpy as np
import logging
import streamlit as st

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    DIAGNOSTIC DATA ENGINE (Pandas Core)
    1. Auto-detects columns using "Fuzzy Matching" (Case-insensitive).
    2. detailed 'diagnose_file' function to show you exactly what is wrong.
    3. Handles 1.73E+12 (Scientific Notation) and '02-12-2024' (Day-First) dates.
    """

    def _clean_header(self, col_name):
        """Turns 'Timestamp IST ' into 'timestamp_ist'."""
        return str(col_name).strip().lower().replace(" ", "_")

    def diagnose_file(self, df, name):
        """
        SELF-DIAGNOSIS: Returns a report of what columns were actually found.
        """
        report = {
            "file": name,
            "rows": len(df),
            "columns_found": list(df.columns),
            "sample_date": df.iloc[0].to_dict() if not df.empty else "EMPTY"
        }
        logger.info(f"DIAGNOSIS [{name}]: {report}")
        return report

    def load_and_process(self, sentiment_source, trades_source):
        try:
            # =========================================================
            # PHASE 1: LOAD & DIAGNOSE TRADES
            # =========================================================
            df_t = pd.read_csv(trades_source)
            # DIAGNOSIS STEP
            self.diagnose_file(df_t, "Trades_Raw")
            
            # 1. Normalize Headers
            # We create a map of {clean_name: original_name} to find columns safely
            clean_map = {self._clean_header(c): c for c in df_t.columns}
            
            # 2. Smart Column Mapping (Looks for keys in clean_map)
            # format: 'internal_name': ['possible_external_name1', 'possible_external_name2']
            target_cols = {
                'time_str': ['timestamp_ist', 'time', 'date', 'created_time'],
                'time_epoch': ['timestamp', 'epoch', 'ts'],
                'account': ['account', 'user', 'wallet'],
                'pnl': ['closed_pnl', 'pnl', 'realized_pnl', 'profit'],
                'size': ['size_usd', 'size', 'amount', 'volume'],
                'leverage': ['leverage', 'lev', 'margin'],
                'side': ['side', 'direction']
            }

            rename_map = {}
            for target, search_list in target_cols.items():
                for search_term in search_list:
                    if search_term in clean_map:
                        original = clean_map[search_term]
                        rename_map[original] = target
                        break # Stop looking for this target once found
            
            df_t.rename(columns=rename_map, inplace=True)
            
            # 3. Validation
            if 'pnl' not in df_t.columns:
                raise ValueError(f"CRITICAL: Could not find PnL column. Found: {df_t.columns.tolist()}")

            # 4. Date Parsing (The "Safe" Method)
            # Priority: Epoch (Scientific Notation)
            if 'time_epoch' in df_t.columns:
                # Coerce to float to handle "1.73E+12" strings
                epoch_series = pd.to_numeric(df_t['time_epoch'], errors='coerce')
                df_t['date_dt'] = pd.to_datetime(epoch_series, unit='ms', errors='coerce')
            
            # Fallback: String (Day-First)
            if 'date_dt' not in df_t.columns or df_t['date_dt'].isnull().all():
                if 'time_str' in df_t.columns:
                    df_t['date_dt'] = pd.to_datetime(df_t['time_str'], dayfirst=True, errors='coerce')
                else:
                    raise ValueError("CRITICAL: No Date/Timestamp column found.")

            df_t = df_t.dropna(subset=['date_dt'])
            
            # 5. Clean Numerics
            for col in ['pnl', 'size', 'leverage']:
                if col in df_t.columns:
                    df_t[col] = pd.to_numeric(df_t[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0.0)
                elif col == 'leverage':
                    df_t['leverage'] = 1.0 # Self-Heal missing leverage

            # 6. Aggregation (Fixes Duplicate Keys)
            df_t['date_dt'] = df_t['date_dt'].dt.normalize()
            df_t['is_win'] = (df_t['pnl'] > 0).astype(int)
            
            df_daily = df_t.groupby(['date_dt', 'account']).agg({
                'pnl': 'sum',
                'leverage': 'mean',
                'size': 'sum',
                'is_win': 'mean',
                'side': 'count'
            }).reset_index()

            # =========================================================
            # PHASE 2: SENTIMENT
            # =========================================================
            df_s = pd.read_csv(sentiment_source)
            self.diagnose_file(df_s, "Sentiment_Raw")
            
            # Normalize headers
            df_s.columns = [self._clean_header(c) for c in df_s.columns]
            
            # Simple Rename
            s_map = {'date': 'date_dt', 'timestamp': 'date_dt', 'value': 'fng_val', 'classification': 'regime'}
            df_s.rename(columns=s_map, inplace=True)
            
            # Parse Date
            df_s['date_dt'] = pd.to_datetime(df_s['date_dt'], errors='coerce')
            df_s = df_s.dropna(subset=['date_dt']).drop_duplicates('date_dt')

            # =========================================================
            # PHASE 3: MERGE
            # =========================================================
            df_final = pd.merge(df_daily, df_s[['date_dt', 'fng_val', 'regime']], on='date_dt', how='left')
            
            # Impute
            df_final['fng_val'] = df_final['fng_val'].ffill().bfill().fillna(50)
            df_final['regime'] = df_final['regime'].ffill().bfill().fillna('Neutral')
            
            # Renaming for App
            df_final.rename(columns={'pnl': 'closedPnL', 'regime': 'value_classification', 'fng_val': 'value'}, inplace=True)
            
            # ARROW FIX: Ensure Date is standard datetime64[ns]
            df_final['date_dt'] = df_final['date_dt'].astype('datetime64[ns]')
            
            return df_final

        except Exception as e:
            # Capture the full traceback for the UI
            import traceback
            st.error(f"PIPELINE CRASHED: {e}")
            st.code(traceback.format_exc())
            raise RuntimeError(f"Engine Failed: {e}")
