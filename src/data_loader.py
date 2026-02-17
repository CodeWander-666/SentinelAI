import pandas as pd
from src.cleaner import DataCleaner

class DataLoader:
    def _fuzzy_find(self, df, candidates):
        """Finds the first matching column (Case-Insensitive)."""
        # Create a map: {clean_name: real_name}
        clean_cols = {str(c).strip().lower(): c for c in df.columns}
        
        for cand in candidates:
            cand_clean = str(cand).strip().lower()
            if cand_clean in clean_cols:
                return clean_cols[cand_clean]
        return None

    def load_and_process(self, sentiment_source, trades_source, tracker):
        try:
            cleaner = DataCleaner(tracker)
            
            # --- PHASE 1: TRADES ---
            tracker.log("Ingesting Trades...", 10)
            df_t = pd.read_csv(trades_source)
            
            # Normalization
            df_t.columns = [str(c).strip() for c in df_t.columns] # Clean spaces only
            
            # Smart Date Finding
            t_col = self._fuzzy_find(df_t, ['Timestamp IST', 'Timestamp', 'time', 'date'])
            if not t_col: raise ValueError(f"No Date found in Trades. Cols: {list(df_t.columns)}")
            
            tracker.log(f"Trades: Locked onto '{t_col}'", 15)
            df_t.rename(columns={t_col: 'date_dt'}, inplace=True)

            # Smart Mapping
            col_map = {
                'account': ['account', 'user'],
                'closedPnL': ['closed pnl', 'pnl'],
                'size': ['size usd', 'size'],
                'leverage': ['leverage', 'lev'],
                'side': ['side', 'direction']
            }
            
            for target, search_list in col_map.items():
                found = self._fuzzy_find(df_t, search_list)
                if found:
                    df_t.rename(columns={found: target}, inplace=True)
                elif target == 'leverage':
                    tracker.log("Injecting missing 'Leverage'...", 20)
                    df_t['leverage'] = 1.0

            # Clean
            tracker.log("Cleaning Trades...", 30)
            df_t = cleaner.clean_dataset(df_t)
            
            # Aggregate
            tracker.log("Aggregating High-Frequency Data...", 45)
            df_t['is_win'] = (df_t['closedPnL'] > 0).astype(int)
            # Use 'closedPnL' instead of 'pnl' as per map
            df_daily = df_t.groupby([df_t['date_dt'].dt.normalize(), 'account']).agg({
                'closedPnL': 'sum', 'leverage': 'mean', 'size': 'sum',
                'is_win': 'mean', 'side': 'count'
            }).reset_index()
            df_daily.rename(columns={'date_dt': 'date_dt'}, inplace=True)

            # --- PHASE 2: SENTIMENT (The Fix) ---
            tracker.log("Ingesting Sentiment...", 60)
            df_s = pd.read_csv(sentiment_source)
            
            # Diagnostic Log
            tracker.log(f"Sentiment Columns: {list(df_s.columns)}", 62)
            
            # Find Date: Look for 'date' OR 'timestamp'
            s_date = self._fuzzy_find(df_s, ['date', 'timestamp', 'time'])
            
            if s_date:
                tracker.log(f"Sentiment: Locked onto '{s_date}'", 65)
                df_s.rename(columns={s_date: 'date_dt'}, inplace=True)
            else:
                # If fuzzy fail, try manual override for known structure
                if 'timestamp' in df_s.columns: df_s.rename(columns={'timestamp': 'date_dt'}, inplace=True)
                elif 'date' in df_s.columns: df_s.rename(columns={'date': 'date_dt'}, inplace=True)
                else:
                    raise ValueError(f"FATAL: No Date in Sentiment file. Found: {list(df_s.columns)}")
            
            # Map Value/Class
            s_map = {'value': ['value', 'fng'], 'value_classification': ['classification', 'label']}
            for target, search in s_map.items():
                found = self._fuzzy_find(df_s, search)
                if found: df_s.rename(columns={found: target}, inplace=True)

            df_s = cleaner.clean_dataset(df_s)
            
            # --- PHASE 3: MERGE ---
            tracker.log("Merging Datasets...", 80)
            df_daily['date_dt'] = pd.to_datetime(df_daily['date_dt'])
            df_s['date_dt'] = pd.to_datetime(df_s['date_dt'])
            
            df_final = pd.merge(df_daily, df_s, on='date_dt', how='left')
            
            # Final Polish
            df_final['value'] = df_final['value'].ffill().bfill().fillna(50)
            df_final['value_classification'] = df_final['value_classification'].ffill().fillna('Neutral')
            
            tracker.log("Pipeline Success.", 100)
            return df_final

        except Exception as e:
            tracker.log(f"FATAL: {str(e)}", 0)
            raise e
