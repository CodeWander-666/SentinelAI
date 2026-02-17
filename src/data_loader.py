import pandas as pd
from src.cleaner import DataCleaner

class DataLoader:
    def _fuzzy_find(self, df, candidates):
        """Finds the first matching column from a list of candidates."""
        clean_cols = {str(c).strip().lower(): c for c in df.columns}
        for cand in candidates:
            if cand in clean_cols:
                return clean_cols[cand]
        return None

    def load_and_process(self, sentiment_source, trades_source, tracker):
        try:
            cleaner = DataCleaner(tracker)
            
            # --- PHASE 1: TRADES ---
            tracker.log("Ingesting Trades...", 10)
            df_t = pd.read_csv(trades_source)
            
            # 1. Smart Date Finding
            # We look for 'timestamp ist' first because it's higher precision string in your file
            t_col = self._fuzzy_find(df_t, ['timestamp ist', 'timestamp', 'time', 'date'])
            if not t_col: raise ValueError(f"No Date found in Trades. Cols: {list(df_t.columns)}")
            
            tracker.log(f"Trades: Locked onto '{t_col}'", 15)
            df_t.rename(columns={t_col: 'date_dt'}, inplace=True)

            # 2. Smart Mapping
            col_map = {
                'account': ['account', 'user'],
                'closedPnL': ['closed pnl', 'pnl', 'realized'],
                'size': ['size usd', 'size', 'volume'],
                'leverage': ['leverage', 'lev'], # Likely missing
                'side': ['side', 'direction']
            }
            
            for target, search_list in col_map.items():
                found = self._fuzzy_find(df_t, search_list)
                if found:
                    df_t.rename(columns={found: target}, inplace=True)
                elif target == 'leverage':
                    # SELF HEAL: Inject Leverage if missing
                    tracker.log("Injecting missing 'Leverage'...", 20)
                    df_t['leverage'] = 1.0

            # 3. Clean
            tracker.log("Cleaning Trades (Dedupe & Impute)...", 30)
            df_t = cleaner.clean_dataset(df_t)
            
            # 4. Aggregate (Daily)
            tracker.log("Aggregating High-Frequency Data...", 45)
            df_t['is_win'] = (df_t['closedPnL'] > 0).astype(int)
            df_daily = df_t.groupby([df_t['date_dt'].dt.normalize(), 'account']).agg({
                'closedPnL': 'sum', 'leverage': 'mean', 'size': 'sum',
                'is_win': 'mean', 'side': 'count'
            }).reset_index()
            # Rename back the normalized date
            df_daily.rename(columns={'date_dt': 'date_dt'}, inplace=True)

            # --- PHASE 2: SENTIMENT ---
            tracker.log("Ingesting Sentiment...", 60)
            df_s = pd.read_csv(sentiment_source)
            
            s_date = self._fuzzy_find(df_s, ['date', 'timestamp'])
            if not s_date: raise ValueError("No Date in Sentiment file")
            
            df_s.rename(columns={s_date: 'date_dt'}, inplace=True)
            
            s_map = {'value': ['value', 'fng'], 'value_classification': ['classification', 'label']}
            for target, search in s_map.items():
                found = self._fuzzy_find(df_s, search)
                if found: df_s.rename(columns={found: target}, inplace=True)

            df_s = cleaner.clean_dataset(df_s)
            
            # --- PHASE 3: MERGE ---
            tracker.log("Merging Datasets...", 80)
            # Ensure types match before merge
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
