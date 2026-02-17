import polars as pl
import pandas as pd
import io

class DataLoader:
    def standardize_columns(self, lf):
        """
        Smartly renames columns to match the internal engine requirements.
        Handles variations like 'Timestamp' vs 'time' or 'Closed PnL' vs 'closedPnL'.
        """
        # Map of {Internal_Name: [Possible_User_Names]}
        column_map = {
            "time": ["Timestamp", "Timestamp IST", "date", "Date", "time"],
            "account": ["Account", "User ID", "Wallet", "account"],
            "closedPnL": ["Closed PnL", "Realized PnL", "pnl", "closedPnL", "Profit"],
            "size": ["Size USD", "Volume", "Amount", "size", "Size"],
            "side": ["Side", "Direction", "side"],
            "leverage": ["Leverage", "Lev", "leverage"]
        }
        
        # Get actual columns in the dataset
        actual_cols = lf.columns
        rename_map = {}
        
        for internal, variants in column_map.items():
            # Find which variant exists in the user's file
            match = next((col for col in variants if col in actual_cols), None)
            if match:
                rename_map[match] = internal
            elif internal == "leverage":
                # Special Case: If leverage is missing (e.g. Spot data), fill with 1.0
                lf = lf.with_columns(pl.lit(1.0).alias("leverage"))

        # Apply renaming
        if rename_map:
            lf = lf.rename(rename_map)
            
        return lf

    def load_and_process(self, sentiment_source, trades_source):
        try:
            # --- 1. Load Sentiment ---
            if isinstance(sentiment_source, str):
                lf_sent = pl.scan_csv(sentiment_source)
            else:
                lf_sent = pl.read_csv(sentiment_source).lazy()

            # Clean Sentiment Columns
            lf_sent = (
                lf_sent
                .with_columns(pl.col("Date").str.strptime(pl.Date, "%d-%m-%Y", strict=False).alias("date_dt"))
                .drop_nulls(subset=["date_dt"])
                .select(["date_dt", "value", "value_classification"])
            )

            # --- 2. Load Trades (Robust Mode) ---
            if isinstance(trades_source, str):
                lf_trades = pl.scan_csv(trades_source)
            else:
                lf_trades = pl.read_csv(trades_source).lazy()

            # APPLY THE FIX: Rename user columns to standard names
            lf_trades = self.standardize_columns(lf_trades)

            # Debug: Verify critical columns exist after rename
            required = ["time", "account", "closedPnL", "size"]
            missing = [c for c in required if c not in lf_trades.columns]
            if missing:
                raise ValueError(f"Critical columns missing even after auto-fix: {missing}. Check your CSV headers.")

            # Process Trades: Parse timestamps (Handle 'Timestamp IST' or standard)
            # Try parsing multiple formats automatically
            lf_metrics = (
                lf_trades
                .with_columns(
                    pl.coalesce([
                        pl.col("time").str.to_datetime(strict=False),  # Try standard ISO
                        pl.col("time").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S", strict=False) # Try common format
                    ]).dt.date().alias("date_dt")
                )
                .drop_nulls(subset=["date_dt"])
                .group_by(["date_dt", "account"])
                .agg([
                    pl.col("closedPnL").cast(pl.Float32).sum().alias("closedPnL"),
                    pl.col("leverage").cast(pl.Float32).mean().alias("leverage"),
                    pl.col("size").cast(pl.Float32).sum().alias("size"),
                    pl.col("side").count().alias("trade_count"),
                    (pl.col("closedPnL") > 0).mean().cast(pl.Float32).alias("win_rate")
                ])
            )

            # --- 3. Merge ---
            lf_final = lf_metrics.join(lf_sent, on="date_dt", how="left")

            # --- 4. Collect to Pandas ---
            df_final = lf_final.collect().to_pandas()
            
            # Post-processing
            df_final['date_dt'] = pd.to_datetime(df_final['date_dt'])
            df_final['value'] = df_final['value'].ffill()
            df_final['value_classification'] = df_final['value_classification'].ffill()
            
            return df_final.dropna(subset=['value'])

        except Exception as e:
            raise RuntimeError(f"Engine Failure: {str(e)}")
