import polars as pl
import pandas as pd
import os
import tempfile

class DataLoader:
    def standardize_columns(self, lf):
        """
        Auto-fixes column names (e.g., maps 'Timestamp' -> 'time').
        Prevents 'Column Not Found' errors.
        """
        # Map of {Internal_Name: [Possible_User_Names]}
        column_map = {
            "time": ["Timestamp", "Timestamp IST", "date", "Date", "time", "Time"],
            "account": ["Account", "User ID", "Wallet", "account", "User"],
            "closedPnL": ["Closed PnL", "Realized PnL", "pnl", "closedPnL", "Profit", "PnL"],
            "size": ["Size USD", "Volume", "Amount", "size", "Size", "Notional"],
            "side": ["Side", "Direction", "side", "Type"],
            "leverage": ["Leverage", "Lev", "leverage"]
        }
        
        # Get actual columns
        actual_cols = lf.columns
        rename_map = {}
        
        for internal, variants in column_map.items():
            match = next((col for col in variants if col in actual_cols), None)
            if match:
                rename_map[match] = internal
        
        # Apply renaming
        if rename_map:
            lf = lf.rename(rename_map)
            
        # SELF-HEALING: If leverage is missing (Spot data), add it as 1.0
        if "leverage" not in lf.columns and "leverage" not in rename_map.values():
            lf = lf.with_columns(pl.lit(1.0).alias("leverage"))
            
        return lf

    def _get_lazy_frame(self, source):
        """
        Returns a LazyFrame from either a path or an uploaded file object.
        Writes uploaded bytes to a temp file to enable 'scan_csv' (Low RAM usage).
        """
        if isinstance(source, str):
            # It's a file path
            return pl.scan_csv(source)
        else:
            # It's an uploaded file object (Streamlit)
            # Create a temp file to store it on disk, not RAM
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(source.getvalue())
                tmp_path = tmp.name
            return pl.scan_csv(tmp_path)

    def load_and_process(self, sentiment_source, trades_source):
        try:
            # --- 1. Load Sentiment ---
            lf_sent = self._get_lazy_frame(sentiment_source)

            # Clean Sentiment
            lf_sent = (
                lf_sent
                .with_columns(
                    pl.col("Date").str.strptime(pl.Date, "%d-%m-%Y", strict=False).alias("date_dt")
                )
                .drop_nulls(subset=["date_dt"])
                .select(["date_dt", "value", "value_classification"])
            )

            # --- 2. Load Trades ---
            lf_trades = self._get_lazy_frame(trades_source)

            # APPLY FIX: Standardize Columns
            lf_trades = self.standardize_columns(lf_trades)

            # Check for critical columns
            required = ["time", "account", "closedPnL", "size"]
            missing = [c for c in required if c not in lf_trades.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}. Found: {lf_trades.columns}")

            # Process Trades
            lf_metrics = (
                lf_trades
                .with_columns(
                    pl.coalesce([
                        pl.col("time").str.to_datetime(strict=False),
                        pl.col("time").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S", strict=False),
                        pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
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

            # --- 4. Collect ---
            df_final = lf_final.collect().to_pandas()
            
            # Final Polish
            df_final['date_dt'] = pd.to_datetime(df_final['date_dt'])
            df_final['value'] = df_final['value'].ffill()
            df_final['value_classification'] = df_final['value_classification'].ffill()
            
            return df_final.dropna(subset=['value'])

        except Exception as e:
            raise RuntimeError(f"Engine Error: {str(e)}")
