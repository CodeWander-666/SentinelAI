import polars as pl
import pandas as pd
import os
import io

class DataLoader:
    def load_and_process(self, sentiment_source, trades_source):
        """
        Hyper-fast data loader using Polars for 1GB+ CSVs.
        Accepts file paths (str) OR file objects (uploaded bytes).
        """
        try:
            # --- 1. Load Sentiment (Small file, can use standard load) ---
            # Polars handles both paths and file-like objects seamlessly
            if isinstance(sentiment_source, str):
                lf_sent = pl.scan_csv(sentiment_source)
            else:
                lf_sent = pl.read_csv(sentiment_source).lazy()

            # Process Sentiment: Rename Date -> date_dt, Cast types
            lf_sent = (
                lf_sent
                .with_columns(pl.col("Date").str.strptime(pl.Date, "%d-%m-%Y", strict=False).alias("date_dt"))
                .drop_nulls(subset=["date_dt"])
                .select(["date_dt", "value", "value_classification"])
            )

            # --- 2. Load Trades (THE HUGE FILE) ---
            # LazyFrame allows us to process 1GB+ without crashing RAM
            if isinstance(trades_source, str):
                lf_trades = pl.scan_csv(trades_source)
            else:
                lf_trades = pl.read_csv(trades_source).lazy()

            # Process Trades: Parse timestamps, floor to date, aggregate
            # We aggregate IN POLARS (Rust) before bringing data to Python
            lf_metrics = (
                lf_trades
                .with_columns(
                    pl.col("time").str.to_datetime(time_unit="us", strict=False).dt.date().alias("date_dt")
                )
                .drop_nulls(subset=["date_dt"])
                .group_by(["date_dt", "account"])
                .agg([
                    pl.col("closedPnL").sum().cast(pl.Float32),
                    pl.col("leverage").mean().cast(pl.Float32),
                    pl.col("size").sum().cast(pl.Float32),
                    pl.col("side").count().alias("trade_count"),
                    (pl.col("closedPnL") > 0).mean().alias("win_rate").cast(pl.Float32)
                ])
            )

            # --- 3. Merge (Join) ---
            # Left join trades with sentiment
            lf_final = lf_metrics.join(lf_sent, on="date_dt", how="left")

            # --- 4. Collect ---
            # Only NOW do we load data into RAM (but it's already aggregated and tiny!)
            df_final = lf_final.collect().to_pandas()

            # Final cleanup in Pandas (Date conversion for plotting)
            df_final['date_dt'] = pd.to_datetime(df_final['date_dt'])
            df_final['value'] = df_final['value'].ffill()
            df_final['value_classification'] = df_final['value_classification'].ffill()
            
            return df_final.dropna(subset=['value'])

        except Exception as e:
            # Fallback for debug
            raise RuntimeError(f"Polars Engine Error: {str(e)}")
