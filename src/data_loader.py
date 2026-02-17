import polars as pl
import pandas as pd
import tempfile
import csv
import os

class DataLoader:
    """
    Enterprise Data Engine (Decoupled Logic Edition).
    Handles Sentiment and Trade data separately to prevent schema conflicts.
    """

    def _detect_separator(self, file_path: str) -> str:
        """Sniffs file separator (CSV vs TSV)."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(2048)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample, delimiters=[',', '\t', ';', '|'])
                return dialect.delimiter
        except Exception:
            return ','

    def _get_lazy_frame(self, source) -> pl.LazyFrame:
        """Creates a Polars LazyFrame from path or uploaded file."""
        path_to_scan = source
        if not isinstance(source, str):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(source.getvalue())
                path_to_scan = tmp.name

        sep = self._detect_separator(path_to_scan)
        return pl.scan_csv(path_to_scan, separator=sep, ignore_errors=True, infer_schema_length=10000)

    def process_sentiment(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Specific logic for Sentiment Data.
        Expected: 'date', 'value', 'classification'
        """
        # 1. Normalize Column Names
        schema_map = {
            "date_str": ["date", "Date", "timestamp", "Timestamp"],
            "value": ["value", "Value", "fng_value"],
            "value_classification": ["classification", "Classification", "label", "class"]
        }
        
        # Apply Rename
        current_cols = lf.collect_schema().names()
        rename_map = {}
        for target, aliases in schema_map.items():
            match = next((col for col in aliases if col in current_cols), None)
            if match:
                rename_map[match] = target
        
        if rename_map:
            lf = lf.rename(rename_map)

        # 2. Parse Date (Prioritize 'date_str')
        lf = lf.with_columns(
            pl.coalesce([
                pl.col("date_str").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                pl.col("date_str").str.strptime(pl.Date, "%d-%m-%Y", strict=False),
                pl.from_epoch(pl.col("date_str").cast(pl.Int64), time_unit="s").dt.date()
            ]).alias("date_dt")
        )

        return lf.drop_nulls(subset=["date_dt"]).select(["date_dt", "value", "value_classification"])

    def process_trades(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Specific logic for Trade Data.
        Expected: 'Timestamp IST', 'Account', 'Closed PnL', 'Size USD'
        """
        # 1. Normalize Column Names
        schema_map = {
            "time_str": ["Timestamp IST", "date", "Date", "time", "Time"],
            "time_epoch": ["Timestamp", "timestamp"],
            "account": ["Account", "User ID", "account"],
            "closedPnL": ["Closed PnL", "Realized PnL", "closedPnL", "pnl"],
            "size": ["Size USD", "Size Tokens", "Volume", "size"],
            "side": ["Side", "Direction", "side"],
            "leverage": ["Leverage", "Lev", "leverage"]
        }

        current_cols = lf.collect_schema().names()
        rename_map = {}
        for target, aliases in schema_map.items():
            matches = [col for col in aliases if col in current_cols]
            if matches:
                rename_map[matches[0]] = target 
        
        if rename_map:
            lf = lf.rename(rename_map)

        # 2. Self-Healing: Add Leverage if missing (Spot Data)
        if "leverage" not in lf.collect_schema().names():
            lf = lf.with_columns(pl.lit(1.0).alias("leverage"))

        # 3. Parse Date (Robust Logic for your specific formats)
        time_exprs = []
        
        # Handle "02-12-2024 22:50"
        if "time_str" in lf.collect_schema().names():
            time_exprs.append(pl.col("time_str").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M", strict=False))
            time_exprs.append(pl.col("time_str").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S", strict=False))
        
        # Handle "1.73E+12" (Epoch MS)
        if "time_epoch" in lf.collect_schema().names():
            time_exprs.append(pl.from_epoch(pl.col("time_epoch").cast(pl.Int64), time_unit="ms"))

        if not time_exprs:
             # Fallback if no valid timestamp column found
             raise ValueError("No valid timestamp column found (Expected 'Timestamp IST' or 'Timestamp').")

        lf = lf.with_columns(
            pl.coalesce(time_exprs).dt.date().alias("date_dt")
        )

        # 4. Aggregate Daily Metrics
        return (
            lf.drop_nulls(subset=["date_dt"])
            .group_by(["date_dt", "account"])
            .agg([
                pl.col("closedPnL").cast(pl.Float32).sum().alias("closedPnL"),
                pl.col("leverage").cast(pl.Float32).mean().alias("leverage"),
                pl.col("size").cast(pl.Float32).sum().alias("size"),
                pl.col("side").count().alias("trade_count"),
                (pl.col("closedPnL") > 0).mean().cast(pl.Float32).alias("win_rate")
            ])
        )

    def load_and_process(self, sentiment_source, trades_source) -> pd.DataFrame:
        try:
            # 1. Ingest
            lf_sent = self._get_lazy_frame(sentiment_source)
            lf_trades = self._get_lazy_frame(trades_source)

            # 2. Process Separately (The Fix)
            lf_sent = self.process_sentiment(lf_sent)
            lf_trades = self.process_trades(lf_trades)

            # 3. Merge
            lf_final = lf_trades.join(lf_sent, on="date_dt", how="left")

            # 4. Materialize
            df_final = lf_final.collect().to_pandas()

            # 5. Post-Process
            df_final['date_dt'] = pd.to_datetime(df_final['date_dt'])
            df_final['value'] = df_final['value'].ffill()
            df_final['value_classification'] = df_final['value_classification'].ffill()
            
            return df_final.dropna(subset=['value'])

        except Exception as e:
            raise RuntimeError(f"Engine Failure: {str(e)}")
