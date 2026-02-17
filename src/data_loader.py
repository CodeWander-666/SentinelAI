import polars as pl
import pandas as pd
import tempfile
import csv
import os

class DataLoader:
    """
    Enterprise Data Ingestion Engine.
    Features:
    - Auto-detects delimiters (CSV, TSV, pipe-separated)
    - Normalizes column names (maps aliases to internal schema)
    - Self-heals missing critical columns (e.g., defaults Leverage to 1.0)
    - Lazy execution for large files (1GB+)
    """

    def _detect_separator(self, file_path: str) -> str:
        """
        Sniffs the file header to determine the correct delimiter.
        Fallback to comma if detection fails.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first 2k bytes to analyze structure
                sample = f.read(2048)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample, delimiters=[',', '\t', ';', '|'])
                return dialect.delimiter
        except Exception:
            return ','  # Default to CSV

    def _get_lazy_frame(self, source) -> pl.LazyFrame:
        """
        Creates a Polars LazyFrame from a path or file buffer.
        Handles separator detection automatically.
        """
        path_to_scan = source

        # Handle Streamlit UploadedFile object
        if not isinstance(source, str):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(source.getvalue())
                path_to_scan = tmp.name

        # Detect Separator
        separator = self._detect_separator(path_to_scan)

        # Robust Scan: Ignore parsing errors to prevent full crash on bad lines
        return pl.scan_csv(
            path_to_scan, 
            separator=separator, 
            ignore_errors=True, 
            infer_schema_length=10000,
            null_values=["NA", "null", "None", "-"]
        )

    def standardize_schema(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Maps diverse user column names to strict internal schema.
        """
        # Internal_Name : [List of possible aliases found in wild datasets]
        schema_map = {
            "time": ["Timestamp", "Timestamp IST", "date", "Date", "time", "Time", "timestamp (utc)", "fill_time"],
            "account": ["Account", "User ID", "Wallet", "account", "User", "subaccount"],
            "closedPnL": ["Closed PnL", "Realized PnL", "pnl", "closedPnL", "Profit", "PnL", "profit_loss"],
            "size": ["Size USD", "Volume", "Amount", "size", "Size", "Notional", "quantity"],
            "side": ["Side", "Direction", "side", "Type", "order_type"],
            "leverage": ["Leverage", "Lev", "leverage", "margin_mult"],
            "value": ["Value", "value", "fng_value", "fear_greed_index"],
            "classification": ["Classification", "class", "label", "sentiment_label"]
        }

        # 1. Fetch actual columns from the LazyFrame
        actual_cols = lf.collect_schema().names()
        rename_map = {}

        # 2. Build Rename Map
        for internal, aliases in schema_map.items():
            # Find the first alias that exists in the file
            match = next((col for col in aliases if col in actual_cols), None)
            if match:
                rename_map[match] = internal

        # 3. Apply Renaming
        if rename_map:
            lf = lf.rename(rename_map)

        # 4. Self-Healing: Inject defaults for missing optional columns
        current_cols = lf.collect_schema().names()
        
        # If 'leverage' is missing, assume Spot Trading (1x)
        if "leverage" not in current_cols:
            lf = lf.with_columns(pl.lit(1.0).alias("leverage"))
            
        return lf

    def load_and_process(self, sentiment_source, trades_source) -> pd.DataFrame:
        try:
            # --- PIPELINE STEP 1: INGEST SENTIMENT ---
            lf_sent = self._get_lazy_frame(sentiment_source)
            lf_sent = self.standardize_schema(lf_sent)

            # Smart Date Parsing for Sentiment
            # Tries multiple formats: DD-MM-YYYY, YYYY-MM-DD, Unix Timestamp
            lf_sent = (
                lf_sent
                .with_columns(
                    pl.coalesce([
                        pl.col("date").str.strptime(pl.Date, "%d-%m-%Y", strict=False),
                        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                        pl.from_epoch(pl.col("date").cast(pl.Int64), time_unit="s").dt.date() 
                    ]).alias("date_dt")
                )
                .drop_nulls(subset=["date_dt"])
                .select(["date_dt", "value", "classification"])
            )

            # --- PIPELINE STEP 2: INGEST TRADES ---
            lf_trades = self._get_lazy_frame(trades_source)
            lf_trades = self.standardize_schema(lf_trades)

            # Validation: Ensure critical columns exist after normalization
            required_cols = {"time", "closedPnL", "size"}
            available_cols = set(lf_trades.collect_schema().names())
            if not required_cols.issubset(available_cols):
                missing = required_cols - available_cols
                raise ValueError(f"CRITICAL ERROR: Could not find columns {missing} even after normalization. Check file headers.")

            # Smart Date Parsing for Trades
            lf_metrics = (
                lf_trades
                .with_columns(
                    pl.coalesce([
                        pl.col("time").str.to_datetime(strict=False), 
                        pl.col("time").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S", strict=False),
                        pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                        # Handle 'Timestamp IST' format if specific regex needed, or fallback
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

            # --- PIPELINE STEP 3: MERGE & MATERIALIZE ---
            # Left join ensures we calculate metrics even if sentiment is missing for that day
            lf_final = lf_metrics.join(lf_sent, on="date_dt", how="left")

            # Collect into Pandas DataFrame (Memory Efficient)
            df_final = lf_final.collect().to_pandas()

            # --- PIPELINE STEP 4: POST-PROCESS ---
            df_final['date_dt'] = pd.to_datetime(df_final['date_dt'])
            df_final['value'] = df_final['value'].ffill()
            df_final['value_classification'] = df_final['value_classification'].ffill()
            
            # Final rename to match analysis expectations
            df_final.rename(columns={'classification': 'value_classification'}, inplace=True)

            return df_final.dropna(subset=['value'])

        except Exception as e:
            # Re-raise with clean context
            raise RuntimeError(f"Data Pipeline Failure: {str(e)}")
