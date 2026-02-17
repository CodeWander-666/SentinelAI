import polars as pl
import pandas as pd
import tempfile
import csv
import os
import re

class DataLoader:
    """
    PRIME TRADE SENTINEL - INDUSTRIAL GRADE DATA ENGINE (V5.0)
    
    CAPABILITIES:
    1. Auto-detects CSV/TSV/Pipe delimiters.
    2. Maps 50+ column variations to strict internal schema.
    3. 'Nuclear' Date Parsing: Tries 7 different date formats + Epochs simultaneously.
    4. Safe Casting: Never crashes on type mismatch (str -> int), just handles it.
    5. string cleaning: Removes '$', ',', and whitespace from numeric columns.
    """

    # ------------------------------------------------------------------
    # 1. CORE UTILITIES
    # ------------------------------------------------------------------
    
    def _detect_separator(self, file_path: str) -> str:
        """Sniffs file separator (CSV vs TSV) with fallback."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(4096)
                sniffer = csv.Sniffer()
                # Prioritize Tab and Comma
                if "\t" in sample: return "\t"
                if "," in sample: return ","
                
                dialect = sniffer.sniff(sample, delimiters=[',', '\t', ';', '|'])
                return dialect.delimiter
        except Exception:
            return ','

    def _get_lazy_frame(self, source) -> pl.LazyFrame:
        """
        Creates a Polars LazyFrame from path or uploaded file.
        Writes buffers to disk to handle 1GB+ files with low RAM.
        """
        path_to_scan = source
        if not isinstance(source, str):
            # Streamlit UploadedFile -> Temp File
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(source.getvalue())
                path_to_scan = tmp.name

        sep = self._detect_separator(path_to_scan)
        
        # Robust Scan: All columns as String first to prevent Load Errors
        # We will cast them manually later.
        return pl.scan_csv(
            path_to_scan, 
            separator=sep, 
            ignore_errors=True, 
            infer_schema_length=0, # Force all strings
            missing_utf8_is_empty_string=True
        )

    def _normalize_column_names(self, lf: pl.LazyFrame, dataset_type: str) -> pl.LazyFrame:
        """
        Aggressive Column Mapper.
        dataset_type: 'sentiment' or 'trades'
        """
        current_cols = lf.collect_schema().names()
        
        # 1. CLEAN RAW HEADERS (Strip whitespace, lower case)
        # We create a map of {clean_name: original_name}
        clean_map = {c.strip().lower().replace(" ", "").replace("_", ""): c for c in current_cols}
        
        rename_map = {}

        # 2. DEFINE ALIASES
        if dataset_type == 'sentiment':
            target_map = {
                "date_col": ["date", "timestamp", "time", "day", "dt"],
                "value": ["value", "fng", "index", "score", "fngvalue"],
                "classification": ["class", "label", "sentiment", "group", "classification"]
            }
        else: # trades
            target_map = {
                "time_col": ["timestampist", "timestamp", "time", "date", "createdat", "filledat"],
                "account": ["account", "user", "wallet", "address", "userid", "subaccount"],
                "closedPnL": ["closedpnl", "realizedpnl", "pnl", "profit", "loss", "return"],
                "size": ["sizeusd", "sizetokens", "volume", "amount", "quantity", "notional", "size"],
                "side": ["side", "direction", "type", "ordertype"],
                "leverage": ["leverage", "lev", "margin", "mult"]
            }

        # 3. MAP
        for target, aliases in target_map.items():
            found = False
            for alias in aliases:
                # Check exact match in clean headers
                if alias in clean_map:
                    original_name = clean_map[alias]
                    rename_map[original_name] = target
                    found = True
                    break
            
            # If still not found, try partial match (dangerous but necessary for messy data)
            if not found:
                for alias in aliases:
                    for clean_col, orig_col in clean_map.items():
                        if alias in clean_col:
                            rename_map[orig_col] = target
                            found = True
                            break
                    if found: break

        return lf.rename(rename_map)

    # ------------------------------------------------------------------
    # 2. NUCLEAR DATE PARSER
    # ------------------------------------------------------------------
    
    def _parse_dates_robust(self, col_name: str) -> pl.Expr:
        """
        The 'Nuclear Option' for Date Parsing.
        Attempts 6 string formats AND Epoch conversion in a single Coalesce.
        Crucial: Uses strict=False to prevent crashes on bad formats.
        """
        c = pl.col(col_name)
        
        return pl.coalesce([
            # 1. Standard ISO (YYYY-MM-DD)
            c.str.to_date(strict=False),
            
            # 2. Common Euro/Indian Formats (DD-MM-YYYY)
            c.str.strptime(pl.Date, "%d-%m-%Y", strict=False),
            c.str.strptime(pl.Date, "%d/%m/%Y", strict=False),
            
            # 3. Datetime formats (DD-MM-YYYY HH:MM)
            c.str.strptime(pl.Datetime, "%d-%m-%Y %H:%M", strict=False).dt.date(),
            c.str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S", strict=False).dt.date(),
            c.str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).dt.date(),

            # 4. EPOCH Handling (The Crash Fix)
            # We first try to cast to Int64. If it fails (because it's "01-02-2018"), 
            # strict=False returns NULL instead of Crashing.
            pl.from_epoch(c.cast(pl.Int64, strict=False), time_unit="s").dt.date(),
            pl.from_epoch(c.cast(pl.Int64, strict=False), time_unit="ms").dt.date()
        ])

    def _clean_numeric(self, col_name: str) -> pl.Expr:
        """Removes '$', ',', and casts to Float32."""
        return (
            pl.col(col_name)
            .cast(pl.String)
            .str.replace_all(",", "")
            .str.replace_all("$", "")
            .cast(pl.Float32, strict=False) # Returns null on failure
            .fill_null(0.0)
        )

    # ------------------------------------------------------------------
    # 3. PIPELINE EXECUTION
    # ------------------------------------------------------------------

    def process_sentiment(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        lf = self._normalize_column_names(lf, 'sentiment')
        
        # Validate
        cols = lf.collect_schema().names()
        if "date_col" not in cols:
             # Last ditch effort: Look for 'timestamp' if logic missed it
             if "timestamp" in cols: lf = lf.rename({"timestamp": "date_col"})
             else: raise ValueError(f"Sentiment file missing Date column. Found: {cols}")

        return (
            lf
            .with_columns(
                self._parse_dates_robust("date_col").alias("date_dt")
            )
            .drop_nulls("date_dt")
            .select([
                pl.col("date_dt"),
                pl.col("value").cast(pl.Int32, strict=False).alias("value"),
                pl.col("classification").cast(pl.String).alias("value_classification")
            ])
        )

    def process_trades(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        lf = self._normalize_column_names(lf, 'trades')
        
        # Validate
        cols = lf.collect_schema().names()
        required = ["time_col", "closedPnL"]
        missing = [x for x in required if x not in cols]
        if missing:
            raise ValueError(f"Trades file missing critical columns: {missing}. Found: {cols}")

        # Add defaults
        if "leverage" not in cols:
            lf = lf.with_columns(pl.lit(1.0).alias("leverage"))
        if "size" not in cols:
            lf = lf.with_columns(pl.lit(0.0).alias("size"))
        if "side" not in cols:
             lf = lf.with_columns(pl.lit("Unknown").alias("side"))

        # Clean Numerics & Parse Date
        return (
            lf
            .with_columns([
                self._parse_dates_robust("time_col").alias("date_dt"),
                self._clean_numeric("closedPnL").alias("closedPnL"),
                self._clean_numeric("leverage").alias("leverage"),
                self._clean_numeric("size").alias("size")
            ])
            .drop_nulls("date_dt")
            .group_by(["date_dt", "account"])
            .agg([
                pl.col("closedPnL").sum(),
                pl.col("leverage").mean(),
                pl.col("size").sum(),
                pl.col("side").count().alias("trade_count"),
                (pl.col("closedPnL") > 0).mean().cast(pl.Float32).alias("win_rate")
            ])
        )

    def load_and_process(self, sentiment_source, trades_source) -> pd.DataFrame:
        try:
            # 1. Ingest Raw
            lf_sent_raw = self._get_lazy_frame(sentiment_source)
            lf_trades_raw = self._get_lazy_frame(trades_source)

            # 2. Process Isolated
            lf_sent_clean = self.process_sentiment(lf_sent_raw)
            lf_trades_clean = self.process_trades(lf_trades_raw)

            # 3. Merge
            lf_final = lf_trades_clean.join(lf_sent_clean, on="date_dt", how="left")

            # 4. Materialize
            df = lf_final.collect().to_pandas()

            # 5. Final Polish
            df['date_dt'] = pd.to_datetime(df['date_dt'])
            df['value'] = df['value'].ffill().bfill() # Fill gaps
            df['value_classification'] = df['value_classification'].ffill().bfill()
            
            return df

        except Exception as e:
            # Descriptive Error
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"PIPELINE CRASH: {str(e)}")
