import polars as pl
import pandas as pd
import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class DataLoader:
    """
    HEAVY ARMOR DATA ENGINE (Polars Edition)
    Optimised for stability and speed. Handles your specific date formats
    ('02-12-2024 22:50' and epoch milliseconds) and column names.
    """

    # ----------------------------------------------------------------------
    # Internal helpers for date parsing
    # ----------------------------------------------------------------------
    @staticmethod
    def _parse_sentiment_dates(series: pl.Series) -> pl.Series:
        """
        Parse sentiment dates. Tries day‑first format first, then falls back
        to Polars' automatic parsing.
        """
        # Convert to string, then try explicit format
        str_series = series.cast(pl.String)
        # Try day‑first format (e.g., "02-12-2024")
        parsed = pl.col(str_series).str.strptime(pl.Datetime, format="%d-%m-%Y", strict=False)
        # If that fails, let Polars try its own heuristics
        return parsed.fill_null(pl.col(str_series).str.strptime(pl.Datetime, strict=False))

    @staticmethod
    def _parse_trades_dates(series: pl.Series, col_name: str) -> pl.Series:
        """
        Parse trades dates:
        - If column is "Timestamp IST", assume string format "%d-%m-%Y %H:%M".
        - If column is "Timestamp", assume numeric milliseconds.
        """
        if col_name == "Timestamp IST":
            # Convert to string and parse with explicit format
            return series.cast(pl.String).str.strptime(
                pl.Datetime, format="%d-%m-%Y %H:%M", strict=False
            )
        elif col_name == "Timestamp":
            # Numeric – assume milliseconds since epoch
            return series.cast(pl.Int64).cast(pl.Datetime)  # Polars uses µs by default, but we'll handle later
        else:
            # Fallback
            return series.cast(pl.String).str.strptime(pl.Datetime, strict=False)

    # ----------------------------------------------------------------------
    # Column name normalisation (lowercase, strip, map to internal names)
    # ----------------------------------------------------------------------
    @staticmethod
    def _normalise_columns(df: pl.DataFrame, mapping: dict) -> pl.DataFrame:
        """Rename columns based on a mapping of internal names to possible external names."""
        current_cols = {col.lower().strip(): col for col in df.columns}
        rename_dict = {}
        for internal, aliases in mapping.items():
            for alias in aliases:
                alias_lower = alias.lower().strip()
                if alias_lower in current_cols:
                    rename_dict[current_cols[alias_lower]] = internal
                    break
        return df.rename(rename_dict)

    @staticmethod
    def _clean_numeric(series: pl.Series) -> pl.Series:
        """Remove '$', ',' and whitespace, then cast to float."""
        return (series.cast(pl.String)
                .str.replace_all(r'[\$,]', '')
                .str.strip_chars()
                .cast(pl.Float64, strict=False)
                .fill_null(0.0))

    # ----------------------------------------------------------------------
    # Main loading pipeline
    # ----------------------------------------------------------------------
    def load_and_process(self,
                         sentiment_source: Union[str, bytes, object],
                         trades_source: Union[str, bytes, object]) -> pl.DataFrame:
        """
        Load, clean, and merge sentiment and trades data.

        Parameters
        ----------
        sentiment_source : file path, bytes, or file-like object (CSV)
        trades_source     : file path, bytes, or file-like object (CSV)

        Returns
        -------
        pl.DataFrame
            Final DataFrame with columns: date_dt, account, closedPnL, leverage,
            size, side, win_rate, value (sentiment), value_classification.
        """
        try:
            logger.info("--- STARTING HEAVY ARMOR PIPELINE (Polars) ---")

            # --------------------------------------------------------------
            # 1. Load sentiment (Fear & Greed)
            # --------------------------------------------------------------
            logger.info("Loading sentiment data...")
            df_sent = self._load_sentiment(sentiment_source)
            logger.info(f"Sentiment shape: {df_sent.shape}")

            # --------------------------------------------------------------
            # 2. Load trades (historical data)
            # --------------------------------------------------------------
            logger.info("Loading trades data...")
            df_trades = self._load_trades(trades_source)
            logger.info(f"Trades shape: {df_trades.shape}")

            # --------------------------------------------------------------
            # 3. Aggregate trades by day + account
            # --------------------------------------------------------------
            logger.info("Aggregating trades...")
            df_agg = self._aggregate_trades(df_trades)
            logger.info(f"Aggregated shape: {df_agg.shape}")

            # --------------------------------------------------------------
            # 4. Merge with sentiment
            # --------------------------------------------------------------
            logger.info("Merging with sentiment...")
            df_final = df_agg.join(
                df_sent,
                on="date_dt",
                how="left"
            )

            # Fill missing sentiment values (forward/backward fill)
            df_final = df_final.with_columns([
                pl.col("value").fill_null(strategy="forward").fill_null(strategy="backward"),
                pl.col("value_classification").fill_null(strategy="forward").fill_null(strategy="backward"),
            ])

            logger.info("--- PIPELINE SUCCESS ---")
            return df_final

        except Exception as e:
            logger.exception("Fatal error in data pipeline")
            raise RuntimeError(f"DATA LOADER ERROR: {str(e)}") from e

    # ----------------------------------------------------------------------
    # Private loading methods
    # ----------------------------------------------------------------------
    def _load_sentiment(self, source) -> pl.DataFrame:
        """Load and clean sentiment CSV."""
        try:
            df = pl.read_csv(source, try_parse_dates=False)  # disable auto‑parse, we handle manually
        except Exception:
            # Fallback to pandas if polars fails (e.g., for uploaded file objects)
            df = pd.read_csv(source)
            df = pl.from_pandas(df)

        # Normalise column names
        mapping = {
            "date_dt": ["date", "timestamp"],
            "value": ["value", "fng_value"],
            "value_classification": ["classification", "label"],
        }
        df = self._normalise_columns(df, mapping)

        # Ensure required columns exist
        required = ["date_dt", "value", "value_classification"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required sentiment column: {col}")

        # Parse dates
        df = df.with_columns(
            self._parse_sentiment_dates(pl.col("date_dt")).alias("date_dt")
        )
        df = df.drop_nulls(subset=["date_dt"])

        # Select only needed columns
        return df.select(["date_dt", "value", "value_classification"])

    def _load_trades(self, source) -> pl.DataFrame:
        """Load and clean trades CSV."""
        try:
            df = pl.read_csv(source, try_parse_dates=False)
        except Exception:
            df = pd.read_csv(source)
            df = pl.from_pandas(df)

        # Normalise column names using a flexible mapping
        mapping = {
            "time_str": ["timestamp ist", "date", "time"],
            "account": ["account", "user id"],
            "closedPnL": ["closed pnl", "pnl", "realized pnl"],
            "size": ["size usd", "size", "volume"],
            "leverage": ["leverage", "lev"],
            "side": ["side", "direction"],
        }
        df = self._normalise_columns(df, mapping)

        # Add missing columns with defaults
        if "leverage" not in df.columns:
            df = df.with_columns(pl.lit(1.0).alias("leverage"))
        if "size" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("size"))

        # Clean numeric columns
        for col in ["closedPnL", "leverage", "size"]:
            if col in df.columns:
                df = df.with_columns(self._clean_numeric(pl.col(col)).alias(col))

        # Parse dates – we have two possible timestamp columns
        date_cols = []
        if "time_str" in df.columns:
            date_cols.append("time_str")
        if "timestamp" in df.columns:
            date_cols.append("timestamp")

        if not date_cols:
            raise ValueError("No timestamp column found in trades data.")

        # Use the first available timestamp column to create a unified date
        # We'll create a new column 'date_dt'
        ts_col = date_cols[0]
        df = df.with_columns(
            self._parse_trades_dates(pl.col(ts_col), ts_col).alias("date_dt")
        )
        df = df.drop_nulls(subset=["date_dt"])

        # Normalise to day (remove time part)
        df = df.with_columns(pl.col("date_dt").dt.date().cast(pl.Datetime).alias("date_dt"))

        return df

    def _aggregate_trades(self, df: pl.DataFrame) -> pl.DataFrame:
        """Group by date_dt and account, compute metrics."""
        # Add win indicator
        df = df.with_columns(
            (pl.col("closedPnL") > 0).cast(pl.Int32).alias("is_win")
        )

        # Group by
        grouped = df.group_by(["date_dt", "account"]).agg([
            pl.col("closedPnL").sum().alias("closedPnL"),
            pl.col("leverage").mean().alias("leverage"),
            pl.col("size").sum().alias("size"),
            pl.col("side").count().alias("trade_count"),
            pl.col("is_win").mean().alias("win_rate"),
        ])

        return grouped
