"""
Data Loader Module for SentinelAI.
Loads and normalizes sentiment and trade CSV files using Polars.
Handles column name variations, date parsing, and data cleaning.
"""

import logging
from typing import Dict, List, Optional

import polars as pl

# Import our custom exceptions
from src.exceptions import DataLoadError, error_context

# Configure logger
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and preprocessing of sentiment and trade data.
    """

    def __init__(self):
        pass

    # -------------------- Public API --------------------

    def load_and_process(self, sentiment_path: str, trades_path: str) -> pl.DataFrame:
        """
        Load sentiment and trades, merge them, and return a single Polars DataFrame.
        Returns:
            Merged Polars DataFrame with columns: date_dt, value, value_classification,
            trade_price, trade_volume, etc.
        """
        try:
            with error_context("data pipeline"):
                logger.info("--- STARTING DATA PIPELINE (Polars) ---")

                # Load and normalize sentiment data
                df_sent = self._load_sentiment(sentiment_path)
                logger.info(f"Sentiment loaded: {df_sent.shape}")

                # Load and normalize trade data
                df_trades = self._load_trades(trades_path)
                logger.info(f"Trades loaded: {df_trades.shape}")

                # Merge on date_dt (inner join by default)
                df_merged = df_sent.join(df_trades, on="date_dt", how="inner")
                logger.info(f"Merged data shape: {df_merged.shape}")

                # Optional: sort by date
                df_merged = df_merged.sort("date_dt")

                return df_merged

        except Exception as e:
            logger.exception("Fatal error in data pipeline")
            raise DataLoadError(f"Data pipeline failed: {str(e)}") from e

    # -------------------- Private Loaders --------------------

    def _load_sentiment(self, path: str) -> pl.DataFrame:
        """
        Load sentiment CSV and standardize columns.
        Expected columns: timestamp, value, classification, date (or variations).
        """
        try:
            df = pl.read_csv(path, try_parse_dates=True)
            logger.info(f"Sentiment original columns: {df.columns}")

            # Define mapping: target -> list of possible source columns (in priority order)
            column_mapping = {
                "date_dt": ["timestamp", "date", "datetime", "time"],
                "value": ["value", "fng_value", "sentiment_value", "score"],
                "value_classification": ["classification", "label", "class", "sentiment_class"],
            }

            # Normalize columns to our standard names
            df = self._normalize_columns(df, column_mapping)

            # Check that required columns exist after normalization
            required = ["date_dt", "value", "value_classification"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise DataLoadError(f"Missing required sentiment columns after normalization: {missing}")

            # Parse dates (ensure datetime type)
            df = df.with_columns(self._parse_dates(pl.col("date_dt")))

            # Clean numeric columns (value might be string with % or other symbols)
            df = df.with_columns(self._clean_numeric(pl.col("value")).alias("value"))

            # Drop rows with null date_dt
            df = df.drop_nulls(subset=["date_dt"])

            return df

        except pl.PolarsError as e:
            raise DataLoadError(f"Failed to read sentiment CSV: {e}") from e

    def _load_trades(self, path: str) -> pl.DataFrame:
        """
        Load trades CSV and standardize columns.
        Expected columns: date, price, volume, etc.
        """
        try:
            df = pl.read_csv(path, try_parse_dates=True)
            logger.info(f"Trades original columns: {df.columns}")

            # Define mapping for trade data
            column_mapping = {
                "date_dt": ["date", "timestamp", "datetime", "trade_date"],
                "trade_price": ["price", "close", "trade_price", "value"],
                "trade_volume": ["volume", "vol", "trade_volume", "quantity"],
            }

            df = self._normalize_columns(df, column_mapping)

            required = ["date_dt", "trade_price"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise DataLoadError(f"Missing required trade columns after normalization: {missing}")

            # Parse date
            df = df.with_columns(self._parse_dates(pl.col("date_dt")))

            # Clean numeric columns
            if "trade_price" in df.columns:
                df = df.with_columns(self._clean_numeric(pl.col("trade_price")).alias("trade_price"))
            if "trade_volume" in df.columns:
                df = df.with_columns(self._clean_numeric(pl.col("trade_volume")).alias("trade_volume"))

            df = df.drop_nulls(subset=["date_dt"])

            return df

        except pl.PolarsError as e:
            raise DataLoadError(f"Failed to read trades CSV: {e}") from e

    # -------------------- Helper Methods --------------------

    def _normalize_columns(self, df: pl.DataFrame, mapping: Dict[str, List[str]]) -> pl.DataFrame:
        """
        Rename columns based on a mapping of target -> list of possible source names.
        Only the first existing source column for each target is used.
        """
        rename_dict = {}
        for target, sources in mapping.items():
            for src in sources:
                if src in df.columns:
                    rename_dict[src] = target
                    break  # Stop after first match

        if rename_dict:
            logger.info(f"Renaming columns: {rename_dict}")
            df = df.rename(rename_dict)
        else:
            logger.warning("No columns were renamed – none of the source patterns matched.")

        return df

    @staticmethod
    def _clean_numeric(series: pl.Expr) -> pl.Expr:
        """
        Clean numeric columns: remove non-numeric characters and cast to float.
        """
        # Remove common non-numeric characters (%, $, commas)
        return (
            series.cast(pl.Utf8)
            .str.replace_all(r"[^\d.-]", "")
            .cast(pl.Float64)
            .alias("cleaned")
        )

    @staticmethod
    def _parse_dates(date_expr: pl.Expr) -> pl.Expr:
        """
        Parse date columns to Polars datetime type.
        Handles multiple formats gracefully.
        """
        return (
            date_exstr.cast(pl.Utf8)
            .str.to_datetime(
                format=None,  # let Polars infer
                strict=False,  # don't fail on errors
                time_unit="us"
            )
        )
