"""
Professional data handler for large historical trade datasets.
Handles CSV loading, cleaning, type conversion, feature engineering, and validation.
Uses Polars for performance and memory efficiency.
"""

import logging
import re
from typing import Optional, Dict, List, Any, Union
from pathlib import Path

import polars as pl

from src.exceptions import DataLoadError, error_context

logger = logging.getLogger(__name__)


class HistoricalDataHandler:
    """
    High-performance handler for loading and cleaning historical trade data.
    Supports large files, automatic delimiter detection, and robust data cleaning.
    """

    # Expected column names and their standardized versions
    COLUMN_MAPPING = {
        "account": ["account", "trader", "address"],
        "coin": ["coin", "symbol", "asset", "token"],
        "execution_price": ["execution price", "price", "avg price", "trade price"],
        "size_tokens": ["size tokens", "amount", "quantity", "size", "vol"],
        "size_usd": ["size usd", "notional", "value", "usd value"],
        "side": ["side", "type", "action"],
        "timestamp_ist": ["timestamp ist", "date", "time", "datetime", "timestamp"],
        "start_position": ["start position", "position"],
        "direction": ["direction", "order type"],
        "closed_pnl": ["closed pnl", "pnl", "profit", "realized pnl"],
        "transaction_hash": ["transaction hash", "tx hash", "hash"],
        "order_id": ["order id", "order", "id"],
        "crossed": ["crossed", "is_crossed"],
        "fee": ["fee", "commission", "taker fee"],
        "trade_id": ["trade id", "trade"],
        "timestamp_raw": ["timestamp"],  # fallback
    }

    # Required columns after standardization
    REQUIRED_COLUMNS = [
        "coin",
        "execution_price",
        "size_tokens",
        "size_usd",
        "side",
        "timestamp_ist",
    ]

    # Columns that should be numeric
    NUMERIC_COLUMNS = [
        "execution_price",
        "size_tokens",
        "size_usd",
        "start_position",
        "closed_pnl",
        "fee",
    ]

    # Date formats to try (in order)
    DATE_FORMATS = [
        "%d-%m-%Y %H:%M",      # 02-12-2024 22:50
        "%Y-%m-%d %H:%M:%S",   # 2024-12-02 22:50:00
        "%d/%m/%Y %H:%M",      # 02/12/2024 22:50
        "%Y/%m/%d %H:%M:%S",   # 2024/12/02 22:50:00
        "%d-%m-%Y",            # 02-12-2024
        "%Y-%m-%d",            # 2024-12-02
    ]

    def __init__(self, drop_duplicates: bool = True, fill_missing: Optional[Dict] = None):
        """
        Args:
            drop_duplicates: Whether to drop duplicate rows based on trade_id (if exists).
            fill_missing: Dictionary mapping column -> fill value. If None, defaults to:
                          numeric: 0, string: "UNKNOWN", boolean: False.
        """
        self.drop_duplicates = drop_duplicates
        self.fill_missing = fill_missing or {}
        self._data: Optional[pl.DataFrame] = None

    def load_data(self, filepath: Union[str, Path]) -> pl.DataFrame:
        """
        Load, clean, and validate the historical data CSV.
        Returns a cleaned Polars DataFrame.
        """
        with error_context(f"Loading historical data from {filepath}"):
            logger.info(f"Loading data from {filepath}")

            # Auto-detect delimiter and read CSV
            df = self._read_csv_auto_delimiter(filepath)
            logger.info(f"Initial shape: {df.shape}")

            # Standardize column names
            df = self._standardize_columns(df)
            logger.info(f"Columns after standardization: {df.columns}")

            # Check required columns exist
            self._validate_columns(df)

            # Parse dates
            df = self._parse_dates(df)

            # Clean numeric columns
            df = self._clean_numeric_columns(df)

            # Handle missing values
            df = self._handle_missing(df)

            # Drop duplicates if requested
            if self.drop_duplicates and "trade_id" in df.columns:
                before = df.height
                df = df.unique(subset=["trade_id"], keep="first")
                logger.info(f"Dropped {before - df.height} duplicate trade_id rows")

            # Optional feature engineering
            df = self._add_features(df)

            # Final validation
            self._validate_data(df)

            self._data = df
            logger.info(f"Final shape after cleaning: {df.shape}")
            return df

    # ----------------------------------------------------------------------
    # Private helper methods
    # ----------------------------------------------------------------------

    def _read_csv_auto_delimiter(self, path: Union[str, Path]) -> pl.DataFrame:
        """Attempt to read CSV with automatic delimiter detection."""
        delimiters = [',', '\t', ';', ' ', '|']
        best_df = None
        max_cols = 0

        for delim in delimiters:
            try:
                df = pl.read_csv(
                    path,
                    separator=delim,
                    try_parse_dates=False,  # we'll parse manually
                    infer_schema_length=10000,
                    ignore_errors=True,
                )
                if df.width > max_cols:
                    max_cols = df.width
                    best_df = df
            except Exception:
                continue

        if best_df is None or best_df.width == 0:
            raise DataLoadError(f"Could not read CSV with any common delimiter: {path}")

        logger.info(f"Detected delimiter with {best_df.width} columns.")
        return best_df

    def _standardize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Rename columns to our internal standardized names.
        Also cleans column names (lowercase, underscores, removes special chars).
        """
        # First, clean original column names: lowercase, replace spaces/special chars
        clean_names = {}
        for col in df.columns:
            # Remove leading/trailing spaces, lowercase, replace spaces with underscore
            cleaned = col.strip().lower()
            cleaned = re.sub(r'[^\w\s]', '', cleaned)  # remove punctuation
            cleaned = re.sub(r'\s+', '_', cleaned)    # spaces to underscore
            clean_names[col] = cleaned

        df = df.rename(clean_names)

        # Now map to our standard names using COLUMN_MAPPING
        rename_dict = {}
        for standard, candidates in self.COLUMN_MAPPING.items():
            for cand in candidates:
                # candidate may be with underscores already
                cand_clean = cand.strip().lower().replace(' ', '_')
                if cand_clean in df.columns:
                    rename_dict[cand_clean] = standard
                    break

        if rename_dict:
            logger.info(f"Renaming columns to standard: {rename_dict}")
            df = df.rename(rename_dict)
        else:
            logger.warning("No columns were mapped to standard names.")

        return df

    def _validate_columns(self, df: pl.DataFrame) -> None:
        """Check that all required columns exist."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise DataLoadError(f"Missing required columns after standardization: {missing}")

    def _parse_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Parse timestamp columns into proper datetime.
        Handles multiple date columns and falls back to various formats.
        """
        date_cols = [col for col in df.columns if 'timestamp' in col or 'date' in col]
        if not date_cols:
            logger.warning("No date columns found; cannot parse dates.")
            return df

        # Prefer 'timestamp_ist' if present
        target_col = 'timestamp_ist' if 'timestamp_ist' in df.columns else date_cols[0]

        # Try to convert to datetime using Polars' built-in parser (fast)
        try:
            df = df.with_columns(
                pl.col(target_col).str.to_datetime(
                    format=None,  # let Polars infer
                    strict=False,
                    time_unit='us'
                ).alias('timestamp')
            )
            # If conversion succeeded (no nulls), we can drop the original
            if df['timestamp'].null_count() == 0:
                logger.info(f"Successfully parsed dates using Polars inference.")
                return df.drop(target_col)
        except Exception:
            pass

        # If Polars fails, try custom formats with slow path
        logger.info("Polars date inference failed; trying custom formats...")
        series = df[target_col].cast(pl.Utf8)

        for fmt in self.DATE_FORMATS:
            try:
                converted = series.str.strptime(pl.Datetime, fmt, strict=False)
                if converted.null_count() < series.len():  # at least some succeeded
                    df = df.with_columns(converted.alias('timestamp'))
                    logger.info(f"Successfully parsed dates with format '{fmt}'")
                    return df.drop(target_col)
            except Exception:
                continue

        # Last resort: keep as string but log warning
        logger.warning(f"Could not parse dates in column '{target_col}'. Keeping as string.")
        df = df.rename({target_col: 'timestamp_raw'})
        return df

    def _clean_numeric_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert numeric columns to proper float, removing non-numeric characters.
        Handles percentages, currency symbols, commas, and scientific notation.
        """
        for col in self.NUMERIC_COLUMNS:
            if col not in df.columns:
                continue

            # Check if column is already numeric (int/float)
            if df[col].dtype in (pl.Int64, pl.Float64):
                continue

            # Clean: remove all non-numeric except dot, minus, and e/E for scientific
            cleaned = (
                df[col]
                .cast(pl.Utf8)
                .str.replace_all(r"[^\d.eE\-]", "")  # keep digits, dot, minus, exponent
                .str.replace_all(r"^$", "0")         # empty strings become "0"
                .cast(pl.Float64, strict=False)
            )

            # If conversion produced all nulls, log warning and keep original
            if cleaned.null_count() == df.height:
                logger.warning(f"Column '{col}' could not be converted to numeric. Keeping as is.")
                continue

            df = df.with_columns(cleaned.alias(col))
            logger.debug(f"Cleaned numeric column '{col}'")

        return df

    def _handle_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Fill missing values according to fill_mapping or sensible defaults.
        """
        for col in df.columns:
            if col in self.fill_missing:
                fill_val = self.fill_missing[col]
            else:
                # Defaults based on column name hints
                if col in self.NUMERIC_COLUMNS or 'price' in col or 'size' in col or 'pnl' in col:
                    fill_val = 0.0
                elif col in ('side', 'direction', 'coin'):
                    fill_val = "UNKNOWN"
                elif col in ('crossed',):
                    fill_val = False
                else:
                    fill_val = "MISSING"

            df = df.with_columns(
                pl.col(col).fill_null(fill_val).alias(col)
            )

        # Also drop rows where required columns are still null (shouldn't happen after fill)
        df = df.drop_nulls(subset=self.REQUIRED_COLUMNS)
        return df

    def _add_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Optional feature engineering: extract date parts, compute derived metrics.
        Add only if relevant columns exist.
        """
        if 'timestamp' in df.columns:
            # Extract date components
            df = df.with_columns([
                pl.col('timestamp').dt.year().alias('year'),
                pl.col('timestamp').dt.month().alias('month'),
                pl.col('timestamp').dt.day().alias('day'),
                pl.col('timestamp').dt.hour().alias('hour'),
                pl.col('timestamp').dt.weekday().alias('weekday'),
            ])

        if 'fee' in df.columns and 'size_usd' in df.columns:
            # Fee percentage
            df = df.with_columns(
                (pl.col('fee') / pl.col('size_usd') * 100).alias('fee_pct')
            )

        if 'closed_pnl' in df.columns and 'size_usd' in df.columns:
            # Return on trade
            df = df.with_columns(
                (pl.col('closed_pnl') / pl.col('size_usd') * 100).alias('return_pct')
            )

        return df

    def _validate_data(self, df: pl.DataFrame) -> None:
        """
        Final sanity checks after cleaning.
        """
        # Check for unreasonable values
        if 'execution_price' in df.columns:
            neg_prices = df.filter(pl.col('execution_price') < 0).height
            if neg_prices > 0:
                logger.warning(f"Found {neg_prices} negative execution prices.")

        if 'size_tokens' in df.columns:
            zero_sizes = df.filter(pl.col('size_tokens') == 0).height
            if zero_sizes > 0:
                logger.warning(f"Found {zero_sizes} zero token sizes.")

        # Log basic stats
        logger.info(f"Data validation passed. Final shape: {df.shape}")

    # ----------------------------------------------------------------------
    # Public utility methods
    # ----------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a dictionary with data quality metrics."""
        if self._data is None:
            return {"error": "No data loaded yet."}

        df = self._data
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "missing_values": {col: df[col].null_count() for col in df.columns},
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
        }

        # Basic stats for numeric columns
        numeric_cols = [col for col in df.columns if df[col].dtype in (pl.Float64, pl.Int64)]
        if numeric_cols:
            stats = df.select([pl.col(c).describe() for c in numeric_cols])
            summary["numeric_stats"] = stats.to_dict(as_series=False)

        return summary

    def to_pandas(self) -> 'pandas.DataFrame':
        """Convert internal Polars DataFrame to Pandas (for Streamlit)."""
        if self._data is None:
            raise DataLoadError("No data loaded. Call load_data() first.")
        return self._data.to_pandas()

    @property
    def data(self) -> Optional[pl.DataFrame]:
        return self._data
