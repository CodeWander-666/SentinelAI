"""
HEAVY ARMOR DATA ENGINE (Polars Edition)
Loads, cleans, and merges sentiment and trades data.
"""
import polars as pl
import pandas as pd
import logging
from typing import Union

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class DataLoader:
    """Load and preprocess sentiment and trades data."""

    # ----------------------------------------------------------------------
    # Internal helpers for date parsing
    # ----------------------------------------------------------------------
    @staticmethod
    def _parse_sentiment_dates(series: pl.Series) -> pl.Series:
        """Parse sentiment dates. Try day‑first format, then fallback."""
        str_series = series.cast(pl.String)
        parsed = str_series.str.strptime(pl.Datetime, format="%d-%m-%Y", strict=False)
        return parsed.fill_null(str_series.str.strptime(pl.Datetime, strict=False))

    @staticmethod
    def _parse_trades_dates(series: pl.Series, col_name: str) -> pl.Series:
        """
        Parse trades dates:
        - "Timestamp IST" -> string "%d-%m-%Y %H:%M"
        - "Timestamp" -> numeric milliseconds
        """
        if col_name == "Timestamp IST":
            return series.cast(pl.String).str.strptime(
                pl.Datetime, format="%d-%m-%Y %H:%M", strict=False
            )
        elif col_name == "Timestamp":
            # Convert milliseconds to microseconds (Polars uses µs)
            return (series.cast(pl.Int64) // 1000).cast(pl.Datetime)
        else:
            return series.cast(pl.String).str.strptime(pl.Datetime, strict=False)

    @staticmethod
    def _normalise_columns(df: pl.DataFrame, mapping: dict) -> pl.DataFrame:
        """Rename columns based on mapping (internal: list of possible external)."""
        current = {col.lower().strip(): col for col in df.columns}
        rename = {}
        for internal, aliases in mapping.items():
            for alias in aliases:
                if alias.lower().strip() in current:
                    rename[current[alias.lower().strip()]] = internal
                    break
        return df.rename(rename)

    @staticmethod
    def _clean_numeric(series: pl.Series) -> pl.Series:
        """Remove '$', ',' and whitespace, cast to float, fill nulls with 0."""
        return (series.cast(pl.String)
                .str.replace_all(r'[\$,]', '')
                .str.strip_chars()
                .cast(pl.Float64, strict=False)
                .fill_null(0.0))

    # ----------------------------------------------------------------------
    # Main pipeline
    # ----------------------------------------------------------------------
    def load_and_process(self,
                         sentiment_source: Union[str, bytes, object],
                         trades_source: Union[str, bytes, object]) -> pl.DataFrame:
        """Load, clean, and merge datasets. Returns Polars DataFrame."""
        try:
            logger.info("--- STARTING DATA PIPELINE (Polars) ---")

            df_sent = self._load_sentiment(sentiment_source)
            logger.info(f"Sentiment loaded: {df_sent.shape}")

            df_trades = self._load_trades(trades_source)
            logger.info(f"Trades loaded: {df_trades.shape}")

            df_agg = self._aggregate_trades(df_trades)
            logger.info(f"Trades aggregated: {df_agg.shape}")

            df_final = df_agg.join(df_sent, on="date_dt", how="left")
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
    # Private loaders
    # ----------------------------------------------------------------------
    def _load_sentiment(self, source) -> pl.DataFrame:
        try:
            df = pl.read_csv(source, try_parse_dates=False)
        except Exception:
            df = pd.read_csv(source)
            df = pl.from_pandas(df)

        mapping = {
            "date_dt": ["date", "timestamp"],
            "value": ["value", "fng_value"],
            "value_classification": ["classification", "label"],
        }
        df = self._normalise_columns(df, mapping)

        required = ["date_dt", "value", "value_classification"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required sentiment column: {col}")

        df = df.with_columns(self._parse_sentiment_dates(pl.col("date_dt")).alias("date_dt"))
        df = df.drop_nulls(subset=["date_dt"])
        df = df.select(["date_dt", "value", "value_classification"])
        return df

    def _load_trades(self, source) -> pl.DataFrame:
        try:
            df = pl.read_csv(source, try_parse_dates=False)
        except Exception:
            df = pd.read_csv(source)
            df = pl.from_pandas(df)

        mapping = {
            "time_str": ["timestamp ist", "date", "time"],
            "account": ["account", "user id"],
            "closedPnL": ["closed pnl", "pnl", "realized pnl"],
            "size": ["size usd", "size", "volume"],
            "leverage": ["leverage", "lev"],
            "side": ["side", "direction"],
        }
        df = self._normalise_columns(df, mapping)

        if "leverage" not in df.columns:
            df = df.with_columns(pl.lit(1.0).alias("leverage"))
        if "size" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("size"))

        for col in ["closedPnL", "leverage", "size"]:
            if col in df.columns:
                df = df.with_columns(self._clean_numeric(pl.col(col)).alias(col))

        # Determine timestamp column
        date_cols = [c for c in ["time_str", "timestamp"] if c in df.columns]
        if not date_cols:
            raise ValueError("No timestamp column found in trades data.")
        ts_col = date_cols[0]

        df = df.with_columns(self._parse_trades_dates(pl.col(ts_col), ts_col).alias("date_dt"))
        df = df.drop_nulls(subset=["date_dt"])
        df = df.with_columns(pl.col("date_dt").dt.date().cast(pl.Datetime).alias("date_dt"))
        return df

    def _aggregate_trades(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns((pl.col("closedPnL") > 0).cast(pl.Int32).alias("is_win"))
        grouped = df.group_by(["date_dt", "account"]).agg([
            pl.col("closedPnL").sum().alias("total_pnl"),
            pl.col("leverage").mean().alias("avg_leverage"),
            pl.col("size").sum().alias("total_size"),
            pl.col("side").count().alias("trade_count"),
            pl.col("is_win").mean().alias("win_rate"),
            (pl.col("side") == "BUY").sum().alias("long_count"),
            (pl.col("side") == "SELL").sum().alias("short_count"),
        ]).with_columns(
            (pl.col("long_count") / (pl.col("long_count") + pl.col("short_count"))).alias("long_ratio")
        )
        return grouped
