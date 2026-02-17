"""
Wrapper for backward compatibility with the original DataLoader interface.
Uses the new HistoricalDataHandler internally.
"""

import logging
from typing import Optional, Union
from pathlib import Path

import polars as pl
import pandas as pd

from src.data_handler import HistoricalDataHandler
from src.exceptions import DataLoadError, error_context

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Backward-compatible DataLoader that uses the professional HistoricalDataHandler.
    Provides the same public methods as the original version.
    """

    def __init__(self, drop_duplicates: bool = True, fill_missing: Optional[dict] = None):
        """
        Args:
            drop_duplicates: Whether to drop duplicate trade_id rows.
            fill_missing: Custom fill values for missing columns.
        """
        self.handler = HistoricalDataHandler(
            drop_duplicates=drop_duplicates,
            fill_missing=fill_missing
        )

    def load_and_process(self, sentiment_path: str, trades_path: str) -> pl.DataFrame:
        """
        Load and process both sentiment and trades files.
        This method mimics the original signature but only uses the trades path,
        as sentiment data is handled separately in the new architecture.
        
        Args:
            sentiment_path: Path to sentiment CSV (ignored, kept for compatibility).
            trades_path: Path to the historical trades CSV.
        
        Returns:
            Polars DataFrame with cleaned trade data.
        """
        with error_context("loading and processing data"):
            logger.info("Loading trades data...")
            df = self.handler.load_data(trades_path)
            logger.info(f"Trades data loaded, shape: {df.shape}")
            return df

    def to_pandas(self) -> pd.DataFrame:
        """Convert the internal Polars DataFrame to Pandas."""
        return self.handler.to_pandas()

    @property
    def data(self) -> Optional[pl.DataFrame]:
        """Return the internal Polars DataFrame if loaded."""
        return self.handler.data
