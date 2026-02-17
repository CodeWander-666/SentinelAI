import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class DataCleaner:
    """
    Enterprise Data Cleaning Suite.
    """
    
    def __init__(self, tracker=None):
        self.tracker = tracker

    def _log(self, msg, pct):
        if self.tracker: self.tracker.log(msg, pct)

    def clean_financial_data(self, df):
        """
        Master pipeline that runs all cleaning steps in sequence.
        """
        # 1. Structural Fixes (10-30%)
        self._log("Standardizing Date Formats...", 10)
        if 'date_dt' in df.columns:
            df['date_dt'] = pd.to_datetime(df['date_dt'], errors='coerce', dayfirst=True)
            df = df.dropna(subset=['date_dt'])
        
        # 2. Deduplication (30-50%)
        self._log(f"Scanning {len(df)} rows for duplicates...", 30)
        df = df.drop_duplicates()
        if 'date_dt' in df.columns:
            # Functional duplicates: keep last entry per timestamp
            df = df.drop_duplicates(subset=['date_dt'], keep='last')

        # 3. Missing Values (50-70%)
        self._log("Imputing Missing Values (Statistical Mean)...", 50)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())

        # 4. Outlier Handling (70-90%)
        self._log("Clipping Outliers (IQR Method)...", 70)
        if 'closedPnL' in df.columns:
            Q1 = df['closedPnL'].quantile(0.25)
            Q3 = df['closedPnL'].quantile(0.75)
            IQR = Q3 - Q1
            # Cap at 3x IQR (Soft cap)
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            df['closedPnL'] = df['closedPnL'].clip(lower=lower, upper=upper)

        self._log("Finalizing Dataset Structure...", 90)
        return df
