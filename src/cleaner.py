import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, tracker=None):
        self.tracker = tracker

    def _log(self, msg):
        if self.tracker: self.tracker.log(msg, 0)

    def clean_financial_data(self, df):
        # 1. Date Standardization
        if 'date_dt' in df.columns:
            if df['date_dt'].astype(str).str.contains('E').any():
                df['date_dt'] = pd.to_datetime(pd.to_numeric(df['date_dt'], errors='coerce'), unit='ms', errors='coerce')
            else:
                df['date_dt'] = pd.to_datetime(df['date_dt'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['date_dt'])
            df['date_dt'] = df['date_dt'].dt.normalize()

        # 2. Numeric Cleaning
        for col in ['closedPnL', 'size', 'leverage']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r'[$,]', '', regex=True),
                    errors='coerce'
                ).fillna(0.0)

        # 3. Leverage Injection
        if 'leverage' not in df.columns:
            df['leverage'] = 1.0

        # 4. Outlier Handling (Soft Cap)
        if 'closedPnL' in df.columns:
            Q1 = df['closedPnL'].quantile(0.25)
            Q3 = df['closedPnL'].quantile(0.75)
            IQR = Q3 - Q1
            df['closedPnL'] = df['closedPnL'].clip(lower=Q1 - 5*IQR, upper=Q3 + 5*IQR)

        return df
