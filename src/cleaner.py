import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# ==========================================
# 1. DUPLICATE HANDLING
# ==========================================
class DuplicateRemover:
    @staticmethod
    def remove_exact(df):
        return df.drop_duplicates()

    @staticmethod
    def remove_functional(df, subset):
        """Removes duplicates based on keys (e.g. same timestamp/account)."""
        if not subset: return df
        # Keep 'last' assuming it's the latest update
        return df.drop_duplicates(subset=subset, keep='last')

# ==========================================
# 2. MISSING VALUE HANDLING
# ==========================================
class MissingValueHandler:
    @staticmethod
    def impute_statistical(df, strategy='mean'):
        """Fills missing values with Mean/Median/Mode."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if strategy == 'mean':
                val = df[col].mean()
            elif strategy == 'median':
                val = df[col].median()
            else:
                val = 0
            df[col] = df[col].fillna(val)
        return df

    @staticmethod
    def impute_knn(df, neighbors=5):
        """Advanced: Uses neighbors to guess missing values."""
        numeric = df.select_dtypes(include=[np.number])
        if not numeric.empty and numeric.isnull().sum().sum() > 0:
            imputer = KNNImputer(n_neighbors=neighbors)
            df[numeric.columns] = imputer.fit_transform(numeric)
        return df

# ==========================================
# 3. OUTLIER HANDLING
# ==========================================
class OutlierHandler:
    @staticmethod
    def clip_iqr(df, col):
        """Clips values to 1.5x IQR to prevent chart distortion."""
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR # Aggressive buffer
            upper = Q3 + 3 * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)
        return df

# ==========================================
# 4. STRUCTURAL FIXING
# ==========================================
class StructuralFixer:
    @staticmethod
    def standardize_dates(df, col):
        if col in df.columns:
            # Handle Scientific Notation (e.g. 1.73E12)
            if df[col].dtype == 'object' and df[col].astype(str).str.contains('E\+').any():
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = pd.to_datetime(df[col], unit='ms', errors='coerce')
            else:
                # Handle String (Day First)
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
        return df

    @staticmethod
    def clean_numerics(df, cols):
        """Removes '$', ',' and whitespace."""
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r'[$,\s]', '', regex=True),
                    errors='coerce'
                ).fillna(0.0)
        return df

# ==========================================
# 5. MASTER FACADE
# ==========================================
class DataCleaner:
    def __init__(self, tracker=None):
        self.tracker = tracker

    def _log(self, msg):
        if self.tracker: self.tracker.log(msg, 0) # Percent handled by loader

    def clean_dataset(self, df):
        # 1. Structural
        if 'date_dt' in df.columns:
            df = StructuralFixer.standardize_dates(df, 'date_dt')
            df = df.dropna(subset=['date_dt'])
        
        df = StructuralFixer.clean_numerics(df, ['closedPnL', 'size', 'leverage'])

        # 2. Duplicates
        df = DuplicateRemover.remove_exact(df)
        if 'date_dt' in df.columns and 'account' in df.columns:
            # Functional Dupes: Same user, same second
            df = DuplicateRemover.remove_functional(df, ['date_dt', 'account'])

        # 3. Missing Values (Fast Stats)
        df = MissingValueHandler.impute_statistical(df, 'mean')

        # 4. Outliers
        df = OutlierHandler.clip_iqr(df, 'closedPnL')

        return df
