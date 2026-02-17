import pandas as pd
import numpy as np
import logging
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. CORE DATA CLEANING METHODS
# ==============================================================================

class DuplicateRemover:
    """Eliminates identical or functional duplicates."""
    
    @staticmethod
    def remove_exact_duplicates(df):
        before = len(df)
        df = df.drop_duplicates()
        logger.info(f"Dropped {before - len(df)} exact duplicates.")
        return df

    @staticmethod
    def remove_functional_duplicates(df, subset_cols):
        """Removes records that are duplicates based on specific key columns."""
        if not subset_cols: return df
        # Keep the last entry (assuming it's the most recent)
        df = df.drop_duplicates(subset=subset_cols, keep='last')
        return df

class MissingValueHandler:
    """Handles deletion, simple imputation, and advanced ML imputation."""
    
    @staticmethod
    def delete_missing(df, threshold=0.5):
        """Drops columns with > 50% missing data, then drops rows with any missing."""
        # Drop columns with too many NaNs
        df = df.dropna(thresh=int(threshold * len(df)), axis=1)
        # Drop remaining rows
        df = df.dropna()
        return df

    @staticmethod
    def impute_statistical(df, strategy='mean'):
        """Fills missing cells with mean, median, or mode."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
        return df

    @staticmethod
    def impute_advanced_knn(df, n_neighbors=5):
        """Uses K-Nearest Neighbors to predict missing values."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df

class OutlierHandler:
    """Identifies and handles extreme values."""
    
    @staticmethod
    def handle_iqr(df, col, action='clip'):
        """Uses Interquartile Range to handle outliers."""
        if col not in df.columns: return df
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if action == 'remove':
            return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        elif action == 'clip':
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        return df

class StructuralFixer:
    """Fixes formats, typos, and standardization."""
    
    @staticmethod
    def standardize_date(df, col):
        """Unifies inconsistent date formats."""
        if col in df.columns:
            # Coerce errors to NaT, then handle
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
        return df

    @staticmethod
    def fix_typos(df, col, mapping_dict):
        """Uses a dictionary to map variations (e.g., CA -> California)."""
        if col in df.columns:
            df[col] = df[col].replace(mapping_dict)
        return df

class DataValidator:
    """Checks constraints and rules."""
    
    @staticmethod
    def validate_range(df, col, min_val, max_val):
        """Ensures values fall within a specific range."""
        if col in df.columns:
            mask = (df[col] >= min_val) & (df[col] <= max_val)
            return df[mask]
        return df

# ==============================================================================
# 2. TECHNICAL & STATISTICAL TECHNIQUES
# ==============================================================================

class FeatureScaler:
    """Normalization and Scaling."""
    
    @staticmethod
    def min_max_scale(df, cols):
        """Rescales data to 0-1 range."""
        scaler = MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols])
        return df

    @staticmethod
    def standardize_zscore(df, cols):
        """Standardizes data (Mean=0, SD=1)."""
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
        return df

class DataSmoother:
    """Reduces noise."""
    
    @staticmethod
    def binning(df, col, bins=5, labels=None):
        """Converts continuous variables into categorical bins."""
        if col in df.columns:
            df[f'{col}_binned'] = pd.cut(df[col], bins=bins, labels=labels)
        return df

class CategoryEncoder:
    """Converts text to numbers."""
    
    @staticmethod
    def one_hot_encode(df, col):
        """Creates binary columns for categories."""
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
        return df

class DataParser:
    """Breaks down complex strings."""
    
    @staticmethod
    def parse_string_col(df, col, separator=' '):
        """Splits a column into list items."""
        if col in df.columns:
            df[f'{col}_parsed'] = df[col].astype(str).str.split(separator)
        return df

# ==============================================================================
# 3. MASTER FACADE
# ==============================================================================

class AutomatedCleaner:
    """
    The 'Button-Click' class that orchestrates all the above.
    """
    def clean_financial_data(self, df):
        """
        Specific pipeline for your Trade/Sentiment Data.
        """
        # 1. Structural Fixes
        if 'date_dt' in df.columns:
            df = StructuralFixer.standardize_date(df, 'date_dt')
        
        # 2. Duplicate Removal
        df = DuplicateRemover.remove_exact_duplicates(df)
        if 'date_dt' in df.columns:
            # In financial time-series, same timestamp = functional duplicate
            df = DuplicateRemover.remove_functional_duplicates(df, ['date_dt'])

        # 3. Missing Values
        df = MissingValueHandler.impute_statistical(df, strategy='mean') # Safe for PnL

        # 4. Outliers (Clip extreme PnL to avoid chart skew)
        if 'closedPnL' in df.columns:
            df = OutlierHandler.handle_iqr(df, 'closedPnL', action='clip')

        return df
