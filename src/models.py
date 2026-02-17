"""
Bonus: Clustering and predictive modeling.
"""
import polars as pl
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


class TraderClustering:
    """Cluster traders into behavioral archetypes."""

    def __init__(self, df: pd.DataFrame):
        """
        df: daily aggregated DataFrame (from analytics) with at least:
            account, avg_leverage, win_rate, trade_count, long_ratio, total_pnl
        """
        self.df = df
        self.trader_features = None
        self.labels = None
        self.model = None

    def prepare_features(self):
        """Aggregate per trader and standardize."""
        trader_agg = self.df.groupby('account').agg({
            'avg_leverage': 'mean',
            'win_rate': 'mean',
            'trade_count': 'sum',
            'long_ratio': 'mean',
            'total_pnl': 'sum'
        }).reset_index()
        trader_agg.columns = ['account', 'leverage', 'win_rate', 'total_trades', 'long_ratio', 'total_pnl']
        self.trader_features = trader_agg.drop('account', axis=1)
        self.trader_ids = trader_agg['account']
        scaler = StandardScaler()
        self.features_scaled = scaler.fit_transform(self.trader_features)

    def cluster(self, n_clusters: int = 3):
        """Apply K‑Means clustering."""
        if self.features_scaled is None:
            self.prepare_features()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.labels = kmeans.fit_predict(self.features_scaled)
        self.model = kmeans
        # Add labels to trader dataframe
        self.trader_features['cluster'] = self.labels
        self.trader_features['account'] = self.trader_ids
        return self.trader_features

    def describe_clusters(self):
        """Return mean of features per cluster."""
        return self.trader_features.groupby('cluster').mean()


class ProfitabilityPredictor:
    """Predict next‑day profitability bucket using XGBoost."""

    def __init__(self, df: pd.DataFrame):
        """
        df: daily aggregated DataFrame (with date, account, total_pnl, sentiment, etc.)
        Must have columns: date_dt, account, total_pnl, value (sentiment score).
        """
        self.df = df.copy()
        self.df['date_dt'] = pd.to_datetime(self.df['date_dt'])
        self.df = self.df.sort_values(['account', 'date_dt'])
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None

    def create_features(self):
        """Create features for each account‑day: lagged PnL, rolling stats, sentiment."""
        # Sort per account
        df = self.df
        # Lagged PnL (previous day)
        df['pnl_lag1'] = df.groupby('account')['total_pnl'].shift(1)
        # 3‑day rolling average
        df['pnl_roll3'] = df.groupby('account')['total_pnl'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        # 7‑day rolling std
        df['pnl_roll7_std'] = df.groupby('account')['total_pnl'].transform(lambda x: x.rolling(7, min_periods=1).std())
        # Current sentiment
        df['sentiment'] = df['value']
        # Day of week
        df['dow'] = df['date_dt'].dt.dayofweek

        # Target: next‑day PnL bucket (quantiles)
        df['pnl_next'] = df.groupby('account')['total_pnl'].shift(-1)
        # Bucket: low, medium, high based on overall PnL distribution
        bins = df['pnl_next'].quantile([0, 0.33, 0.67, 1]).values
        bins[0] = -np.inf
        bins[-1] = np.inf
        df['pnl_bucket'] = pd.cut(df['pnl_next'], bins=bins, labels=[0, 1, 2]).astype(float)

        # Drop rows with NaN in features or target
        feature_cols = ['pnl_lag1', 'pnl_roll3', 'pnl_roll7_std', 'sentiment', 'dow']
        self.feature_cols = feature_cols
        df = df.dropna(subset=feature_cols + ['pnl_bucket'])
        self.df = df
        return df

    def train_test_split(self, test_size=0.2, random_state=42):
        """Split by time (last 20% of dates)."""
        dates = self.df['date_dt'].unique()
        dates = np.sort(dates)
        split_idx = int(len(dates) * (1 - test_size))
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        train = self.df[self.df['date_dt'].isin(train_dates)]
        test = self.df[self.df['date_dt'].isin(test_dates)]
        self.X_train = train[self.feature_cols]
        self.y_train = train['pnl_bucket']
        self.X_test = test[self.feature_cols]
        self.y_test = test['pnl_bucket']

    def train_xgboost(self):
        """Train XGBoost classifier."""
        self.model = xgb.XGBClassifier(objective='multi:softprob', random_state=42, n_estimators=100)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Print classification report and confusion matrix."""
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred, target_names=['Low', 'Medium', 'High']))
        cm = confusion_matrix(self.y_test, y_pred)
        return cm
