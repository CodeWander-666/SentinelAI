"""
Analytics module: statistical tests, segmentation, and plotting.
"""
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Analytics:
    def __init__(self, df: pl.DataFrame):
        """
        df : Polars DataFrame with columns:
            date_dt, account, total_pnl, avg_leverage, total_size,
            trade_count, win_rate, long_ratio, value (sentiment), value_classification
        """
        self.df = df
        self.df_pd = df.to_pandas()
        self.df_pd['date_dt'] = pd.to_datetime(self.df_pd['date_dt'])

    # ----------------------------------------------------------------------
    # Segmentation functions
    # ----------------------------------------------------------------------
    def segment_by_leverage(self, threshold: float = 5.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split traders into high‑leverage (avg >= threshold) and low‑leverage."""
        avg_lev = self.df_pd.groupby('account')['avg_leverage'].mean()
        high = avg_lev[avg_lev >= threshold].index
        low = avg_lev[avg_lev < threshold].index
        df_high = self.df_pd[self.df_pd['account'].isin(high)]
        df_low = self.df_pd[self.df_pd['account'].isin(low)]
        return df_high, df_low

    def segment_by_frequency(self, percentile: float = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split traders by median trade count: frequent (above median) vs infrequent."""
        med_count = self.df_pd.groupby('account')['trade_count'].median()
        threshold = med_count.quantile(percentile / 100)
        frequent = med_count[med_count > threshold].index
        infrequent = med_count[med_count <= threshold].index
        df_freq = self.df_pd[self.df_pd['account'].isin(frequent)]
        df_infreq = self.df_pd[self.df_pd['account'].isin(infrequent)]
        return df_freq, df_infreq

    def segment_by_consistency(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Consistent winners: win rate > 0.5 and std of PnL below median."""
        trader_stats = self.df_pd.groupby('account').agg(
            win_rate_mean=('win_rate', 'mean'),
            pnl_std=('total_pnl', 'std')
        ).reset_index()
        median_std = trader_stats['pnl_std'].median()
        consistent = trader_stats[(trader_stats['win_rate_mean'] > 0.5) & (trader_stats['pnl_std'] <= median_std)]['account']
        inconsistent = trader_stats[~trader_stats['account'].isin(consistent)]['account']
        df_cons = self.df_pd[self.df_pd['account'].isin(consistent)]
        df_incons = self.df_pd[self.df_pd['account'].isin(inconsistent)]
        return df_cons, df_incons

    # ----------------------------------------------------------------------
    # Statistical tests: Fear vs Greed
    # ----------------------------------------------------------------------
    def fear_vs_greed_metrics(self) -> Dict:
        """Compute metrics on Fear vs Greed days and run t‑tests."""
        fear = self.df_pd[self.df_pd['value_classification'].str.lower().str.contains('fear')]
        greed = self.df_pd[self.df_pd['value_classification'].str.lower().str.contains('greed')]

        results = {}
        metrics = ['total_pnl', 'win_rate', 'avg_leverage', 'trade_count', 'long_ratio']
        for m in metrics:
            f_mean = fear[m].mean()
            g_mean = greed[m].mean()
            stat, p = stats.ttest_ind(fear[m].dropna(), greed[m].dropna(), equal_var=False)
            results[m] = {
                'fear_mean': f_mean,
                'greed_mean': g_mean,
                'diff': g_mean - f_mean,
                'p_value': p,
                'significant': p < 0.05
            }
        return results

    # ----------------------------------------------------------------------
    # Drawdown proxy (max daily loss per trader)
    # ----------------------------------------------------------------------
    def add_drawdown_proxy(self) -> pd.DataFrame:
        """Add a column 'max_daily_loss' (minimum daily PnL per account)."""
        min_pnl = self.df_pd.groupby('account')['total_pnl'].min().rename('max_daily_loss')
        return self.df_pd.merge(min_pnl, on='account')

    # ----------------------------------------------------------------------
    # Plotting helpers
    # ----------------------------------------------------------------------
    def plot_pnl_by_sentiment(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=self.df_pd, x='value_classification', y='total_pnl', ax=ax)
        ax.set_title('Daily PnL Distribution by Sentiment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def plot_metric_by_sentiment(self, metric: str):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=self.df_pd, x='value_classification', y=metric, estimator=np.mean, errorbar='ci', ax=ax)
        ax.set_title(f'Average {metric} on Fear vs Greed Days')
        plt.tight_layout()
        return fig

    def plot_leverage_distribution(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=self.df_pd, x='avg_leverage', hue='value_classification', bins=30, kde=True, ax=ax)
        ax.set_title('Leverage Distribution by Sentiment')
        plt.tight_layout()
        return fig

    def plot_correlation_heatmap(self, cols: List[str]):
        corr = self.df_pd[cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlation Matrix of Key Metrics')
        plt.tight_layout()
        return fig
