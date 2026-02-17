import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ModelEngine:
    def cluster_traders(self, df):
        """
        Part B: Segmentation
        Clusters traders into 'Behavioral Archetypes' using K-Means.
        """
        # Aggregate daily records to Trader Profiles
        profiles = df.groupby('account').agg({
            'closedPnL': 'sum',
            'leverage': 'mean',
            'trade_count': 'sum',
            'is_win': 'mean',
            'size': 'mean'
        }).dropna()
        
        if len(profiles) < 3: 
            df['Cluster'] = "Unclassified"
            return df 
        
        # Normalize Data for AI
        scaler = StandardScaler()
        X = scaler.fit_transform(profiles)
        
        # Create 3 distinct segments
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        profiles['Cluster'] = kmeans.fit_predict(X)
        
        # Map back to main dataframe
        cluster_map = profiles['Cluster'].to_dict()
        df['Cluster'] = df['account'].map(cluster_map).fillna(-1).astype(int).astype(str)
        
        return df

    def analyze_regimes(self, df):
        """
        Part B: Performance & Behavior Diff (Fear vs Greed)
        """
        required = ['value_classification', 'closedPnL', 'is_win', 'leverage', 'trade_count', 'long_ratio', 'size']
        if not all(col in df.columns for col in required): return pd.DataFrame()

        # Group by Market Regime
        stats = df.groupby('value_classification')[required[1:]].mean()
        stats['Day Count'] = df['value_classification'].value_counts()
        return stats

    def calculate_kpis(self, df):
        """
        Part A: Key Metrics
        """
        total_pnl = df['closedPnL'].sum()
        wins = df[df['closedPnL'] > 0]['closedPnL'].sum()
        losses = abs(df[df['closedPnL'] < 0]['closedPnL'].sum())
        profit_factor = wins / losses if losses > 0 else 0
        
        # Drawdown
        cum = df.sort_values('date_dt')['closedPnL'].cumsum()
        peak = cum.cummax()
        dd = (cum - peak).min()
        
        return {
            "Total PnL": total_pnl,
            "Profit Factor": profit_factor,
            "Max Drawdown": dd,
            "Win Rate": df['is_win'].mean() * 100,
            "Avg Leverage": df['leverage'].mean(),
            "Total Trades": df['trade_count'].sum()
        }
