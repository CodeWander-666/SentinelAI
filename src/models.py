import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ModelEngine:
    def cluster_traders(self, df):
        """Segments traders into Behavioral Archetypes."""
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
        
        scaler = StandardScaler()
        X = scaler.fit_transform(profiles)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        profiles['Cluster'] = kmeans.fit_predict(X)
        
        cluster_map = profiles['Cluster'].to_dict()
        df['Cluster'] = df['account'].map(cluster_map).fillna(-1).astype(int).astype(str)
        return df

    def analyze_regimes(self, df):
        """Part B: Performance diff between Fear vs Greed."""
        required = ['value_classification', 'closedPnL', 'is_win', 'leverage', 'trade_count']
        if not all(col in df.columns for col in required): return pd.DataFrame()

        stats = df.groupby('value_classification')[required[1:]].mean()
        stats['Day Count'] = df['value_classification'].value_counts()
        return stats

    def calculate_kpis(self, df):
        """Part A: Global Metrics."""
        total_pnl = df['closedPnL'].sum()
        wins = df[df['closedPnL'] > 0]['closedPnL'].sum()
        losses = abs(df[df['closedPnL'] < 0]['closedPnL'].sum())
        profit_factor = wins / losses if losses > 0 else 0
        
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
