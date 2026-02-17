import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ModelEngine:
    def cluster_traders(self, df):
        """Segments traders based on PnL, Leverage, and Volume."""
        profiles = df.groupby('account').agg({
            'closedPnL': 'sum',
            'leverage': 'mean',
            'trade_count': 'sum',
            'is_win': 'mean',
            'size': 'mean'
        }).dropna()
        
        if len(profiles) < 3: return df 
        
        scaler = StandardScaler()
        X = scaler.fit_transform(profiles)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        profiles['Cluster'] = kmeans.fit_predict(X)
        
        cluster_map = profiles['Cluster'].to_dict()
        df['Cluster'] = df['account'].map(cluster_map).fillna(-1).astype(str)
        return df

    def calculate_stats(self, df):
        """Returns dictionary of key trading metrics."""
        total_pnl = df['closedPnL'].sum()
        wins = df[df['closedPnL'] > 0]['closedPnL'].sum()
        losses = abs(df[df['closedPnL'] < 0]['closedPnL'].sum())
        profit_factor = wins / losses if losses > 0 else 0
        
        # Max Drawdown
        cum_pnl = df.sort_values('date_dt')['closedPnL'].cumsum()
        peak = cum_pnl.cummax()
        dd = cum_pnl - peak
        max_dd = dd.min()
        
        return {
            "Total PnL": total_pnl,
            "Profit Factor": profit_factor,
            "Max Drawdown": max_dd,
            "Win Rate": df['is_win'].mean() * 100
        }
