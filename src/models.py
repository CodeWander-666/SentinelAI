import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ModelEngine:
    def cluster_traders(self, df):
        """
        Part B/Bonus: Segments traders into behavioral archetypes.
        """
        # Aggregate to Account Level
        profiles = df.groupby('account').agg({
            'closedPnL': 'sum',
            'leverage': 'mean',
            'trade_count': 'sum',
            'is_win': 'mean',
            'size': 'mean'
        }).dropna()
        
        if len(profiles) < 3: return df # Not enough data
        
        # Scale
        scaler = StandardScaler()
        X = scaler.fit_transform(profiles)
        
        # Cluster
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        profiles['Cluster'] = kmeans.fit_predict(X)
        
        # Map labels back to main DF
        cluster_map = profiles['Cluster'].to_dict()
        df['Cluster'] = df['account'].map(cluster_map).fillna(-1).astype(str)
        
        return df
