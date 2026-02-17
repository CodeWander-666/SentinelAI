import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ModelEngine:
    def cluster_traders(self, df):
        """Segments daily trading behavior."""
        if len(df) < 5:
            return pd.DataFrame()
            
        # Feature Engineering
        features = df[['leverage', 'closedPnL', 'size', 'is_win']].fillna(0)
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        
        # Clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        df['Cluster'] = df['Cluster'].astype(str) # Convert to string for categorical coloring
        
        return df
