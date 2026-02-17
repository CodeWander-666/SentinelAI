from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

class ModelEngine:
    def cluster_traders(self, df, n_clusters=3):
        """
        Segments traders using K-Means clustering based on performance history.
        """
        try:
            # Create trader profiles (aggregate over time)
            profiles = df.groupby('account').agg({
                'closedPnL': 'sum',
                'leverage': 'mean',
                'side': 'mean',
                'is_win': 'mean'
            }).dropna()

            if profiles.empty:
                raise ValueError("Not enough data to form trader profiles.")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(profiles)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            profiles['Cluster'] = kmeans.fit_predict(X_scaled)

            return profiles
        except Exception as e:
            print(f"Clustering error: {e}")
            return pd.DataFrame()

    def train_predictor(self, df):
        """
        Trains XGBoost to predict next-day profitability.
        Returns the model and feature importance map.
        """
        try:
            # Feature Engineering: Lagged Target
            df = df.copy()
            df['target'] = (df.groupby('account')['closedPnL'].shift(-1) > 0).astype(int)
            
            features = ['leverage', 'size', 'side', 'is_win', 'value']
            
            # Drop NaNs created by shift
            df_clean = df.dropna(subset=['target'] + features)

            if len(df_clean) < 50:
                 raise ValueError("Insufficient data for training prediction model.")

            X = df_clean[features]
            y = df_clean['target']

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)

            score = model.score(X_test, y_test)
            
            return model, dict(zip(features, model.feature_importances_)), score

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, {}, 0.0
