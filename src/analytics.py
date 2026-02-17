from scipy import stats
import pandas as pd
import numpy as np

class Analyzer:
    def compare_regimes(self, df, metric='closedPnL'):
        """
        Safely compares Fear vs Greed.
        Returns 'N/A' if data is insufficient instead of Crashing.
        """
        try:
            # 1. Check Data Sufficiency
            if df.empty or 'value_classification' not in df.columns:
                return {'status': 'error', 'message': 'No data available'}
            
            # 2. Segment Data
            fear = df[df['value_classification'].str.contains('Fear', case=False, na=False)][metric]
            greed = df[df['value_classification'].str.contains('Greed', case=False, na=False)][metric]
            
            # 3. Validation
            if len(fear) < 2 or len(greed) < 2:
                return {
                    'status': 'warning',
                    'message': 'Insufficient data points for T-Test',
                    'fear_count': len(fear),
                    'greed_count': len(greed)
                }
                
            # 4. Statistical Test
            t_stat, p_val = stats.ttest_ind(fear, greed, equal_var=False)
            
            return {
                'status': 'success',
                't_statistic': t_stat,
                'p_value': p_val,
                'is_significant': p_val < 0.05,
                'fear_mean': fear.mean(),
                'greed_mean': greed.mean()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
