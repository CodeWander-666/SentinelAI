from scipy import stats
import pandas as pd
import numpy as np

class Analyzer:
    def compare_regimes(self, df, metric='closedPnL'):
        """
        Performs a T-Test to check for statistically significant differences
        between 'Fear' and 'Greed' regimes.
        """
        try:
            # Segregate data
            fear_data = df[df['value_classification'] == 'Fear'][metric]
            greed_data = df[df['value_classification'] == 'Greed'][metric]

            # Validation: Ensure we have enough data points
            if len(fear_data) < 2 or len(greed_data) < 2:
                return {
                    'status': 'error',
                    'message': 'Insufficient data points for T-Test'
                }

            # Independent T-Test (assuming unequal variance)
            t_stat, p_val = stats.ttest_ind(fear_data, greed_data, equal_var=False)

            return {
                'status': 'success',
                't_statistic': t_stat,
                'p_value': p_val,
                'is_significant': p_val < 0.05,
                'fear_mean': fear_data.mean(),
                'greed_mean': greed_data.mean()
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def get_correlation(self, df):
        """Calculates correlation matrix for key metrics."""
        cols = ['closedPnL', 'leverage', 'size', 'value']
        return df[cols].corr()
