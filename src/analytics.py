from scipy import stats
import pandas as pd

class Analyzer:
    def compare_regimes(self, df):
        if 'value_classification' not in df.columns: return None
        
        fear = df[df['value_classification'] == 'Fear']['closedPnL']
        greed = df[df['value_classification'] == 'Greed']['closedPnL']
        
        if len(fear) < 2 or len(greed) < 2: return None
        
        t_stat, p_val = stats.ttest_ind(fear, greed, equal_var=False)
        return {'t_stat': t_stat, 'p_value': p_val}
