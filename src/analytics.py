import pandas as pd
import numpy as np
from scipy import stats

class MathEngine:
    """
    Advanced Quantitative Metrics Engine.
    """
    
    @staticmethod
    def calculate_drawdown(series):
        """Calculates Maximum Drawdown (MDD)."""
        # Cumulative PnL curve
        cum_pnl = series.cumsum()
        # High Water Mark
        hwm = cum_pnl.cummax()
        # Drawdown
        dd = cum_pnl - hwm
        return dd.min()

    @staticmethod
    def calculate_volatility(series, window=7):
        """Calculates Rolling Volatility (Std Dev)."""
        return series.rolling(window=window).std().fillna(0)

    @staticmethod
    def sharpe_proxy(pnl_series):
        """Simple Sharpe Ratio Proxy (Mean PnL / Std Dev PnL)."""
        if pnl_series.std() == 0: return 0
        return pnl_series.mean() / pnl_series.std()

    @staticmethod
    def profit_factor(pnl_series):
        """Gross Profit / Gross Loss."""
        gross_profit = pnl_series[pnl_series > 0].sum()
        gross_loss = abs(pnl_series[pnl_series < 0].sum())
        if gross_loss == 0: return 0
        return gross_profit / gross_loss
