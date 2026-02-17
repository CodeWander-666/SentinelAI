import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import DataLoader
from src.utils import PipelineTracker
from src.models import ModelEngine
from src.analytics import MathEngine  # New Math Engine

# --- CONFIGURATION ---
st.set_page_config(page_title="SentinelAI Pro", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .stApp { background-color: #050505; color: #E0E0E0; font-family: 'JetBrains Mono', monospace; }
    div[data-testid="stMetric"] { background-color: #111; border: 1px solid #333; padding: 15px; border-radius: 4px; }
    h1, h2, h3 { color: #00FF99; }
</style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state: st.session_state.df = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("🦅 SENTINEL PRO")
    st.caption("Advanced Quantitative Analytics")
    source = st.radio("SOURCE", ["Repository", "Manual Upload"], index=1)
    
    u_t, u_s = None, None
    if source == "Manual Upload":
        u_t = st.file_uploader("Trades", type="csv")
        u_s = st.file_uploader("Sentiment", type="csv")
    
    if st.button("EXECUTE PIPELINE"):
        loader = DataLoader()
        tracker = PipelineTracker(st.empty(), st.progress(0))
        try:
            if source == "Manual Upload" and u_t and u_s:
                st.session_state.df = loader.load_and_process(u_s, u_t, tracker)
            else:
                import os
                base = os.path.dirname(os.path.abspath(__file__))
                st.session_state.df = loader.load_and_process(
                    os.path.join(base, 'data', 'sentiment.csv'),
                    os.path.join(base, 'data', 'trades.csv'),
                    tracker
                )
            st.success("Compute Complete")
        except Exception as e:
            st.error(f"Error: {e}")

# --- ANALYTICS DASHBOARD ---
if st.session_state.df is not None:
    df = st.session_state.df.sort_values('date_dt')
    math = MathEngine()

    # --- TOP LEVEL KPI ---
    total_pnl = df['closedPnL'].sum()
    max_dd = math.calculate_drawdown(df['closedPnL'])
    sharpe = math.sharpe_proxy(df['closedPnL'])
    pf = math.profit_factor(df['closedPnL'])

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total PnL", f"${total_pnl:,.0f}", delta="Net Yield")
    k2.metric("Profit Factor", f"{pf:.2f}", delta="> 1.5 is Good")
    k3.metric("Sharpe Proxy", f"{sharpe:.2f}", delta="Risk-Adj Return")
    k4.metric("Max Drawdown", f"${max_dd:,.0f}", delta="Peak-to-Valley", delta_color="inverse")
    k5.metric("Win Rate", f"{df['is_win'].mean()*100:.1f}%")

    st.markdown("---")

    # --- ADVANCED TABS ---
    tab_overview, tab_risk, tab_dist, tab_corr = st.tabs([
        "📈 PERFORMANCE CURVE", 
        "⚠️ RISK & VOLATILITY", 
        "📊 DISTRIBUTION MATH", 
        "🔗 CORRELATION MATRIX"
    ])

    with tab_overview:
        # Cumulative PnL Chart (Area)
        df['cum_pnl'] = df['closedPnL'].cumsum()
        
        fig_equity = px.area(df, x='date_dt', y='cum_pnl', 
                             title="Cumulative Equity Curve", template='plotly_dark')
        # Add coloring based on sentiment
        fig_equity.update_traces(line_color='#00FF99', fillcolor='rgba(0, 255, 153, 0.1)')
        st.plotly_chart(fig_equity, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            # Bar Chart: Daily PnL
            fig_daily = px.bar(df, x='date_dt', y='closedPnL', color='value_classification',
                               title="Daily PnL by Regime", template='plotly_dark')
            st.plotly_chart(fig_daily, use_container_width=True)
        with c2:
            # Scatter: Win Rate vs Trade Count
            fig_scatter = px.scatter(df, x='trade_count', y='closedPnL', size='leverage',
                                     color='value_classification', title="Volume vs Profitability",
                                     template='plotly_dark')
            st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_risk:
        st.subheader("Risk Analytics Engine")
        r1, r2 = st.columns(2)
        
        with r1:
            # Rolling Volatility
            df['rolling_vol'] = math.calculate_volatility(df['closedPnL'], window=7)
            fig_vol = px.line(df, x='date_dt', y='rolling_vol', title="7-Day Rolling Volatility (Std Dev)",
                              template='plotly_dark')
            fig_vol.update_traces(line_color='#FF5555')
            st.plotly_chart(fig_vol, use_container_width=True)
            
        with r2:
            # Drawdown Underwater Plot
            df['hwm'] = df['cum_pnl'].cummax()
            df['drawdown'] = df['cum_pnl'] - df['hwm']
            fig_dd = px.area(df, x='date_dt', y='drawdown', title="Underwater Drawdown Plot",
                             template='plotly_dark')
            fig_dd.update_traces(fillcolor='rgba(255, 85, 85, 0.3)', line_color='#FF5555')
            st.plotly_chart(fig_dd, use_container_width=True)

    with tab_dist:
        st.subheader("Statistical Distribution Analysis")
        d1, d2 = st.columns(2)
        
        with d1:
            # PnL Histogram with KDE
            import plotly.figure_factory as ff
            hist_data = [df['closedPnL'].dropna()]
            group_labels = ['Daily PnL']
            fig_hist = ff.create_distplot(hist_data, group_labels, bin_size=500, show_rug=False)
            fig_hist.update_layout(title="PnL Distribution (Gaussian Check)", template='plotly_dark')
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with d2:
            # Box Plot by Sentiment
            fig_box = px.box(df, x='value_classification', y='closedPnL', points="all",
                             title="PnL Spread by Market Regime", template='plotly_dark')
            st.plotly_chart(fig_box, use_container_width=True)

    with tab_corr:
        st.subheader("Correlation & Heatmaps")
        
        # Correlation Matrix
        corr_cols = ['closedPnL', 'leverage', 'size', 'trade_count', 'value'] # 'value' is Fear/Greed Index
        corr_matrix = df[corr_cols].corr()
        
        fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                title="Feature Correlation Matrix", template='plotly_dark',
                                color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.info("💡 **Insight:** High correlation between 'value' (Sentiment) and 'leverage' indicates emotional trading.")

else:
    st.info("Awaiting Data Stream.")
