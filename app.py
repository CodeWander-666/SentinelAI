import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.data_loader import DataLoader
from src.models import ModelEngine

# --- UI SETTINGS ---
st.set_page_config(page_title="PrimeTrade Sentinel", layout="wide", page_icon="📈")

# High-Contrast Terminal Styling
st.markdown("""
<style>
    .stApp { background-color: #0B0E11; color: #EAECEF; font-family: 'Inter', sans-serif; }
    div[data-testid="stMetric"] { background-color: #1E2329; border-left: 5px solid #F0B90B; padding: 15px; border-radius: 4px; }
    h1, h2, h3 { color: #F0B90B; font-weight: 700; }
    .stButton>button { background-color: #F0B90B; color: black; font-weight: bold; border: none; width: 100%; border-radius: 4px; }
    .stAlert { background-color: #1E2329; color: #EAECEF; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state: st.session_state.df = None

# --- SIDEBAR CONTROL ---
st.sidebar.title("⚡ QUANT CORE")
source_mode = st.sidebar.radio("Data Stream", ["Standard Repository", "Custom Upload"], horizontal=True)

u_t, u_s = None, None
if source_mode == "Custom Upload":
    u_t = st.sidebar.file_uploader("Trades CSV", type="csv")
    u_s = st.sidebar.file_uploader("Sentiment CSV", type="csv")

if st.sidebar.button("INITIALIZE TERMINAL"):
    loader = DataLoader()
    try:
        with st.spinner("Decoding Market Regime Patterns..."):
            if source_mode == "Custom Upload" and u_t and u_s:
                st.session_state.df = loader.load_and_process(u_s, u_t)
            else:
                base = os.path.dirname(os.path.abspath(__file__))
                st.session_state.df = loader.load_and_process(
                    os.path.join(base, 'data', 'sentiment.csv'),
                    os.path.join(base, 'data', 'trades.csv')
                )
            st.toast("Connection Established", icon="⚡")
    except Exception as e:
        st.error(f"Engine Fault: {e}")

# --- DASHBOARD RENDERING ---
if st.session_state.df is not None:
    df = st.session_state.df
    st.title("SENTINEL INTELLIGENCE TERMINAL")
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Net PnL", f"${df['closedPnL'].sum():,.0f}")
    k2.metric("Win Rate", f"{df['is_win'].mean()*100:.1f}%")
    k3.metric("Avg Leverage", f"{df['leverage'].mean():.2f}x")
    k4.metric("Dataset Size", f"{len(df):,} Rows")

    st.markdown("---")

    # Visuals
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Profitability by Sentiment")
        fig = px.box(df, x='value_classification', y='closedPnL', template="plotly_dark", color='value_classification')
        st.plotly_chart(fig, width='stretch')
    
    with c_right:
        st.subheader("Leverage Correlation")
        fig2 = px.scatter(df, x='leverage', y='closedPnL', color='value_classification', template="plotly_dark", size='size', opacity=0.7)
        st.plotly_chart(fig2, width='stretch')

    # Advanced AI Section
    st.markdown("---")
    st.subheader("Behavioral Archetypes (AI Cluster)")
    if st.button("EXECUTE CLUSTERING MODEL"):
        engine = ModelEngine()
        clusters = engine.cluster_traders(df)
        st.plotly_chart(px.scatter(clusters, x='leverage', y='closedPnL', color='Cluster', template="plotly_dark", title="Risk/Reward Segmentation"), width='stretch')
        # Professional Data Table (Standard Table instead of Styler to avoid Matplotlib error)
        st.table(clusters.groupby('Cluster')[['leverage', 'closedPnL', 'is_win']].mean().style.format("{:.2f}"))

else:
    st.info("🛰️ Awaiting connection. Please select a source and click 'Initialize Terminal' to begin analysis.")
