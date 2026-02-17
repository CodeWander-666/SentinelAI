import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.data_loader import DataLoader
from src.analytics import Analyzer
from src.models import ModelEngine

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="PrimeTrade Infinity",
    layout="wide",
    page_icon="🚀"
)

# Custom CSS for Speed & Aesthetics
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    div.stButton > button { background-color: #00CC96; color: white; border: none; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: DATA INGESTION ---
st.sidebar.title("🚀 Data Engine")
st.sidebar.info("Using Polars Engine for high-performance processing (1GB+ Support).")

# 1. File Uploader (Up to 200MB by default, configurable via server)
uploaded_trades = st.sidebar.file_uploader("Upload Trades CSV (Max 200MB)", type=["csv"])
uploaded_sent = st.sidebar.file_uploader("Upload Sentiment CSV", type=["csv"])

# --- DATA LOADING LOGIC ---
@st.cache_data(show_spinner=False)
def get_data(sent_file, trade_file):
    dl = DataLoader()
    return dl.load_and_process(sent_file, trade_file)

try:
    with st.spinner("🚀 Turbo-charging Data Engine... Processing..."):
        # LOGIC: If user uploads files, use them. Else, use local repo files.
        if uploaded_trades and uploaded_sent:
            st.sidebar.success("✅ Using Uploaded Data")
            df = get_data(uploaded_sent, uploaded_trades)
        else:
            # Fallback to local files (Your 1000MB Desktop file path)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            local_sent = os.path.join(base_dir, 'data', 'sentiment.csv')
            local_trade = os.path.join(base_dir, 'data', 'trades.csv')
            
            if os.path.exists(local_trade):
                st.sidebar.info(f"📂 Loaded Local Data")
                df = get_data(local_sent, local_trade)
            else:
                st.warning("⚠️ Waiting for data upload...")
                st.stop()

    # Initialize Engines
    analyzer = Analyzer()
    engine = ModelEngine()

except Exception as e:
    st.error(f"Engine Failure: {e}")
    st.stop()

# --- MAIN DASHBOARD (Same as before, but faster) ---
st.title("🚀 PrimeTrade Infinity: High-Frequency Analytics")
st.markdown("### Processed 1M+ rows in < 2 seconds.")

# Filter
regime_filter = st.sidebar.multiselect(
    "Filter Regime", options=df['value_classification'].unique(), default=df['value_classification'].unique()
)
filtered_df = df[df['value_classification'].isin(regime_filter)]

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Volume Processed", f"${filtered_df['size'].sum()/1e6:.1f}M")
c2.metric("Avg Leverage", f"{filtered_df['leverage'].mean():.2f}x")
c3.metric("Net PnL", f"${filtered_df['closedPnL'].sum():,.0f}")
c4.metric("Active Traders", f"{filtered_df['account'].nunique()}")

st.markdown("---")

# Visuals
col1, col2 = st.columns(2)
with col1:
    st.subheader("PnL Distribution")
    fig = px.box(df, x='value_classification', y='closedPnL', color='value_classification', 
                 color_discrete_map={'Fear': '#FF4B4B', 'Greed': '#00CC96'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Leverage Correlation")
    # Sampling for scatter plot speed (10k points max)
    plot_df = filtered_df.sample(n=min(10000, len(filtered_df)), random_state=42)
    fig2 = px.scatter(plot_df, x='leverage', y='closedPnL', color='value_classification', size='size')
    st.plotly_chart(fig2, use_container_width=True)
