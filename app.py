import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import DataLoader
from src.utils import PipelineTracker
from src.models import ModelEngine

st.set_page_config(page_title="PrimeTrade Sentinel", layout="wide", page_icon="📡")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    /* Terminal Look for Logs */
    .stCode { background-color: #000 !important; color: #0f0 !important; }
</style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state: st.session_state.df = None

# --- SIDEBAR ---
st.sidebar.title("📡 SYSTEM CONTROL")
src = st.sidebar.radio("Source", ["Repository", "Upload"], index=1)

u_t, u_s = None, None
if src == "Upload":
    u_t = st.sidebar.file_uploader("Trades CSV", type="csv")
    u_s = st.sidebar.file_uploader("Sentiment CSV", type="csv")

if st.sidebar.button("▶ INITIATE SEQUENCE"):
    # 1. Create Placeholders for Real-Time Logs
    st.subheader("⚙️ Pipeline Operations")
    prog_bar = st.progress(0, text="Starting engine...")
    log_box = st.empty() # This container will hold the logs
    
    # 2. Initialize Tracker
    tracker = PipelineTracker(log_box, prog_bar)
    loader = DataLoader()
    
    try:
        if src == "Upload" and u_t and u_s:
            st.session_state.df = loader.load_and_process(u_s, u_t, tracker)
        else:
            # Repo Logic
            import os
            base = os.path.dirname(os.path.abspath(__file__))
            st.session_state.df = loader.load_and_process(
                os.path.join(base, 'data', 'sentiment.csv'),
                os.path.join(base, 'data', 'trades.csv'),
                tracker
            )
        st.success("✅ System Online")
        
    except Exception as e:
        st.error("❌ Sequence Aborted")

# --- DASHBOARD ---
if st.session_state.df is not None:
    df = st.session_state.df
    st.markdown("---")
    st.title("📊 LIVE INTELLIGENCE")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("PnL", f"${df['closedPnL'].sum():,.0f}")
    k2.metric("Win Rate", f"{df['is_win'].mean()*100:.1f}%")
    k3.metric("Leverage", f"{df['leverage'].mean():.2f}x")
    k4.metric("Days", f"{len(df)}")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Performance")
        st.plotly_chart(px.box(df, x='value_classification', y='closedPnL', template="plotly_dark"), use_container_width=True)
    with c2:
        st.subheader("Market Depth")
        st.plotly_chart(px.scatter(df, x='leverage', y='closedPnL', color='value_classification', template="plotly_dark"), use_container_width=True)

else:
    st.info("Ready. Connect data streams to begin.")
