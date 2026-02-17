import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.data_loader import DataLoader
from src.utils import PipelineTracker
from src.models import ModelEngine

st.set_page_config(page_title="Sentinel AI", layout="wide", page_icon="🛡️")

# Styling
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div[data-testid="stMetric"] { background-color: #1F2937; border: 1px solid #374151; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state: st.session_state.df = None

# Sidebar
st.sidebar.title("🛡️ COMMAND CENTER")
mode = st.sidebar.radio("Input Mode", ["Repository", "Upload"], index=1)

u_t, u_s = None, None
if mode == "Upload":
    u_t = st.sidebar.file_uploader("Trades Data", type="csv")
    u_s = st.sidebar.file_uploader("Sentiment Data", type="csv")

if st.sidebar.button("▶ START ANALYSIS"):
    # 1. Setup Live Diagnostics
    st.subheader("⚙️ System Operations")
    p_bar = st.progress(0, text="Initializing...")
    log_area = st.empty()
    tracker = PipelineTracker(log_area, p_bar)
    
    loader = DataLoader()
    
    try:
        if mode == "Upload" and u_t and u_s:
            st.session_state.df = loader.load_and_process(u_s, u_t, tracker)
        elif mode == "Repository":
            base = os.path.dirname(os.path.abspath(__file__))
            st.session_state.df = loader.load_and_process(
                os.path.join(base, 'data', 'sentiment.csv'),
                os.path.join(base, 'data', 'trades.csv'),
                tracker
            )
        st.success("Analysis Complete")
    except Exception as e:
        st.error("Operation Failed")

# Dashboard
if st.session_state.df is not None:
    df = st.session_state.df
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["DASHBOARD", "SELF-DIAGNOSIS"])
    
    with tab1:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Net Profit", f"${df['closedPnL'].sum():,.0f}")
        k2.metric("Win Rate", f"{df['is_win'].mean()*100:.1f}%")
        k3.metric("Leverage", f"{df['leverage'].mean():.2f}x")
        k4.metric("Days", f"{len(df)}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Performance by Regime")
            fig = px.box(df, x='value_classification', y='closedPnL', template="plotly_dark", color='value_classification')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Volume Analysis")
            fig2 = px.scatter(df, x='size', y='closedPnL', color='value_classification', template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
            
        if st.button("Run AI Cluster"):
            eng = ModelEngine()
            cl = eng.cluster_traders(df)
            st.plotly_chart(px.scatter(cl, x='leverage', y='closedPnL', color='Cluster', template="plotly_dark"), use_container_width=True)

    with tab2:
        st.write("### 🩺 Data Health Report")
        st.write(f"**Rows Processed:** {len(df)}")
        st.write("**Columns:**", list(df.columns))
        st.write("**Sample Data:**")
        st.dataframe(df.head())
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())

else:
    st.info("Awaiting Data. Use sidebar to upload.")
