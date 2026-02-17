import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.data_loader import DataLoader
from src.models import ModelEngine

st.set_page_config(page_title="PrimeTrade Sentinel", layout="wide", page_icon="🛡️")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stMetric { background-color: #1F2937; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE STATE ---
if 'df' not in st.session_state: st.session_state.df = None

# --- SIDEBAR ---
st.sidebar.title("🛡️ SENTINEL CORE")
source = st.sidebar.radio("Data Stream", ["Local Repository", "Upload CSV"], index=1)

u_t, u_s = None, None
if source == "Upload CSV":
    u_t = st.sidebar.file_uploader("Trades Data", type="csv")
    u_s = st.sidebar.file_uploader("Sentiment Data", type="csv")

# --- EXECUTION BUTTON ---
if st.sidebar.button("RUN DIAGNOSTICS & LOAD"):
    loader = DataLoader()
    try:
        with st.spinner("Running System Check..."):
            if source == "Upload CSV" and u_t and u_s:
                st.session_state.df = loader.load_and_process(u_s, u_t)
            elif source == "Local Repository":
                base = os.path.dirname(os.path.abspath(__file__))
                t_path = os.path.join(base, 'data', 'trades.csv')
                s_path = os.path.join(base, 'data', 'sentiment.csv')
                
                # Check file existence
                if not os.path.exists(t_path): st.error(f"Missing: {t_path}"); st.stop()
                
                st.session_state.df = loader.load_and_process(s_path, t_path)
            
            st.toast("System Healthy - Data Loaded", icon="✅")
    except Exception as e:
        # Error is already printed by loader, just stop
        st.stop()

# --- MAIN DASHBOARD ---
if st.session_state.df is not None:
    df = st.session_state.df
    
    # TABS FOR VIEW
    tab_dash, tab_diag = st.tabs(["📊 DASHBOARD", "🛠️ SELF-DIAGNOSIS"])
    
    with tab_dash:
        st.title("TRADING INTELLIGENCE")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Net PnL", f"${df['closedPnL'].sum():,.0f}")
        k2.metric("Win Rate", f"{df['is_win'].mean()*100:.1f}%")
        k3.metric("Avg Leverage", f"{df['leverage'].mean():.2f}x")
        k4.metric("Days Analyzed", f"{len(df):,}")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Profit by Sentiment")
            fig = px.box(df, x='value_classification', y='closedPnL', template='plotly_dark', color='value_classification')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Leverage Impact")
            fig2 = px.scatter(df, x='leverage', y='closedPnL', template='plotly_dark', color='value_classification', size='size')
            st.plotly_chart(fig2, use_container_width=True)
            
        # AI Cluster
        if st.button("RUN AI SEGMENTATION"):
            eng = ModelEngine()
            cl = eng.cluster_traders(df)
            st.plotly_chart(px.scatter(cl, x='leverage', y='closedPnL', color='Cluster', template='plotly_dark'), use_container_width=True)

    with tab_diag:
        st.markdown("### 🩺 Data Health Check")
        st.write("This section helps you verify the data quality.")
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.write("**First 5 Rows (Check Columns):**")
            st.dataframe(df.head())
        with col_d2:
            st.write("**Column Data Types:**")
            st.write(df.dtypes.astype(str))
            
        st.write("**Missing Values Scan:**")
        st.write(df.isnull().sum())

else:
    st.info("System Idle. Upload files and click 'RUN DIAGNOSTICS & LOAD' to begin.")
