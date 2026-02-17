import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Local Modules
from src.data_loader import DataLoader
# Ensure these exist or create placeholders
try:
    from src.models import ModelEngine
except ImportError:
    class ModelEngine: 
        def cluster_traders(self, df): return pd.DataFrame() # Placeholder

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="PrimeTrade Sentinel", layout="wide", page_icon="⚡")

# Professional Dark Mode CSS
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; font-family: 'Inter', sans-serif; }
    div[data-testid="stMetric"] { 
        background-color: #1F2937; 
        border: 1px solid #374151; 
        padding: 15px; 
        border-radius: 8px; 
    }
    h1, h2, h3 { color: #60A5FA; }
    .stButton>button { 
        background-color: #2563EB; 
        color: white; 
        border: none; 
        border-radius: 6px; 
        height: 3rem; 
        width: 100%;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state: st.session_state.df = None

# --- 2. SIDEBAR CONTROL ---
st.sidebar.header("DATA FEED CONTROL")
data_mode = st.sidebar.radio("Input Source", ["Local Repository", "Manual Upload"], index=1)

uploaded_trades, uploaded_sent = None, None
if data_mode == "Manual Upload":
    uploaded_trades = st.sidebar.file_uploader("Upload Trades (CSV)", type="csv")
    uploaded_sent = st.sidebar.file_uploader("Upload Sentiment (CSV)", type="csv")

if st.sidebar.button("INITIALIZE ANALYTICS"):
    loader = DataLoader()
    try:
        with st.spinner("Ingesting & Normalizing Data..."):
            # Determine Source
            if data_mode == "Manual Upload" and uploaded_trades and uploaded_sent:
                st.session_state.df = loader.load_and_process(uploaded_sent, uploaded_trades)
                st.toast("Pipeline Active: Uploaded Data", icon="✅")
            
            elif data_mode == "Local Repository":
                base_dir = os.path.dirname(os.path.abspath(__file__))
                f_t = os.path.join(base_dir, 'data', 'trades.csv')
                f_s = os.path.join(base_dir, 'data', 'sentiment.csv')
                if os.path.exists(f_t) and os.path.exists(f_s):
                    st.session_state.df = loader.load_and_process(f_s, f_t)
                    st.toast("Pipeline Active: Local Data", icon="✅")
                else:
                    st.error("Local files not found. Please upload manually.")
            
            else:
                st.warning("Please upload both CSV files.")

    except Exception as e:
        st.error(f"System Failure: {e}")

# --- 3. MAIN DASHBOARD ---
if st.session_state.df is not None:
    df = st.session_state.df
    
    st.title("SENTINEL INTELLIGENCE TERMINAL")
    st.caption(f"System Status: ONLINE | Records Analyzed: {len(df):,}")
    
    # KPIS
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Net Profit (PnL)", f"${df['closedPnL'].sum():,.0f}")
    k2.metric("Win Rate", f"{df['is_win'].mean()*100:.1f}%")
    k3.metric("Avg Leverage", f"{df['leverage'].mean():.2f}x")
    k4.metric("Active Days", f"{len(df):,}")

    st.markdown("---")

    # CHARTS (Using width='stretch' to fix deprecation warning)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Profit vs. Sentiment")
        fig = px.box(df, x='value_classification', y='closedPnL', 
                     color='value_classification', template='plotly_dark',
                     color_discrete_map={'Fear': '#EF4444', 'Greed': '#10B981', 'Neutral': '#6B7280'})
        st.plotly_chart(fig, width='stretch') # FIXED PARAMETER

    with c2:
        st.subheader("Leverage Impact")
        fig2 = px.scatter(df, x='leverage', y='closedPnL', 
                          color='value_classification', template='plotly_dark',
                          size='size', opacity=0.7)
        st.plotly_chart(fig2, width='stretch') # FIXED PARAMETER

    # AI SECTION
    st.markdown("---")
    st.subheader("🧠 Algorithmic Segmentation")
    
    if st.button("RUN CLUSTERING MODEL"):
        try:
            from src.models import ModelEngine
            engine = ModelEngine()
            # Ensure model can handle the dataframe
            clusters = engine.cluster_traders(df)
            
            if not clusters.empty:
                st.success("Segmentation Complete")
                st.plotly_chart(px.scatter(clusters, x='leverage', y='closedPnL', color='Cluster', template='plotly_dark'), width='stretch')
                
                # Safe Display without Styler (avoids Matplotlib error)
                st.write("**Cluster Performance Metrics**")
                stats = clusters.groupby('Cluster')[['leverage', 'closedPnL', 'is_win']].mean()
                st.dataframe(stats, use_container_width=True) 
        except Exception as e:
            st.warning(f"Modeling Module Unavailable or Error: {e}")

else:
    # Empty State
    st.info("👋 Welcome. Please select your data source on the left to begin.")
