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
    page_icon="⚡"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    div.stButton > button { 
        background-color: #00CC96; color: white; border: none; 
        font-size: 18px; padding: 10px 24px;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'df' not in st.session_state:
    st.session_state.df = None

# --- SIDEBAR: DATA INGESTION ---
st.sidebar.title("🚀 Data Engine")

# 1. Check for Local Data
base_dir = os.path.dirname(os.path.abspath(__file__))
local_sent = os.path.join(base_dir, 'data', 'sentiment.csv')
local_trade = os.path.join(base_dir, 'data', 'trades.csv')
local_exists = os.path.exists(local_sent) and os.path.exists(local_trade)

uploaded_trades = None
uploaded_sent = None

if not local_exists:
    st.sidebar.warning("⚠️ Local data not found.")
    st.sidebar.info("Please upload your datasets to begin.")
    uploaded_trades = st.sidebar.file_uploader("Upload Trades CSV", type=["csv"])
    uploaded_sent = st.sidebar.file_uploader("Upload Sentiment CSV", type=["csv"])
else:
    st.sidebar.success("✅ Local Data Detected")
    use_upload = st.sidebar.checkbox("Upload New Data Instead?")
    if use_upload:
        uploaded_trades = st.sidebar.file_uploader("Upload New Trades", type=["csv"])
        uploaded_sent = st.sidebar.file_uploader("Upload New Sentiment", type=["csv"])

# --- START ANALYSIS BUTTON ---
start_button = st.sidebar.button("🚀 Start Analysis")

# --- PROCESSING LOGIC ---
if start_button:
    dl = DataLoader()
    try:
        with st.spinner("🔄 Ingesting & Analyzing Data..."):
            # Determine source
            if uploaded_trades and uploaded_sent:
                df = dl.load_and_process(uploaded_sent, uploaded_trades)
                st.toast("Using Uploaded Data", icon="📂")
            elif local_exists and not (uploaded_trades or uploaded_sent):
                df = dl.load_and_process(local_sent, local_trade)
                st.toast("Using Local Data", icon="💻")
            else:
                st.error("❌ No data source selected. Please upload files.")
                st.stop()

            # Store in Session State
            st.session_state.df = df
            st.session_state.data_processed = True
            st.rerun() # Force refresh to show dashboard

    except RuntimeError as e:
        st.error("🚨 Analysis Failed")
        st.warning(f"Engine Error: {e}")
        st.info("Tip: Ensure your CSV has columns like 'Timestamp' and 'Closed PnL'.")
    except Exception as e:
        st.error(f"Unexpected Error: {e}")

# --- MAIN DASHBOARD (Only shows after processing) ---
if st.session_state.data_processed and st.session_state.df is not None:
    df = st.session_state.df
    analyzer = Analyzer()
    engine = ModelEngine()

    st.title("⚡ PrimeTrade Infinity Dashboard")
    
    # Filter
    regime_filter = st.multiselect(
        "Filter Regime", 
        options=df['value_classification'].unique(), 
        default=df['value_classification'].unique()
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
        fig = px.box(filtered_df, x='value_classification', y='closedPnL', 
                     color='value_classification',
                     color_discrete_map={'Fear': '#FF4B4B', 'Greed': '#00CC96'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Leverage Correlation")
        # Sample for performance if > 10k rows
        plot_df = filtered_df.sample(n=min(10000, len(filtered_df)))
        fig2 = px.scatter(plot_df, x='leverage', y='closedPnL', 
                          color='value_classification', size='size')
        st.plotly_chart(fig2, use_container_width=True)

else:
    # Landing Page State
    st.title("👋 Welcome to PrimeTrade Sentinel")
    st.markdown("""
    ### Ready to Analyze?
    1. Check the **Sidebar** on the left.
    2. Upload your **Trades** and **Sentiment** CSV files.
    3. Click **"Start Analysis"** to generate the dashboard.
    """)
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)
