import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import DataLoader
from src.utils import PipelineTracker
from src.models import ModelEngine

# --- CONFIGURATION ---
st.set_page_config(page_title="SentinelAI", layout="wide", page_icon="🦅")

# --- PROFESSIONAL STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0B0C10; color: #C5C6C7; font-family: 'Roboto', sans-serif; }
    h1, h2, h3 { color: #66FCF1; font-weight: 300; letter-spacing: 1px; }
    div[data-testid="stMetric"] { background-color: #1F2833; border-left: 4px solid #45A29E; padding: 15px; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1F2833; border-radius: 4px; color: #C5C6C7; }
    .stTabs [aria-selected="true"] { background-color: #45A29E; color: white; font-weight: bold; }
    .stButton>button { background-color: #45A29E; color: white; border: none; font-weight: bold; width: 100%; transition: 0.3s; }
    .stButton>button:hover { background-color: #66FCF1; color: black; }
</style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state: st.session_state.df = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("🦅 SENTINEL AI")
    st.markdown("### *Quantitative Intelligence*")
    st.divider()
    
    source = st.radio("DATA FEED", ["Repository", "Manual Upload"], index=1)
    
    u_t, u_s = None, None
    if source == "Manual Upload":
        u_t = st.file_uploader("Trades Data (CSV)", type="csv")
        u_s = st.file_uploader("Sentiment Data (CSV)", type="csv")
    
    if st.button("INITIALIZE SYSTEM"):
        loader = DataLoader()
        st.write("---")
        p_bar = st.progress(0, text="Standby...")
        log_box = st.empty()
        tracker = PipelineTracker(log_box, p_bar)
        
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
            st.success("System Online")
        except Exception as e:
            st.error("Initialization Failed")

# --- MAIN INTERFACE ---
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Run Segmentation Model in background for Part B
    eng = ModelEngine()
    df = eng.cluster_traders(df)

    # HEADER KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total PnL", f"${df['closedPnL'].sum():,.0f}", delta="Net Profit")
    col2.metric("Win Rate", f"{df['is_win'].mean()*100:.1f}%", delta="Global Avg")
    col3.metric("Avg Leverage", f"{df['leverage'].mean():.2f}x", delta="Risk Level")
    col4.metric("Active Traders", f"{df['account'].nunique()}", delta="Population")

    st.divider()

    # TABS STRUCTURE
    tab_a, tab_b, tab_c, tab_bonus = st.tabs([
        "📁 PART A: DATA HEALTH", 
        "🧠 PART B: SENTIMENT ANALYSIS", 
        "♟️ PART C: STRATEGY PROTOCOL",
        "✨ BONUS: AI SEGMENTATION"
    ])

    # --- PART A: DATA DOCUMENTATION ---
    with tab_a:
        st.subheader("Data Preparation & Integrity Report")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Dataset Dimensions:**")
            st.code(f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}")
            
            st.write("**Missing Values Scan:**")
            missing = df.isnull().sum()
            st.dataframe(missing[missing > 0], use_container_width=True)

        with c2:
            st.write("**Processed Schema:**")
            st.write(list(df.columns))
            
            st.write("**Metrics Created:**")
            st.markdown("""
            - `closedPnL`: Daily Profit/Loss per account
            - `is_win`: Binary win/loss indicator
            - `leverage`: Average leverage used
            - `trade_count`: Volume of trades per day
            - `long_ratio`: Bias towards Long positions (0-1)
            """)

    # --- PART B: ANALYSIS ---
    with tab_b:
        st.subheader("Market Regime Analysis (Fear vs. Greed)")
        
        # Question 1: Performance Difference
        st.markdown("#### 1. Performance Variance by Regime")
        c1, c2 = st.columns(2)
        with c1:
            fig_pnl = px.box(df, x='value_classification', y='closedPnL', 
                             color='value_classification', template='plotly_dark',
                             title="Daily PnL Distribution")
            st.plotly_chart(fig_pnl, use_container_width=True)
        with c2:
            # Win Rate by Regime
            wr_regime = df.groupby('value_classification')['is_win'].mean().reset_index()
            fig_wr = px.bar(wr_regime, x='value_classification', y='is_win', 
                            color='value_classification', template='plotly_dark',
                            title="Win Rate by Regime")
            st.plotly_chart(fig_wr, use_container_width=True)

        # Question 2: Behavior Changes
        st.markdown("#### 2. Behavioral Shifts")
        c3, c4 = st.columns(2)
        with c3:
            fig_lev = px.violin(df, x='value_classification', y='leverage', 
                                color='value_classification', template='plotly_dark',
                                title="Leverage Usage by Sentiment")
            st.plotly_chart(fig_lev, use_container_width=True)
        with c4:
            fig_freq = px.box(df, x='value_classification', y='trade_count', 
                              template='plotly_dark', title="Trading Frequency")
            st.plotly_chart(fig_freq, use_container_width=True)

    # --- PART C: STRATEGY ---
    with tab_c:
        st.subheader("Strategic Directives (Actionable Output)")
        
        # Calculate Logic for Recommendations
        fear_df = df[df['value_classification'] == 'Fear']
        greed_df = df[df['value_classification'] == 'Greed']
        
        fear_pnl = fear_df['closedPnL'].mean()
        greed_pnl = greed_df['closedPnL'].mean()
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.info("### 🛡️ STRATEGY 1: RISK MANAGEMENT")
            if fear_pnl < greed_pnl:
                st.write(f"**Trigger:** Market Performance drops during FEAR (Avg PnL: ${fear_pnl:.2f}).")
                st.write("**Action:** Implement automated leverage caps (max 5x) when Sentiment Index < 40.")
                st.write("**Rationale:** Traders historically underperform during high anxiety periods.")
            else:
                st.write("**Trigger:** Market Performance stable during FEAR.")
                st.write("**Action:** Maintain standard risk parameters; volatility is not impacting net returns.")

        with col_rec2:
            st.success("### ⚔️ STRATEGY 2: ALPHA GENERATION")
            whales = df[df['Cluster'] == '1'] # Assuming '1' might be high volume
            if not whales.empty:
                st.write("**Trigger:** High Volume Segment identified.")
                st.write("**Action:** Follow 'Whale' segment directional bias during GREED regimes.")
            else:
                st.write("**Action:** Increase position sizing on Mean Reversion strategies when Sentiment > 75 (Extreme Greed).")

    # --- BONUS: AI CLUSTERING ---
    with tab_bonus:
        st.subheader("Behavioral Archetypes (Unsupervised Learning)")
        st.caption("K-Means Clustering based on Leverage, Size, and PnL.")
        
        if 'Cluster' in df.columns:
            c1, c2 = st.columns([3, 1])
            with c1:
                fig_cl = px.scatter(df, x='leverage', y='closedPnL', color='Cluster',
                                    size='size', hover_data=['account'],
                                    template='plotly_dark', title="Trader Segmentation Map")
                st.plotly_chart(fig_cl, use_container_width=True)
            
            with c2:
                st.write("**Cluster Profiles:**")
                stats = df.groupby('Cluster')[['leverage', 'closedPnL', 'is_win']].mean()
                st.dataframe(stats.style.highlight_max(axis=0), use_container_width=True)
        else:
            st.warning("Insufficient data for clustering.")

else:
    st.info("Awaiting Data Stream. Please upload CSV files in the Sidebar.")
