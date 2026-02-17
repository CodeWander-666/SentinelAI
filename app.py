import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import DataLoader
from src.utils import PipelineTracker
from src.models import ModelEngine

# --- CONFIGURATION ---
st.set_page_config(page_title="SentinelAI Expert", layout="wide", page_icon="🦅")

# --- PROFESSIONAL STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E0E0E0; font-family: 'Segoe UI', sans-serif; }
    div[data-testid="stMetric"] { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    h1, h2, h3 { color: #58a6ff; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #58a6ff; border-bottom: 3px solid #58a6ff; }
</style>
""", unsafe_allow_html=True)

if 'df' not in st.session_state: st.session_state.df = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("🦅 SENTINEL AI")
    st.caption("Quantitative Trading Expert")
    st.divider()
    source = st.radio("DATA SOURCE", ["Repository", "Manual Upload"], index=1)
    
    u_t, u_s = None, None
    if source == "Manual Upload":
        u_t = st.file_uploader("Trades Data", type="csv")
        u_s = st.file_uploader("Sentiment Data", type="csv")
    
    if st.button("RUN ANALYSIS", type="primary"):
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
            st.toast("Data Loaded Successfully", icon="✅")
        except Exception as e:
            st.error("Error Loading Data")

# --- MAIN INTERFACE ---
if st.session_state.df is not None:
    df = st.session_state.df.sort_values('date_dt')
    engine = ModelEngine()
    
    # Run AI Clustering automatically
    df = engine.cluster_traders(df)
    
    # Calculate Global Stats
    kpi = engine.calculate_kpis(df)

    # --- HEADER METRICS ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total PnL", f"${kpi['Total PnL']:,.0f}")
    c2.metric("Profit Factor", f"{kpi['Profit Factor']:.2f}")
    c3.metric("Win Rate", f"{kpi['Win Rate']:.1f}%")
    c4.metric("Avg Leverage", f"{kpi['Avg Leverage']:.1f}x")
    c5.metric("Total Volume", f"{kpi['Total Trades']:,}")
    
    st.divider()

    # --- PROJECT REQUIREMENTS TABS ---
    tab_a, tab_b, tab_c = st.tabs([
        "📂 PART A: DATA PREP", 
        "🔍 PART B: ANALYSIS", 
        "💡 PART C: ACTIONABLE STRATEGY"
    ])

    # --- PART A: DATA PREP ---
    with tab_a:
        st.subheader("Data Integrity & Health")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Dataset Dimensions:**")
            st.code(f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}")
            st.write("**Missing Values (After Cleaning):**")
            st.dataframe(df.isnull().sum()[df.isnull().sum() > 0], use_container_width=True)
        
        with c2:
            st.write("**Aligned Data Sample (Daily Aggregated):**")
            st.dataframe(df[['date_dt', 'account', 'closedPnL', 'leverage', 'value_classification']].head(), use_container_width=True)
            st.info("✅ Timestamps converted and datasets aligned by Date.")

    # --- PART B: ANALYSIS ---
    with tab_b:
        st.subheader("Evidence-Based Insights")
        
        # 1. FEAR vs GREED ANALYSIS
        st.markdown("#### 1. Performance & Behavior by Sentiment")
        regime_stats = engine.analyze_regimes(df)
        
        # Display Comparative Table
        st.dataframe(regime_stats.style.background_gradient(cmap='RdYlGn', subset=['closedPnL', 'is_win']), use_container_width=True)
        
        b1, b2 = st.columns(2)
        with b1:
            # PnL Distribution
            fig_box = px.box(df, x='value_classification', y='closedPnL', color='value_classification', 
                             title="Does PnL differ by Sentiment?", template="plotly_dark")
            st.plotly_chart(fig_box, use_container_width=True)
        with b2:
            # Behavior: Leverage
            fig_lev = px.violin(df, x='value_classification', y='leverage', box=True, color='value_classification',
                                title="Do traders change Leverage based on Sentiment?", template="plotly_dark")
            st.plotly_chart(fig_lev, use_container_width=True)

        # 2. SEGMENTATION
        st.markdown("#### 2. Trader Segmentation (Clusters)")
        st.caption("Identifying segments: High vs Low Leverage, Consistent vs Inconsistent")
        
        s1, s2 = st.columns([2, 1])
        with s1:
            fig_clus = px.scatter(df, x='leverage', y='closedPnL', color='Cluster', size='size',
                                  title="Cluster Map: Leverage vs Profitability", template="plotly_dark")
            st.plotly_chart(fig_clus, use_container_width=True)
        with s2:
            st.write("**Segment Profiles:**")
            cluster_stats = df.groupby('Cluster')[['closedPnL', 'leverage', 'is_win', 'trade_count']].mean()
            st.dataframe(cluster_stats.style.highlight_max(axis=0), use_container_width=True)

        st.markdown("#### 3. Key Insights")
        st.markdown("""
        * **Insight 1:** Check the 'PnL' column in the table above. If 'Greed' is higher, traders perform better in bullish sentiment.
        * **Insight 2:** Observe the 'Leverage' violin plot. A wider distribution in one regime indicates inconsistent risk management.
        * **Insight 3:** The Clusters reveal distinct groups. Usually, one cluster represents 'High Leverage / High Risk' traders.
        """)

    # --- PART C: ACTIONABLE OUTPUT ---
    with tab_c:
        st.subheader("Strategic Directives & Rules of Thumb")
        
        # DYNAMIC LOGIC GENERATION
        fear_data = regime_stats.loc['Fear'] if 'Fear' in regime_stats.index else None
        greed_data = regime_stats.loc['Greed'] if 'Greed' in regime_stats.index else None
        
        col_strat1, col_strat2 = st.columns(2)
        
        with col_strat1:
            st.success("### 🛡️ STRATEGY 1: REGIME ADAPTATION")
            if fear_data is not None and greed_data is not None:
                if fear_data['closedPnL'] < greed_data['closedPnL']:
                    st.write("**Observation:** Performance degrades during FEAR regimes.")
                    st.write(f"**Rule of Thumb:** When Sentiment Index < 40 (Fear), reduce position sizing by 50%.")
                    st.write(f"**Data Backing:** Fear Avg PnL (${fear_data['closedPnL']:.2f}) < Greed Avg PnL (${greed_data['closedPnL']:.2f}).")
                else:
                    st.write("**Observation:** Performance is robust during FEAR.")
                    st.write("**Rule of Thumb:** Maintain standard leverage during volatility; opportunity cost is high if reduced.")
            else:
                st.write("Insufficient data across both regimes to form a rule.")

        with col_strat2:
            st.info("### 🎯 STRATEGY 2: SEGMENT OPTIMIZATION")
            # Analyze Clusters
            best_cluster = cluster_stats['closedPnL'].idxmax()
            worst_cluster = cluster_stats['closedPnL'].idxmin()
            
            st.write(f"**Observation:** Segment {best_cluster} outperforms Segment {worst_cluster}.")
            st.write(f"**Rule of Thumb:** Mimic behavior of Segment {best_cluster} (Avg Lev: {cluster_stats.loc[best_cluster, 'leverage']:.1f}x).")
            st.write(f"**Action:** Flag traders matching Segment {worst_cluster} profile for risk review.")

else:
    st.info("Ready to Analyze. Please upload data via the Sidebar.")
