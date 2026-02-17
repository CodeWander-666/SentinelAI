import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import DataLoader
from src.utils import PipelineTracker
from src.models import ModelEngine


st.set_page_config(page_title="SentinelAI ", layout="wide", page_icon="AI")


st.markdown("""
<style>
    /* Dark Mode Global */
    .stApp { background-color: #0E1117; color: #E0E0E0; font-family: 'Segoe UI', sans-serif; }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.9rem; }
    div[data-testid="stMetricValue"] { color: #58a6ff; font-weight: 600; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border: none; color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #58a6ff; border-bottom: 2px solid #58a6ff; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #010409; border-right: 1px solid #30363d; }
    
    /* Tables */
    div[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'df' not in st.session_state: st.session_state.df = None

# --- SIDEBAR ---
with st.sidebar:
    st.title(" SENTINEL AI")
    st.caption("Trader Performance vs Market Sentiment")
    st.divider()
    
    source = st.radio("DATA SOURCE", ["Repository", "Manual Upload"], index=1)
    
    u_t, u_s = None, None
    if source == "Manual Upload":
        u_t = st.file_uploader("Upload Trades", type="csv")
        u_s = st.file_uploader("Upload Sentiment", type="csv")
    
    if st.button("CONNECT FEED", type="primary"):
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
            st.toast("System Online", icon="✅")
        except Exception as e:
            st.error("Connection Failed")

# --- MAIN DASHBOARD ---
if st.session_state.df is not None:
    df = st.session_state.df.sort_values('date_dt')
    engine = ModelEngine()
    stats = engine.calculate_stats(df)
    
    # 1. TOP KPI ROW
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Net Profit (PnL)", f"${stats['Total PnL']:,.0f}")
    c2.metric("Profit Factor", f"{stats['Profit Factor']:.2f}")
    c3.metric("Win Rate", f"{stats['Win Rate']:.1f}%")
    c4.metric("Max Drawdown", f"${stats['Max Drawdown']:,.0f}")
    c5.metric("Avg Leverage", f"{df['leverage'].mean():.1f}x")
    
    st.divider()

    # 2. MAIN TABS
    tab_perf, tab_edge, tab_data, tab_ai = st.tabs([
        " PERFORMANCE", " SENTIMENT EDGE", " TRADE JOURNAL", " AI STRATEGY"
    ])

    # TAB 1: PERFORMANCE (The Trader View)
    with tab_perf:
        # Equity Curve
        df['Equity'] = df['closedPnL'].cumsum()
        fig_eq = px.area(df, x='date_dt', y='Equity', title="Cumulative Equity Curve", template="plotly_dark")
        fig_eq.update_traces(line_color='#238636', fillcolor='rgba(35, 134, 54, 0.1)')
        st.plotly_chart(fig_eq, use_container_width=True)
        
        c_left, c_right = st.columns(2)
        with c_left:
            # Daily PnL Bar
            fig_bar = px.bar(df, x='date_dt', y='closedPnL', title="Daily PnL", 
                             color='closedPnL', color_continuous_scale=['#da3633', '#238636'], template="plotly_dark")
            fig_bar.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        with c_right:
            # Leverage vs PnL
            fig_lev = px.scatter(df, x='leverage', y='closedPnL', size='size', 
                                 title="Leverage Efficiency", template="plotly_dark", opacity=0.7)
            st.plotly_chart(fig_lev, use_container_width=True)

    # TAB 2: SENTIMENT (Part B Requirements)
    with tab_edge:
        st.subheader("Market Regime Analysis (Fear vs Greed)")
        
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Does Fear/Greed impact returns?")
            fig_box = px.box(df, x='value_classification', y='closedPnL', 
                             color='value_classification', template="plotly_dark",
                             color_discrete_map={'Fear': '#da3633', 'Greed': '#238636', 'Neutral': 'gray'})
            st.plotly_chart(fig_box, use_container_width=True)
        
        with c2:
            st.caption("Do we over-leverage in Greed?")
            fig_vio = px.violin(df, x='value_classification', y='leverage', box=True, 
                                color='value_classification', template="plotly_dark")
            st.plotly_chart(fig_vio, use_container_width=True)

    # TAB 3: DATA (Transparency)
    with tab_data:
        st.subheader("Daily Trading Log")
        # Clean table for display
        display_df = df[['date_dt', 'account', 'closedPnL', 'leverage', 'trade_count', 'value_classification']].copy()
        display_df['date_dt'] = display_df['date_dt'].dt.date
        st.dataframe(
            display_df.style.background_gradient(subset=['closedPnL'], cmap='RdYlGn'),
            use_container_width=True,
            height=500
        )

    # TAB 4: STRATEGY (Part C & Bonus)
    with tab_ai:
        st.subheader("Algorithmic Insights")
        
        # Run Clustering
        df_ai = engine.cluster_traders(df)
        
        col_viz, col_txt = st.columns([2, 1])
        
        with col_viz:
            fig_clus = px.scatter(df_ai, x='leverage', y='closedPnL', color='Cluster', 
                                  title="Trader Segmentation (Risk vs Reward)", template="plotly_dark")
            st.plotly_chart(fig_clus, use_container_width=True)
            
        with col_txt:
            st.info("###  RISK PROTOCOL")
            fear_pnl = df[df['value_classification']=='Fear']['closedPnL'].mean()
            if fear_pnl < 0:
                st.write("**Condition:** Negative Expectancy during FEAR.")
                st.write("**Action:** Reduce leverage by 50% when Sentiment < 25.")
            else:
                st.write("**Condition:** Stable performance in all regimes.")
                st.write("**Action:** Maintain standard sizing.")
                
            st.success("###  ALPHA SIGNAL")
            st.write("**Observation:** High leverage clusters correlate with higher volatility but not necessarily higher Sharpes.")
            st.write("**Action:** Cap leverage at 3x for consistent compounding.")

else:
    # Empty State
    st.markdown("""
    <div style='text-align: center; padding: 50px; color: #30363d;'>
        <h1>Waiting for Data Stream</h1>
        <p>Please upload your trading history and sentiment data in the sidebar to activate the terminal.</p>
    </div>
    """, unsafe_allow_html=True)
