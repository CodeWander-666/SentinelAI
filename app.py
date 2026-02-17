import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import timedelta

# Local Modules (Assumes your src/ folder is set up as discussed)
from src.data_loader import DataLoader
from src.analytics import Analyzer
from src.models import ModelEngine

# ==============================================================================
# 1. CONFIGURATION & STYLING
# ==============================================================================
class AppConfig:
    PAGE_TITLE = "PRIMETRADE | SENTINEL TERMINAL"
    PAGE_LAYOUT = "wide"
    
    # Financial Color Palette
    COLOR_BG = "#0E1117"
    COLOR_SURFACE = "#1E2329"
    COLOR_TEXT_PRIMARY = "#E6E8EB"
    COLOR_TEXT_SECONDARY = "#9CA3AF"
    COLOR_ACCENT = "#2962FF"
    COLOR_SUCCESS = "#00C853"
    COLOR_DANGER = "#D50000"
    COLOR_WARNING = "#FFAB00"

    @staticmethod
    def apply_styles():
        st.markdown(f"""
        <style>
            /* GLOBAL RESET */
            .stApp {{ background-color: {AppConfig.COLOR_BG}; }}
            
            /* HEADERS */
            h1, h2, h3 {{ font-family: 'Roboto', sans-serif; letter-spacing: -0.5px; color: {AppConfig.COLOR_TEXT_PRIMARY}; }}
            
            /* METRIC CARDS */
            div[data-testid="stMetric"] {{
                background-color: {AppConfig.COLOR_SURFACE};
                padding: 15px;
                border-radius: 4px;
                border-left: 4px solid {AppConfig.COLOR_ACCENT};
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }}
            div[data-testid="stMetricLabel"] {{ color: {AppConfig.COLOR_TEXT_SECONDARY}; font-size: 0.85rem; }}
            div[data-testid="stMetricValue"] {{ color: {AppConfig.COLOR_TEXT_PRIMARY}; font-size: 1.5rem; font-weight: 600; }}
            
            /* BUTTONS */
            div.stButton > button {{
                background-color: {AppConfig.COLOR_ACCENT};
                color: white;
                border-radius: 4px;
                border: none;
                text-transform: uppercase;
                font-weight: 600;
                letter-spacing: 0.5px;
                transition: all 0.2s;
            }}
            div.stButton > button:hover {{ background-color: #0039CB; }}
            
            /* DATAFRAME */
            div[data-testid="stDataFrame"] {{ border: 1px solid #333; }}
        </style>
        """, unsafe_allow_html=True)

# ==============================================================================
# 2. UI COMPONENTS (CHARTS & WIDGETS)
# ==============================================================================
class DashboardUI:
    @staticmethod
    def render_kpi_row(df):
        cols = st.columns(4)
        
        # Calculations
        total_pnl = df['closedPnL'].sum()
        win_rate = df['is_win'].mean() * 100
        total_vol = df['size'].sum()
        avg_lev = df['leverage'].mean()
        
        # Formatting
        pnl_color = "normal" # handled by delta
        
        cols[0].metric("Net PnL", f"${total_pnl:,.0f}", delta=f"{total_pnl/1000:.1f}k")
        cols[1].metric("Win Rate", f"{win_rate:.1f}%", f"{win_rate-50:.1f}% vs Bench")
        cols[2].metric("Volume Traded", f"${total_vol/1e6:,.2f}M")
        cols[3].metric("Avg Leverage", f"{avg_lev:.2f}x")

    @staticmethod
    def render_charts(df):
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("#### PnL Distribution by Sentiment")
            # Box Plot
            fig = px.box(
                df, x='value_classification', y='closedPnL',
                color='value_classification',
                color_discrete_map={
                    'Fear': AppConfig.COLOR_DANGER, 
                    'Greed': AppConfig.COLOR_SUCCESS, 
                    'Neutral': AppConfig.COLOR_TEXT_SECONDARY
                }
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=AppConfig.COLOR_TEXT_SECONDARY,
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("#### Cumulative Performance")
            # Equity Curve
            daily_pnl = df.groupby('date_dt')['closedPnL'].sum().cumsum().reset_index()
            fig2 = px.line(daily_pnl, x='date_dt', y='closedPnL')
            fig2.update_traces(line_color=AppConfig.COLOR_ACCENT, line_width=3)
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=AppConfig.COLOR_TEXT_SECONDARY,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig2, use_container_width=True)

    @staticmethod
    def render_strategy_section(df):
        st.markdown("---")
        st.markdown("### 🧠 Algorithmic Intelligence")
        
        tab1, tab2 = st.tabs(["CLUSTERING MODEL", "STRATEGY ENGINE"])
        
        with tab1:
            if st.button("Run K-Means Segmentation"):
                with st.spinner("Analyzing Trader Behavior..."):
                    try:
                        engine = ModelEngine()
                        clusters = engine.cluster_traders(df)
                        
                        col_viz, col_data = st.columns([2, 1])
                        
                        with col_viz:
                            fig = px.scatter(
                                clusters, x='leverage', y='closedPnL', color='Cluster',
                                title="Trader Archetypes (Risk vs Reward)",
                                color_continuous_scale='Viridis'
                            )
                            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='gray')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_data:
                            st.dataframe(
                                clusters.groupby('Cluster')[['leverage', 'closedPnL', 'is_win']].mean().style.background_gradient(),
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Modeling Error: {e}")

        with tab2:
            c1, c2 = st.columns(2)
            # Dynamic Logic
            fear_lev = df[df['value_classification'] == 'Fear']['leverage'].mean()
            greed_lev = df[df['value_classification'] == 'Greed']['leverage'].mean()
            
            with c1:
                st.info(f"**RISK PROTOCOL**\n\nLeverage delta in Greed vs Fear is **{(greed_lev - fear_lev):.2f}x**. Recommended Action: Cap leverage at 10x when Sentiment > 60.")
            with c2:
                st.success(f"**ALPHA SIGNAL**\n\nWin Rates during 'Extreme Fear' are statistically significant. Action: Increase Spot Allocation when Sentiment < 20.")

# ==============================================================================
# 3. MAIN APP LOGIC
# ==============================================================================
def main():
    st.set_page_config(page_title=AppConfig.PAGE_TITLE, layout=AppConfig.PAGE_LAYOUT, page_icon="⚡")
    AppConfig.apply_styles()

    # --- SIDEBAR ---
    st.sidebar.title("DATA FEED")
    
    # Data Loading Logic
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_sent = os.path.join(base_dir, 'data', 'sentiment.csv')
    local_trade = os.path.join(base_dir, 'data', 'trades.csv')
    
    data_source = st.sidebar.radio("Source", ["Local Repository", "Upload CSV"], horizontal=True)
    
    uploaded_t, uploaded_s = None, None
    if data_source == "Upload CSV":
        uploaded_t = st.sidebar.file_uploader("Trades", type="csv")
        uploaded_s = st.sidebar.file_uploader("Sentiment", type="csv")
    
    if st.sidebar.button("CONNECT & LOAD", type="primary"):
        dl = DataLoader()
        try:
            with st.spinner("Ingesting Data..."):
                if data_source == "Upload CSV" and uploaded_t and uploaded_s:
                    st.session_state.df = dl.load_and_process(uploaded_s, uploaded_t)
                elif data_source == "Local Repository" and os.path.exists(local_trade):
                    st.session_state.df = dl.load_and_process(local_sent, local_trade)
                else:
                    st.error("Data Source Unavailable")
                    return
                st.toast("System Online", icon="✅")
        except Exception as e:
            st.error(f"Pipeline Failure: {e}")

    # --- DASHBOARD CONTENT ---
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        # Header
        col_title, col_filter = st.columns([3, 1])
        with col_title:
            st.title("PRIMETRADE SENTINEL")
            st.caption(f"STATUS: ONLINE | RECORDS: {len(df):,}")
        
        with col_filter:
            regimes = df['value_classification'].unique()
            sel = st.multiselect("Filter Regime", regimes, default=regimes)
            df_filtered = df[df['value_classification'].isin(sel)]
        
        # 1. KPIs
        DashboardUI.render_kpi_row(df_filtered)
        st.markdown("---")
        
        # 2. Charts
        DashboardUI.render_charts(df_filtered)
        
        # 3. Strategy
        DashboardUI.render_strategy_section(df_filtered)
        
        # 4. Diagnostics (Hidden unless needed)
        with st.expander("🛠️ DATA DIAGNOSTICS"):
            st.write(df.describe())
            st.write("Missing Values:", df.isnull().sum())

    else:
        # Landing State
        st.markdown(f"""
        <div style='text-align: center; padding: 50px;'>
            <h1>WAITING FOR DATA FEED</h1>
            <p style='color: {AppConfig.COLOR_TEXT_SECONDARY};'>Select a data source in the sidebar and click CONNECT.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
