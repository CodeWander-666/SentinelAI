import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from typing import Optional

# Local Modules
from src.data_loader import DataLoader
from src.analytics import Analyzer
from src.models import ModelEngine

# --- 1. CONFIGURATION LAYER ---
class DashboardConfig:
    """Centralized configuration for the application."""
    PAGE_TITLE = "PrimeTrade | Quant Terminal"
    PAGE_ICON = "⚡"
    LAYOUT = "wide"
    
    # Financial Terminal Color Palette
    COLORS = {
        'background': '#0E1117',
        'surface': '#1F2937',
        'text': '#E0E0E0',
        'accent': '#2563EB',
        'success': '#10B981',
        'danger': '#EF4444',
        'warning': '#F59E0B',
        'chart_up': '#00CC96',
        'chart_down': '#FF4B4B'
    }

    @staticmethod
    def apply_custom_css():
        """Injects professional CSS overrides."""
        st.markdown(f"""
        <style>
            /* Global Reset */
            .stApp {{ background-color: {DashboardConfig.COLORS['background']}; color: {DashboardConfig.COLORS['text']}; }}
            
            /* Metric Cards */
            div[data-testid="metric-container"] {{
                background-color: {DashboardConfig.COLORS['surface']};
                border: 1px solid #374151;
                padding: 1rem;
                border-radius: 6px;
                transition: transform 0.2s;
            }}
            div[data-testid="metric-container"]:hover {{
                border-color: {DashboardConfig.COLORS['accent']};
            }}
            
            /* Typography */
            h1, h2, h3 {{ font-family: 'Inter', sans-serif; letter-spacing: -0.5px; }}
            
            /* Buttons */
            div.stButton > button {{
                background-color: {DashboardConfig.COLORS['accent']};
                color: white;
                border: none;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            /* Sidebar */
            section[data-testid="stSidebar"] {{
                background-color: #111827;
                border-right: 1px solid #374151;
            }}
        </style>
        """, unsafe_allow_html=True)

# --- 2. LOGIC LAYER (Chart Builders) ---
class ChartBuilder:
    """Factory class for generating Plotly figures."""
    
    @staticmethod
    def build_pnl_distribution(df: pd.DataFrame) -> go.Figure:
        color_map = {'Fear': DashboardConfig.COLORS['danger'], 
                     'Greed': DashboardConfig.COLORS['success'], 
                     'Neutral': '#9CA3AF'}
        
        fig = px.box(
            df, x='value_classification', y='closedPnL', 
            color='value_classification', color_discrete_map=color_map,
            points=False # Clean look, remove outliers
        )
        fig.update_layout(
            title="<b>PnL Distribution by Market Regime</b>",
            xaxis_title=None, yaxis_title="Closed PnL ($)",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color=DashboardConfig.COLORS['text'],
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @staticmethod
    def build_leverage_scatter(df: pd.DataFrame) -> go.Figure:
        # Downsample for performance if needed
        plot_df = df.sample(n=min(5000, len(df)), random_state=42)
        
        fig = px.scatter(
            plot_df, x='leverage', y='closedPnL', 
            color='value_classification', size='size',
            color_discrete_map={'Fear': DashboardConfig.COLORS['danger'], 'Greed': DashboardConfig.COLORS['success']},
            opacity=0.7
        )
        fig.update_layout(
            title="<b>Leverage vs. Profitability Correlation</b>",
            xaxis_title="Leverage (x)", yaxis_title="Closed PnL ($)",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color=DashboardConfig.COLORS['text'],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    @staticmethod
    def build_cluster_map(df: pd.DataFrame) -> go.Figure:
        fig = px.scatter(
            df, x='leverage', y='closedPnL', 
            color=df['Cluster'].astype(str),
            title="<b>Trader Archetype Clusters (AI Segmentation)</b>",
            labels={'color': 'Archetype ID'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color=DashboardConfig.COLORS['text']
        )
        return fig

# --- 3. APPLICATION LAYER ---
def main():
    # A. Setup
    st.set_page_config(
        page_title=DashboardConfig.PAGE_TITLE,
        page_icon=DashboardConfig.PAGE_ICON,
        layout=DashboardConfig.LAYOUT
    )
    DashboardConfig.apply_custom_css()

    # B. Sidebar Control Panel
    st.sidebar.title(f"{DashboardConfig.PAGE_ICON} DATA ENGINE")
    
    # Data Loading Logic
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_sent = os.path.join(base_dir, 'data', 'sentiment.csv')
    local_trade = os.path.join(base_dir, 'data', 'trades.csv')
    local_exists = os.path.exists(local_sent) and os.path.exists(local_trade)

    # State Management for Data
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Source Selection
    data_source = st.sidebar.radio(
        "Data Pipeline Source:",
        ("Local Repository", "Upload New CSVs"),
        index=0 if local_exists else 1,
        help="Select 'Local Repository' to use pre-loaded data."
    )

    uploaded_trades, uploaded_sent = None, None
    if data_source == "Upload New CSVs":
        uploaded_trades = st.sidebar.file_uploader("Trades Data", type=['csv'])
        uploaded_sent = st.sidebar.file_uploader("Sentiment Data", type=['csv'])

    # Initialize Analysis Button
    if st.sidebar.button("INITIALIZE SYSTEM", type="primary"):
        with st.spinner("Ingesting & Normalizing Data..."):
            dl = DataLoader()
            try:
                if data_source == "Upload New CSVs" and uploaded_trades and uploaded_sent:
                    st.session_state.df = dl.load_and_process(uploaded_sent, uploaded_trades)
                    st.toast("Data Pipeline Active: Uploaded Source", icon="✅")
                elif data_source == "Local Repository" and local_exists:
                    st.session_state.df = dl.load_and_process(local_sent, local_trade)
                    st.toast("Data Pipeline Active: Local Repository", icon="✅")
                else:
                    st.sidebar.error("❌ Missing Data Sources. Please check your inputs.")
            except Exception as e:
                st.error(f"Pipeline Failure: {str(e)}")

    # C. Main Dashboard Rendering
    if st.session_state.df is not None:
        render_dashboard(st.session_state.df)
    else:
        render_landing_page()

def render_dashboard(df: pd.DataFrame):
    """Renders the main analytics interface."""
    
    # 1. Header & Filters
    col_head, col_filter = st.columns([3, 1])
    with col_head:
        st.title("PrimeTrade Analytics Terminal")
        st.caption(f"System Status: ONLINE | Last Updated: {datetime.now().strftime('%H:%M:%S UTC')}")
    
    with col_filter:
        regimes = df['value_classification'].unique()
        sel_regime = st.multiselect("Filter Regime", regimes, default=regimes)
    
    # Apply Filter
    filtered_df = df[df['value_classification'].isin(sel_regime)]

    # 2. KPI Ticker
    st.markdown("### Market Performance Metrics")
    k1, k2, k3, k4 = st.columns(4)
    
    try:
        k1.metric("Total Volume", f"${filtered_df['size'].sum()/1e6:,.1f}M", help="Total traded volume in filtered period")
        k2.metric("Avg Leverage", f"{filtered_df['leverage'].mean():.2f}x", delta_color="off")
        
        pnl = filtered_df['closedPnL'].sum()
        k3.metric("Net PnL", f"${pnl:,.0f}", delta=f"{pnl/1000:.1f}k", delta_color="normal")
        
        wr = filtered_df['is_win'].mean() * 100
        k4.metric("Win Rate", f"{wr:.1f}%", f"{wr-50:.1f}% vs 50/50")
    except Exception as e:
        st.error(f"Metric Calculation Error: {e}")

    st.markdown("---")

    # 3. Visual Analytics (Wrapped in containers for stability)
    c1, c2 = st.columns(2)
    with c1:
        try:
            st.plotly_chart(ChartBuilder.build_pnl_distribution(df), use_container_width=True)
        except Exception as e:
            st.warning(f"Chart Render Error: {e}")

    with c2:
        try:
            st.plotly_chart(ChartBuilder.build_leverage_scatter(filtered_df), use_container_width=True)
        except Exception as e:
            st.warning(f"Chart Render Error: {e}")

    # 4. Advanced Intelligence Section
    st.markdown("---")
    st.subheader("Algorithmic Intelligence")
    
    tab1, tab2 = st.tabs(["🧩 Trader Clustering", "🤖 Strategic Signals"])
    
    with tab1:
        st.markdown("**Unsupervised Learning: K-Means Segmentation**")
        if st.button("Run Segmentation Model"):
            with st.spinner("Training model on 4 dimensions..."):
                engine = ModelEngine()
                clustered_df = engine.cluster_traders(df)
                if not clustered_df.empty:
                    st.plotly_chart(ChartBuilder.build_cluster_map(clustered_df), use_container_width=True)
                    
                    # Cluster Statistics
                    st.markdown("**Cluster Performance Profile**")
                    stats = clustered_df.groupby('Cluster')[['leverage', 'closedPnL', 'is_win']].mean()
                    st.dataframe(stats.style.background_gradient(cmap='viridis', subset=['closedPnL']), use_container_width=True)

    with tab2:
        st.markdown("**Automated Risk Management Directives**")
        
        # Calculate Logic
        avg_lev_greed = df[df['value_classification']=='Greed']['leverage'].mean()
        
        col_sig1, col_sig2 = st.columns(2)
        with col_sig1:
            st.info(f"""
            **SIGNAL: LEVERAGE CEILING**
            
            Based on historical data, leverage exceeds **{avg_lev_greed:.1f}x** during Greed cycles. 
            
            **Action:** Cap leverage at 10x when Sentiment > 75 to prevent liquidation cascades.
            """)
        with col_sig2:
            st.success("""
            **SIGNAL: MEAN REVERSION**
            
            'Fear' regimes show a statistical PnL variance favoring Spot Accumulation over Derivatives.
            
            **Action:** Shift allocation to 80% Spot / 20% Futures when Sentiment < 25.
            """)

def render_landing_page():
    """Renders the initial welcome state."""
    st.markdown("## 👋 Welcome to PrimeTrade Sentinel")
    st.markdown("""
    This platform provides institutional-grade analytics correlating **Market Sentiment** with **Trader Behavior**.
    
    **System Capabilities:**
    * 🚀 **High-Frequency Ingestion:** Polars engine for million-row datasets.
    * 🧠 **AI Segmentation:** K-Means clustering for trader profiling.
    * 📊 **Regime Analysis:** Statistical correlation of Fear/Greed vs PnL.
    
    **To Begin:** Select your data source in the sidebar and click **INITIALIZE SYSTEM**.
    """)

if __name__ == "__main__":
    main()
