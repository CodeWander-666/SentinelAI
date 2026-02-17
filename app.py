import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import DataLoader
from src.analytics import Analyzer
from src.models import ModelEngine

# --- Page Config ---
st.set_page_config(
    page_title="PrimeTrade Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #41444b;
    }
    h1, h2, h3 { font-family: 'Arial', sans-serif; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---
@st.cache_data
def load_data():
    dl = DataLoader()
    # Ensure these paths match your local setup or Docker container
    return dl.load_and_process('data/sentiment.csv', 'data/trades.csv')

try:
    df = load_data()
    analyzer = Analyzer()
    engine = ModelEngine()
except Exception as e:
    st.error(f"System Error: Failed to load data. {e}")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.title("Configuration")
regime_filter = st.sidebar.multiselect(
    "Filter by Market Regime",
    options=df['value_classification'].unique(),
    default=df['value_classification'].unique()
)

# Filter Data
filtered_df = df[df['value_classification'].isin(regime_filter)]

# --- Main Dashboard ---
st.title("PrimeTrade Analytics Dashboard")
st.markdown("### Historical Trader Performance vs. Market Sentiment")

# 1. KPI Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Volume", f"${filtered_df['size'].sum()/1e6:.2f}M")
col2.metric("Avg Leverage", f"{filtered_df['leverage'].mean():.2f}x")
col3.metric("Net PnL", f"${filtered_df['closedPnL'].sum():,.0f}")
col4.metric("Avg Win Rate", f"{filtered_df['is_win'].mean()*100:.1f}%")

st.markdown("---")

# 2. Analysis Section
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Performance by Regime")
    # T-Test Logic
    ttest_res = analyzer.compare_regimes(df)
    
    if ttest_res['status'] == 'success':
        fig_box = px.box(df, x='value_classification', y='closedPnL', 
                        title="PnL Distribution: Fear vs Greed",
                        color='value_classification',
                        color_discrete_map={'Fear': '#FF4B4B', 'Greed': '#00CC96'})
        st.plotly_chart(fig_box, use_container_width=True)
        
        with st.expander("Statistical Significance (T-Test)"):
            st.write(f"**P-Value:** {ttest_res['p_value']:.4f}")
            if ttest_res['is_significant']:
                st.success("Result: Statistically Significant Difference detected.")
            else:
                st.warning("Result: No Significant Difference detected.")
    else:
        st.warning("Insufficient data for statistical comparison.")

with col_right:
    st.subheader("Behavioral Shifts")
    fig_scatter = px.scatter(filtered_df, x='leverage', y='closedPnL', 
                            color='value_classification', 
                            size='size',
                            title="Leverage vs PnL Correlation",
                            opacity=0.7)
    st.plotly_chart(fig_scatter, use_container_width=True)

# 3. Machine Learning Section
st.markdown("---")
st.subheader("Trader Segmentation (Clustering)")

if st.button("Run K-Means Clustering"):
    profiles = engine.cluster_traders(df)
    
    if not profiles.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_cluster = px.scatter(profiles, x='leverage', y='closedPnL', 
                                    color=profiles['Cluster'].astype(str),
                                    title="Trader Archetypes",
                                    labels={'color': 'Cluster ID'})
            st.plotly_chart(fig_cluster, use_container_width=True)
        with c2:
            st.write("**Cluster Summary**")
            st.dataframe(profiles.groupby('Cluster')[['leverage', 'closedPnL', 'is_win']].mean())
    else:
        st.error("Clustering failed due to data constraints.")

# 4. Strategy Output
st.markdown("---")
st.subheader("Automated Strategy Recommendations")

# Logic-based recommendations
avg_lev_fear = df[df['value_classification']=='Fear']['leverage'].mean()
avg_lev_greed = df[df['value_classification']=='Greed']['leverage'].mean()

rec_text = ""
if avg_lev_greed > avg_lev_fear:
    rec_text += "* **Risk Alert:** Leverage tends to increase during Greed regimes. Recommendation: Implement a 10x hard cap on leverage when Sentiment Index > 60.\n"

if ttest_res.get('fear_mean', 0) > ttest_res.get('greed_mean', 0):
    rec_text += "* **Opportunity:** Historical PnL is higher during Fear. Recommendation: Increase spot allocation strategies when Sentiment Index < 40."

st.info(rec_text if rec_text else "No specific anomalies detected in current dataset.")
