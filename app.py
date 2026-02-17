import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import DataLoader
from src.analytics import Analytics

st.set_page_config(layout="wide")
st.title("📊 Trader Performance vs Market Sentiment")

# Load data
@st.cache_data
def load_data():
    loader = DataLoader()
    df = loader.load_and_process("data/sentiment.csv", "data/trades.csv")
    return df.to_pandas()

df = load_data()
df['date_dt'] = pd.to_datetime(df['date_dt'])

# Sidebar filters
st.sidebar.header("Filters")
selected_accounts = st.sidebar.multiselect("Select Accounts", options=df['account'].unique(), default=df['account'].unique()[:5])
date_range = st.sidebar.date_input("Date Range", [df['date_dt'].min(), df['date_dt'].max()])
sentiment_filter = st.sidebar.multiselect("Sentiment", options=df['value_classification'].unique(), default=df['value_classification'].unique())

filtered_df = df[(df['account'].isin(selected_accounts)) &
                 (df['date_dt'] >= pd.to_datetime(date_range[0])) &
                 (df['date_dt'] <= pd.to_datetime(date_range[1])) &
                 (df['value_classification'].isin(sentiment_filter))]

# Main panel
col1, col2, col3 = st.columns(3)
col1.metric("Total PnL", f"${filtered_df['total_pnl'].sum():,.0f}")
col2.metric("Avg Win Rate", f"{filtered_df['win_rate'].mean():.2%}")
col3.metric("Avg Leverage", f"{filtered_df['avg_leverage'].mean():.1f}x")

# Charts
tab1, tab2, tab3 = st.tabs(["PnL vs Sentiment", "Trader Metrics", "Correlation"])

with tab1:
    fig = px.box(filtered_df, x='value_classification', y='total_pnl', title='Daily PnL by Sentiment')
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.bar(filtered_df.groupby('value_classification')[['total_pnl', 'trade_count']].mean().reset_index(),
                  x='value_classification', y='total_pnl', title='Average PnL on Fear vs Greed Days')
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    metric = st.selectbox("Select Metric", ['avg_leverage', 'win_rate', 'trade_count', 'long_ratio'])
    fig = px.box(filtered_df, x='value_classification', y=metric, title=f'{metric} by Sentiment')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    corr_cols = ['total_pnl', 'win_rate', 'avg_leverage', 'trade_count', 'long_ratio', 'value']
    corr_df = filtered_df[corr_cols].corr()
    fig = px.imshow(corr_df, text_auto=True, color_continuous_scale='RdBu_r', title='Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(filtered_df.head(100))
