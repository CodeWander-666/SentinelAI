# PrimeTrade Analytics Suite

## Overview
This repository contains a comprehensive data analysis pipeline designed to investigate the correlation between Crypto Market Sentiment (Fear & Greed Index) and historical trader performance on the Hyperliquid platform.

## Architecture
The project is structured as a modular Python application:
* **ETL Layer (`src/data_loader.py`)**: robust ingestion, cleaning, and daily aggregation of tick-level trade data.
* **Analytics Layer (`src/analytics.py`)**: Statistical hypothesis testing (T-Tests) to validate performance divergences.
* **Intelligence Layer (`src/models.py`)**: Unsupervised learning (K-Means) for trader segmentation and XGBoost for predictive modeling.
* **Visualization (`app.py`)**: Interactive Streamlit dashboard for real-time insight generation.

## Key Findings
1. **Leverage Behavior**: Analysis indicates a statistically significant increase in leverage usage during "Greed" regimes, often correlating with increased drawdown risk.
2. **Trader Segmentation**: K-Means clustering identified three distinct trader archetypes: High-Frequency Scalpers, Risk-Averse Whales, and High-Leverage Retail.

## Installation & Usage

### Option 1: Local Python Environment
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
