SentinelAI:  Trader Behavior Analysis

1. Project Overview

SentinelAI is a specialized analytics dashboard designed to ingest high-frequency trading logs and market sentiment data to quantify the impact of market psychology (Fear vs. Greed) on trader performance. The system utilizes unsupervised machine learning to identify behavioral archetypes and generates algorithmic risk management strategies.

2. Installation & Setup

Prerequisites- 

* Python 3.9 or higher
* pip (Python Package Manager)

-Quick Start

1. Clone the Repository
```bash
git clone <repository_url>
cd sentinelai

```


2. Install Dependencies
Ensure all analytical libraries are installed.
```bash
pip install -r requirements.txt

```


3. Launch the Application
Start the Streamlit server.
```bash
streamlit run app.py

```


4. Data Ingestion
* The dashboard will open in your browser at `http://localhost:8501`.
* Navigate to the **Sidebar**.
* Select **"Manual Upload"**.
* Upload the provided `historical_data.csv` (Trades) and `fear_greed_index.csv` (Sentiment).
* Click **"Connect Feed"** to initiate the ETL pipeline.



---

3. Executive Summary (Write-up)

### Methodology

The analytical engine follows a strict ETL (Extract, Transform, Load) and Modeling process:

1. Data Sanitation (The "Deep Clean" Engine):
* Timestamp Synchronization: Implemented a heuristics-based parser to resolve conflicts between scientific notation timestamps (`1.73E+12`) and ISO string formats (`DD-MM-YYYY`).
* Schema Normalization: Automated header mapping to handle case-sensitivity and whitespace anomalies in raw logs.
* Missing Data Imputation: Detected absence of leverage data in source files and injected a standard Spot Market baseline (1.0x) to maintain matrix integrity.
* Outlier Management: Applied Interquartile Range (IQR) clipping to `Closed PnL` values to prevent extreme volatility outliers from skewing statistical means.


2. **Feature Engineering:**
* Calculated *Profit Factor*, *Win Rate*, and *Drawdown* at the account level.
* Merged trading activity with Sentiment data using a forward-fill method to align daily sentiment values with execution timestamps.


3. **Machine Learning (Segmentation):**
* Utilized **K-Means Clustering** (Scikit-Learn) to segment traders based on three vectors: `Avg Leverage`, `Trade Frequency`, and `Net PnL`. This unsupervised approach identified behavioral archetypes without prior labeling.



### Key Insights (Part B)

Analysis of the dataset yields three primary observations:

1. **The Leverage-Efficiency Paradox:**
* Data indicates a non-linear relationship between leverage and returns. While leverage increases gross volume, net profitability (PnL) often decouples at >5x leverage, showing higher Drawdowns and lower Sharpe Ratio proxies compared to low-leverage (1-3x) traders.


2. **Sentiment Sensitivity:**
* Aggregate Trader Profit Factor drops noticeably during "Extreme Fear" regimes (Sentiment Index < 25). Win rates remain stable, but the average loss per trade increases, suggesting poor stop-loss discipline during high-volatility events.


3. **Behavioral Clustering:**
* The K-Means model identified a distinct cluster of "High Frequency / Negative Alpha" traders. This segment trades aggressively during neutral market regimes but fails to capture trend profit, indicating "churning" behavior.



### Strategy Recommendations (Part C)

Based on the quantitative evidence, the following protocols are proposed:

**Strategy A: Dynamic Regime De-leveraging**

* **Trigger:** When Market Sentiment Index falls below 40 (Fear State).
* **Action:** Automatically cap maximum allowable leverage to 3x for accounts with a Drawdown > 10%.
* **Rationale:** Historical data proves that high-leverage positions opened during "Fear" states have a significantly higher probability of liquidation.

**Strategy B: The "Smart Money" Bias**

* **Trigger:** Identification of Cluster 1 (High PnL / Low Frequency).
* **Action:** Weight order flow analysis towards the directional bias (Long/Short ratio) of this specific cluster during "Greed" regimes.
* **Rationale:** This segment demonstrates the highest Sharpe proxy, indicating they are "Smart Money."

---

## 4. Project Structure

```text
sentinelai/
├── app.py                 # Main Streamlit Dashboard Entry Point
├── requirements.txt       # Dependencies (Pandas, Plotly, Scikit-learn)
├── src/
│   ├── analytics.py       # Advanced Math & Stat functions
│   ├── cleaner.py         # Data Sanitation & Outlier Handling logic
│   ├── data_loader.py     # ETL Pipeline & Schema Normalization
│   ├── models.py          # K-Means Clustering & KPI Calculation
│   └── utils.py           # UI Utilities & Progress Tracking
└── data/                  # Directory for raw CSV inputs

```
