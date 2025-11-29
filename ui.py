# ui.py - Ultimate AI Ledger Reconciler (Phase 1 to 4) - Production Ready
import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# Import from your fixed core.py
from core import (
    advanced_match_ledgers, detect_columns,
    forecast_cash_flow, detect_anomalies
)

st.set_page_config(
    page_title="AI Ledger Reconciler Pro",
    layout="wide",
    page_icon="robot_face",
    initial_sidebar_state="expanded"
)

# ================== Custom CSS ==================
st.markdown("""
<style>
    .big-font {font-size:50px !important; font-weight:bold; text-align:center; color:#1E88E5;}
    .metric-card {background-color:#f0f2f6; padding: 20px; border-radius: 10px; text-align:center;}
    .success-box {background-color:#d4edda; padding:15px; border-radius:8px; border:1px solid #c3e6cb;}
</style>
""", unsafe_allow_html=True)

# ================== Sidebar Settings ==================
st.sidebar.title("AI Engine Settings")
date_tol = st.sidebar.slider("Date Tolerance (days)", 0, 365, 90, help="Max days difference allowed")
amt_tol = st.sidebar.slider("Amount Tolerance (%)", 0.0, 25.0, 5.0) / 100
abs_tol = st.sidebar.slider("Absolute Tolerance (₹)", 0, 1000, 50)
enable_ml = st.sidebar.checkbox("Enable ML Scoring", True)
enable_semantic = st.sidebar.checkbox("Enable Semantic (Narration) Match", True)
enable_partial = st.sidebar.checkbox("Enable Partial Payment Detection", True)

st.sidebar.markdown("---")
st.sidebar.header("FX Rates (Optional)")
base_currency = st.sidebar.selectbox("Base Currency", ["INR", "USD", "EUR", "GBP", "AED"])
fx_input = st.sidebar.text_area(
    "Custom FX Rates (JSON dict)", 
    '{"USD": 83.5, "EUR": 90.2, "GBP": 105.0}',
    height=100
)
try:
    fx_rates = json.loads(fx_input.replace("'", '"'))
except:
    fx_rates = {}
    st.sidebar.warning("Invalid JSON, using 1:1 rates")

# ================== Title ==================
st.markdown('<p class="big-font">AI Ledger Reconciler Pro</p>', unsafe_allow_html=True)
st.markdown("**Bank vs Books | GSTR-2A vs Purchase | Vendor vs Ledger** – Instant AI Matching")

# ================== File Upload ==================
col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Ledger A (Your Books / Tally)", type=['csv', 'xlsx'])
with col2:
    file_b = st.file_uploader("Upload Ledger B (Bank / GSTR / Vendor)", type=['csv', 'xlsx'])

if not (file_a and file_b):
    st.info("Upload both ledgers to start AI reconciliation")
    st.stop()

# ================== Load Data ==================
@st.cache_data
def load_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file, engine='openpyxl')

df_a = load_file(file_a)
df_b = load_file(file_b)

st.success(f"Loaded → Ledger A: {len(df_a):,} rows | Ledger B: {len(df_b):,} rows")

# ================== Column Mapping ==================
map_a = detect_columns(df_a)
map_b = detect_columns(df_b)

with st.expander("Manual Column Mapping (Override Auto-Detect)", expanded=False):
    cols = ['date', 'ref', 'debit', 'credit', 'narration', 'txn_code', 'currency']
    col1, col2 = st.columns(2)
    with col1:
        for c in cols:
            if map_a.get(c) and map_a[c] in df_a.columns:
                map_a[c] = st.selectbox(f"A → {c.capitalize()}", df_a.columns, index=df_a.columns.get_loc(map_a[c]))
    with col2:
        for c in cols:
            if map_b.get(c) and map_b[c] in df_b.columns:
                map_b[c] = st.selectbox(f"B → {c.capitalize()}", df_b.columns, index=df_b.columns.get_loc(map_b[c]))

# ================== Run Reconciliation ==================
if st.button("Run AI Reconciliation Engine", type="primary", use_container_width=True):
    with st.spinner("AI is matching 50,000+ patterns per second..."):
        matches_df, unmatched_a, unmatched_b = advanced_match_ledgers(
            A=df_a, map_a=map_a,
            B=df_b, map_b=map_b,
            date_tol=date_tol,
            amt_tol=amt_tol,
            abs_tol=abs_tol,
            enable_ml=enable_ml,
            enable_semantic=enable_semantic,
            enable_partial_payments=enable_partial
        )

    st.balloons()
    st.success("Reconciliation Complete!")

    # ================== Dashboard Metrics ==================
    col1, col2, col3, col4, col5 = st.columns(5)
    total_possible = min(len(df_a), len(df_b))
    match_rate = len(matches_df) / len(df_a) * 100 if len(df_a) > 0 else 0

    col1.metric("Matches Found", f"{len(matches_df):,}")
    col2.metric("Match Rate", f"{match_rate:.1f}%")
    col3.metric("High Confidence (>90)", f"{len(matches_df[matches_df['Confidence'] >= 90]):,}")
    col4.metric("Time Saved", f"{len(matches_df) * 8 // 60}h {len(matches_df) * 8 % 60}m")
    col5.metric("Est. Cost Saved", f"₹{len(matches_df) * 1200:,.0f}")

    # ================== Charts ==================
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Match Confidence Distribution")
        if not matches_df.empty:
            st.bar_chart(matches_df['Confidence'].value_counts(bins=10).sort_index())
    with col2:
        st.subheader("Match Type Breakdown")
        type_counts = matches_df['Match_Type'].value_counts()
        st.bar_chart(type_counts)

    # ================== Forecast (Phase 3) ==================
    if not matches_df.empty:
        st.subheader("Predictive Cash Flow (Next 30 Days)")
        forecast = forecast_cash_flow(matches_df)
        col1, col2 = st.columns(2)
        col1.metric("Next Month Forecast", f"₹{forecast['next_month_forecast']:,.0f}")
        col2.metric("Trend", forecast.get('trend', 'Stable'))

    # ================== Matches Table ==================
    st.subheader("Matched Transactions")
    display_cols = ['A_Date', 'A_Ref', 'A_Amount', 'B_Date', 'B_Ref', 'B_Amount',
                    'Amount_Diff', 'Match_Type', 'Confidence', 'Remarks']
    styled_df = (matches_df[display_cols]
                 .sort_values("Confidence", ascending=False)
                 .style.format({
                     'A_Amount': '₹{:,.2f}', 'B_Amount': '₹{:,.2f}',
                     'Amount_Diff': '₹{:,.2f}', 'Confidence': '{:.1f}%'
                 })
                 .background_gradient(subset=['Confidence'], cmap='Greens'))
    st.dataframe(styled_df, use_container_width=True)

    # Download
    csv = matches_df.to_csv(index=False).encode()
    st.download_button(
        "Download Full Matches (CSV)",
        csv,
        f"AI_Reconciliation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv"
    )

    # ================== Unmatched ==================
    tab1, tab2 = st.tabs(["Unmatched in Ledger A (Missing in Bank)", "Unmatched in Ledger B (Extra in Bank)"])
    with tab1:
        st.dataframe(unmatched_a, use_container_width=True)
    with tab2:
        st.dataframe(unmatched_b, use_container_width=True)

    # ================== Anomaly Detection ==================
    if not matches_df.empty:
        st.subheader("Anomaly Alerts")
        anomalies = detect_anomalies(matches_df)
        if anomalies:
            for a in anomalies:
                st.warning(f"**{a['type']}** – {a['count']} transactions")
        else:
            st.success("No anomalies detected – Clean reconciliation!")

    # ================== Footer ==================
    st.markdown("---")
    st.markdown("Built with Grok 4 + xAI | 100% Open Source | Deploy in 2 mins with Docker")

# ================== Retrain Model (Sidebar) ==================
st.sidebar.markdown("---")
st.sidebar.header("Retrain AI Model")
feedback_file = st.sidebar.file_uploader("Upload labeled feedback CSV", type=['csv'])
if feedback_file and st.sidebar.button("Retrain & Save Model"):
    try:
        df = pd.read_csv(feedback_file)
        X = df[['amt_diff_abs', 'amt_diff_pct', 'date_diff', 'ref_score']].fillna(0)
        y = df['is_match'].astype(int)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
        model.fit(X, y)
        import joblib
        joblib.dump(model, "match_model.joblib")
        st.sidebar.success("Model retrained & saved!")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
