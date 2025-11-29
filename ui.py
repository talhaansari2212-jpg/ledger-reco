# ui.py
import streamlit as st
import pandas as pd
from datetime import datetime
from core import advanced_match_ledgers, detect_columns, forecast_cash_flow, detect_anomalies

st.set_page_config(page_title="AI Ledger Reconciler Pro", layout="wide", page_icon="robot")

st.markdown("<h1 style='text-align: center;'>AI Ledger Reconciler Pro</h1>", unsafe_allow_html=True)
st.markdown("**Bank vs Books | GSTR-2A vs Purchase | Vendor Ledger** – Instant Matching")

# Sidebar
st.sidebar.title("Settings")
date_tol = st.sidebar.slider("Date Tolerance (days)", 0, 365, 90)
amt_tol = st.sidebar.slider("Amount Tolerance (%)", 0.0, 25.0, 5.0) / 100
abs_tol = st.sidebar.slider("Absolute Tolerance", 0, 1000, 50)
enable_ml = st.sidebar.checkbox("Enable ML", True)
enable_semantic = st.sidebar.checkbox("Enable Narration Match", True)
enable_partial = st.sidebar.checkbox("Enable Partial Payments", True)

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Ledger A (Your Books)", type=['csv','xlsx'])
with col2:
    file_b = st.file_uploader("Ledger B (Bank / GSTR)", type=['csv','xlsx'])

if not (file_a and file_b):
    st.info("Upload both files to start")
    st.stop()

def load_df(f):
    return pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)

df_a = load_df(file_a)
df_b = load_df(file_b)
st success(f"Loaded A: {len(df_a)} | B: {len(df_b)} rows")

map_a = detect_columns(df_a)
map_b = detect_columns(df_b)

with st.expander("Manual Column Mapping"):
    cols = ['date','ref','debit','credit','narration','txn_code']
    c1, c2 = st.columns(2)
    with c1:
        for c in cols:
            if map_a.get(c): map_a[c] = st.selectbox(f"A → {c}", df_a.columns, index=df_a.columns.get_loc(map_a[c]) if map_a[c] in df_a.columns else 0)
    with c2:
        for c in cols:
            if map_b.get(c): map_b[c] = st.selectbox(f"B → {c}", df_b.columns, index=df_b.columns.get_loc(map_b[c]) if map_b[c] in df_b.columns else 0)

if st.button("Run AI Reconciliation", type="primary"):
    with st.spinner("Matching..."):
        matches, un_a, un_b = advanced_match_ledgers(
            df_a, map_a, df_b, map_b,
            date_tol=date_tol,
            amt_tol=amt_tol,
            abs_tol=abs_tol,
            enable_ml=enable_ml,
            enable_semantic=enable_semantic,
            enable_partial_payments=enable_partial
        )
    st.success(f"Done! {len(matches)} matches found")

    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", len(matches))
    c2.metric("High Confidence", len(matches[matches['Confidence'] >= 90]))
    c3.metric("Forecast Next Month", f"₹{forecast_cash_flow(matches)['next_month_forecast']:,.0f}")

    st.bar_chart(matches['Match_Type'].value_counts())
    st.dataframe(matches.sort_values("Confidence", ascending=False).head(100))

    st.download_button("Download Matches", matches.to_csv(index=False), "matches.csv")

    tab1, tab2 = st.tabs(["Unmatched A", "Unmatched B"])
    with tab1: st.dataframe(un_a)
    with tab2: st.dataframe(un_b)
