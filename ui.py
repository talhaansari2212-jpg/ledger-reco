# ui.py - Complete Phase 1-4 Streamlit UI
import streamlit as st
import pandas as pd
from core import advanced_match_ledgers, detect_columns, apply_fx_conversion, forecast_cash_flow, create_blockchain_record, process_real_time_transaction, detect_reconciliation_anomalies, explain_match_shap, load_model
import io, json

st.set_page_config(page_title="AI Ledger Reconciler", layout="wide", page_icon="💰")
st.title("AI Ledger Reconciliation - Phase 1-4")

# Sidebar Settings
st.sidebar.header("Reconciliation Settings")
date_tol = st.sidebar.slider("Date Tolerance (days)", 0, 180, 30)
amt_tol_pct = st.sidebar.slider("Amount Tolerance (%)", 0.0, 20.0, 5.0)/100
abs_tol = st.sidebar.slider("Absolute Tolerance (₹)", 0, 1000, 50)
enable_ml = st.sidebar.checkbox("Enable ML Boost", value=True)
enable_partial = st.sidebar.checkbox("Enable Partial Payment
