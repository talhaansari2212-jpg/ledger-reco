# ui.py - Fully Integrated Streamlit UI (Phases 1-4)

import streamlit as st
import pandas as pd
import io
import json # For real-time transaction input

# --- CORE IMPORTS ---
# Ensure core.py is in the same directory and all functions are callable
from core import (
    advanced_match_ledgers, 
    detect_columns, 
    forecast_cash_flow,      # Phase 3
    create_blockchain_record,  # Phase 3
    process_real_time_transaction, # Phase 4
    detect_reconciliation_anomalies, # Phase 4
    explain_match_shap,      # Phase 4
    load_model, save_model, RandomForestClassifier # For Retraining
)

st.set_page_config(page_title="AI Ledger Reconciler - Advanced", layout="wide", page_icon="💰")

st.title("AI Ledger Reconciliation - Advanced System")
st.markdown("---")

# ================== SIDEBAR SETTINGS (Phase 1 & 2) ==================

st.sidebar.header("Reconciliation Settings")

# Phase 1 Settings
date_tol = st.sidebar.slider("Date Tolerance (days)", 0, 180, 30)
amt_tol_pct = st.sidebar.slider("Amount Tolerance (%)", 0.0, 20.0, 5.0) / 100
abs_tol = st.sidebar.slider("Absolute Tolerance (₹)", 0, 1000, 50)
enable_ml = st.sidebar.checkbox("Enable ML Boost", value=True)

# Phase 2 Settings
enable_partial = st.sidebar.checkbox("Enable Partial Payment Detection", value=True)
base_currency = st.sidebar.selectbox("Base Currency", ["INR","USD","EUR","GBP","AED"])
fx_rates_input = st.sidebar.text_area(
    "FX Rates (JSON: {('From','Base'):Rate})", "{('USD','INR'): 83.2, ('EUR','INR'): 90.5}"
)
fx_rates = {}
try:
    # Safely evaluate string dictionary to Python dictionary
    fx_rates_temp = eval(fx_rates_input.replace(':',': ').replace("'", '"')) 
    # Ensure keys are tuples of strings (as expected by core.py)
    fx_rates = {tuple(k):v for k,v in fx_rates_temp.items() if isinstance(k, tuple)}
except Exception:
    try:
        # Fallback for simple dict {str: float}
        fx_rates_simple = json.loads(fx_rates_input)
        fx_rates = {tuple(k.split(',')):v for k,v in fx_rates_simple.items()}
    except Exception:
        st.sidebar.error("Error parsing FX Rates format. Using default empty rates.")
        fx_rates = {}


st.sidebar.markdown("---")

# ================== FILE UPLOAD & MAPPING ==================

col1, col2 = st.columns(2)
if 'df_a' not in st.session_state: st.session_state['df_a'] = pd.DataFrame()
if 'df_b' not in st.session_state: st.session_state['df_b'] = pd.DataFrame()

with col1:
    file_a = st.file_uploader("Upload Ledger A (Your Books)", type=['csv','xlsx'])
with col2:
    file_b = st.file_uploader("Upload Ledger B (Bank/GSTR)", type=['csv','xlsx'])

if file_a and file_b:
    try:
        # Read files and store in session state to persist
        st.session_state['df_a'] = pd.read_csv(file_a) if file_a.name.endswith('.csv') else pd.read_excel(file_a)
        st.session_state['df_b'] = pd.read_csv(file_b) if file_b.name.endswith('.csv') else pd.read_excel(file_b)
        st.success(f"Loaded Ledger A: {len(st.session_state['df_a'])} rows | Ledger B: {len(st.session_state['df_b'])} rows")
    except Exception as e:
        st.error(f"Error reading files: {e}")
        st.stop()
        
    # Auto-detect columns
    map_a = detect_columns(st.session_state['df_a'])
    map_b = detect_columns(st.session_state['df_b'])

    # Manual override
    with st.expander("Manual Column Mapping Override"):
        for col in ['date','ref','debit','credit','txn_code','currency','narration']:
            
            # Ledger A Mapping
            default_a = map_a.get(col) or 'None'
            options_a = ['None'] + list(st.session_state['df_a'].columns)
            default_index_a = options_a.index(default_a) if default_a in options_a else 0
            map_a[col] = st.selectbox(f"Ledger A: {col}", options_a, index=default_index_a, key=f"A_{col}")
            
            # Ledger B Mapping
            default_b = map_b.get(col) or 'None'
            options_b = ['None'] + list(st.session_state['df_b'].columns)
            default_index_b = options_b.index(default_b) if default_b in options_b else 0
            map_b[col] = st.selectbox(f"Ledger B: {col}", options_b, index=default_index_b, key=f"B_{col}")

    # Final Map Cleanup: Convert 'None' string to Python None
    for map_dict in [map_a, map_b]:
        for key, value in map_dict.items():
            if value == 'None':
                map_dict[key] = None

    # ================== RUN MATCHING (The core logic wrapper) ==================
    if st.button("Run AI Reconciliation", type="primary"):
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Run core matching function
        matches_df, unmatched_a, unmatched_b = advanced_match_ledgers(
            st.session_state['df_a'], map_a, st.session_state['df_b'], map_b,
            date_tol=date_tol,
            amt_tol=amt_tol_pct,
            abs_tol=abs_tol,
            enable_ml=enable_ml,
            enable_partial_payments=enable_partial,
            fx_rates=fx_rates # Pass FX rates to core
        )

        st.session_state['matches_df'] = matches_df
        st.session_state['unmatched_a'] = unmatched_a
        st.session_state['unmatched_b'] = unmatched_b
        st.session_state['map_b'] = map_b # Store for real-time check

        progress_bar.progress(100)
        progress_text.text("Reconciliation Complete! Results:")

if 'matches_df' in st.session_state and not st.session_state['matches_df'].empty:
    
    matches_df = st.session_state['matches_df']
    unmatched_a = st.session_state['unmatched_a']
    unmatched_b = st.session_state['unmatched_b']

    # --- RESULTS SUMMARY (Phase 1 & 2) ---
    st.subheader("📊 Reconciliation Summary")
    col1, col2, col3, col4 = st.columns(4)
    total_rows_a = len(st.session_state['df_a'])
    
    col1.metric("Total Matches", len(matches_df))
    col2.metric("Match Rate (A)", f"{len(matches_df)/total_rows_a*100:.1f}%" if total_rows_a > 0 else "0%")
    col3.metric("Average Score", f"{matches_df['Score'].mean():.1f}" if not matches_df.empty else "0")
    col4.metric("Partial Matches", len(matches_df[matches_df['Match_Type']=='Partial Payment']))

    # Match Type Chart
    st.subheader("Match Type Distribution")
    st.bar_chart(matches_df['Match_Type'].value_counts())

    # --- PHASE 3: ANALYTICS & AUDIT ---
    st.markdown("---")
    st.header("📈 Advanced Analytics & Audit")
    
    # 1. Predictive Cash Flow Display
    st.subheader("Predictive Cash Flow")
    forecast_info = forecast_cash_flow(matches_df)
    if forecast_info and not forecast_info['monthly_history'].empty:
        st.metric(f"Next Month Forecast ({base_currency})", f"₹{forecast_info['next_month_forecast']:,.2f}")
        st.write("Trend Slope per Month:", f"₹{forecast_info['trend_slope']:,.2f}")
        st.line_chart(forecast_info['monthly_history'])
    else:
        st.info("Not enough historical data for cash flow forecast.")

    # 2. Blockchain Hash column in Matches
    st.subheader("Blockchain Audit Hash")
    # Generate hash column if it doesn't exist (or for partials that need it)
    matches_df['Blockchain_Hash'] = matches_df.apply(lambda r: create_blockchain_record(r.to_dict()), axis=1)
    
    audit_cols = ['A_index', 'B_index', 'A_Ref', 'B_Ref', 'A_Amount', 'Match_Type', 'Score', 'Blockchain_Hash']
    st.dataframe(matches_df[audit_cols])

    # --- DETAILED RESULTS ---
    st.markdown("---")
    st.header("📝 Detailed Results")
    
    st.subheader("Matched Transactions")
    show_cols = ['A_Date','A_Ref','A_Amount','B_Date','B_Ref','B_Amount','Match_Type','Score','Remarks']
    st.dataframe(matches_df[show_cols])

    # Export CSV
    csv_data = matches_df.to_csv(index=False).encode()
    st.download_button("Download Matches CSV", csv_data, "matches.csv", "text/csv")

    # Unmatched Tabs
    tab1, tab2 = st.tabs([f"Unmatched in Ledger A ({len(unmatched_a)})", f"Unmatched in Ledger B ({len(unmatched_b)})"])
    with tab1: st.dataframe(unmatched_a)
    with tab2: st.dataframe(unmatched_b)
    
    # --- PHASE 4: ANOMALY & XAI ---
    st.markdown("---")
    st.header("🔍 Quality Check & Diagnostics")
    
    # 1. Anomaly Detection Dashboard
    st.subheader("Anomaly Detection")
    anomalies = detect_reconciliation_anomalies(matches_df)
    if anomalies:
        for a in anomalies:
            st.warning(f"🚨 Anomaly Type: **{a['type']}** | Rows affected: {len(a['rows'])}")
        st.info("Check affected rows in the Detailed Matches table by their original index.")
    else:
        st.success("✅ No reconciliation anomalies detected.")

    # 2. Explainable AI Panel
    st.subheader("Explain Match Confidence (XAI)")
    if enable_ml:
        match_indices = matches_df.index.tolist()
        match_idx = st.selectbox("Select Match Row Index (Original Index of Match DF)", match_indices)
        
        if st.button("Explain Selected Match", key='explain_btn'):
            
            model = load_model()
            if model and match_idx in matches_df.index:
                
                # NOTE: Features must exactly match training features
                feat_cols = ['amt_diff_abs','amt_diff_pct','date_diff','ref_score','meta_eq']
                
                # Create a sample DataFrame for SHAP explanation
                X = matches_df.loc[[match_idx]][feat_cols].copy() 
                X['meta_eq'] = X['meta_eq'].astype(int) 
                
                shap_vals = explain_match_shap(model, X, match_index=0) # Only one row being explained
                
                if shap_vals is not None:
                    st.success("Feature Contribution (SHAP Values):")
                    feature_names = X.columns.tolist()
                    shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_vals})
                    st.dataframe(shap_df.sort_values('SHAP Value', ascending=False))
                else:
                    st.info("Explainability (SHAP) requires the ML model to be properly loaded and the 'shap' library to be installed.")
            else:
                st.error("ML model not loaded or selected index is invalid.")
    else:
        st.info("ML Boost disabled. Enable it to use XAI.")

# ================== RETRAINING SECTION (Phase 4) ==================

st.sidebar.header("Retrain ML Model")
st.sidebar.info("Use historical matched/unmatched data to improve prediction.")
feedback_file = st.sidebar.file_uploader("Upload Feedback CSV", type=['csv'], key='feedback_upload')
if feedback_file and st.sidebar.button("Retrain Model"):
    if RandomForestClassifier is None:
        st.sidebar.error("Scikit-learn/joblib not installed. Cannot retrain.")
    else:
        try:
            df_feedback = pd.read_csv(feedback_file)
            # NOTE: Feature names in feedback CSV must match ML features!
            X = df_feedback[['amt_diff_abs','amt_diff_pct','date_diff','ref_score','meta_eq']].fillna(0)
            y = df_feedback['is_match'].astype(int)
            
            model = RandomForestClassifier(n_estimators=150, random_state=42)
            model.fit(X, y)
            save_model(model)
            st.sidebar.success("Model retrained successfully and saved to 'match_model.joblib'!")
        except Exception as e:
            st.sidebar.error(f"Retraining failed. Check CSV columns/format. Error: {e}") 

# ----------------- REAL-TIME TRANSACTION SIMULATION (Phase 4) -----------------
st.sidebar.markdown("---")
st.sidebar.header("Real-Time Check (Sim.)")
if 'unmatched_b' in st.session_state:
    real_time_amt = st.sidebar.number_input("Transaction Amount (e.g., Bank Deposit)", value=100.0, step=0.01)
    real_time_ref = st.sidebar.text_input("Transaction Ref/Narration", value="IMPS/CASH")
    
    if st.sidebar.button("Simulate Real-Time Check"):
        # Dummy transaction structure based on expected B ledger (for simplicity)
        rt_txn = {'amt': real_time_amt, 'ref': real_time_ref, 'date': pd.Timestamp.now(), 'narration': real_time_ref}
        
        # NOTE: map_b is required here, stored in session state after run
        if 'map_b' in st.session_state:
            result = process_real_time_transaction(rt_txn, st.session_state['unmatched_b'].copy(), st.session_state['map_b'])
            st.sidebar.success(result)
        else:
            st.sidebar.error("Please run reconciliation first to load unmatched data.")
