# ui.py - Fully Integrated Streamlit UI (Phases 1-4)

import streamlit as st
import pandas as pd
import io
import json # For real-time transaction input

# --- CORE IMPORTS ---
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
    "FX Rates (JSON format, e.g., {('USD','INR'):83.2})", "{}"
)
try:
    # Parsing FX Rates dictionary
    fx_rates = json.loads(fx_rates_input.replace("(", "[").replace(")", "]").replace("'", '"'))
    # Convert list keys back to tuples for the core function's expectation
    fx_rates = {tuple(k):v for k,v in fx_rates.items()}
except Exception as e:
    st.sidebar.error("Error parsing FX Rates JSON/Dict. Using default empty rates.")
    fx_rates = {}

st.sidebar.markdown("---")

# ================== FILE UPLOAD & MAPPING ==================

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Ledger A (Your Books)", type=['csv','xlsx'])
with col2:
    file_b = st.file_uploader("Upload Ledger B (Bank/GSTR)", type=['csv','xlsx'])

if file_a and file_b:
    try:
        df_a = pd.read_csv(file_a) if file_a.name.endswith('.csv') else pd.read_excel(file_a)
        df_b = pd.read_csv(file_b) if file_b.name.endswith('.csv') else pd.read_excel(file_b)
        st.success(f"Loaded Ledger A: {len(df_a)} rows | Ledger B: {len(df_b)} rows")
    except Exception as e:
        st.error(f"Error reading files: {e}")
        st.stop()
        
    # Auto-detect columns (map_a, map_b will contain all detected columns)
    map_a = detect_columns(df_a)
    map_b = detect_columns(df_b)

    # Manual override (Selectbox uses the detected column as default index)
    with st.expander("Manual Column Mapping Override"):
        for col in ['date','ref','debit','credit','txn_code','currency','narration']:
            # Check if the column is present in the DataFrame before creating selectbox
            if map_a.get(col) and map_a[col] in df_a.columns:
                map_a[col] = st.selectbox(f"Ledger A: {col}", df_a.columns, index=df_a.columns.get_loc(map_a[col]), key=f"A_{col}")
            elif col in df_a.columns: # If not auto-detected but present
                 map_a[col] = st.selectbox(f"Ledger A: {col}", ['None'] + list(df_a.columns), index=0, key=f"A_{col}")
                 
            if map_b.get(col) and map_b[col] in df_b.columns:
                map_b[col] = st.selectbox(f"Ledger B: {col}", df_b.columns, index=df_b.columns.get_loc(map_b[col]), key=f"B_{col}")
            elif col in df_b.columns:
                map_b[col] = st.selectbox(f"Ledger B: {col}", ['None'] + list(df_b.columns), index=0, key=f"B_{col}")


    # ================== RUN MATCHING (The core logic wrapper) ==================
    if st.button("Run AI Reconciliation", type="primary"):
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # NOTE: FX conversion logic is now handled internally by advanced_match_ledgers
        
        matches_df, unmatched_a, unmatched_b = advanced_match_ledgers(
            df_a, map_a, df_b, map_b,
            date_tol=date_tol,
            amt_tol=amt_tol_pct,
            abs_tol=abs_tol,
            enable_ml=enable_ml,
            enable_partial_payments=enable_partial,
            fx_rates=fx_rates # Pass FX rates to core
        )

        progress_bar.progress(100)
        progress_text.text("Reconciliation Complete! Results:")

        # --- RESULTS SUMMARY (Phase 1 & 2) ---
        st.subheader("📊 Reconciliation Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        total_rows_a = len(df_a)
        
        col1.metric("Total Matches", len(matches_df))
        col2.metric("Match Rate", f"{len(matches_df)/total_rows_a*100:.1f}%" if total_rows_a > 0 else "0%")
        col3.metric("Average Score", f"{matches_df['Score'].mean():.1f}" if not matches_df.empty else "0")
        col4.metric("Partial Matches", len(matches_df[matches_df['Match_Type']=='Partial Payment']))
        col5.metric("Estimated Savings", f"₹{len(matches_df)*1200:,.0f}") # Placeholder metric

        # Match Type Chart
        st.subheader("Match Type Distribution")
        st.bar_chart(matches_df['Match_Type'].value_counts())

        # --- PHASE 3: ANALYTICS & AUDIT ---
        st.markdown("---")
        st.header("📈 Advanced Analytics & Audit")
        
        # 1. Predictive Cash Flow Display
        st.subheader("Predictive Cash Flow")
        forecast_info = forecast_cash_flow(matches_df)
        if forecast_info and len(forecast_info.get('monthly_history', [])) > 1:
            st.metric(f"Next Month Forecast ({base_currency})", f"{forecast_info['next_month_forecast']:,.2f}")
            st.write("Trend Slope per Month:", f"{forecast_info['trend_slope']:,.2f}")
            st.line_chart(forecast_info['monthly_history'])
        else:
            st.info("Not enough historical data for cash flow forecast.")

        # 2. Blockchain Hash column in Matches
        st.subheader("Blockchain Audit Hash")
        if not matches_df.empty:
            # Use original index columns for better context
            audit_df = matches_df[['A_index', 'B_index', 'A_Ref', 'B_Ref', 'A_Amount', 'Match_Type', 'Score']].copy()
            audit_df['Blockchain_Hash'] = audit_df.apply(lambda r: create_blockchain_record(r.to_dict()), axis=1)
            st.dataframe(audit_df)
        else:
            st.info("No matches to generate audit hashes.")

        # --- DETAILED RESULTS ---
        st.markdown("---")
        st.header("📝 Detailed Results")
        
        st.subheader("Matched Transactions")
        show_cols = ['A_Date','A_Ref','A_Amount','B_Date','B_Ref','B_Amount','Match_Type','Score','Remarks']
        st.dataframe(matches_df[show_cols] if not matches_df.empty else pd.DataFrame(columns=show_cols))

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
            st.info("Check affected rows in the Detailed Matches table.")
        else:
            st.success("✅ No reconciliation anomalies detected.")

        # 2. Explainable AI Panel
        st.subheader("Explain Match Confidence (XAI)")
        if enable_ml and not matches_df.empty:
            match_indices = matches_df.index.tolist()
            match_idx = st.selectbox("Select Match Row Index (Original Index)", match_indices)
            
            if st.button("Explain Selected Match", key='explain_btn'):
                
                model = load_model()
                if model and match_idx in matches_df.index:
                    # Filter data for the match row and select ML features
                    match_row = matches_df.loc[[match_idx]]
                    # NOTE: Features must exactly match training features
                    X = match_row[['amt_diff_abs','amt_diff_pct','date_diff','ref_score','meta_eq']].copy() 
                    X['meta_eq'] = X['meta_eq'].astype(int) # Ensure boolean is converted to int
                    
                    shap_vals = explain_match_shap(model, X, match_index=0) # Only one row being explained
                    
                    if shap_vals is not None:
                        st.success("Feature Contribution (SHAP Values):")
                        feature_names = X.columns.tolist()
                        shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_vals})
                        st.dataframe(shap_df.sort_values('SHAP Value', ascending=False))
                        # 
                    else:
                        st.info("Explainability (SHAP) requires the ML model to be properly loaded and the 'shap' library to be installed.")
                else:
                    st.error("ML model not loaded or selected index is invalid.")
        elif enable_ml and matches_df.empty:
            st.info("No matches found to explain.")


# ================== RETRAINING SECTION (Phase 4) ==================

st.sidebar.header("Retrain ML Model")
st.sidebar.info("Use historical matched/unmatched data to improve prediction.")
feedback_file = st.sidebar.file_uploader("Upload Feedback CSV", type=['csv'], key='feedback_upload')
if feedback_file and st.sidebar.button("Retrain Model"):
    try:
        df_feedback = pd.read_csv(feedback_file)
        # Ensure feedback columns match the features expected by the model
        X = df_feedback[['amt_diff_abs','amt_diff_pct','date_diff_days','ref_score','meta_eq']].fillna(0)
        y = df_feedback['is_match'].astype(int)
        
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X, y)
        save_model(model)
        st.sidebar.success("Model retrained successfully and saved to 'match_model.joblib'!")
    except Exception as e:
        st.sidebar.error(f"Retraining failed. Check CSV columns/format. Error: {e}")
