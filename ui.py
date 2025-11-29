# ui.py - Phase 4 Streamlit UI for Ledger Reconciliation
import streamlit as st
import pandas as pd
import json
from core import (
    advanced_match_ledgers, detect_columns, apply_fx_conversion,
    forecast_cash_flow, create_blockchain_record,
    process_real_time_transaction, detect_reconciliation_anomalies,
    explain_match_shap, load_model, save_model, RandomForestClassifier
)
import io

st.set_page_config(page_title="AI Ledger Reconciler - Phase 4", layout="wide", page_icon="💰")
st.title("AI Ledger Reconciliation - Phase 4")

# ---------------- Sidebar ----------------
st.sidebar.header("Reconciliation Settings")
date_tol = st.sidebar.slider("Date Tolerance (days)", 0, 180, 30)
amt_tol_pct = st.sidebar.slider("Amount Tolerance (%)", 0.0, 20.0, 5.0)/100
abs_tol = st.sidebar.slider("Absolute Tolerance (₹)", 0, 1000, 50)
enable_ml = st.sidebar.checkbox("Enable ML Boost", value=True)
enable_partial = st.sidebar.checkbox("Enable Partial Payment Detection", value=True)

# Currency selection & FX rates
base_currency = st.sidebar.selectbox("Base Currency", ["INR","USD","EUR","GBP","AED"])
fx_rates_input = st.sidebar.text_area("FX Rates (JSON format, e.g., {('USD','INR'):83.2})", "{}")
try: fx_rates = eval(fx_rates_input)
except: fx_rates = {}

# ---------------- Upload Ledgers ----------------
col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Ledger A (Your Books)", type=['csv','xlsx'])
with col2:
    file_b = st.file_uploader("Upload Ledger B (Bank/GSTR)", type=['csv','xlsx'])

if file_a and file_b:
    df_a = pd.read_csv(file_a) if file_a.name.endswith('.csv') else pd.read_excel(file_a)
    df_b = pd.read_csv(file_b) if file_b.name.endswith('.csv') else pd.read_excel(file_b)
    st.success(f"Loaded Ledger A: {len(df_a)} rows | Ledger B: {len(df_b)} rows")

    # Auto-detect columns
    map_a = detect_columns(df_a)
    map_b = detect_columns(df_b)

    # Manual override
    with st.expander("Manual Column Mapping Override"):
        for col in ['date','ref','debit','credit','txn_code','currency']:
            if map_a.get(col) in df_a.columns:
                map_a[col] = st.selectbox(f"Ledger A: {col}", df_a.columns, index=df_a.columns.get_loc(map_a[col]))
            if map_b.get(col) in df_b.columns:
                map_b[col] = st.selectbox(f"Ledger B: {col}", df_b.columns, index=df_b.columns.get_loc(map_b[col]))

    # FX Conversion
    if 'currency' in df_a.columns and 'currency' in df_b.columns:
        df_a = apply_fx_conversion(df_a, currency_col='currency', amt_col='amt', fx_rates=fx_rates, base_currency=base_currency)
        df_b = apply_fx_conversion(df_b, currency_col='currency', amt_col='amt', fx_rates=fx_rates, base_currency=base_currency)

    # ---------------- Run Matching ----------------
    if st.button("Run AI Reconciliation"):
        progress_text = st.empty()
        progress_bar = st.progress(0)
        matches_df, unmatched_a, unmatched_b = advanced_match_ledgers(
            df_a, map_a, df_b, map_b,
            date_tol=date_tol,
            amt_tol=amt_tol_pct,
            abs_tol=abs_tol,
            enable_ml=enable_ml,
            enable_partial_payments=enable_partial
        )
        progress_bar.progress(100)
        progress_text.text("Reconciliation Complete!")

        # ---------------- Summary Metrics ----------------
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Matches", len(matches_df))
        col2.metric("Match Rate", f"{len(matches_df)/len(df_a)*100:.1f}%" if len(df_a)>0 else "0%")
        col3.metric("Average Score", f"{matches_df['Score'].mean():.1f}" if not matches_df.empty else "0")
        col4.metric("Estimated Savings", f"₹{len(matches_df)*1200:,.0f}")
        col5.metric("Partial Matches", len(matches_df[matches_df['Match_Type']=='Partial Payment']))

        # ---------------- Charts ----------------
        st.subheader("Match Type Distribution")
        st.bar_chart(matches_df['Match_Type'].value_counts())

        # Detailed Matches
        st.subheader("Matched Transactions")
        show_cols = ['A_Date','A_Ref','A_Amount','B_Date','B_Ref','B_Amount','Match_Type','Score','Remarks']
        st.dataframe(matches_df[show_cols] if not matches_df.empty else pd.DataFrame(columns=show_cols))

        # Download CSV
        csv_data = matches_df.to_csv(index=False).encode()
        st.download_button("Download Matches CSV", csv_data, "matches.csv", "text/csv")

        # Unmatched Tabs
        tab1, tab2 = st.tabs(["Unmatched in Ledger A", "Unmatched in Ledger B"])
        with tab1: st.dataframe(unmatched_a)
        with tab2: st.dataframe(unmatched_b)

        # ---------------- Phase 3: Predictive Cash Flow ----------------
        st.subheader("Predictive Cash Flow")
        forecast_info = forecast_cash_flow(matches_df)
        if forecast_info:
            st.metric("Next Month Forecast (₹)", f"{forecast_info['next_month_forecast']:.2f}")
            st.write("Trend Slope per Month:", f"{forecast_info['trend_slope']:.2f}")
            st.line_chart(forecast_info['monthly_history'])

        # ---------------- Blockchain Hash ----------------
        st.subheader("Blockchain Audit Hash")
        if not matches_df.empty:
            st.dataframe(matches_df[['A_Ref','B_Ref','Match_Type','Score']].assign(
                Blockchain_Hash=matches_df.apply(lambda r: create_blockchain_record(r.to_dict()), axis=1)
            ))

        # ---------------- Phase 4: Real-Time ----------------
        st.subheader("Real-Time Reconciliation / Streaming")
        st.info("Phase 4 supports live reconciliation via API/WebSocket (simulated).")
        txn_input = st.text_area("Enter new transaction JSON", "{}")
        if st.button("Process Real-Time Transaction"):
            try:
                new_txn = json.loads(txn_input)
                realtime_matches = process_real_time_transaction(new_txn, df_b, map_a, map_b)
                st.success(f"{len(realtime_matches)} matches found for incoming transaction")
                if len(realtime_matches)>0:
                    st.dataframe(realtime_matches)
            except Exception as e:
                st.error(f"Error processing transaction: {e}")

        # ---------------- Anomaly Detection ----------------
        st.subheader("Anomaly Detection")
        anomalies = detect_reconciliation_anomalies(matches_df)
        if anomalies:
            for a in anomalies:
                st.warning(f"Anomaly Type: {a['type']} | Rows affected: {len(a['rows'])}")
        else:
            st.success("No anomalies detected")

        # ---------------- Explainable AI ----------------
        st.subheader("Explain Match Confidence")
        st.info("SHAP/LIME explanations for model-based matches")
        match_idx = st.number_input(
            "Select Match Index", min_value=0,
            max_value=len(matches_df)-1 if not matches_df.empty else 0, step=1
        )
        if st.button("Explain Selected Match") and not matches_df.empty:
            model = load_model()
            if model:
                X = matches_df[['amt_diff_abs','amt_diff_pct','date_diff','ref_score','meta_eq']]
                shap_vals = explain_match_shap(model, X, match_index=match_idx)
                if shap_vals is not None:
                    st.write(shap_vals)
                else:
                    st.info("Explainability not available")
            else:
                st.error("ML model not loaded")

# ---------------- Retraining Section ----------------
st.sidebar.header("Retrain ML Model")
feedback_file = st.sidebar.file_uploader("Upload Feedback CSV", type=['csv'])
if feedback_file and st.sidebar.button("Retrain Model"):
    df_feedback = pd.read_csv(feedback_file)
    X = df_feedback[['amt_diff_abs','amt_diff_pct','date_diff_days','ref_score','meta_eq']].fillna(0)
    y = df_feedback['is_match'].astype(int)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    save_model(model)
    st.sidebar.success("Model retrained successfully!")
