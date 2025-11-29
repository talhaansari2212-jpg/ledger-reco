import streamlit as st
import pandas as pd
import io
from core import detect_columns, advanced_match_ledgers, train_match_model, retrain_with_feedback
from exporter import export_report

st.set_page_config(layout="wide", page_title="Smart Ledger Reconciliation")

# Helper
def get_idx(df, col):
    opts = [None] + list(df.columns)
    return opts.index(col) if col and col in df.columns else 0

lang = st.sidebar.selectbox("زبان / Language", ["English", "اردو"], index=0)

if lang == "اردو":
    st.title("سمارٹ لاجرک ریکونسیلیشن ٹول")
    lbl_my = "اپنا کمپنی لیجر (AP)"
    lbl_sup = "سپلائر اسٹیٹمنٹ (SOA)"
    run_lbl = "اب ریکونسیلیشن چلائیں"
    retrain_lbl = "ماڈل دوبارہ تربیت دیں (فیڈبیک اپلوڈ کریں)"
else:
    st.title("Smart Ledger Reconciliation Tool")
    lbl_my = "My Company Ledger (AP)"
    lbl_sup = "Supplier Statement (SOA)"
    run_lbl = "Run Reconciliation Now"
    retrain_lbl = "Retrain Model (Upload Feedback CSV)"

st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    file_a = st.file_uploader(lbl_my, type=['csv','xlsx'])
with c2:
    file_b = st.file_uploader(lbl_sup, type=['csv','xlsx'])

st.sidebar.markdown("Advanced Options")
enable_ml = st.sidebar.checkbox("Enable ML-based matching", value=True)
date_tol = st.sidebar.slider("Date Tolerance (days)", 0, 30, 7)
amt_tol_pct = st.sidebar.slider("Amount Tolerance (%)", 0.0, 20.0, 5.0) / 100.0
abs_tol = st.sidebar.slider("Absolute Tolerance (₹)", 0, 1000, 50)
debug = st.sidebar.checkbox("Debug Mode (more logging)", value=False)

if file_a and file_b:
    try:
        A = pd.read_csv(file_a) if file_a.name.endswith('.csv') else pd.read_excel(file_a)
        B = pd.read_csv(file_b) if file_b.name.endswith('.csv') else pd.read_excel(file_b)

        cols_a = detect_columns(A)
        cols_b = detect_columns(B)

        with st.expander("Column Mapping (change if needed)", expanded=True):
            ca, cb = st.columns(2)
            with ca:
                map_a = {
                    'debit': st.selectbox("Debit/Invoice (My Ledger)", [None]+list(A.columns), index=get_idx(A, cols_a.get('debit'))),
                    'credit': st.selectbox("Credit/Payment (My Ledger)", [None]+list(A.columns), index=get_idx(A, cols_a.get('credit'))),
                    'date': st.selectbox("Date (My Ledger)", [None]+list(A.columns), index=get_idx(A, cols_a.get('date'))),
                    'ref': st.selectbox("Ref/Invoice No (My Ledger)", [None]+list(A.columns), index=get_idx(A, cols_a.get('ref'))),
                }
            with cb:
                map_b = {
                    'debit': st.selectbox("Debit (Supplier)", [None]+list(B.columns), index=get_idx(B, cols_b.get('debit'))),
                    'credit': st.selectbox("Credit (Supplier)", [None]+list(B.columns), index=get_idx(B, cols_b.get('credit'))),
                    'date': st.selectbox("Date (Supplier)", [None]+list(B.columns), index=get_idx(B, cols_b.get('date'))),
                    'ref': st.selectbox("Ref/Invoice No (Supplier)", [None]+list(B.columns), index=get_idx(B, cols_b.get('ref'))),
                }

        if st.button(run_lbl):
            with st.spinner("Matching in progress..."):
                matches, un_a, un_b = advanced_match_ledgers(A, map_a, B, map_b,
                                                            date_tol=date_tol, amt_tol=amt_tol_pct, abs_tol=abs_tol,
                                                            enable_ml=enable_ml, debug=debug)
                # Export
                buffer = io.BytesIO()
                export_report(matches, un_a, un_b, buffer, include_model_info=True)
                st.success(f"Done! Matches found: {len(matches)}")
                st.download_button("Download Report", buffer.getvalue(), "Reconciliation_Report.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                if not matches.empty:
                    st.subheader("Top Matches")
                    show_cols = ['A_Date','A_Ref','A_Amount','B_Date','B_Ref','B_Amount','Match_Type','Score','Remarks']
                    cols_to_show = [c for c in show_cols if c in matches.columns]
                    st.dataframe(matches[cols_to_show].head(20))
                else:
                    st.info("No matches found")

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.header("Model Retraining / Feedback")
st.write("If you have labeled historical reconciliation pairs (with is_match flag), upload CSV to retrain the RF model.")

feedback_file = st.file_uploader(retrain_lbl, type=['csv'])
if feedback_file is not None:
    try:
        with st.spinner("Training model..."):
            model, auc = retrain_with_feedback(feedback_file, debug=debug)
        st.success(f"Model retrained. AUC: {auc if auc is not None else 'N/A'}")
    except Exception as e:
        st.error(f"Retrain error: {e}")
