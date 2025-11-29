import streamlit as st
import pandas as pd
import io

# --- core.py aur exporter.py ko yahin paste kar diya hai ---
# (neeche dekho)

st.set_page_config(page_title="Ledger Recon - Talha", layout="wide")
st.title("Smart Ledger Reconciliation Tool")
st.caption("Partial Payments + Fuzzy Matching + 100% Free")

c1, c2 = st.columns(2)
with c1:
    file_a = st.file_uploader("My Company Ledger (AP)", type=['csv', 'xlsx'])
with c2:
    file_b = st.file_uploader("Supplier Statement (SOA)", type=['csv', 'xlsx'])

if file_a and file_b:
    try:
        A = pd.read_csv(file_a) if file_a.name.endswith('csv') else pd.read_excel(file_a)
        B = pd.read_csv(file_b) if file_b.name.endswith('csv') else pd.read_excel(file_b)

        # Auto detect columns
        cols_a = detect_columns(A)
        cols_b = detect_columns(B)

        with st.expander("Column Mapping (Auto-detected, change if needed)", expanded=False):
            ca, cb = st.columns(2)
            with ca:
                map_a = {
                    'debit': st.selectbox("A - Debit/Invoice", [None]+list(A.columns), index=get_idx(A, cols_a['debit'])),
                    'credit': st.selectbox("A - Credit/Payment", [None]+list(A.columns), index=get_idx(A, cols_a['credit'])),
                    'date': st.selectbox("A - Date", [None]+list(A.columns), index=get_idx(A, cols_a['date'])),
                    'ref': st.selectbox("A - Ref/Invoice No", [None]+list(A.columns), index=get_idx(A, cols_a['ref'])),
                }
            with cb:
                map_b = {
                    'debit': st.selectbox("B - Debit (Payment Recd)", [None]+list(B.columns), index=get_idx(B, cols_b['debit'])),
                    'credit': st.selectbox("B - Credit (Invoice)", [None]+list(B.columns), index=get_idx(B, cols_b['credit'])),
                    'date': st.selectbox("B - Date", [None]+list(B.columns), index=get_idx(B, cols_b['date'])),
                    'ref': st.selectbox("B - Ref/Invoice No", [None]+list(B.columns), index=get_idx(B, cols_b['ref'])),
                }

        col1, col2 = st.columns(2)
        date_tol = col1.slider("Date Tolerance (days)", 0, 30, 7)
        amt_tol = col2.slider("Amount Tolerance (%)", 0.0, 20.0, 5.0) / 100

        if st.button("Run Reconciliation Now", type="primary"):
            with st.spinner("Matching invoices & partial payments..."):
                matches, un_a, un_b = advanced_match_ledgers(A, map_a, B, map_b, date_tol, amt_tol)
                buffer = io.BytesIO()
                export_report(matches, un_a, un_b, buffer)
                buffer.seek(0)
                st.success(f"Done! {len(matches)} matches found")
                st.download_button(
                    "Download Full Excel Report",
                    buffer.getvalue(),
                    "Ledger_Reconciliation_Result.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                if not matches.empty:
                    st.dataframe(matches[['A_Date', 'A_Ref', 'A_Amount', 'B_Date', 'B_Ref', 'B_Amount', 'Match_Type']].head(10))

    except Exception as e:
        st.error(f"Error: {e}")

# Helper function
def get_idx(df, col):
    opts = [None] + list(df.columns)
    return opts.index(col) if col in df.columns else 0
