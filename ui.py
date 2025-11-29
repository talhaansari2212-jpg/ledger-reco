import streamlit as st
import pandas as pd
import io

# --- FIX: Core Functions ko Import karna zaroori hai ---
try:
    from core import detect_columns, advanced_match_ledgers
except ImportError:
    st.error("Error: core.py file nahi mili. Please check karein ke dono files aik hi folder mein hain aur core.py ka naam sahi hai.")
    st.stop()

# Helper function (isay upar rakhna zaroori tha)
def get_idx(df, col):
    opts = [None] + list(df.columns)
    return opts.index(col) if col and col in df.columns else 0

# Language Selection (Default: English)
lang = st.sidebar.selectbox(
    "زبان / Language", 
    ["English", "اردو"], # English is now the first option
    index=0 # Default index is 0 (English)
) 

if lang == "اردو":
    st.title("سمارٹ لاجرک ریکونسیلیشن ٹول")
    st.caption("پارشیئل پیمنٹس • فیوزی میچنگ • 100% مفت")
    lbl_my = "اپنا کمپنی لیجر (AP)"
    lbl_sup = "سپلائر اسٹیٹمنٹ (SOA)"
    btn_run = "اب ریکونسیلیشن چلائیں"
    success_msg = "ہو گیا! میچز مل گئے"
    download_lbl = "مکمل ایکسل رپورٹ ڈاؤن لوڈ کریں"
    no_match = "کوئی میچ نہیں ملا"
else:
    st.title("Smart Ledger Reconciliation Tool")
    st.caption("Partial Payments + Fuzzy Matching + 100% Free")
    lbl_my = "My Company Ledger (AP)"
    lbl_sup = "Supplier Statement (SOA)"
    btn_run = "Run Reconciliation Now"
    success_msg = "Done! Matches found"
    download_lbl = "Download Full Excel Report"
    no_match = "No matches found"

st.markdown("---")

c1, c2 = st.columns(2)
with c1:
    file_a = st.file_uploader(lbl_my, type=['csv', 'xlsx'])
with c2:
    file_b = st.file_uploader(lbl_sup, type=['csv', 'xlsx'])

if file_a and file_b:
    try:
        # Load Data
        A = pd.read_csv(file_a) if file_a.name.endswith('.csv') else pd.read_excel(file_a)
        B = pd.read_csv(file_b) if file_b.name.endswith('.csv') else pd.read_excel(file_b)

        cols_a = detect_columns(A)
        cols_b = detect_columns(B)

        # Mapping UI
        with st.expander("کالم میپنگ (اگر ضرورت ہو تو تبدیل کریں)" if lang == "اردو" else "Column Mapping (change if needed)", expanded=True):
            ca, cb = st.columns(2)
            with ca:
                map_a = {
                    'debit': st.selectbox("ڈیبٹ/انویس" if lang == "اردو" else "Debit/Invoice", [None]+list(A.columns), index=get_idx(A, cols_a.get('debit'))),
                    'credit': st.selectbox("کریڈٹ/پیمنٹ" if lang == "اردو" else "Credit/Payment", [None]+list(A.columns), index=get_idx(A, cols_a.get('credit'))),
                    'date': st.selectbox("تاریخ" if lang == "اردو" else "Date", [None]+list(A.columns), index=get_idx(A, cols_a.get('date'))),
                    'ref': st.selectbox("ریف/انوائس نمبر" if lang == "اردو" else "Ref/Invoice No", [None]+list(A.columns), index=get_idx(A, cols_a.get('ref'))),
                }
            with cb:
                map_b = {
                    'debit': st.selectbox("ڈیبٹ (پیمنٹ موصول)" if lang == "اردو" else "Debit (Payment Recd)", [None]+list(B.columns), index=get_idx(B, cols_b.get('debit'))),
                    'credit': st.selectbox("کریڈٹ (انوائس)" if lang == "اردو" else "Credit (Invoice)", [None]+list(B.columns), index=get_idx(B, cols_b.get('credit'))),
                    'date': st.selectbox("تاریخ" if lang == "اردو" else "Date", [None]+list(B.columns), index=get_idx(B, cols_b.get('date'))),
                    'ref': st.selectbox("ریف/انوائس نمبر" if lang == "اردو" else "Ref/Invoice No", [None]+list(B.columns), index=get_idx(B, cols_b.get('ref'))),
                }

        col1, col2, col3 = st.columns(3) # NEW: 3 columns for 3 sliders
        date_tol = col1.slider("تاریخ کی رواداری (دن)" if lang == "اردو" else "Date Tolerance (days)", 0, 30, 7)
        amt_tol = col2.slider("رقم کی رواداری (%)" if lang == "اردو" else "Amount Tolerance (%)", 0.0, 20.0, 5.0) / 100
        # NEW SLIDER: Absolute Tolerance
        abs_tol = col3.slider("رقم کی مطلق رواداری (₹)" if lang == "اردو" else "Absolute Tolerance (₹)", 0, 500, 50) 


        if st.button(btn_run, type="primary"):
            with st.spinner("میچنگ ہو رہی ہے..." if lang == "اردو" else "Matching in progress..."):
                
                # RUNNING MATCHING LOGIC - Passing abs_tol
                matches, un_a, un_b = advanced_match_ledgers(A, map_a, B, map_b, date_tol, amt_tol, abs_tol) 
                
                # EXPORT LOGIC
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    # Summary Sheet
                    pd.DataFrame({"حالت": ["مکمل"], "میچز": [len(matches)]} if lang == "اردو" else {"Status": ["Done"], "Matches": [len(matches)]}).to_excel(writer, sheet_name="Summary", index=False)
                    
                    # Matched Sheet
                    if not matches.empty:
                        matches.to_excel(writer, sheet_name="میچ شدہ" if lang == "اردو" else "Matched", index=False)
                    
                    # Unmatched Sheets
                    if not un_a.empty:
                        un_a.to_excel(writer, sheet_name="بقایا (اپنا)" if lang == "اردو" else "Unmatched_Mine", index=False)
                    if not un_b.empty:
                        un_b.to_excel(writer, sheet_name="بقایا (سپلائر)" if lang == "اردو" else "Unmatched_Supplier", index=False)
                
                buffer.seek(0)

                st.success(f"{success_msg}: {len(matches)}")
                
                # Download Button
                st.download_button(
                    download_lbl,
                    buffer.getvalue(),
                    "ریکونسیلیشن_رپورٹ.xlsx" if lang == "اردو" else "Reconciliation_Result.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Preview Table
                if not matches.empty:
                    preview_cols = ['A_Date', 'A_Ref', 'A_Amount', 'B_Date', 'B_Ref', 'B_Amount', 'Match_Type', 'Remarks'] # Added Remarks
                    st.subheader("پہلے 10 میچز" if lang == "اردو" else "Top 10 Matches")
                    cols_to_show = [c for c in preview_cols if c in matches.columns]
                    st.dataframe(matches[cols_to_show].head(10))
                else:
                    st.info(no_match)

    except Exception as e:
        st.error(f"Error: {e}")
