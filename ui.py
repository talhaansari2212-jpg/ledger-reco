# ui.py - FINAL PRODUCTION READY STREAMLIT UI
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from core import advanced_match_ledgers, retrain_with_feedback, detect_columns

st.set_page_config(page_title="AI Ledger Reconciler", layout="wide", page_icon="rocket")

# Custom CSS for pro look
st.markdown("""
<style>
    .big-font {font-size:50px !important; font-weight:bold; color:#1E88E5;}
    .metric-card {background-color:#f0f2f6; padding:20px; border-radius:10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    .success-box {background-color:#d4edda; padding:15px; border-radius:8px; border-left:6px solid #28a745;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='big-font'>AI-Powered Ledger Reconciliation</h1>", unsafe_allow_html=True)
st.markdown("**Made in India | GST Ready | ML + Phonetic + Tiered Matching**")

# Sidebar
with st.sidebar:
    st.header("Settings & Controls")
    
    date_tol = st.slider("Date Tolerance (days)", 1, 365, 180, help="Max date gap allowed (default 6 months)")
    amt_tol = st.slider("Amount Tolerance (%)", 0.0, 20.0, 5.0, 0.5) / 100
    abs_tol = st.slider("Absolute Tolerance (₹)", 0, 500, 50)
    
    enable_ml = st.checkbox("Enable ML Boost", value=True)
    min_confidence = st.slider("Minimum Confidence %", 50, 99, 65)
    
    st.markdown("---")
    st.caption(f"Model: {'Loaded' if os.path.exists('match_model.joblib') else 'Not found'}")

# File Upload
col1, col2 = st.columns(2)
with col1:
    st.subheader("Ledger A (Your Books)")
    file_a = st.file_uploader("Upload Ledger A (Excel/CSV)", type=['csv','xlsx'], key="a")
with col2:
    st.subheader("Ledger B (Bank/GSTR-2A)")
    file_b = st.file_uploader("Upload Ledger B (Excel/CSV)", type=['csv','xlsx'], key="b")

if file_a and file_b:
    try:
        df_a = pd.read_csv(file_a) if file_a.name.endswith('.csv') else pd.read_excel(file_a)
        df_b = pd.read_csv(file_b) if file_b.name.endswith('.csv') else pd.read_excel(file_b)
        
        st.success(f"Loaded: {len(df_a):,} rows | {len(df_b):,} rows")
        
        # Auto Column Detection
        with st.spinner("Detecting columns automatically..."):
            map_a = detect_columns(df_a)
            map_b = detect_columns(df_b)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Ledger A Mapping**")
            st.json(map_a, expanded=False)
        with col2:
            st.write("**Ledger B Mapping**")
            st.json(map_b, expanded=False)
        with col3:
            st.write("**Adjust if needed**")
            with st.expander("Manual Override"):
                map_a['date'] = st.selectbox("Date Column (A)", df_a.columns, index=df_a.columns.get_loc(map_a['date']) if map_a['date'] else 0)
                map_a['ref'] = st.selectbox("Reference/Narration (A)", df_a.columns, index=df_a.columns.get_loc(map_a['ref']) if map_a['ref'] else 0)
                map_b['date'] = st.selectbox("Date Column (B)", df_b.columns, index=df_b.columns.get_loc(map_b['date']) if map_b['date'] else 0)
                map_b['ref'] = st.selectbox("Reference/Narration (B)", df_b.columns, index=df_b.columns.get_loc(map_b['ref']) if map_b['ref'] else 0)

        if st.button("Run AI Reconciliation", type="primary", use_container_width=True):
            with st.spinner("AI matching in progress... (yeh thoda waqt lega, chai pi lo)"):
                matches_df, unmatched_a, unmatched_b = advanced_match_ledgers(
                    df_a, map_a, df_b, map_b,
                    date_tol=date_tol,
                    amt_tol=amt_tol,
                    abs_tol=abs_tol,
                    enable_ml=enable_ml
                )
            
            # Results
            st.markdown("<div class='success-box'>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Matches", len(matches_df))
            col2.metric("Match Rate", f"{len(matches_df)/len(df_a)*100:.1f}%")
            col3.metric("Avg Confidence", f"{matches_df['Score'].mean():.1f}%" if not matches_df.empty else "0%")
            col4.metric("Savings (est.)", f"₹{len(matches_df) * 1200:,.0f}")
            st.markdown("</div>", unsafe_allow_html=True)

            # Charts
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Match Type Distribution")
                chart_data = matches_df['Match_Type'].value_counts()
                st.bar_chart(chart_data)
            with col2:
                st.subheader("Confidence Score Trend")
                st.line_chart(matches_df['Score'])

            st.subheader("Detailed Matches")
            show_cols = ['A_Date','A_Ref','A_Amount','B_Date','B_Ref','B_Amount','Match_Type','Score','Remarks']
            st.dataframe(
                matches_df[show_cols].sort_values("Score", ascending=False),
                use_container_width=True,
                hide_index=True
            )

            # Export
            csv = matches_df.to_csv(index=False).encode()
            st.download_button("Download Matches CSV", csv, "matches.csv", "text/csv")

            # Unmatched
            tab1, tab2 = st.tabs(["Unmatched in Your Books", "Unmatched in Bank/GSTR"])
            with tab1:
                st.dataframe(unmatched_a, use_container_width=True)
            with tab2:
                st.dataframe(unmatched_b, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Tip: Make sure both files have Date, Amount (Debit/Credit), and Narration columns")

# Retraining Section
with st.sidebar:
    st.markdown("---")
    st.header("Model Retraining")
    feedback = st.file_uploader("Upload Feedback CSV", type=['csv'], help="Columns: amt_diff_abs,amt_diff_pct,date_diff_days,ref_score,phonetic_eq,is_match")
    if feedback and st.button("Retrain Model"):
        with st.spinner("Training new model..."):
            msg = retrain_with_feedback(feedback)
            st.success(msg)

st.markdown("---")
st.caption("Built with love by Indian Developer | Open Source | No SaaS Trap | 100% Offline")
