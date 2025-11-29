# ui.py - Streamlit UI for Phase 1 Ledger Reconciliation
import streamlit as st
import pandas as pd
from core import advanced_match_ledgers, detect_columns
import io

st.set_page_config(page_title="AI Ledger Reconciler", layout="wide", page_icon="💰")

# Sidebar: Settings
st.sidebar.header("Reconciliation Settings")
date_tol = st.sidebar.slider("Date Tolerance (days)", 0, 180, 30)
amt_tol_pct = st.sidebar.slider("Amount Tolerance (%)", 0.0, 20.0, 5.0) / 100
abs_tol = st.sidebar.slider("Absolute Tolerance (₹)", 0, 500, 50)
enable_ml = st.sidebar.checkbox("Enable ML Boost", value=True)

# Upload Files
st.title("AI Ledger Reconciliation")
col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Ledger A (Your Books)", type=['csv','xlsx'])
with col2:
    file_b = st.file_uploader("Upload Ledger B (Bank/GSTR)", type=['csv','xlsx'])

if file_a and file_b:
    # Read files
    df_a = pd.read_csv(file_a) if file_a.name.endswith('.csv') else pd.read_excel(file_a)
    df_b = pd.read_csv(file_b) if file_b.name.endswith('.csv') else pd.read_excel(file_b)
    st.success(f"Loaded Ledger A: {len(df_a)} rows | Ledger B: {len(df_b)} rows")

    # Auto-detect columns
    map_a = detect_columns(df_a)
    map_b = detect_columns(df_b)

    # Manual override
    with st.expander("Manual Column Mapping Override"):
        for col in ['date','ref','debit','credit']:
            if map_a.get(col):
                map_a[col] = st.selectbox(f"Ledger A: {col}", df_a.columns, index=df_a.columns.get_loc(map_a[col]))
            if map_b.get(col):
                map_b[col] = st.selectbox(f"Ledger B: {col}", df_b.columns, index=df_b.columns.get_loc(map_b[col]))

    # Run Matching
    if st.button("Run AI Reconciliation"):
        progress_text = st.empty()
        progress_bar = st.progress(0)
        matches_df, unmatched_a, unmatched_b = advanced_match_ledgers(
            df_a, map_a, df_b, map_b,
            date_tol=date_tol,
            amt_tol=amt_tol_pct,
            abs_tol=abs_tol,
            enable_ml=enable_ml
        )
        progress_bar.progress(100)
        progress_text.text("Reconciliation Complete!")

        # Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Matches", len(matches_df))
        col2.metric("Match Rate", f"{len(matches_df)/len(df_a)*100:.1f}%")
        col3.metric("Average Score", f"{matches_df['Score'].mean():.1f}" if not matches_df.empty else "0")
        col4.metric("Estimated Savings", f"₹{len(matches_df)*1200:,.0f}")

        # Match Type Chart
        st.subheader("Match Type Distribution")
        st.bar_chart(matches_df['Match_Type'].value_counts())

        # Detailed Matches
        st.subheader("Matched Transactions")
        show_cols = ['A_Date','A_Ref','A_Amount','B_Date','B_Ref','B_Amount','Match_Type','Score','Remarks']
        st.dataframe(matches_df[show_cols] if not matches_df.empty else pd.DataFrame(columns=show_cols))

        # Export CSV
        csv_data = matches_df.to_csv(index=False).encode()
        st.download_button("Download Matches CSV", csv_data, "matches.csv", "text/csv")

        # Unmatched Tabs
        tab1, tab2 = st.tabs(["Unmatched in Ledger A", "Unmatched in Ledger B"])
        with tab1: st.dataframe(unmatched_a)
        with tab2: st.dataframe(unmatched_b)

# Retraining Section
st.sidebar.header("Retrain ML Model")
feedback_file = st.sidebar.file_uploader("Upload Feedback CSV (historical labels)", type=['csv'])
if feedback_file and st.sidebar.button("Retrain Model"):
    from core import load_model, save_model, RandomForestClassifier
    df_feedback = pd.read_csv(feedback_file)
    X = df_feedback[['amt_diff_abs','amt_diff_pct','date_diff_days','ref_score','meta_eq']].fillna(0)
    y = df_feedback['is_match'].astype(int)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    save_model(model)
    st.sidebar.success("Model retrained successfully!")
