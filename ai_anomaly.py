# ai_anomaly.py
# Phase 1: Anomaly detection & analysis helpers for reconciliation
# - Duplicate detection
# - Fraud pattern heuristics (round amounts, sequential invoices)
# - Timing differences (using matched pairs)
# - Missing sequences detection
# - Statistical outliers (z-score + IsolationForest)
# - Integrates with nlp_utils for semantic matching & summarization hooks

import re
import numpy as np
import pandas as pd
from collections import defaultdict

# Optional ML libs
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
except Exception:
    IsolationForest = None
    StandardScaler = None

# Local NLP utilities (optional heavy deps inside nlp_utils)
try:
    from nlp_utils import get_embedding, semantic_similarity_matrix, categorize_texts, detect_payment_terms, summarize_texts
except Exception:
    get_embedding = None
    semantic_similarity_matrix = None
    categorize_texts = None
    detect_payment_terms = None
    summarize_texts = None

# Invoice pattern helper (reuse similar to core)
INV_NUM_RE = re.compile(r'(\d{2,4}[-/]\d{1,4}[-/]\d{1,6}|\bINV[-]?\d{1,8}\b|\d{4}\d{3,6})', re.IGNORECASE)

def normalize_ref(ref):
    if pd.isna(ref):
        return ""
    s = str(ref).strip().lower()
    s = re.sub(r'[^a-z0-9\-\/]', '', s)
    return s

def extract_invoice_number(ref):
    if pd.isna(ref) or not ref:
        return None
    m = INV_NUM_RE.search(str(ref))
    if m:
        return m.group(0)
    # attempt to extract trailing digits
    m2 = re.search(r'(\d{3,8})$', str(ref))
    return m2.group(1) if m2 else None

# 1) Duplicate detection
def detect_duplicates(df, ref_col=None, date_col=None, amt_col='amt', amt_tolerance=1.0):
    """
    Detect potential duplicate entries within a single dataframe.
    Returns DataFrame with duplicate groups and a boolean flag 'is_dup'.
    """
    d = df.copy()
    d['_norm_ref'] = d[ref_col].astype(str).apply(normalize_ref) if ref_col else ''
    # Approx duplicate by (same normalized ref) OR (same amount within tol and same date)
    d['_dup_key'] = d['_norm_ref']
    # For blank refs, fallback to amount+date
    mask_blank = d['_dup_key'].str.len() == 0
    if date_col:
        fallback = d.loc[mask_blank].apply(lambda r: f"{int(round(float(r.get(amt_col) or 0)/max(1,amt_tolerance)))}_{str(r.get(date_col))[:10]}", axis=1)
    else:
        fallback = d.loc[mask_blank].apply(lambda r: f"{int(round(float(r.get(amt_col) or 0)/max(1,amt_tolerance)))}", axis=1)
    d.loc[mask_blank, '_dup_key'] = fallback
    # group
    groups = d.groupby('_dup_key').filter(lambda x: len(x) > 1)
    if groups.empty:
        d['is_dup'] = False
        return d[['is_dup']], pd.DataFrame()
    # Mark duplicates
    d['is_dup'] = d['_dup_key'].isin(groups['_dup_key'].unique())
    dup_details = d[d['is_dup']].copy()
    return d[['is_dup']], dup_details

# 2) Fraud pattern heuristics
def detect_fraud_patterns(df, ref_col=None, amt_col='amt'):
    """
    Flags:
      - round_amount: amounts that end with many zeros or are exact multiples of 1000/10000
      - repeat_amounts: same amount repeated many times in a short window
      - sequential_invoices: invoice numbers that increment by 1
    Returns a DataFrame with columns: round_amount, repeated_amount_count, seq_group
    """
    d = df.copy()
    d['_amt'] = pd.to_numeric(d.get(amt_col), errors='coerce').fillna(0).abs()
    d['round_amount'] = d['_amt'].apply(lambda x: (x % 1000 == 0 and x != 0) or (x % 100 == 0 and x != 0 and x >= 10000))
    # repeat amounts over dataset
    amt_counts = d['_amt'].value_counts()
    d['repeated_amount_count'] = d['_amt'].map(amt_counts).fillna(0).astype(int)
    # sequential invoice detection
    d['_inv'] = d[ref_col].astype(str).apply(extract_invoice_number) if ref_col else None
    d['_inv_num'] = d['_inv'].apply(lambda x: int(re.sub(r'\D','',x)) if x and re.search(r'\d',x) else np.nan)
    d = d.sort_values(by=['_inv_num']).reset_index(drop=True)
    d['seq_gap'] = d['_inv_num'].diff().fillna(0)
    # mark likely sequences where consecutive diffs == 1
    d['seq_group'] = (d['seq_gap'] == 1).cumsum()
    # entries part of sequences longer than 2
    seq_sizes = d.groupby('seq_group')['_inv_num'].transform('count')
    d['part_of_seq'] = seq_sizes >= 3
    # return cleaned flags
    return d[['round_amount','repeated_amount_count','part_of_seq','_inv','_inv_num']]

# 3) Timing differences across matched pairs
def analyze_timing(matches_df, a_date_col='A_Date', b_date_col='B_Date'):
    """
    matches_df: DataFrame with A_Date and B_Date (datetime or parseable)
    Returns timing summary and dataframe with 'date_diff_days' and flags where diff exceeds typical threshold
    """
    m = matches_df.copy()
    m['A_date_parsed'] = pd.to_datetime(m.get(a_date_col), errors='coerce')
    m['B_date_parsed'] = pd.to_datetime(m.get(b_date_col), errors='coerce')
    m['date_diff_days'] = (m['B_date_parsed'] - m['A_date_parsed']).dt.days
    # summary stats
    stats = {
        'median_lag': float(m['date_diff_days'].median(skipna=True) if not m['date_diff_days'].dropna().empty else np.nan),
        'mean_lag': float(m['date_diff_days'].mean(skipna=True) if not m['date_diff_days'].dropna().empty else np.nan),
        'std_lag': float(m['date_diff_days'].std(skipna=True) if not m['date_diff_days'].dropna().empty else np.nan),
    }
    # flag large lags ( > mean + 2*std ) or negative gaps (advanced)
    if not np.isnan(stats['mean_lag']) and not np.isnan(stats['std_lag']):
        m['timing_alert'] = m['date_diff_days'].apply(lambda x: abs(x - stats['mean_lag']) > 2*stats['std_lag'] if pd.notna(x) else False)
    else:
        m['timing_alert'] = False
    return stats, m[['A_Date','B_Date','date_diff_days','timing_alert']]

# 4) Missing transaction sequences per supplier or global
def detect_missing_sequences(df, ref_col, group_col=None):
    """
    For each group (or entire df), extract invoice numbers, sort and look for numeric gaps.
    Returns a list/dict of gaps per group.
    """
    d = df.copy()
    d['_inv'] = d[ref_col].astype(str).apply(extract_invoice_number)
    d['_num'] = d['_inv'].apply(lambda x: int(re.sub(r'\D','',x)) if x and re.search(r'\d',x) else np.nan)
    results = {}
    if group_col and group_col in d.columns:
        groups = d.groupby(group_col)
    else:
        groups = [('ALL', d)]
    for gname, gdf in groups:
        nums = sorted(gdf['_num'].dropna().unique())
        gaps = []
        if len(nums) <= 1:
            results[gname] = {'gaps': [], 'count': len(nums)}
            continue
        for i in range(len(nums)-1):
            if nums[i+1] - nums[i] > 1:
                gaps.append({'from': int(nums[i]), 'to': int(nums[i+1]), 'missing': int(nums[i+1] - nums[i] - 1)})
        results[gname] = {'gaps': gaps, 'count': len(nums)}
    return results

# 5) Statistical outliers (z-score + optional IsolationForest)
def detect_outliers(df, amt_col='amt', method='zscore', z_thresh=3.0, iforest_opts=None):
    """
    Returns DataFrame with is_outlier flag and anomaly_score (if available)
    method: 'zscore' or 'isolation'
    """
    d = df.copy()
    d['_amt'] = pd.to_numeric(d.get(amt_col), errors='coerce').fillna(0)
    if method == 'zscore':
        mean = d['_amt'].mean()
        std = d['_amt'].std(ddof=0) if d['_amt'].size>0 else 0
        if std == 0 or np.isnan(std):
            d['is_outlier'] = False
            d['outlier_score'] = 0.0
            return d[['is_outlier','outlier_score']]
        d['zscore'] = (d['_amt'] - mean)/std
        d['is_outlier'] = d['zscore'].abs() > z_thresh
        d['outlier_score'] = d['zscore'].abs()
        return d[['is_outlier','outlier_score']]
    elif method == 'isolation' and IsolationForest is not None:
        X = d[['_amt']].values.reshape(-1,1)
        iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        try:
            preds = iso.fit_predict(X)
            scores = -iso.decision_function(X)  # higher -> more anomalous
            d['is_outlier'] = preds == -1
            d['outlier_score'] = scores
            return d[['is_outlier','outlier_score']]
        except Exception:
            # fallback to zscore
            return detect_outliers(df, amt_col=amt_col, method='zscore', z_thresh=z_thresh)
    else:
        # fallback
        return detect_outliers(df, amt_col=amt_col, method='zscore', z_thresh=z_thresh)

# 6) Semantic matching across ledgers (uses nlp_utils if available)
def semantic_match_transactions(dfA, dfB, text_col_a, text_col_b, top_k=1, min_score=0.7):
    """
    Compute semantic similarity between description/narration fields of A and B.
    Returns list of top matches (A_index, B_index, score) for scores above min_score.
    Uses get_embedding/semantic_similarity_matrix from nlp_utils if available; falls back to simple fuzzy ratio.
    """
    from rapidfuzz import fuzz

    results = []
    if semantic_similarity_matrix and get_embedding:
        # compute embeddings
        textsA = dfA[text_col_a].astype(str).fillna('').tolist()
        textsB = dfB[text_col_b].astype(str).fillna('').tolist()
        sim = semantic_similarity_matrix(textsA, textsB)  # expects [lenA x lenB] matrix of cosine sims
        # iterate rows
        for i,row in enumerate(sim):
            top_idxs = np.argsort(row)[::-1][:top_k]
            for idx in top_idxs:
                score = float(row[idx])
                if score >= min_score:
                    results.append({'A_idx': dfA.index[i], 'B_idx': dfB.index[idx], 'score': score})
    else:
        # fallback to fuzzy token_set_ratio
        for a_idx, a_row in dfA.iterrows():
            a_txt = str(a_row.get(text_col_a,''))
            best = None
            best_score = 0
            for b_idx, b_row in dfB.iterrows():
                b_txt = str(b_row.get(text_col_b,''))
                s = fuzz.token_set_ratio(a_txt, b_txt)/100.0
                if s > best_score:
                    best_score = s
                    best = b_idx
            if best_score >= min_score:
                results.append({'A_idx': a_idx, 'B_idx': best, 'score': best_score})
    return pd.DataFrame(results)

# 7) High-level orchestrator
def run_anomaly_analysis(A, B, matches_df=None, map_a=None, map_b=None, text_col_a=None, text_col_b=None, debug=False):
    """
    Run a suite of anomaly checks and NLP analyses and return a dict of DataFrames & summaries.
    - A, B: raw dataframes (should already have 'amt' column or provide mapping)
    - matches_df: match results produced by advanced_match_ledgers (optional)
    - map_a/map_b: mapping dicts with 'ref', 'date', 'debit','credit' if needed
    - text_col_a/text_col_b: narration/description columns for NLP
    """
    results = {}
    # Ensure amount fields exist
    if 'amt' not in A.columns:
        # mapping fallback
        if map_a:
            A['amt'] = A.apply(lambda r: (pd.to_numeric(r.get(map_a.get('debit')), errors='coerce') or 0) - (pd.to_numeric(r.get(map_a.get('credit')), errors='coerce') or 0), axis=1).abs()
        else:
            A['amt'] = 0
    if 'amt' not in B.columns:
        if map_b:
            B['amt'] = B.apply(lambda r: (pd.to_numeric(r.get(map_b.get('debit')), errors='coerce') or 0) - (pd.to_numeric(r.get(map_b.get('credit')), errors='coerce') or 0), axis=1).abs()
        else:
            B['amt'] = 0

    # Duplicate detection
    ref_a = (map_a.get('ref') if map_a else None) if map_a else None
    ref_b = (map_b.get('ref') if map_b else None) if map_b else None
    dup_flags_a, dup_details_a = detect_duplicates(A, ref_col=ref_a, date_col=(map_a.get('date') if map_a else None))
    dup_flags_b, dup_details_b = detect_duplicates(B, ref_col=ref_b, date_col=(map_b.get('date') if map_b else None))
    results['dup_flags_a'] = dup_flags_a
    results['dup_flags_b'] = dup_flags_b
    results['dup_details_a'] = dup_details_a
    results['dup_details_b'] = dup_details_b

    # Fraud heuristics
    fraud_a = detect_fraud_patterns(A, ref_col=ref_a, amt_col='amt')
    fraud_b = detect_fraud_patterns(B, ref_col=ref_b, amt_col='amt')
    results['fraud_a'] = fraud_a
    results['fraud_b'] = fraud_b

    # Outliers
    outliers_a = detect_outliers(A, amt_col='amt', method='isolation' if IsolationForest else 'zscore')
    outliers_b = detect_outliers(B, amt_col='amt', method='isolation' if IsolationForest else 'zscore')
    results['outliers_a'] = outliers_a
    results['outliers_b'] = outliers_b

    # Missing sequences (global)
    seqs_a = detect_missing_sequences(A, ref_col=ref_a)
    seqs_b = detect_missing_sequences(B, ref_col=ref_b)
    results['seqs_a'] = seqs_a
    results['seqs_b'] = seqs_b

    # Timing analysis if matches provided
    if matches_df is not None and not matches_df.empty:
        timing_stats, timing_detail = analyze_timing(matches_df)
        results['timing_stats'] = timing_stats
        results['timing_detail'] = timing_detail
    else:
        results['timing_stats'] = {}
        results['timing_detail'] = pd.DataFrame()

    # Semantic match candidates (if narration columns provided)
    if text_col_a and text_col_b:
        try:
            sem_df = semantic_match_transactions(A, B, text_col_a, text_col_b, top_k=3, min_score=0.65)
            results['semantic_matches'] = sem_df
        except Exception:
            results['semantic_matches'] = pd.DataFrame()

    # NLP categorization & payment term extraction
    if text_col_a:
        try:
            cats_a = categorize_texts(A[text_col_a].astype(str).fillna('')) if categorize_texts else pd.Series(index=A.index, data=['Unknown']*len(A))
            payterms_a = A[text_col_a].astype(str).apply(lambda t: detect_payment_terms(t) if detect_payment_terms else None)
            results['cats_a'] = cats_a
            results['payterms_a'] = payterms_a
        except Exception:
            results['cats_a'] = pd.Series(index=A.index, data=['Unknown']*len(A))
            results['payterms_a'] = pd.Series(index=A.index, data=[''])
    if text_col_b:
        try:
            cats_b = categorize_texts(B[text_col_b].astype(str).fillna('')) if categorize_texts else pd.Series(index=B.index, data=['Unknown']*len(B))
            payterms_b = B[text_col_b].astype(str).apply(lambda t: detect_payment_terms(t) if detect_payment_terms else None)
            results['cats_b'] = cats_b
            results['payterms_b'] = payterms_b
        except Exception:
            results['cats_b'] = pd.Series(index=B.index, data=['Unknown']*len(B))
            results['payterms_b'] = pd.Series(index=B.index, data=[''])

    # Smart remarks (summaries) for anomalies using summarizer if available
    if summarize_texts and text_col_a:
        try:
            # Example: summarize top anomalous descriptions from A
            candidates = A.loc[outliers_a['is_outlier']].head(50)[text_col_a].astype(str).tolist()
            if candidates:
                summary = summarize_texts(candidates)
                results['anomaly_summary_a'] = summary
        except Exception:
            results['anomaly_summary_a'] = None

    return results
