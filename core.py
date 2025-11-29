# core.py - Phase 1, 2, 3, 4 Integrated
import re, io, os, hashlib
from datetime import timedelta
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

try:
    import jellyfish
except Exception:
    jellyfish = None

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import joblib
except Exception:
    RandomForestClassifier = None
    joblib = None

# Optional NLP embeddings
try:
    from sentence_transformers import SentenceTransformer
    # 'sentence-transformers/all-MiniLM-L6-v2' is a good balance of speed and performance
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 
except Exception:
    sbert_model = None

MODEL_PATH = "match_model.joblib"
INV_PATTERN = re.compile(r'(inv(?:oice)?[-\s:]*)?(\d{2,4}[-/]\d{1,4}[-/]\d{1,6}|\d{4}-\d{3,6}|\bINV[-]?\d{3,6}\b)', re.IGNORECASE)
BANK_CODE_PATTERN = re.compile(r'\b(NEFT|RTGS|IMPS|UPI)\b', re.IGNORECASE)

# ================== Helpers ==================

def detect_columns(df):
    cols = {'debit': None, 'credit': None, 'date': None, 'ref': None, 'gstin': None, 'txn_code': None, 'narration': None, 'currency': None}
    keywords = {
        'debit': ['debit','dr','db','expense','purchase','charge','invoice','inv_amt'],
        'credit': ['credit','cr','payment','receipt','refund'],
        'date': ['date','trans','posting','value_date','txn','transaction'],
        'ref': ['ref','invoice','inv','vno','voucher','po','bill','cheque','chq','document','narration'],
        'gstin': ['gstin','gst','tin'],
        'txn_code': ['txn_code','type','mode','bank_code'],
        'narration': ['narration','description','remarks','details'],
        'currency': ['currency','curr','ccy']
    }
    lower_cols = {c.lower(): c for c in df.columns}
    for key, words in keywords.items():
        for word in words:
            for low, orig in lower_cols.items():
                if word in low and cols[key] is None:
                    cols[key] = orig
                    break
    # Date fallback
    if not cols['date']:
        for c in df.columns:
            try:
                if pd.to_datetime(df[c], errors='coerce').notna().sum() > len(df)*0.4:
                    cols['date'] = c
                    break
            except: pass
    # Ref fallback
    if not cols['ref']:
        for c in df.select_dtypes('object').columns:
            try:
                if df[c].astype(str).str.contains(r'\d{3,}', regex=True).mean() > 0.4:
                    cols['ref'] = c
                    break
            except: pass
    return cols

def phonetic_codes(s):
    if not s or pd.isna(s): return ("","")
    s = str(s)
    meta = jellyfish.metaphone(s) if jellyfish else ""
    sound = jellyfish.soundex(s) if jellyfish else ""
    return (meta, sound)

def extract_invoice_pattern(ref):
    if pd.isna(ref): return ""
    m = INV_PATTERN.search(str(ref))
    return m.group(2) if m else ""

def extract_bank_code(text):
    if pd.isna(text): return ""
    m = BANK_CODE_PATTERN.search(str(text))
    return m.group(1).upper() if m else ""

def get_amount_from_row(row, mapping):
    d, c = 0, 0
    try:
        if mapping.get('debit'): d = pd.to_numeric(row.get(mapping.get('debit')), errors='coerce') or 0
    except: d=0
    try:
        if mapping.get('credit'): c = pd.to_numeric(row.get(mapping.get('credit')), errors='coerce') or 0
    except: c=0
    return abs(d - c)

def load_model(path=MODEL_PATH):
    if joblib and os.path.exists(path):
        try: return joblib.load(path)
        except: return None
    return None

def optimize_dataframes(df):
    for col in df.columns:
        if df[col].dtype=='object': df[col] = df[col].astype('string')
        elif pd.api.types.is_numeric_dtype(df[col]): df[col] = pd.to_numeric(df[col], downcast='float')
    return df

def apply_fx_conversion(df, mapping, fx_rates=None, base_currency='INR'):
    """Convert amounts to base currency if currency column exists and is mapped."""
    currency_col = mapping.get('currency')
    amt_col = 'amt'
    
    if currency_col is None or fx_rates is None or df.empty:
        return df

    def convert(row):
        cur = row.get(currency_col)
        amt = row.get(amt_col, 0)
        
        if pd.isna(cur) or cur == base_currency: return amt
        
        # Look up rate for (From_Currency, Base_Currency)
        rate_key = (cur, base_currency)
        rate = fx_rates.get(rate_key, 1.0)
        
        # Basic currency check (e.g., USD -> INR rate should be > 1)
        if rate <= 0: rate = 1.0 
        
        return amt * rate
        
    df[amt_col] = df.apply(convert, axis=1)
    
    # Optional: Update currency column to base currency for clarity
    df[currency_col] = base_currency
    
    return df

def detect_partial_payments(sub_A, sub_B, used_a_indices, used_b_indices, ref_col='ref', tolerance=0.1, allocation='FIFO'):
    """
    Detect partial payments where transactions sharing the same reference 
    have similar total amounts in Ledger A and B.
    """
    partial_matches=[]
    
    # Filter out already matched transactions
    unmatched_A = sub_A[~sub_A.index.isin(used_a_indices)]
    unmatched_B = sub_B[~sub_B.index.isin(used_b_indices)]
    
    a_ref_groups = unmatched_A.groupby(ref_col)
    b_ref_groups = unmatched_B.groupby(ref_col)
    
    for ref, a_group in a_ref_groups:
        if ref and ref in b_ref_groups.groups: # Check if ref is non-empty and present in B
            b_group = b_ref_groups.get_group(ref)
            
            a_total = a_group['amt'].sum()
            b_total = b_group['amt'].sum()
            
            if a_total > 0 and b_total > 0 and (abs(a_total - b_total) / max(a_total, b_total)) < tolerance:
                
                # Allocate payments (e.g., using FIFO approach for simplicity)
                a_sorted = a_group.sort_index(ascending=(allocation=='FIFO'))
                b_sorted = b_group.sort_index(ascending=(allocation=='FIFO'))
                
                # Simple one-to-one assignment for partials with similar reference and total
                min_len = min(len(a_sorted), len(b_sorted))
                
                for i in range(min_len):
                    a_idx = a_sorted.iloc[i].name
                    b_idx = b_sorted.iloc[i].name
                    
                    # Mark as used for partials
                    used_a_indices.add(a_idx)
                    used_b_indices.add(b_idx)
                    
                    partial_matches.append({
                        'A_index': int(a_idx),
                        'B_index': int(b_idx),
                        'A_Amount': float(a_sorted.loc[a_idx,'amt']),
                        'B_Amount': float(b_sorted.loc[b_idx,'amt']),
                        'Ref': ref,
                        'Match_Type': 'Partial Payment'
                    })
    return partial_matches


def compute_semantic_similarity(narration_a, narration_b):
    """Return 0-1 similarity score using sentence-transformers"""
    if sbert_model is None: return 0.0
    if pd.isna(narration_a) or pd.isna(narration_b): return 0.0
    
    try:
        embs = sbert_model.encode([str(narration_a), str(narration_b)])
        embs_a, embs_b = embs[0], embs[1]
        
        # Calculate Cosine Similarity
        # Use np.dot for dot product and np.linalg.norm for L2 norm
        sim = np.dot(embs_a, embs_b) / (np.linalg.norm(embs_a) * np.linalg.norm(embs_b) + 1e-9)
        return float(sim)
    except Exception:
        return 0.0

# ------------------------- Main Matching Function -------------------------

def advanced_match_ledgers(A, map_a, B, map_b,
                         date_tol=180, amt_tol=0.05, abs_tol=50,
                         enable_ml=True, ml_model_path=MODEL_PATH,
                         enable_partial_payments=True, fx_rates=None,
                         debug=False):

    A, B = optimize_dataframes(A.copy()), optimize_dataframes(B.copy())
    
    A['amt'] = A.apply(lambda r: get_amount_from_row(r, map_a), axis=1)
    B['amt'] = B.apply(lambda r: get_amount_from_row(r, map_b), axis=1)
    A['_idx'] = A.index; B['_idx'] = B.index
    
    # Filter for non-zero amounts
    sub_A = A[A['amt']>0].copy(); sub_B = B[B['amt']>0].copy()
    if sub_A.empty or sub_B.empty: return pd.DataFrame(), A, B

    # --- Phase 1: Pre-processing & Feature Extraction ---
    
    # Date Handling
    date_a_col = map_a.get('date'); date_b_col = map_b.get('date')
    sub_A['date'] = pd.to_datetime(sub_A[date_a_col], errors='coerce') if date_a_col else pd.NaT
    sub_B['date'] = pd.to_datetime(sub_B[date_b_col], errors='coerce') if date_b_col else pd.NaT

    # Reference Handling
    ref_a_col = map_a.get('ref'); ref_b_col = map_b.get('ref')
    sub_A['ref'] = sub_A[ref_a_col].astype(str).fillna('').str.lower() if ref_a_col else ''
    sub_B['ref'] = sub_B[ref_b_col].astype(str).fillna('').str.lower() if ref_b_col else ''

    # Phonetic Codes
    sub_A[['meta_a','sound_a']] = sub_A['ref'].apply(lambda x: pd.Series(list(phonetic_codes(x))))
    sub_B[['meta_b','sound_b']] = sub_B['ref'].apply(lambda x: pd.Series(list(phonetic_codes(x))))

    # Invoice Patterns
    sub_A['inv_pattern'] = sub_A['ref'].apply(extract_invoice_pattern)
    sub_B['inv_pattern'] = sub_B['ref'].apply(extract_invoice_pattern)
    
    # Narration (for Semantic Sim)
    narration_a_col = map_a.get('narration'); narration_b_col = map_b.get('narration')
    sub_A['narration'] = sub_A[narration_a_col].astype(str).fillna('') if narration_a_col else ''
    sub_B['narration'] = sub_B[narration_b_col].astype(str).fillna('') if narration_b_col else ''

    # Bank Code (Phase 2)
    txn_a_col = map_a.get('txn_code'); txn_b_col = map_b.get('txn_code')
    sub_A['bank_code'] = sub_A[txn_a_col].astype(str).apply(extract_bank_code) if txn_a_col else ''
    sub_B['bank_code'] = sub_B[txn_b_col].astype(str).apply(extract_bank_code) if txn_b_col else ''

    # --- Phase 2: FX Conversion ---
    if fx_rates is not None:
        sub_A = apply_fx_conversion(sub_A, map_a, fx_rates=fx_rates)
        sub_B = apply_fx_conversion(sub_B, map_b, fx_rates=fx_rates)

    # --- Main Loop Setup ---
    used_b = set(); used_a=set(); matches=[]
    model = load_model(ml_model_path) if enable_ml else None

    def candidate_filter(a_row):
        a_amt = a_row['amt']; 
        # Window size based on percentage and absolute tolerance
        pct_window = a_amt * amt_tol * 3
        low = a_amt - pct_window - abs_tol; high = a_amt + pct_window + abs_tol
        
        a_date=a_row['date']; date_low=date_high=None
        if pd.notna(a_date): 
            date_low = a_date - timedelta(days=date_tol)
            date_high = a_date + timedelta(days=date_tol)
            
        cand = sub_B[~sub_B.index.isin(used_b)].copy()
        cand = cand[(cand['amt'] >= low) & (cand['amt'] <= high)]
        
        if pd.notna(a_date): 
            cand = cand[(cand['date'] >= date_low) & (cand['date'] <= date_high)]
            
        return cand

    for a_idx, a_row in sub_A.iterrows():
        if a_idx in used_a: continue
        
        cand = candidate_filter(a_row)
        if cand.empty: continue
        
        # --- Scoring Feature Calculation ---
        cand['amt_diff_abs'] = (cand['amt'] - a_row['amt']).abs()
        cand['amt_diff_pct'] = cand['amt_diff_abs'] / (a_row['amt'] + 1e-9)
        cand['date_diff'] = (cand['date'] - a_row['date']).dt.days.abs().fillna(999)
        cand['ref_score'] = cand['ref'].apply(lambda x: fuzz.ratio(x, a_row['ref']))
        
        # Binary Match Indicators
        cand['meta_eq'] = cand['meta_b'] == a_row['meta_a']
        cand['sound_eq'] = cand['sound_b'] == a_row['sound_a']
        cand['inv_eq'] = cand['inv_pattern'] == a_row['inv_pattern']
        cand['bank_eq'] = cand['bank_code'] == a_row['bank_code'] # Phase 2 Integration
        
        # Semantic Similarity (Phase 3 Integration) - Apply only if SBERT model is available
        if sbert_model is not None and narration_a_col and narration_b_col:
            # Applying semantic check to candidates - NOTE: Can be slow
            cand['semantic_sim'] = cand.apply(
                lambda r: compute_semantic_similarity(a_row['narration'], r['narration']), axis=1
            )
        else:
            cand['semantic_sim'] = 0.0

        # --- Rule-Based Scoring ---
        cand['score_rule'] = (1 - cand['amt_diff_pct'].clip(0, 1)) * 0.5 + \
                             (1 - cand['date_diff'].clip(0, date_tol) / date_tol) * 0.3 + \
                             (cand['ref_score'] / 100) * 0.2
                             
        # Small boosts for exact indicators
        cand['score_rule'] += cand['meta_eq'].astype(int) * 0.03 
        cand['score_rule'] += cand['sound_eq'].astype(int) * 0.02 
        cand['score_rule'] += cand['inv_eq'].astype(int) * 0.08
        cand['score_rule'] += cand['bank_eq'].astype(int) * 0.05 # Phase 2 Bank Code Boost

        # --- ML/Combined Scoring ---
        if model is not None:
            # ML features for the pre-trained model
            feat_cols=['amt_diff_abs','amt_diff_pct','date_diff','ref_score','meta_eq']
            Xm = cand[feat_cols].copy(); Xm['meta_eq'] = Xm['meta_eq'].astype(int)
            try: 
                probs = model.predict_proba(Xm)[:,1]
            except: 
                probs=np.zeros(len(Xm))
                
            cand['ml_prob']=probs
            
            # Combine all scores (Phase 3 Semantic Sim added)
            cand['score_combined'] = cand['score_rule'] * 0.4 + \
                                     cand['ml_prob'] * 0.4 + \
                                     cand['semantic_sim'] * 0.2
        else:
            cand['ml_prob']=0.0
            # If no ML, rule + semantic sim
            cand['score_combined'] = cand['score_rule'] * 0.8 + cand['semantic_sim'] * 0.2 
            
        best_idx = cand['score_combined'].idxmax()
        if best_idx is None: continue # Should not happen if cand is not empty
        best = cand.loc[best_idx]

        # --- Tiered Decision Logic ---
        tier=None; remark=""; conf=best['score_combined']
        amt_ok = best['amt_diff_abs'] <= abs_tol
        date_exact = best['date_diff'] == 0
        ref_ok = best['ref_score'] >= 98 or best['inv_eq']

        if amt_ok and date_exact and ref_ok: 
            tier="Tier1-Exact"; conf=conf; remark="Exact match"
        elif best['amt_diff_pct']<=amt_tol*1.5 and best['date_diff']<=date_tol and best['ref_score']>=85: 
            tier="Tier2-Fuzzy"; conf=conf*0.92; remark="Fuzzy amount/date"
        elif (best['amt_diff_pct']<=amt_tol*3 or best['amt_diff_abs']<=abs_tol) and conf >= 0.5: 
            tier="Tier3-AmountOnly"; conf=conf*0.8; remark="Amount only"
        elif (best['ref_score']>=70 or best['meta_eq'] or best['semantic_sim'] >= 0.7): 
            tier="Tier4-RefMatch"; conf=conf*0.65; remark="Ref/phonetic/semantic match"

        if tier and (conf>=0.55 or amt_ok):
            
            # Retrieve original values for the match record
            A_row_orig = A.loc[a_row['_idx']]
            B_row_orig = B.loc[best['_idx']]
            
            matches.append({
                "A_index": int(a_row['_idx']),
                "B_index": int(best['_idx']),
                "A_Date": A_row_orig.get(date_a_col, a_row.get('date',"")),
                "A_Ref": A_row_orig.get(ref_a_col,a_row.get('ref',"")),
                "A_Amount": float(a_row['amt']),
                "B_Date": B_row_orig.get(date_b_col, best.get('date',"")),
                "B_Ref": B_row_orig.get(ref_b_col,best.get('ref',"")),
                "B_Amount": float(best['amt']),
                "Match_Type": tier,
                "Score": round(conf*100,1),
                "Remarks": remark,
                "Hash": hashlib.sha256(f"{a_row['_idx']}_{best['_idx']}".encode()).hexdigest()[:16]
            })
            used_b.add(best.name); used_a.add(a_idx)

    # --- Phase 2: Partial Payments After Main Loop ---
    if enable_partial_payments:
        partials = detect_partial_payments(sub_A, sub_B, used_a, used_b, ref_col='ref')
        
        for p in partials:
            # Check if this index was not used in main loop (to prevent double-matching)
            if p['A_index'] not in used_a and p['B_index'] not in used_b:
                
                A_row_orig = A.loc[p['A_index']]
                B_row_orig = B.loc[p['B_index']]
                
                matches.append({
                    "A_index": p['A_index'],
                    "B_index": p['B_index'],
                    "A_Date": A_row_orig.get(date_a_col, sub_A.loc[p['A_index'], 'date']),
                    "A_Ref": p['Ref'],
                    "A_Amount": p['A_Amount'],
                    "B_Date": B_row_orig.get(date_b_col, sub_B.loc[p['B_index'], 'date']),
                    "B_Ref": p['Ref'],
                    "B_Amount": p['B_Amount'],
                    "Match_Type": p['Match_Type'],
                    "Score": 75.0, # Fixed score for rule-based partial match
                    "Remarks": "Partial Payment/Group Match",
                    "Hash": hashlib.sha256(f"partial_{p['A_index']}_{p['B_index']}".encode()).hexdigest()[:16]
                })
                used_a.add(p['A_index'])
                used_b.add(p['B_index'])


    match_df=pd.DataFrame(matches)
    
    # Drop temp columns and return unmatched ledgers
    unmatched_A = A[~A['_idx'].isin(used_a)].drop(columns=['_idx','amt'], errors='ignore')
    unmatched_B = B[~B['_idx'].isin(used_b)].drop(columns=['_idx','amt'], errors='ignore')
    
    return match_df, unmatched_A.reset_index(drop=True), unmatched_B.reset_index(drop=True)
    
# Phase 3 & 4 remain as separate functions (forecast_cash_flow, create_blockchain_record, 
# process_real_time_transaction, detect_reconciliation_anomalies, explain_match_shap)
# for usage outside the main matching process.
