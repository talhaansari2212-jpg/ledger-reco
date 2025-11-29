# core.py - Complete Phase 1-4

import re, io, os, hashlib
from datetime import timedelta
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

try:
    import jellyfish
except:
    jellyfish = None

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import joblib
except:
    RandomForestClassifier = None
    joblib = None

# NLP embeddings
try:
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except:
    sbert_model = None

MODEL_PATH = "match_model.joblib"

# ================== Helpers ==================
def detect_columns(df):
    cols = {'debit': None, 'credit': None, 'date': None, 'ref': None, 'gstin': None, 'txn_code': None, 'narration': None}
    keywords = {
        'debit': ['debit','dr','db','expense','purchase','charge','invoice','inv_amt'],
        'credit': ['credit','cr','payment','receipt','refund'],
        'date': ['date','trans','posting','value_date','txn','transaction'],
        'ref': ['ref','invoice','inv','vno','voucher','po','bill','cheque','chq','document','narration'],
        'gstin': ['gstin','gst','tin'],
        'txn_code': ['txn_code','type','mode','bank_code'],
        'narration': ['narration','description','remarks','details']
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

INV_PATTERN = re.compile(r'(inv(?:oice)?[-\s:]*)?(\d{2,4}[-/]\d{1,4}[-/]\d{1,6}|\d{4}-\d{3,6}|\bINV[-]?\d{3,6}\b)', re.IGNORECASE)

def extract_invoice_pattern(ref):
    if pd.isna(ref): return ""
    m = INV_PATTERN.search(str(ref))
    return m.group(2) if m else ""

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

def save_model(model, path=MODEL_PATH):
    if joblib:
        try: joblib.dump(model, path)
        except: pass

def optimize_dataframes(df):
    for col in df.columns:
        if df[col].dtype=='object': df[col] = df[col].astype('string')
        elif pd.api.types.is_numeric_dtype(df[col]): df[col] = pd.to_numeric(df[col], downcast='float')
    return df

# ================== Tiered Matcher Class ==================
class TieredMatcher:
    def phonetic_similarity(self, a,b):
        if not a or not b: return 0.0
        scores=[]
        if jellyfish:
            if jellyfish.soundex(a)==jellyfish.soundex(b): scores.append(1.0)
            if jellyfish.metaphone(a)==jellyfish.metaphone(b): scores.append(1.0)
        return max(scores) if scores else 0.0
def detect_partial_payments(df_a, df_b, used_a, used_b):
    """Placeholder for complex partial payment detection logic."""
    # Is function mein partial payment ki complex logic aayegi.
    # Abhi ke liye yeh sirf ek khaali list return karega.
    return []
# ================== Advanced Ledger Matching ==================
def advanced_match_ledgers(A, map_a, B, map_b,
                          date_tol=180, amt_tol=0.05, abs_tol=50,
                          enable_ml=True, ml_model_path=MODEL_PATH,
                          enable_partial_payments=True,
                          debug=False):
    A, B = optimize_dataframes(A.copy()), optimize_dataframes(B.copy())
    matcher = TieredMatcher()
    A['amt'] = A.apply(lambda r: get_amount_from_row(r, map_a), axis=1)
    B['amt'] = B.apply(lambda r: get_amount_from_row(r, map_b), axis=1)
    A['_idx'] = A.index; B['_idx'] = B.index
    sub_A = A[A['amt']>0]; sub_B = B[B['amt']>0]
    if sub_A.empty or sub_B.empty: return pd.DataFrame(), A, B
    if map_a.get('date'): sub_A['date'] = pd.to_datetime(sub_A[map_a['date']], errors='coerce')
    else: sub_A['date'] = pd.NaT
    if map_b.get('date'): sub_B['date'] = pd.to_datetime(sub_B[map_b['date']], errors='coerce')
    else: sub_B['date'] = pd.NaT
    sub_A['ref'] = sub_A[map_a.get('ref')].astype(str).fillna('').str.lower() if map_a.get('ref') else ''
    sub_B['ref'] = sub_B[map_b.get('ref')].astype(str).fillna('').str.lower() if map_b.get('ref') else ''
    sub_A[['meta_a','sound_a']] = sub_A['ref'].apply(lambda x: pd.Series(list(phonetic_codes(x))))
    sub_B[['meta_b','sound_b']] = sub_B['ref'].apply(lambda x: pd.Series(list(phonetic_codes(x))))
    sub_A['inv_pattern'] = sub_A['ref'].apply(extract_invoice_pattern)
    sub_B['inv_pattern'] = sub_B['ref'].apply(extract_invoice_pattern)
    used_b = set(); used_a=set(); matches=[]
    model = load_model(ml_model_path) if enable_ml else None

    # FX Conversion if currency exists
    if 'currency' in sub_A.columns and 'currency' in sub_B.columns:
        from core import apply_fx_conversion
        fx_rates = {('USD','INR'):83.2, ('EUR','INR'):90.1} # example
        sub_A = apply_fx_conversion(sub_A, currency_col='currency', amt_col='amt', fx_rates=fx_rates)
        sub_B = apply_fx_conversion(sub_B, currency_col='currency', amt_col='amt', fx_rates=fx_rates)

    # Main matching loop
    def candidate_filter(a_row):
        a_amt = a_row['amt']; pct_window=max(a_amt*amt_tol*3,0)
        low = a_amt-pct_window-abs_tol; high = a_amt+pct_window+abs_tol
        a_date=a_row['date']; date_low=date_high=None
        if pd.notna(a_date): date_low = a_date-timedelta(days=date_tol); date_high=a_date+timedelta(days=date_tol)
        cand=sub_B[~sub_B.index.isin(used_b)].copy()
        cand=cand[(cand['amt']>=low)&(cand['amt']<=high)]
        if pd.notna(a_date): cand=cand[(cand['date']>=date_low)&(cand['date']<=date_high)]
        return cand

    for a_idx, a_row in sub_A.iterrows():
        cand = candidate_filter(a_row)
        if cand.empty: continue
        cand['amt_diff_abs'] = (cand['amt']-a_row['amt']).abs()
        cand['amt_diff_pct'] = cand['amt_diff_abs'] / (a_row['amt']+1e-9)
        cand['date_diff'] = (cand['date']-a_row['date']).dt.days.abs().fillna(999)
        cand['ref_score'] = cand['ref'].apply(lambda x: fuzz.ratio(x,a_row['ref']))
        cand['meta_eq'] = cand['meta_b']==a_row['meta_a']
        cand['sound_eq'] = cand['sound_b']==a_row['sound_a']
        cand['inv_eq'] = cand['inv_pattern']==a_row['inv_pattern']
        cand['score_rule'] = (1-cand['amt_diff_pct'].clip(0,1))*0.5 + (1-cand['date_diff'].clip(0,180)/180)*0.3 + (cand['ref_score']/100)*0.2
        cand['score_rule'] += cand['meta_eq'].astype(int)*0.03 + cand['sound_eq'].astype(int)*0.02 + cand['inv_eq'].astype(int)*0.08
        if model is not None:
            feat_cols=['amt_diff_abs','amt_diff_pct','date_diff','ref_score','meta_eq']
            Xm = cand[feat_cols].copy(); Xm['meta_eq'] = Xm['meta_eq'].astype(int)
            try: probs = model.predict_proba(Xm)[:,1]
            except: probs=np.zeros(len(Xm))
            cand['ml_prob']=probs
            cand['score_combined'] = cand['score_rule']*0.4 + cand['ml_prob']*0.6
        else:
            cand['ml_prob']=0.0; cand['score_combined']=cand['score_rule']
        best_idx=cand['score_combined'].idxmax(); best=cand.loc[best_idx]

        # Tiered decision logic
        tier=None; remark=""; conf=best['score_combined']
        amt_ok = best['amt_diff_abs']<=abs_tol
        date_exact = best['date_diff']==0
        ref_ok = best['ref_score']>=98 or best['inv_eq']
        if amt_ok and date_exact and ref_ok: tier="Tier1-Exact"; remark="Exact match"
        elif best['amt_diff_pct']<=amt_tol*1.5 and best['date_diff']<=date_tol and best['ref_score']>=85: tier="Tier2-Fuzzy"; conf*=0.92; remark="Fuzzy amount/date"
        elif best['amt_diff_pct']<=amt_tol*3 or best['amt_diff_abs']<=abs_tol: tier="Tier3-AmountOnly"; conf*=0.8; remark="Amount only"
        elif best['ref_score']>=70 or best['meta_eq']: tier="Tier4-RefMatch"; conf*=0.65; remark="Ref/phonetic match"

        if tier:
            A_date_val=a_row.get(map_a.get('date'), a_row.get('date',""))
            B_date_val=best.get(map_b.get('date'), best.get('date',""))
            A_ref_val=a_row.get(map_a.get('ref'),a_row.get('ref',""))
            B_ref_val=best.get(map_b.get('ref'),best.get('ref',""))
            matches.append({
                "A_index": int(a_row['_idx']),
                "B_index": int(best['_idx']),
                "A_Date": A_date_val,
                "A_Ref": A_ref_val,
                "A_Amount": float(a_row['amt']),
                "B_Date": B_date_val,
                "B_Ref": B_ref_val,
                "B_Amount": float(best['amt']),
                "Match_Type": tier,
                "Score": round(conf*100,1),
                "Remarks": remark,
                "Hash": hashlib.sha256(f"{a_row['_idx']}_{best['_idx']}".encode()).hexdigest()[:16]
            })
            used_b.add(best.name); used_a.add(a_idx)

    # Partial Payments Detection
    if enable_partial_payments:
        from core import detect_partial_payments
        partials = detect_partial_payments(sub_A, sub_B, used_a, used_b)
        for p in partials:
            matches.append({
                "A_index": p['A_index'],
                "B_index": p['B_index'],
                "A_Date": sub_A.loc[p['A_index'], 'date'] if 'date' in sub_A.columns else "",
                "A_Ref": p['Ref'],
                "A_Amount": p['A_Amount'],
                "B_Date": sub_B.loc[p['B_index'], 'date'] if 'date' in sub_B.columns else "",
                "B_Ref": p['Ref'],
                "B_Amount": p['B_Amount'],
                "Match_Type": p['Match_Type'],
                "Score": 75.0,
                "Remarks": "Partial Payment",
                "Hash": hashlib.sha256(f"partial_{p['Ref']}_{p['A_Amount']}".encode()).hexdigest()[:16]
            })

    match_df=pd.DataFrame(matches)
    unmatched_A = A[~A['_idx'].isin(match_df['A_index'])].drop(columns=['_idx','amt'], errors='ignore')
    unmatched_B = B[~B['_idx'].isin(match_df['B_index'])].drop(columns=['_idx','amt'], errors='ignore')
    return match_df, unmatched_A.reset_index(drop=True), unmatched_B.reset_index(drop=True)

# ----------------- Phase 3 & 4 helpers -----------------
def forecast_cash_flow(matches_df, date_col='A_Date', amt_col='A_Amount'):
    if matches_df.empty: return {}
    df = matches_df.copy()
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    df['amt'] = pd.to_numeric(df[amt_col], errors='coerce')
    df = df.dropna(subset=['date','amt'])
    if df.empty: return {}
    monthly = df.groupby(df['date'].dt.to_period('M'))['amt'].sum()
    x = np.arange(len(monthly))
    y = monthly.values
    if len(x)<2: return {'next_month_forecast': y[-1] if len(y)>0 else 0}
    trend = np.polyfit(x,y,1)
    forecast = trend[0]*len(x)+trend[1]
    return {'next_month_forecast': forecast, 'trend_slope': trend[0], 'monthly_history': monthly}

def create_blockchain_record(match_data):
    data_str = str(match_data)
    return hashlib.sha256(data_str.encode()).hexdigest()

def compute_semantic_similarity(narration_a, narration_b):
    if sbert_model is None: return 0.0
    if pd.isna(narration_a) or pd.isna(narration_b): return 0.0
    embs_a = sbert_model.encode([str(narration_a)])
    embs_b = sbert_model.encode([str(narration_b)])
    sim = np.dot(embs_a, embs_b.T) / (np.linalg.norm(embs_a)*np.linalg.norm(embs_b)+1e-9)
    return float(sim[0][0])

def process_real_time_transaction(new_txn, ledger_B, map_a, map_b):
    df_new = pd.DataFrame([new_txn])
    matches, _, _ = advanced_match_ledgers(df_new, map_a, ledger_B, map_b)
    return matches

def detect_reconciliation_anomalies(df):
    df = df.copy()
    anomalies=[]
    duplicate_rows=df[df.duplicated(['A_Ref','A_Amount'], keep=False)]
    if not duplicate_rows.empty: anomalies.append({'type':'Duplicate','rows':duplicate_rows.index.tolist()})
    if 'A_Amount' in df.columns:
        mean_amt=df['A_Amount'].mean(); std_amt=df['A_Amount'].std()
        outliers=df[(df['A_Amount']>(mean_amt+3*std_amt)) | (df['A_Amount']<(mean_amt-3*std_amt))]
        if not outliers.empty: anomalies.append({'type':'Amount Outlier','rows':outliers.index.tolist()})
    if 'date_diff' in df.columns:
        timing_issues=df[df['date_diff']>180]
        if not timing_issues.empty: anomalies.append({'type':'Timing Issue','rows':timing_issues.index.tolist()})
    return anomalies

def explain_match_shap(model, X, match_index=0):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return shap_values[1][match_index]
    except:
        return None
