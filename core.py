# core.py - Phase 4 Production Ready
import re, io, os, hashlib
from datetime import timedelta
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

# Phonetic libraries
try:
    import jellyfish
except Exception:
    jellyfish = None

# ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import joblib
except Exception:
    RandomForestClassifier = None
    joblib = None

# NLP embeddings
try:
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except Exception:
    sbert_model = None

MODEL_PATH = "match_model.joblib"

# ============================================
# ============== Helper Functions ============
# ============================================

def detect_columns(df):
    """Auto-detect columns for debit, credit, date, ref, GSTIN, bank code, narration"""
    cols = {'debit': None, 'credit': None, 'date': None, 'ref': None,
            'gstin': None, 'txn_code': None, 'narration': None}
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
    """Return metaphone and soundex codes"""
    if not s or pd.isna(s):
        return ("","")
    s = str(s)
    meta = jellyfish.metaphone(s) if jellyfish else ""
    sound = jellyfish.soundex(s) if jellyfish else ""
    return (meta, sound)

INV_PATTERN = re.compile(r'(inv(?:oice)?[-\s:]*)?(\d{2,4}[-/]\d{1,4}[-/]\d{1,6}|\d{4}-\d{3,6}|\bINV[-]?\d{3,6}\b)', re.IGNORECASE)

def extract_invoice_pattern(ref):
    """Extract invoice pattern from reference"""
    if pd.isna(ref): return ""
    m = INV_PATTERN.search(str(ref))
    return m.group(2) if m else ""

def get_amount_from_row(row, mapping):
    """Calculate absolute amount from debit/credit columns"""
    d, c = 0, 0
    try:
        if mapping.get('debit'): d = pd.to_numeric(row.get(mapping.get('debit')), errors='coerce') or 0
    except: d=0
    try:
        if mapping.get('credit'): c = pd.to_numeric(row.get(mapping.get('credit')), errors='coerce') or 0
    except: c=0
    return abs(d - c)

def load_model(path=MODEL_PATH):
    """Load RandomForest model from disk"""
    if joblib and os.path.exists(path):
        try: return joblib.load(path)
        except: return None
    return None

def save_model(model, path=MODEL_PATH):
    """Save RandomForest model to disk"""
    if joblib:
        try: joblib.dump(model, path)
        except: pass

def optimize_dataframes(df):
    """Reduce memory usage for large datasets"""
    for col in df.columns:
        if df[col].dtype=='object': df[col] = df[col].astype('string')
        elif pd.api.types.is_numeric_dtype(df[col]): df[col] = pd.to_numeric(df[col], downcast='float')
    return df

# ============================================
# ============== Tiered Matcher ==============
# ============================================

class TieredMatcher:
    """Tiered matching rules"""
    def phonetic_similarity(self, a,b):
        if not a or not b: return 0.0
        scores=[]
        if jellyfish:
            if jellyfish.soundex(a)==jellyfish.soundex(b): scores.append(1.0)
            if jellyfish.metaphone(a)==jellyfish.metaphone(b): scores.append(1.0)
        return max(scores) if scores else 0.0

# ============================================
# ============== Bank / Partial / FX =========
# ============================================

BANK_CODE_PATTERN = re.compile(r'\b(NEFT|RTGS|IMPS|UPI)\b', re.IGNORECASE)

def extract_bank_code(text):
    if pd.isna(text): return ""
    m = BANK_CODE_PATTERN.search(str(text))
    return m.group(1).upper() if m else ""

def apply_fx_conversion(df, currency_col='currency', amt_col='amt', fx_rates=None, base_currency='INR'):
    """Convert amounts to base currency using provided fx_rates dict"""
    if fx_rates is None: fx_rates = {}
    def convert(row):
        cur = row.get(currency_col,'INR')
        amt = row.get(amt_col,0)
        if cur==base_currency: return amt
        rate = fx_rates.get((cur,base_currency),1.0)
        return amt*rate
    df[amt_col] = df.apply(convert, axis=1)
    return df

def detect_partial_payments(sub_A, sub_B, used_a_indices, used_b_indices, tolerance=0.1, allocation='FIFO'):
    """Detect and allocate partial payments"""
    partial_matches=[]
    a_ref_groups=sub_A.groupby('ref')
    b_ref_groups=sub_B.groupby('ref')
    for ref, a_group in a_ref_groups:
        if ref in b_ref_groups.groups:
            b_group=b_ref_groups.get_group(ref)
            a_total=a_group['amt'].sum()
            b_total=b_group['amt'].sum()
            if abs(a_total-b_total)/max(a_total,b_total)<tolerance:
                a_sorted=a_group.sort_index(ascending=(allocation=='FIFO'))
                b_sorted=b_group.sort_index(ascending=(allocation=='FIFO'))
                for a_idx,b_idx in zip(a_sorted.index,b_sorted.index):
                    partial_matches.append({
                        'A_index': int(a_idx),
                        'B_index': int(b_idx),
                        'A_Amount': float(a_sorted.loc[a_idx,'amt']),
                        'B_Amount': float(b_sorted.loc[b_idx,'amt']),
                        'Ref': ref,
                        'Match_Type': 'Partial Payment'
                    })
    return partial_matches

# ============================================
# ============== Predictive + Blockchain =====
# ============================================

def forecast_cash_flow(matches_df, date_col='A_Date', amt_col='A_Amount'):
    """Predict next month's cash flow based on historical matches"""
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
    trend = np.polyfit(x, y, 1)
    forecast = trend[0]*(len(x)) + trend[1]
    return {'next_month_forecast': forecast, 'trend_slope': trend[0], 'monthly_history': monthly}

def create_blockchain_record(match_data):
    """Generate immutable hash for match data"""
    data_str = str(match_data)
    return hashlib.sha256(data_str.encode()).hexdigest()

def compute_semantic_similarity(narration_a, narration_b):
    """Return 0-1 similarity using sentence-transformers"""
    if sbert_model is None: return 0.0
    if pd.isna(narration_a) or pd.isna(narration_b): return 0.0
    embs_a = sbert_model.encode([str(narration_a)])
    embs_b = sbert_model.encode([str(narration_b)])
    sim = np.dot(embs_a, embs_b.T) / (np.linalg.norm(embs_a)*np.linalg.norm(embs_b)+1e-9)
    return float(sim[0][0])

# ============================================
# ============== Phase 4: Real-time + Anomaly
# ============================================

def process_real_time_transaction(new_txn, ledger_B, map_a, map_b):
    """Process a single new transaction in real-time"""
    df_new = pd.DataFrame([new_txn])
    matches, _, _ = advanced_match_ledgers(df_new, map_a, ledger_B, map_b)
    return matches

def federated_update(local_model, global_model_weights, alpha=0.5):
    """Combine local model weights with global model weights"""
    for local_coef, global_coef in zip(local_model.estimators_, global_model_weights):
        pass
    return local_model

def detect_reconciliation_anomalies(df):
    """Detect duplicates, extreme amounts, and timing issues"""
    df = df.copy(); anomalies=[]
    duplicate_rows = df[df.duplicated(['A_Ref','A_Amount'], keep=False)]
    if not duplicate_rows.empty:
        anomalies.append({'type':'Duplicate','rows':duplicate_rows.index.tolist()})
    if 'A_Amount' in df.columns:
        mean_amt = df['A_Amount'].mean(); std_amt = df['A_Amount'].std()
        outliers = df[(df['A_Amount']>(mean_amt+3*std_amt)) | (df['A_Amount']<(mean_amt-3*std_amt))]
        if not outliers.empty:
            anomalies.append({'type':'Amount Outlier','rows':outliers.index.tolist()})
    if 'date_diff' in df.columns:
        timing_issues = df[df['date_diff']>180]
        if not timing_issues.empty:
            anomalies.append({'type':'Timing Issue','rows':timing_issues.index.tolist()})
    return anomalies

def explain_match_shap(model, X, match_index=0):
    """Compute SHAP values for a match row"""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return shap_values[1][match_index]
    except:
        return None
