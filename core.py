# core.py - FINAL PRODUCTION VERSION
import re
import io
import os
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from datetime import timedelta
import hashlib

try:
    import jellyfish
except Exception:
    jellyfish = None

# ML & Transformers
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import joblib
except Exception:
    RandomForestClassifier = None
    joblib = None

try:
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except Exception:
    sbert_model = None

MODEL_PATH = "match_model.joblib"

# ======================== HELPER FUNCTIONS ========================

def detect_columns(df):
    # ... (tumhara wahi excellent function)
    # Already perfect — no change needed
    pass  # keep as is

def phonetic_codes(s):
    if not s or pd.isna(s):
        return ("", "")
    s = str(s).upper()
    meta = jellyfish.metaphone(s) if jellyfish else ""
    sound = jellyfish.soundex(s) if jellyfish else ""
    return (meta, sound)

def extract_invoice_pattern(ref):
    if pd.isna(ref): return ""
    patterns = [
        r'(?:INV|INVOICE)[\s:-]*(\d{4,10})',
        r'(\d{2,4}[/-]\d{1,4}[/-]\d{1,6})',
        r'(\d{4}-\d{3,8})',
        r'([A-Z]{2,4}\d{6,10})'
    ]
    text = str(ref).upper()
    for p in patterns:
        m = re.search(p, text)
        if m: return m.group(1)
    return ""

def extract_gst_pattern(text):
    if pd.isna(text): return ""
    m = re.search(r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b', str(text))
    return m.group(0) if m else ""

def extract_bank_code(text):
    if pd.isna(text): return ""
    m = re.search(r'\b(NEFT|RTGS|IMPS|UPI|NACH|ACH)\b', str(text), re.I)
    return m.group(1).upper() if m else ""

def get_amount_from_row(row, mapping):
    debit = pd.to_numeric(row.get(mapping.get('debit')), errors='coerce') or 0
    credit = pd.to_numeric(row.get(mapping.get('credit')), errors='coerce') or 0
    return abs(debit - credit)

# ======================== TIERED MATCHER CLASS ========================

class TieredMatcher:
    def __init__(self):
        self.weights = {'tier1': 1.0, 'tier2': 0.88, 'tier3': 0.75, 'tier4': 0.62}

    def phonetic_similarity(self, a, b):
        if not a or not b: return 0.0
        a, b = str(a).upper(), str(b).upper()
        scores = []
        if jellyfish:
            if jellyfish.soundex(a) == jellyfish.soundex(b): scores.append(1.0)
            if jellyfish.metaphone(a) == jellyfish.metaphone(b): scores.append(1.0)
            if jellyfish.nysiis(a) == jellyfish.nysiis(b): scores.append(1.0)
        return max(scores) if scores else 0.0

    def tier1_exact(self, a, b, abs_tol=1.0):
        if (abs(a['amt'] - b['amt']) <= abs_tol and
            abs((a['date'] - b['date']).days) <= 1 if pd.notna(a['date']) and pd.notna(b['date']) else False and
            fuzz.ratio(str(a.get('ref','')), str(b.get('ref',''))) >= 95):
            return {"tier": "Tier 1 - Exact", "score": 0.99, "remark": "Perfect match"}

    def tier2_fuzzy(self, a, b, amt_tol=0.05, date_tol=7):
        amt_diff = abs(a['amt'] - b['amt']) / max(a['amt'], 1)
        date_diff = abs((a['date'] - b['date']).days) if pd.notna(a['date']) and pd.notna(b['date']) else 999
        ref_score = fuzz.ratio(str(a.get('ref','')), str(b.get('ref','')))
        if amt_diff <= amt_tol and date_diff <= date_tol and ref_score >= 85:
            return {"tier": "Tier 2 - Strong Fuzzy", "score": 0.88, "remark": f"Amount ±{amt_diff:.1%}, Date ±{date_diff}d"}

    def tier3_amount(self, a, b, amt_tol=0.05, abs_tol=50):
        diff = abs(a['amt'] - b['amt'])
        if diff <= max(a['amt'] * amt_tol, abs_tol):
            return {"tier": "Tier 3 - Amount Only", "score": 0.75, "remark": f"Amount diff ₹{diff:.2f}"}

    def tier4_reference(self, a, b):
        inv_a = extract_invoice_pattern(a.get('ref'))
        inv_b = extract_invoice_pattern(b.get('ref'))
        if inv_a and inv_a == inv_b:
            return {"tier": "Tier 4 - Invoice Pattern", "score": 0.70, "remark": f"INV Match: {inv_a}"}
        if self.phonetic_similarity(a.get('ref'), b.get('ref')) >= 0.9:
            return {"tier": "Tier 4 - Phonetic", "score": 0.68, "remark": "Phonetic match"}

# ======================== MAIN MATCHING FUNCTION ========================

def advanced_match_ledgers(A, map_a, B, map_b,
                          date_tol=180, amt_tol=0.05, abs_tol=50,
                          enable_ml=True, debug=False):
    
    A, B = A.copy(), B.copy()
    matcher = TieredMatcher()

    # Prepare data
    for df, m in [(A, map_a), (B, map_b)]:
        df['amt'] = df.apply(lambda r: get_amount_from_row(r, m), axis=1)
        df['date'] = pd.to_datetime(df[m.get('date')], errors='coerce') if m.get('date') else pd.NaT
        df['ref'] = df[m.get('ref')].astype(str).fillna('').str.lower() if m.get('ref') else ''

    A['_idx'] = A.index
    B['_idx'] = B.index
    sub_A = A[A['amt'] > 0]
    sub_B = B[B['amt'] > 0]

    matches = []
    used_b = set()

    model = joblib.load(MODEL_PATH) if enable_ml and joblib and os.path.exists(MODEL_PATH) else None

    for _, a_row in sub_A.iterrows():
        candidates = sub_B[~sub_B.index.isin(used_b)]
        if pd.notna(a_row['date']):
            candidates = candidates[
                (candidates['date'] >= a_row['date'] - timedelta(days=date_tol)) &
                (candidates['date'] <= a_row['date'] + timedelta(days=date_tol))
            ]
        candidates = candidates[
            (candidates['amt'] >= a_row['amt'] * (1 - amt_tol*3)) &
            (candidates['amt'] <= a_row['amt'] * (1 + amt_tol*3))
        ]

        if candidates.empty: continue

        best = None
        best_score = 0

        for _, b_row in candidates.iterrows():
            result = (matcher.tier1_exact(a_row, b_row, abs_tol) or
                     matcher.tier2_fuzzy(a_row, b_row, amt_tol, date_tol) or
                     matcher.tier3_amount(a_row, b_row, amt_tol, abs_tol) or
                     matcher.tier4_reference(a_row, b_row))

            if result:
                score = result['score']
                if model:
                    features = [
                        abs(a_row['amt'] - b_row['amt']),
                        abs(a_row['amt'] - b_row['amt']) / max(a_row['amt'], 1),
                        abs((a_row['date'] - b_row['date']).days) if pd.notna(a_row['date']) and pd.notna(b_row['date']) else 999,
                        fuzz.ratio(a_row['ref'], b_row['ref']),
                        matcher.phonetic_similarity(a_row['ref'], b_row['ref'])
                    ]
                    ml_prob = model.predict_proba([features])[0][1] if len(features) == 5 else 0.5
                    score = score * 0.4 + ml_prob * 0.6

                if score > best_score:
                    best = result
                    best_score = score
                    best_b = b_row

        if best and best_score >= 0.58:
            matches.append({
                "A_idx": a_row['_idx'],
                "B_idx": best_b['_idx'],
                "A_Date": a_row.get(map_a.get('date')),
                "A_Ref": a_row.get(map_a.get('ref')),
                "A_Amount": a_row['amt'],
                "B_Date": best_b.get(map_b.get('date')),
                "B_Ref": best_b.get(map_a.get('ref')),
                "B_Amount": best_b['amt'],
                "Match_Type": best['tier'],
                "Score": round(best_score * 100, 1),
                "Remarks": best['remark'],
                "Hash": hashlib.sha256(str(sorted([a_row['_idx'], best_b['_idx']])).encode()).hexdigest()[:16]
            })
            used_b.add(best_b.name)

    match_df = pd.DataFrame(matches)
    return match_df, A[~A['_idx'].isin(match_df['A_idx'])], B[~B['_idx'].isin(match_df['B_idx'])]

# ======================== RETRAIN FUNCTION ========================

def retrain_with_feedback(csv_file):
    df = pd.read_csv(csv_file)
    X = df[['amt_diff_abs','amt_diff_pct','date_diff_days','ref_score','phonetic_eq']].fillna(0)
    y = df['is_match']
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return "Model retrained successfully!"
