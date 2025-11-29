# core.py - Fully Production Ready Ledger Reconciliation Engine (2025)
import re
import os
import hashlib
from datetime import timedelta
from typing import Dict, Tuple, Optional, Any, List
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

# Optional dependencies with graceful fallback
try:
    import jellyfish
except ImportError:
    jellyfish = None

try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
except ImportError:
    RandomForestClassifier = None
    joblib = None

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
except Exception:
    _SBERT_AVAILABLE = False
    sbert_model = None

MODEL_PATH = "match_model.joblib"

# ================== Constants & Patterns ==================
INV_PATTERN = re.compile(
    r'(inv(?:oice)?[-\s:]*)?(?:'
    r'\d{2,4}[-/]\d{1,4}[-/]\d{1,6}|'
    r'\d{4}-\d{3,6}|'
    r'\bINV[-]?\d{3,10}\b|'
    r'\b\d{10,15}\b'  # GST Invoice style
    r')', re.IGNORECASE
)

BANK_CODE_PATTERN = re.compile(r'\b(NEFT|RTGS|IMPS|UPI|FT)\b', re.IGNORECASE)

# ================== Helper Functions ==================
def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Auto-detect common columns with smart fallbacks"""
    cols = {'debit': None, 'credit': None, 'date': None, 'ref': None,
              'gstin': None, 'txn_code': None, 'narration': None, 'currency': None}
    keywords = {
        'debit': ['debit', 'dr', 'db', 'expense', 'purchase', 'charge', 'invoice', 'inv_amt'],
        'credit': ['credit', 'cr', 'payment', 'receipt', 'refund', 'deposit'],
        'date': ['date', 'trans', 'posting', 'value', 'txn', 'transaction', 'entry'],
        'ref': ['ref', 'invoice', 'inv', 'vno', 'voucher', 'po', 'bill', 'cheque', 'chq', 'narration', 'particulars'],
        'gstin': ['gstin', 'gst', 'tin', 'pan'],
        'txn_code': ['mode', 'type', 'instrument', 'bank', 'code'],
        'narration': ['narration', 'description', 'remarks', 'details', 'particulars'],
        'currency': ['currency', 'cur', 'ccy']
    }
    lower_cols = {c.lower(): c for c in df.columns}

    for key, words in keywords.items():
        for word in words:
            for low, orig in lower_cols.items():
                if word in low and cols[key] is None:
                    cols[key] = orig
                    break
            else:
                continue
            break

    # Date fallback
    if not cols['date']:
        for c in df.columns:
            if df[c].dtype == 'object':
                parsed = pd.to_datetime(df[c], errors='coerce')
                if parsed.notna().sum() > len(df) * 0.3:
                    cols['date'] = c
                    break

    # Ref fallback
    if not cols['ref']:
        for c in df.select_dtypes(include='object').columns:
            if df[c].astype(str).str.contains(r'\d{4,}', na=False).mean() > 0.4:
                cols['ref'] = c
                break

    return cols


def phonetic_codes(s: str) -> Tuple[str, str]:
    if not s or pd.isna(s):
        return ("", "")
    s = str(s).upper()
    meta = jellyfish.metaphone(s) if jellyfish else ""
    sound = jellyfish.soundex(s) if jellyfish else ""
    return (meta or "", sound or "")


def extract_invoice_number(text: str) -> str:
    if pd.isna(text):
        return ""
    m = INV_PATTERN.search(str(text))
    return m.group(0).upper().replace(" ", "") if m else ""


def get_amount(row: pd.Series, mapping: Dict) -> float:
    debit = pd.to_numeric(row.get(mapping.get('debit')), errors='coerce') or 0
    credit = pd.to_numeric(row.get(mapping.get('credit')), errors='coerce') or 0
    return abs(float(debit - credit))


def optimize_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('string')
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df


# ================== Tiered Matcher ==================
class TieredMatcher:
    @staticmethod
    def phonetic_match(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        a, b = str(a).upper(), str(b).upper()
        score = 0.0
        if jellyfish:
            if jellyfish.soundex(a) == jellyfish.soundex(b):
                score += 1.0
            if jellyfish.metaphone(a) == jellyfish.metaphone(b):
                score += 1.0
        return score / 2 if score > 0 else 0.0


matcher = TieredMatcher()


# ================== Main Matching Engine ==================
def advanced_match_ledgers(
    A: pd.DataFrame, map_a: Dict,
    B: pd.DataFrame, map_b: Dict,
    date_tol: int = 180,
    amt_tol: float = 0.05,
    abs_tol: float = 100,
    enable_ml: bool = True,
    enable_semantic: bool = True,
    enable_partial_payments: bool = True,   # <-- YE NAAM HONA CHAHIYE
    ml_model_path: str = MODEL_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    A = optimize_df(A.copy())
    B = optimize_df(B.copy())

    # Extract amounts
    A['amt'] = A.apply(lambda r: get_amount(r, map_a), axis=1)
    B['amt'] = B.apply(lambda r: get_amount(r, map_b), axis=1)

    A['_idx'] = A.index
    B['_idx'] = B.index

    sub_A = A[A['amt'] > 1].copy()
    sub_B = B[B['amt'] > 1].copy()

    if sub_A.empty or sub_B.empty:
        return pd.DataFrame(), A, B

    # Parse dates
    date_col_a = map_a.get('date')
    date_col_b = map_b.get('date')
    sub_A['date'] = pd.to_datetime(sub_A[date_col_a] if date_col_a else None, errors='coerce')
    sub_B['date'] = pd.to_datetime(sub_B[date_col_b] if date_col_b else None, errors='coerce')

    # References & features
    ref_a = map_a.get('ref')
    ref_b = map_b.get('ref')
    sub_A['ref'] = sub_A[ref_a].fillna('').astype(str).str.lower() if ref_a else ''
    sub_B['ref'] = sub_B[ref_b].fillna('').astype(str).str.lower() if ref_b else ''

    sub_A[['meta', 'sound']] = sub_A['ref'].apply(lambda x: pd.Series(phonetic_codes(x)))
    sub_B[['meta', 'sound']] = sub_B['ref'].apply(lambda x: pd.Series(phonetic_codes(x)))

    sub_A['inv'] = sub_A['ref'].apply(extract_invoice_number)
    sub_B['inv'] = sub_B['ref'].apply(extract_invoice_number)

    # Narration & bank code
    narr_a = map_a.get('narration')
    narr_b = map_b.get('narration')
    if narr_a: sub_A['narr'] = sub_A[narr_a].fillna('').astype(str)
    if narr_b: sub_B['narr'] = sub_B[narr_b].fillna('').astype(str)

    txn_a = map_a.get('txn_code')
    txn_b = map_b.get('txn_code')
    if txn_a: sub_A['bank_code'] = sub_A[txn_a].astype(str).str.extract(BANK_CODE_PATTERN, expand=False).fillna('')
    if txn_b: sub_B['bank_code'] = sub_B[txn_b].astype(str).str.extract(BANK_CODE_PATTERN, expand=False).fillna('')

    # Load ML model
    model = joblib.load(ml_model_path) if enable_ml and joblib and os.path.exists(ml_model_path) else None

    used_A = set()
    used_B = set()
    matches = []

    for idx_a, row_a in sub_A.iterrows():
        if idx_a in used_A:
            continue

        # Candidate filtering
        amt = row_a['amt']
        window = max(amt * amt_tol * 3, abs_tol * 2)
        candidates = sub_B[
            (sub_B['amt'].between(amt - window - abs_tol, amt + window + abs_tol)) &
            (~sub_B.index.isin(used_B))
        ]

        if pd.notna(row_a['date']):
            dlow = row_a['date'] - timedelta(days=date_tol)
            dhigh = row_a['date'] + timedelta(days=date_tol)
            candidates = candidates[candidates['date'].between(dlow, dhigh)]

        if candidates.empty:
            continue

        # Feature engineering
        c = candidates.copy()
        c['amt_diff_abs'] = (c['amt'] - amt).abs()
        c['amt_diff_pct'] = c['amt_diff_abs'] / (amt + 1e-9)
        c['date_diff'] = (c['date'] - row_a['date']).dt.days.abs().fillna(999)
        c['ref_score'] = c['ref'].apply(lambda x: fuzz.ratio(x, row_a['ref']) / 100)
        c['inv_match'] = (c['inv'] == row_a['inv']).astype(int)
        c['phonetic'] = c.apply(lambda r: matcher.phonetic_match(row_a['ref'], r['ref']), axis=1)
        c['bank_match'] = (c['bank_code'] == row_a.get('bank_code', '')).astype(int) * 0.8

        # Semantic similarity
        c['semantic'] = 0.0
        if enable_semantic and _SBERT_AVAILABLE and 'narr' in row_a and 'narr' in c.columns:
            texts_a = [row_a['narr']] * len(c)
            texts_b = c['narr'].tolist()
            embs_a = sbert_model.encode(texts_a)
            embs_b = sbert_model.encode(texts_b)
            c['semantic'] = np.dot(embs_a, embs_b.T).diagonal() / (
                np.linalg.norm(embs_a, axis=1) * np.linalg.norm(embs_b, axis=1) + 1e-9)

        # Rule score
        c['rule_score'] = (
            (1 - c['amt_diff_pct'].clip(0, 1)) * 0.45 +
            (1 - c['date_diff'].clip(0, 180) / 180) * 0.25 +
            c['ref_score'] * 0.15 +
            c['inv_match'] * 0.08 +
            c['phonetic'] * 0.05 +
            c['bank_match'] * 0.02 +
            c['semantic'] * 0.15
        )

        # ML score
        if model:
            features = c[['amt_diff_abs', 'amt_diff_pct', 'date_diff', 'ref_score',
                          'inv_match', 'phonetic', 'semantic']].fillna(0)
            try:
                prob = model.predict_proba(features)[:, 1]
                c['ml_score'] = prob
                c['final_score'] = c['rule_score'] * 0.4 + c['ml_score'] * 0.6
            except:
                c['final_score'] = c['rule_score']
        else:
            c['final_score'] = c['rule_score']

        best = c.loc[c['final_score'].idxmax()]
        score = best['final_score']

        if score < 0.5 and best['amt_diff_abs'] > abs_tol:
            continue

        # Tier classification
        tier = "Tier5-Low"
        remark = "Weak match"
        if (best['amt_diff_abs'] <= abs_tol and best['date_diff'] <= 3 and
            (best['ref_score'] >= 0.95 or best['inv_match'])):
            tier = "Tier1-Exact"
            remark = "Exact match"
        elif score >= 0.85:
            tier = "Tier2-High"
            remark = "Strong fuzzy"
        elif score >= 0.7:
            tier = "Tier3-Medium"
            remark = "Moderate match"
        elif best['amt_diff_abs'] <= abs_tol:
            tier = "Tier4-AmountOnly"
            remark = "Amount matched"

        matches.append({
            "A_index": int(row_a['_idx']),
            "B_index": int(best['_idx']),
            "A_Date": row_a.get(date_col_a, row_a['date']),
            "B_Date": best.get(date_col_b, best['date']),
            "A_Ref": row_a.get(ref_a, row_a['ref']),
            "B_Ref": best.get(ref_b, best['ref']),
            "A_Amount": float(amt),
            "B_Amount": float(best['amt']),
            "Amount_Diff": float(best['amt_diff_abs']),
            "Match_Type": tier,
            "Confidence": round(score * 100, 2),
            "Remarks": remark,
            "Hash": hashlib.sha256(f"{row_a['_idx']}_{best['_idx']}_{score}".encode()).hexdigest()[:12]
        })

        used_A.add(idx_a)
        used_B.add(best.name)

    match_df = pd.DataFrame(matches)
    unmatched_A = A[~A['_idx'].isin(match_df['A_index'])].drop(columns=['_idx', 'amt'], errors='ignore')
    unmatched_B = B[~B['_idx'].isin(match_df['B_index'])].drop(columns=['_idx', 'amt'], errors='ignore')

    return match_df, unmatched_A.reset_index(drop=True), unmatched_B.reset_index(drop=True)


# ================== Extra Features ==================
def forecast_cash_flow(matches_df: pd.DataFrame) -> Dict:
    if matches_df.empty:
        return {"forecast": 0, "trend": "N/A"}
    df = matches_df.copy()
    df['date'] = pd.to_datetime(df['A_Date'], errors='coerce')
    df['amt'] = pd.to_numeric(df['A_Amount'], errors='coerce')
    df = df.dropna(subset=['date', 'amt'])
    monthly = df.groupby(df['date'].dt.to_period('M'))['amt'].sum()
    if len(monthly) < 2:
        return {"next_month": float(monthly.iloc[-1]) if not monthly.empty else 0}
    x = np.arange(len(monthly))
    slope, intercept = np.polyfit(x, monthly.values, 1)
    next_month = slope * len(monthly) + intercept
    return {"next_month_forecast": round(next_month, 2), "trend": "Up" if slope > 0 else "Down"}


def detect_anomalies(matches_df: pd.DataFrame) -> List[Dict]:
    anomalies = []
    if 'A_Amount' in matches_df.columns:
        mean = matches_df['A_Amount'].mean()
        std = matches_df['A_Amount'].std()
        outliers = matches_df[matches_df['A_Amount'].abs() > (mean + 3*std)]
        if not outliers.empty:
            anomalies.append({"type": "Large Amount", "count": len(outliers)})
    return anomalies
