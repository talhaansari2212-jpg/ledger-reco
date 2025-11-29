import re
import io
import os
from datetime import timedelta
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
try:
    import jellyfish
except Exception:
    jellyfish = None

# Optional ML libs
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import joblib
except Exception:
    RandomForestClassifier = None
    train_test_split = None
    roc_auc_score = None
    joblib = None

MODEL_PATH = "match_model.joblib"

# -------------------------
# Column detection helper
# -------------------------
def detect_columns(df):
    cols = {'debit': None, 'credit': None, 'date': None, 'ref': None}
    keywords = {
        'debit': ['debit','dr','db','expense','purchase','charge','invoice','inv_amt'],
        'credit': ['credit','cr','payment','receipt','refund'],
        'date': ['date','trans','posting','value_date','txn','transaction'],
        'ref': ['ref','invoice','inv','vno','voucher','po','bill','cheque','chq','document','narration']
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

# -------------------------
# Phonetic helpers
# -------------------------
def phonetic_codes(s):
    if s is None:
        return ("","")
    s = str(s)
    if not s:
        return ("","")
    meta = ""
    sound = ""
    if jellyfish:
        try:
            meta = jellyfish.metaphone(s)
        except Exception:
            meta = ""
        try:
            sound = jellyfish.soundex(s)
        except Exception:
            sound = ""
    return (meta, sound)

# -------------------------
# Pattern recognition helpers
# -------------------------
INV_PATTERN = re.compile(r'(inv(?:oice)?[-\s:]*)?(\d{2,4}[-/]\d{1,4}[-/]\d{1,6}|\d{4}[-]\d{3,6}|\bINV[-]?\d{3,6}\b)', re.IGNORECASE)

def extract_invoice_pattern(ref):
    if pd.isna(ref):
        return ""
    m = INV_PATTERN.search(str(ref))
    return m.group(2) if m else ""

# -------------------------
# Amount extraction
# -------------------------
def get_amount_from_row(row, mapping):
    # mapping keys might be None
    d = 0
    c = 0
    try:
        if mapping.get('debit'):
            d = pd.to_numeric(row.get(mapping.get('debit')), errors='coerce') or 0
    except Exception:
        d = 0
    try:
        if mapping.get('credit'):
            c = pd.to_numeric(row.get(mapping.get('credit')), errors='coerce') or 0
    except Exception:
        c = 0
    # Amount as absolute of debit-credit (works for ledger where invoice in debit/credit columns)
    return abs((d or 0) - (c or 0))

# -------------------------
# ML model helpers
# -------------------------
def load_model(path=MODEL_PATH):
    if joblib and os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

def save_model(model, path=MODEL_PATH):
    if joblib:
        try:
            joblib.dump(model, path)
        except Exception:
            pass

def train_match_model(pairs_df, debug=False, model_path=MODEL_PATH):
    """
    Train a RandomForest model given a historical labeled DataFrame.
    Expected columns in pairs_df:
      - amt_diff_abs, amt_diff_pct, date_diff_days, ref_score, phonetic_eq (0/1), is_match (0/1)
    Returns trained model and AUC (if possible)
    """
    if RandomForestClassifier is None:
        raise RuntimeError("scikit-learn not available in environment")

    required = {'amt_diff_abs','amt_diff_pct','date_diff_days','ref_score','phonetic_eq','is_match'}
    if not required.issubset(set(pairs_df.columns)):
        raise ValueError(f"pairs_df must contain columns: {required}")

    X = pairs_df[['amt_diff_abs','amt_diff_pct','date_diff_days','ref_score','phonetic_eq']].fillna(0)
    y = pairs_df['is_match'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None)
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    auc = None
    try:
        probs = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = None
    save_model(model, model_path)
    if debug:
        return model, auc
    return model, auc

# -------------------------
# Core tiered matching
# -------------------------
def advanced_match_ledgers(A, map_a, B, map_b,
                          date_tol=7, amt_tol=0.05, abs_tol=50,
                          enable_ml=True, ml_model_path=MODEL_PATH,
                          debug=False):
    """
    Tiered matching with fallback mechanisms and optional ML scoring.
    Returns: match_df, unmatched_A, unmatched_B
    """

    # Work on copies
    A = A.copy()
    B = B.copy()

    # Prepare amounts
    A['amt'] = A.apply(lambda r: get_amount_from_row(r, map_a), axis=1)
    B['amt'] = B.apply(lambda r: get_amount_from_row(r, map_b), axis=1)

    # Keep original indices
    A['_orig_idx'] = A.index
    B['_orig_idx'] = B.index

    # Filter positive amounts
    sub_A = A[A['amt'] > 0].copy()
    sub_B = B[B['amt'] > 0].copy()

    if sub_A.empty or sub_B.empty:
        # nothing to match
        return pd.DataFrame(), A.drop(columns=['_orig_idx','amt'], errors='ignore'), B.drop(columns=['_orig_idx','amt'], errors='ignore')

    # Parse dates
    if map_a.get('date'):
        sub_A['date'] = pd.to_datetime(sub_A[map_a['date']], errors='coerce')
    else:
        sub_A['date'] = pd.NaT

    if map_b.get('date'):
        sub_B['date'] = pd.to_datetime(sub_B[map_b['date']], errors='coerce')
    else:
        sub_B['date'] = pd.NaT

    # References normalized
    sub_A['ref'] = sub_A[map_a.get('ref')].astype(str).fillna('').str.lower().str.strip() if map_a.get('ref') else ''
    sub_B['ref'] = sub_B[map_b.get('ref')].astype(str).fillna('').str.lower().str.strip() if map_b.get('ref') else ''

    # phonetic codes
    sub_A[['meta_a','sound_a']] = sub_A['ref'].apply(lambda x: pd.Series(list(phonetic_codes(x))))
    sub_B[['meta_b','sound_b']] = sub_B['ref'].apply(lambda x: pd.Series(list(phonetic_codes(x))))

    # invoice pattern extraction
    sub_A['inv_pattern'] = sub_A['ref'].apply(extract_invoice_pattern)
    sub_B['inv_pattern'] = sub_B['ref'].apply(extract_invoice_pattern)

    used_b_indices = set()
    used_a_indices = set()
    matches = []

    # Load ML model if enabled
    model = None
    if enable_ml:
        try:
            model = load_model(ml_model_path)
        except Exception:
            model = None

    # Candidate selection helper (narrows down to reasonable candidates)
    def candidate_filter(a_row):
        # Amount window
        a_amt = a_row['amt']
        pct_window = max(amt_tol * 3 * a_amt, 0)
        low = a_amt - pct_window - abs_tol
        high = a_amt + pct_window + abs_tol
        # Date window
        a_date = a_row['date']
        date_low = None
        date_high = None
        if pd.notna(a_date):
            date_low = a_date - pd.Timedelta(days=date_tol*4)
            date_high = a_date + pd.Timedelta(days=date_tol*4)

        cand = sub_B[~sub_B.index.isin(used_b_indices)].copy()
        # filter by amount window
        cand = cand[(cand['amt'] >= low) & (cand['amt'] <= high)]
        # filter by date if available
        if pd.notna(a_date):
            cand = cand[(cand['date'] >= date_low) & (cand['date'] <= date_high)]
        return cand

    # Iterate through A rows
    for a_index, a_row in sub_A.iterrows():
        cand = candidate_filter(a_row)
        if cand.empty:
            continue

        # compute features against candidates
        cand = cand.copy()
        cand['amt_diff_abs'] = (cand['amt'] - a_row['amt']).abs()
        cand['amt_diff_pct'] = cand['amt_diff_abs'] / (a_row['amt'] + 1e-9)
        cand['date_diff'] = (cand['date'] - a_row['date']).dt.days.abs().fillna(999)
        cand['ref_score'] = cand['ref'].apply(lambda x: fuzz.ratio(x, a_row['ref']))
        cand['meta_eq'] = cand['meta_b'] == a_row['meta_a']
        cand['sound_eq'] = cand['sound_b'] == a_row['sound_a']
        cand['inv_eq'] = cand['inv_pattern'] == a_row['inv_pattern']

        # Compute a baseline rule-based score (weighted composite)
        # weights: amount 0.5, date 0.3, ref 0.2
        cand['score_rule'] = (1 - cand['amt_diff_pct'].clip(0,1)) * 0.5 + \
                              (1 - cand['date_diff'].clip(0,60) / 60) * 0.3 + \
                              (cand['ref_score'] / 100) * 0.2

        # If phonetic equal give a small boost
        cand['score_rule'] = cand['score_rule'] + (cand['meta_eq'].astype(int) * 0.03) + (cand['sound_eq'].astype(int) * 0.02)
        # If invoice pattern exact match, boost strongly
        cand['score_rule'] = cand['score_rule'] + (cand['inv_eq'].astype(int) * 0.08)

        # If ML available, predict probability and combine
        if model is not None:
            feat_cols = ['amt_diff_abs','amt_diff_pct','date_diff','ref_score','meta_eq']
            Xm = cand[['amt_diff_abs','amt_diff_pct','date_diff','ref_score','meta_eq']].copy()
            # numeric meta_eq -> int
            Xm['meta_eq'] = Xm['meta_eq'].astype(int)
            try:
                probs = model.predict_proba(Xm)[:,1]
            except Exception:
                probs = np.zeros(len(Xm))
            cand['ml_prob'] = probs
            # Combine rule and ML: weighted average (ML 0.6 if available)
            cand['score_combined'] = (cand['score_rule'] * 0.4) + (cand['ml_prob'] * 0.6)
        else:
            cand['ml_prob'] = 0.0
            cand['score_combined'] = cand['score_rule']

        # Choose best candidate by combined score
        best_idx = cand['score_combined'].idxmax()
        best = cand.loc[best_idx]

        # Tiered decisions
        # Tier 1: Exact match (amount within abs_tol and date exact and ref nearly equal)
        tier = None
        remark = ""
        chosen_score = best['score_combined']

        amt_ok_exact = best['amt_diff_abs'] <= abs_tol
        date_exact = best['date_diff'] == 0
        ref_exact = best['ref_score'] >= 98 or best['inv_eq']

        if amt_ok_exact and date_exact and ref_exact and best['score_combined'] > 0.7:
            tier = "Tier 1 - Exact"
            confidence = chosen_score
            remark = "Exact amount & date & ref"
        else:
            # Tier 2: Fuzzy amount + date range
            if (best['amt_diff_pct'] <= amt_tol * 1.5 or best['amt_diff_abs'] <= abs_tol) and (best['date_diff'] <= date_tol) and best['ref_score'] >= 85:
                tier = "Tier 2 - Fuzzy Amount + Date"
                confidence = chosen_score * 0.9
                remark = f"Amount fuzzy ({best['amt_diff_abs']:.2f}), Date gap {int(best['date_diff'])}d"
            else:
                # Tier 3: Amount only with tolerance
                if (best['amt_diff_pct'] <= amt_tol * 3) or (best['amt_diff_abs'] <= abs_tol):
                    tier = "Tier 3 - Amount Only"
                    confidence = chosen_score * 0.75
                    remark = f"Amount close ({best['amt_diff_abs']:.2f})"
                else:
                    # Tier 4: Reference pattern / phonetic match
                    if best['inv_eq'] or best['meta_eq'] or best['sound_eq'] or best['ref_score'] >= 70:
                        tier = "Tier 4 - Reference Match"
                        confidence = chosen_score * 0.6
                        remark = f"Ref pattern/phonetic match (score {best['ref_score']:.1f})"
                    else:
                        tier = None
                        confidence = chosen_score * 0.2
                        remark = "No reliable tier match"

        # Enforce minimum thresholds
        # If ML present, be conservative: require combined score > 0.55 or specific tier overrides
        min_score_req = 0.55 if model is not None else 0.58
        # Also allow absolute amt tolerance bypass
        amt_bypass = best['amt_diff_abs'] <= abs_tol

        if tier and (best['score_combined'] >= min_score_req or amt_bypass):
            match_type = tier
            final_score = float(round(confidence * 100, 1))
            # Compose match record
            matches.append({
                "A_index": a_row['_orig_idx'],
                "B_index": best['_orig_idx'],
                "A_Date": a_row.get(map_a.get('date'), ""),
                "A_Ref": a_row.get(map_a.get('ref'), ""),
                "A_Amount": a_row['amt'],
                "B_Date": best.get(map_b.get('date'), ""),
                "B_Ref": best.get(map_b.get('ref'), ""),
                "B_Amount": best['amt'],
                "Match_Type": match_type,
                "Score": final_score,
                "Remarks": remark
            })
            used_b_indices.add(best.name)
            used_a_indices.add(a_index)
        else:
            # No match - skip for now
            if debug:
                # Keep track for debugging (could add to unmatched remarks later)
                pass

    # Prepare DataFrames
    match_df = pd.DataFrame(matches)

    unmatched_A = A[~A.index.isin(used_a_indices)].drop(columns=['_orig_idx','amt'], errors='ignore')
    unmatched_B = B[~B.index.isin(used_b_indices)].drop(columns=['_orig_idx','amt'], errors='ignore')

    # Add remarks to unmatched
    if not unmatched_A.empty:
        unmatched_A['Remarks'] = 'Missing from Supplier Statement (entry not found)'
    if not unmatched_B.empty:
        unmatched_B['Remarks'] = 'Missing from Our Ledger (entry not found)'

    return match_df, unmatched_A, unmatched_B

# -------------------------
# Retraining from user feedback
# -------------------------
def retrain_with_feedback(feedback_pairs_csv, debug=False, model_path=MODEL_PATH):
    """
    feedback_pairs_csv: path or file-like containing rows:
      A_ref,B_ref,amt_diff_abs,amt_diff_pct,date_diff_days,ref_score,phonetic_eq,is_match
    is_match: 1 for true match, 0 for not match
    Returns new model and auc if trained.
    """
    if RandomForestClassifier is None:
        raise RuntimeError("scikit-learn not available in environment")

    if isinstance(feedback_pairs_csv, (str, bytes)):
        df = pd.read_csv(feedback_pairs_csv)
    else:
        df = pd.read_csv(feedback_pairs_csv)

    model, auc = train_match_model(df, debug=debug, model_path=model_path)
    return model, auc
