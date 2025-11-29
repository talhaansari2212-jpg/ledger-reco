# core.py - FULL WORKING VERSION

import pandas as pd
from rapidfuzz import fuzz
import numpy as np

def detect_columns(df):
    cols = {'debit': None, 'credit': None, 'date': None, 'ref': None}
    keywords = {
        'debit': ['debit','dr','db','expense','purchase','charge'],
        'credit': ['credit','cr','payment','receipt','refund'],
        'date': ['date','trans','posting','value_date','txn'],
        'ref': ['ref','invoice','inv','vno','voucher','po','bill','cheque','chq']
    }
    lower_cols = {c.lower(): c for c in df.columns}
    for key, words in keywords.items():
        for word in words:
            for low, orig in lower_cols.items():
                if word in low and cols[key] is None:
                    cols[key] = orig

    # fallback for date
    if not cols['date']:
        for c in df.columns:
            try:
                if pd.to_datetime(df[c], errors='coerce').notna().sum() > len(df) * 0.5:
                    cols['date'] = c
                    break
            except: pass

    # fallback for ref
    if not cols['ref']:
        for c in df.select_dtypes(include='object').columns:
            if df[c].astype(str).str.contains(r'\d{3,}', regex=True).mean() > 0.4:
                cols['ref'] = c
                break

    return cols


def advanced_match_ledgers(A, map_a, B, map_b, date_tol=7, amt_tol=0.05):
    A = A.copy()
    B = B.copy()

    # Normalize amounts
    def get_amt(row, mapping):
        deb = pd.to_numeric(row.get(mapping.get('debit')), errors='coerce') or 0
        cre = pd.to_numeric(row.get(mapping.get('credit')), errors='coerce') or 0
        return abs(deb - cre)

    A['amt'] = A.apply(lambda row: get_amt(row, map_a), axis=1)
    B['amt'] = B.apply(lambda row: get_amt(row, map_b), axis=1)

    A = A[A['amt'] > 0].reset_index(drop=True)
    B = B[B['amt'] > 0].reset_index(drop=True)

    A['date'] = pd.to_datetime(A[map_a['date']], errors='coerce')
    B['date'] = pd.to_datetime(B[map_b['date']], errors='coerce')
    A['ref'] = A[map_a['ref']].astype(str).str.lower().str.strip()
    B['ref'] = B[map_b['ref']].astype(str).str.lower().str.strip()

    matches = []
    used_b = set()

    for i, row_a in A.iterrows():
        if i in used_b: continue
        candidates = B[~B.index.isin(used_b)].copy()

        candidates['amt_diff'] = (candidates['amt'] - row_a['amt']).abs() / (row_a['amt'] + 1)
        candidates['date_diff'] = (candidates['date'] - row_a['date']).dt.days.abs().fillna(999)
        candidates['ref_score'] = candidates['ref'].apply(lambda x: fuzz.ratio(x, row_a['ref']) if x and row_a['ref'] else 0)

        candidates['score'] = (
            (1 - np.clip(candidates['amt_diff'], 0, 1)) * 0.5 +
            (1 - np.clip(candidates['date_diff'], 0, 60)/60) * 0.3 +
            (candidates['ref_score']/100) * 0.2
        )

        best = candidates.loc[candidates['score'].idxmax()]

        if (best['score'] > 0.55 and 
            best['amt_diff'] <= amt_tol * 2 and 
            best['date_diff'] <= date_tol * 3):
            
            match_type = "Exact" if best['amt_diff'] < 0.01 and best['date_diff'] <= 1 else "Fuzzy/Partial"
            matches.append({
                "A_Date": row_a[map_a['date']],
                "A_Ref": row_a[map_a['ref']],
                "A_Amount": row_a['amt'],
                "B_Date": best[map_b['date']],
                "B_Ref": best[map_b['ref']],
                "B_Amount": best['amt'],
                "Match_Type": match_type,
                "Score": round(best['score']*100, 1)
            })
            used_b.add(best.name)

    return pd.DataFrame(matches) if matches else pd.DataFrame(), A, B
