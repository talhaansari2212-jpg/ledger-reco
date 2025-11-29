import pandas as pd
from rapidfuzz import fuzz
import numpy as np

def detect_columns(df):
    cols = {'debit': None, 'credit': None, 'date': None, 'ref': None}
    keywords = {
        'debit': ['debit','dr','db','expense','purchase','charge'],
        'credit': ['credit','cr','payment','receipt','refund'],
        'date': ['date','trans','posting','value_date','txn','transaction'],
        'ref': ['ref','invoice','inv','vno','voucher','po','bill','cheque','chq','document']
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
            if df[c].astype(str).str.contains(r'\d{3,}', regex=True).mean() > 0.4:
                cols['ref'] = c
                break

    return cols

def advanced_match_ledgers(A, map_a, B, map_b, date_tol=7, amt_tol=0.05):
    # Data clean copy
    A = A.copy()
    B = B.copy()
    
    # Track original indices to filter Unmatched rows later
    A['_orig_idx'] = A.index
    B['_orig_idx'] = B.index

    def get_amount(row, mapping):
        d = pd.to_numeric(row.get(mapping.get('debit')), errors='coerce') or 0
        c = pd.to_numeric(row.get(mapping.get('credit')), errors='coerce') or 0
        return abs(d - c)

    A['amt'] = A.apply(lambda r: get_amount(r, map_a), axis=1)
    B['amt'] = B.apply(lambda r: get_amount(r, map_b), axis=1)

    # Filter for matching process (only positive amounts)
    sub_A = A[A['amt'] > 0].copy()
    sub_B = B[B['amt'] > 0].copy()

    # Prepare fields for matching
    sub_A['date'] = pd.to_datetime(sub_A[map_a['date']], errors='coerce')
    sub_B['date'] = pd.to_datetime(sub_B[map_b['date']], errors='coerce')
    sub_A['ref'] = sub_A[map_a['ref']].astype(str).str.lower().str.strip()
    sub_B['ref'] = sub_B[map_b['ref']].astype(str).str.lower().str.strip()

    matches = []
    used_b_indices = set()
    used_a_indices = set()

    for a_index, a_row in sub_A.iterrows():
        # Look in B where index not used yet
        cand = sub_B[~sub_B.index.isin(used_b_indices)].copy()
        if cand.empty: continue

        # Calculate scores
        cand['amt_diff'] = (cand['amt'] - a_row['amt']).abs() / (a_row['amt'] + 1)
        cand['date_diff'] = (cand['date'] - a_row['date']).dt.days.abs().fillna(999)
        cand['ref_score'] = cand['ref'].apply(lambda x: fuzz.ratio(x, a_row['ref']))

        cand['score'] = (1 - cand['amt_diff'].clip(0,1)) * 0.5 + \
                        (1 - cand['date_diff'].clip(0,60)/60) * 0.3 + \
                        (cand['ref_score']/100) * 0.2

        best = cand.loc[cand['score'].idxmax()]
        
        # Threshold Check
        if best['score'] > 0.58 and best['amt_diff'] <= amt_tol*3 and best['date_diff'] <= date_tol*4:
            matches.append({
                "A_Date": a_row[map_a['date']], "A_Ref": a_row[map_a['ref']], "A_Amount": a_row['amt'],
                "B_Date": best[map_b['date']], "B_Ref": best[map_b['ref']], "B_Amount": best['amt'],
                "Match_Type": "Exact" if best['amt_diff']<0.01 else "Fuzzy/Partial",
                "Score": round(best['score']*100,1)
            })
            used_b_indices.add(best.name) 
            used_a_indices.add(a_index)

    # Prepare Final Dataframes
    match_df = pd.DataFrame(matches)
    
    # Filter Unmatched Rows using original indices (removing helper columns)
    unmatched_A = A[~A.index.isin(used_a_indices)].drop(columns=['_orig_idx', 'amt'], errors='ignore')
    unmatched_B = B[~B.index.isin(used_b_indices)].drop(columns=['_orig_idx', 'amt'], errors='ignore')

    return match_df, unmatched_A, unmatched_B
