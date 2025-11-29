import pandas as pd
from rapidfuzz import fuzz
import numpy as np

def detect_columns(df):
    cols = {'debit': None, 'credit': None, 'date': None, 'ref': None}
    keywords = {
        'debit': ['debit','dr','expense','purchase'],
        'credit': ['credit','cr','payment','receipt'],
        'date': ['date','trans','posting','value'],
        'ref': ['ref','invoice','inv','vno','po','bill','chq']
    }
    lower = {c.lower(): c for c in df.columns}
    for key, words in keywords.items():
        for w in words:
            for l, o in lower.items():
                if w in l and not cols[key]:
                    cols[key] = o
    if not cols['date']:
        for c in df.columns:
            if pd.to_datetime(df[c], errors='coerce').notna().sum() > len(df)//2:
                cols['date'] = c; break
    if not cols['ref']:
        for c in df.select_dtypes('object').columns:
            if df[c].astype(str).str.contains(r'\d{3,}').mean() > 0.5:
                cols['ref'] = c; break
    return cols

def advanced_match_ledgers(A, map_a, B, map_b, date_tol=7, amt_tol=0.05):
    A = A.copy(); B = B.copy()
    A['amt'] = (A[map_a['debit']].fillna(0) - A[map_a['credit']].fillna(0)).abs() if map_a.get('debit') else 0
    B['amt'] = (B[map_b['debit']].fillna(0) - B[map_b['credit']].fillna(0)).abs() if map_b.get('debit') else 0
    A['date'] = pd.to_datetime(A[map_a['date']], errors='coerce')
    B['date'] = pd.to_datetime(B[map_b['date']], errors='coerce')
    A['ref'] = A[map_a['ref']].astype(str).str.lower().fillna('')
    B['ref'] = B[map_b['ref']].astype(str).str.lower().fillna('')

    matches = []
    used = set()

    for i, row_a in A.iterrows():
        candidates = B[~B.index.isin(used)]
        if candidates.empty: continue
        candidates['amt_diff'] = (candidates['amt'] - row_a['amt']).abs() / (row_a['amt'] + 1)
        candidates['date_diff'] = (candidates['date'] - row_a['date']).dt.days.abs()
        candidates['ref_score'] = candidates['ref'].apply(lambda x: fuzz.ratio(x, row_a['ref']))
        candidates['score'] = (
            (1 - candidates['amt_diff'].clip(0,1)) * 0.5 +
            (1 - candidates['date_diff'].clip(0,60)/60) * 0.3 +
            (candidates['ref_score']/100) * 0.2
        )
        best = candidates.loc[candidates['score'].idxmax()]
        if best['score'] > 0.6 and best['amt_diff'] < amt_tol*2 and best['date_diff'] <= date_tol*3:
            matches.append({**{f"A_{k}": v for k,v in row_a.items()},
                           **{f"B_{k}": v for k,v in best.items()},
                           "Match_Type": "Partial/Fuzzy" if best['amt_diff']>0 else "Exact"})
            used.add(best.name)

    matches_df = pd.DataFrame(matches) if matches else pd.DataFrame()
    return matches_df, A[~A.index.isin([m.get('A_amt') for m in matches])], B[~B.index.isin(used)]
