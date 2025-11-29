import requests
import datetime
import time
from functools import lru_cache
import pandas as pd
import re
from decimal import Decimal, ROUND_HALF_UP

# ===========================
# Exchange Rate Service
# ===========================
# Primary: exchangerate.host (free, supports historical)
# Fallback hooks for RBI/ECB (placeholders — can be implemented if API credentials / endpoints available)
EXCHANGERATE_HOST = "https://api.exchangerate.host"

# Supported currencies for this module
SUPPORTED_CURRENCIES = {"INR", "USD", "EUR", "GBP", "AED"}

# volatility multipliers (higher -> looser tolerances)
CURRENCY_VOLATILITY = {
    "INR": 1.0,
    "USD": 0.8,
    "EUR": 0.9,
    "GBP": 0.95,
    "AED": 1.2  # example: AED might be treated slightly more volatile in some corridors
}

# simple LRU cache for exchange rates for performance
@lru_cache(maxsize=4096)
def _fetch_rates_from_exchangerate_host(date_str: str, base: str = "EUR"):
    # date_str in YYYY-MM-DD
    url = f"{EXCHANGERATE_HOST}/{date_str}"
    params = {"base": base}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        payload = r.json()
        if payload.get("success", True) is False:
            raise RuntimeError("ExchangeRate.host returned failure")
        rates = payload.get("rates", {})
        return rates
    except Exception as e:
        # bubble up to caller to try other providers if desired
        raise

def get_exchange_rate_on(date: datetime.date, from_ccy: str, to_ccy: str, prefer_provider: str = "exchangerate.host"):
    """
    Get historical exchange rate for a date (date: datetime.date).
    Attempts providers in order. Returns Decimal rate (1 from_ccy = rate to_ccy).
    """
    if from_ccy == to_ccy:
        return Decimal("1.0")

    date_str = date.strftime("%Y-%m-%d")
    # Primary: exchangerate.host
    try:
        # fetch rates with base=from_ccy to get direct conversion
        rates = _fetch_rates_from_exchangerate_host(date_str, base=from_ccy)
        if to_ccy in rates:
            return Decimal(str(rates[to_ccy]))
        # If not present, try reciprocal via base=to_ccy
        rates2 = _fetch_rates_from_exchangerate_host(date_str, base=to_ccy)
        if from_ccy in rates2:
            # rate_to = 1 / rates2[from]
            return (Decimal("1.0") / Decimal(str(rates2[from_ccy])))
    except Exception:
        # Could try RBI/ECB here as fallback (placeholders)
        pass

    # If still not available, raise
    raise RuntimeError(f"Exchange rate not available for {from_ccy}->{to_ccy} on {date_str}")

def convert_amount(amount: float, from_ccy: str, to_ccy: str, date: datetime.date = None, rounding_places: int = 2):
    """
    Convert a numeric amount from one currency to another using historical rates.
    date: datetime.date or None (uses today)
    Returns Decimal rounded to rounding_places
    """
    if date is None:
        date = datetime.date.today()
    rate = get_exchange_rate_on(date, from_ccy.upper(), to_ccy.upper())
    amt = Decimal(str(amount)) * Decimal(rate)
    quant = Decimal("1." + "0" * rounding_places)
    return amt.quantize(Decimal(1) / (Decimal(10) ** rounding_places), rounding=ROUND_HALF_UP)

# ===========================
# Currency-specific tolerances
# ===========================
def currency_tolerance(amount: float, currency: str, base_pct_tol: float = 0.05, base_abs_tol: float = 50.0):
    """
    Returns (pct_tol, abs_tol) adjusted for currency volatility.
    base_pct_tol is fraction (e.g., 0.05 for 5%)
    """
    mult = CURRENCY_VOLATILITY.get(currency.upper(), 1.0)
    return float(base_pct_tol * mult), float(base_abs_tol * mult)

# FX gain/loss calculation
def compute_fx_gain_loss(amount_a: float, ccy_a: str, date_a: datetime.date,
                         amount_b: float, ccy_b: str, date_b: datetime.date,
                         base_currency: str = "INR"):
    """
    Convert both amounts to base_currency using historical rates and compute gain/loss.
    Returns dict: {amount_a_base, amount_b_base, diff_base, sign}
    """
    a_base = convert_amount(amount_a, ccy_a, base_currency, date=date_a)
    b_base = convert_amount(amount_b, ccy_b, base_currency, date=date_b)
    diff = a_base - b_base
    sign = "gain" if diff > 0 else ("loss" if diff < 0 else "none")
    return {
        "A_in_base": float(a_base),
        "B_in_base": float(b_base),
        "diff_in_base": float(diff),
        "sign": sign
    }

# Cross-currency amount validation with real-time rates
def is_cross_currency_match(amount_a, ccy_a, date_a, amount_b, ccy_b, date_b,
                            base_currency="INR", pct_tol=0.05, abs_tol=50.0):
    """
    Validate whether two transactions in different currencies can be considered matching.
    - Converts both to base_currency using historical rates on respective dates
    - Uses percentage and absolute tolerances (currency-specific adjustments)
    Returns (bool, details)
    """
    try:
        details = compute_fx_gain_loss(amount_a, ccy_a, date_a, amount_b, ccy_b, date_b, base_currency=base_currency)
    except Exception as e:
        return False, {"error": str(e)}

    # Adjust tolerances according to base currency volatility
    pct_adj, abs_adj = currency_tolerance(base_currency, base_currency, base_pct_tol=pct_tol, base_abs_tol=abs_tol)
    a_to_b_pct_diff = abs(details["diff_in_base"]) / (abs(details["B_in_base"]) + 1e-9)
    is_ok = (a_to_b_pct_diff <= pct_adj) or (abs(details["diff_in_base"]) <= abs_adj)
    details.update({"pct_diff": a_to_b_pct_diff, "pct_tol": pct_adj, "abs_tol": abs_adj, "is_match": is_ok})
    return is_ok, details

# ===========================
# Data enrichment helpers for reconciliation
# ===========================
def enrich_dataframe_with_currency(df: pd.DataFrame, currency_column: str = None, amount_column: str = "amt",
                                   assumed_currency: str = "INR", base_currency: str = "INR", date_column: str = None):
    """
    Ensure dataframe has:
      - currency column (use currency_column if provided else assume assumed_currency)
      - amount (numeric) column
      - amount_in_base (converted to base_currency using date_column)
    Returns new DataFrame copy with added columns: '_currency', '_amount_numeric', 'amount_in_base', '_rate_used'
    """
    out = df.copy()
    out['_currency'] = out[currency_column].fillna(assumed_currency) if currency_column and currency_column in out.columns else assumed_currency
    out['_amount_numeric'] = pd.to_numeric(out.get(amount_column), errors='coerce').fillna(0).astype(float)
    # compute conversion per-row
    rates = []
    converted = []
    for idx, row in out.iterrows():
        cur = str(row['_currency']).upper()
        amt = float(row['_amount_numeric'])
        dt = None
        if date_column and date_column in out.columns:
            try:
                dt = pd.to_datetime(row[date_column]).date()
            except Exception:
                dt = datetime.date.today()
        else:
            dt = datetime.date.today()
        try:
            conv = convert_amount(amt, cur, base_currency, date=dt)
            converted.append(float(conv))
            rates.append(float(get_exchange_rate_on(dt, cur, base_currency)))
        except Exception:
            converted.append(None)
            rates.append(None)
    out['amount_in_' + base_currency.lower()] = converted
    out['_rate_used'] = rates
    return out

# ===========================
# GST / Tax-related helpers
# ===========================
GSTIN_REGEX = re.compile(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$', re.IGNORECASE)
# Note: above regex is a common pattern; full checksum validation not implemented here.

def validate_gstin(gstin: str):
    """
    Basic GSTIN format validation.
    Returns (is_valid_format, reason). Full checksum validation is complex and optional.
    """
    if not gstin or not isinstance(gstin, str):
        return False, "Empty or not a string"
    gstin = gstin.strip().upper()
    if not GSTIN_REGEX.match(gstin):
        return False, "GSTIN format invalid"
    # TODO: Implement checksum validation if required
    return True, "Format valid (checksum not validated)"

# GST invoice pattern recognition
GST_INVOICE_PATTERN = re.compile(r'((?:(?:GST)?\s*INV[:/\-\s]?)?[A-Z0-9\-/]{3,30}\d+)', re.IGNORECASE)

def extract_gst_invoice_number(text: str):
    if not text:
        return None
    m = GST_INVOICE_PATTERN.search(str(text))
    if m:
        return m.group(1)
    # fallback: trailing invoice-like token
    m2 = re.search(r'INV[-\s]?\d{2,6}', str(text), re.IGNORECASE)
    return m2.group(0) if m2 else None

# Tax breakup detection
def detect_tax_breakup(row: dict, cgst_col=None, sgst_col=None, igst_col=None, tax_col=None, tax_rate_col=None):
    """
    Given a row (pandas Series or dict) attempts to extract CGST/SGST/IGST amounts.
    Returns a dict: {'cgst':val, 'sgst':val, 'igst':val, 'tax_total':val, 'tax_from_rate':bool}
    """
    def _num(v):
        try:
            return float(v)
        except Exception:
            return 0.0

    cgst = _num(row.get(cgst_col)) if cgst_col in row else 0.0
    sgst = _num(row.get(sgst_col)) if sgst_col in row else 0.0
    igst = _num(row.get(igst_col)) if igst_col in row else 0.0
    tax_total = _num(row.get(tax_col)) if tax_col in row else cgst + sgst + igst
    tax_from_rate = False
    if tax_total == 0 and tax_rate_col and tax_rate_col in row:
        # attempt compute from taxable value if present
        rate = _num(row.get(tax_rate_col))
        # attempt find taxable base
        # typical columns: taxable_value, base_amount, net_amount
        taxable = None
        for k in ['taxable_value', 'base', 'taxable', 'amount', 'invoice_amount', 'inv_amount']:
            if k in row:
                taxable = _num(row.get(k))
                if taxable:
                    break
        if taxable:
            tax_total = taxable * rate / 100.0
            tax_from_rate = True
    return {
        'cgst': cgst,
        'sgst': sgst,
        'igst': igst,
        'tax_total': tax_total,
        'tax_from_rate': tax_from_rate
    }

# GSTIN matching across ledgers (fuzzy)
def gstin_match(gstin_a: str, gstin_b: str):
    """
    Compare two GSTINs. Returns boolean and reason.
    """
    if not gstin_a or not gstin_b:
        return False, "Missing GSTIN"
    a = str(gstin_a).strip().upper()
    b = str(gstin_b).strip().upper()
    if a == b:
        return True, "Exact match"
    # partial match on PAN portion (positions 3-12)
    try:
        pan_a = a[2:12]
        pan_b = b[2:12]
        if pan_a == pan_b:
            return True, "PAN matches (likely same entity)"
    except Exception:
        pass
    return False, "Different GSTINs"

# Automated GSTR-2A vs purchase ledger matching (prototype)
def match_gstr2a(purchase_df: pd.DataFrame, gstr2a_df: pd.DataFrame,
                 map_purchase: dict, map_gstr: dict,
                 date_tolerance_days: int = 15, amount_tolerance_pct: float = 0.02):
    """
    Attempts to match purchase ledger entries with supplier GSTR-2A records.
    map_purchase and map_gstr should contain keys: 'invoice', 'gstin', 'date', 'amount', 'cgst','sgst','igst'
    Returns matched_pairs (DataFrame) and unmatched indices (purchase_unmatched, gstr_unmatched)
    """
    P = purchase_df.copy()
    G = gstr2a_df.copy()

    # normalize invoice numbers
    P['_inv'] = P[map_purchase.get('invoice')].astype(str).apply(lambda x: (x or "").strip().upper())
    G['_inv'] = G[map_gstr.get('invoice')].astype(str).apply(lambda x: (x or "").strip().upper())

    # parse dates
    P['_date'] = pd.to_datetime(P[map_purchase.get('date')], errors='coerce')
    G['_date'] = pd.to_datetime(G[map_gstr.get('date')], errors='coerce')

    P['_amount'] = pd.to_numeric(P[map_purchase.get('amount')], errors='coerce').fillna(0)
    G['_amount'] = pd.to_numeric(G[map_gstr.get('amount')], errors='coerce').fillna(0)

    # first attempt exact invoice & gstin & amount within tolerance
    matches = []
    used_g = set()
    for p_idx, prow in P.iterrows():
        p_inv = prow['_inv']
        p_gstin = prow.get(map_purchase.get('gstin'))
        p_date = prow['_date']
        p_amt = prow['_amount']
        found = False
        # find candidates in G with same invoice (direct)
        cand = G[G['_inv'] == p_inv].copy()
        if cand.empty:
            # fallback: gstin match + amount close + date within tolerance
            cand = G[(G.get(map_gstr.get('gstin')) == p_gstin)]
        # compute amount tolerance
        for g_idx, grow in cand.iterrows():
            if g_idx in used_g:
                continue
            # date tolerance
            g_date = grow['_date']
            date_ok = True
            if pd.notna(p_date) and pd.notna(g_date):
                date_ok = abs((p_date - g_date).days) <= date_tolerance_days
            # amount tolerance
            pct_diff = abs(p_amt - grow['_amount']) / (abs(grow['_amount']) + 1e-9)
            if date_ok and pct_diff <= amount_tolerance_pct:
                matches.append({
                    'purchase_idx': p_idx,
                    'gstr_idx': g_idx,
                    'purchase_invoice': p_inv,
                    'gstr_invoice': grow['_inv'],
                    'purchase_amount': p_amt,
                    'gstr_amount': grow['_amount'],
                    'pct_diff': pct_diff
                })
                used_g.add(g_idx)
                found = True
                break
        # continue loop
    match_df = pd.DataFrame(matches)
    purchase_unmatched = P[~P.index.isin(match_df['purchase_idx'].tolist() if not match_df.empty else [])]
    gstr_unmatched = G[~G.index.isin(match_df['gstr_idx'].tolist() if not match_df.empty else [])]
    return match_df, purchase_unmatched, gstr_unmatched

# Tax period wise reconciliation reports
def tax_period_report(purchase_df: pd.DataFrame, map_purchase: dict, period_column: str = None):
    """
    Aggregates tax amounts per period for reconciliation reporting.
    map_purchase keys: 'date','amount','cgst','sgst','igst','gstin'
    period_column: optional precomputed period (e.g., '2024-07') else derive from date
    Returns aggregated DataFrame: period, total_taxable, total_tax, cgst, sgst, igst, count
    """
    P = purchase_df.copy()
    if period_column and period_column in P.columns:
        P['_period'] = P[period_column]
    else:
        P['_date'] = pd.to_datetime(P.get(map_purchase.get('date')), errors='coerce')
        P['_period'] = P['_date'].dt.to_period('M').astype(str).fillna('Unknown')

    P['_taxable'] = pd.to_numeric(P.get(map_purchase.get('amount')), errors='coerce').fillna(0)
    P['_cgst'] = pd.to_numeric(P.get(map_purchase.get('cgst')), errors='coerce').fillna(0)
    P['_sgst'] = pd.to_numeric(P.get(map_purchase.get('sgst')), errors='coerce').fillna(0)
    P['_igst'] = pd.to_numeric(P.get(map_purchase.get('igst')), errors='coerce').fillna(0)
    agg = P.groupby('_period').agg(
        total_taxable=pd.NamedAgg(column='_taxable', aggfunc='sum'),
        total_cgst=pd.NamedAgg(column='_cgst', aggfunc='sum'),
        total_sgst=pd.NamedAgg(column='_sgst', aggfunc='sum'),
        total_igst=pd.NamedAgg(column='_igst', aggfunc='sum'),
        count=pd.NamedAgg(column='_taxable', aggfunc='count')
    ).reset_index().rename(columns={'_period': 'period'})
    agg['total_tax'] = agg['total_cgst'] + agg['total_sgst'] + agg['total_igst']
    return agg

# ===========================
# Bank-specific helpers (basic)
# ===========================
# Detect NEFT/RTGS/IMPS/UPI from narration/txn_code
BANK_CODE_RE = re.compile(r'\b(NEFT|RTGS|IMPS|UPI|IMPS-PAY|NACH|ACH)\b', re.IGNORECASE)

def detect_bank_txn_type(text: str):
    if not text:
        return None
    m = BANK_CODE_RE.search(text)
    return m.group(1).upper() if m else None

def parse_bank_date_flexible(date_text: str):
    """
    Attempts to parse a variety of date formats common in bank statements.
    Returns datetime.date or None
    """
    from dateutil import parser
    try:
        dt = parser.parse(str(date_text), dayfirst=False)
        return dt.date()
    except Exception:
        try:
            dt = parser.parse(str(date_text), dayfirst=True)
            return dt.date()
        except Exception:
            return None

# ===========================
# Utilities
# ===========================
def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default
