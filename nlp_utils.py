# nlp_utils.py
# Lightweight NLP utilities:
# - Embeddings (sentence-transformers if available, fallback to TF-IDF)
# - Semantic similarity computation
# - Simple categorizer using keyword heuristics or k-NN over embeddings
# - Payment terms detection (regex)
# - Summarization hook (OpenAI or local transformer if installed)
import os
import re
import numpy as np
import pandas as pd

# Optional heavy deps
try:
    from sentence_transformers import SentenceTransformer, util
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except Exception:
    sbert_model = None

try:
    from transformers import pipeline
    summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception:
    summarizer_pipeline = None

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# simple TF-IDF fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    tfidf_vect = None
except Exception:
    TfidfVectorizer = None
    NearestNeighbors = None
    tfidf_vect = None

# Embedding getter
def get_embedding(texts):
    """
    Returns numpy array of embeddings for list of texts.
    Tries sentence-transformers then TF-IDF fallback (dense projection).
    """
    if sbert_model:
        if isinstance(texts, str):
            texts = [texts]
        embs = sbert_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs
    # fallback to TF-IDF as proxy embeddings
    if TfidfVectorizer:
        global tfidf_vect
        if tfidf_vect is None:
            tfidf_vect = TfidfVectorizer(max_features=2048, analyzer='word', ngram_range=(1,2))
            tfidf_vect.fit(texts if isinstance(texts, list) else [texts])
        X = tfidf_vect.transform(texts if isinstance(texts, list) else [texts]).toarray()
        # Normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        return X / norms
    # last fallback: char-level hash vector
    if isinstance(texts, str):
        texts = [texts]
    out = []
    for t in texts:
        v = np.array([hash(t) % 1000], dtype=float)
        out.append(v / (np.linalg.norm(v)+1e-9))
    return np.vstack(out)

def semantic_similarity_matrix(textsA, textsB):
    """
    Returns a [lenA x lenB] matrix of cosine similarities
    """
    emA = get_embedding(textsA)
    emB = get_embedding(textsB)
    # Normalize
    def norm_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        return X / norms
    emA = norm_rows(emA)
    emB = norm_rows(emB)
    sim = np.dot(emA, emB.T)
    return sim

# Categorization (simple prototype centroids or keyword)
DEFAULT_LABELS = {
    'invoice': ['invoice','inv','bill','charges','supply'],
    'payment': ['payment','paid','recd','receipt','remittance','credited'],
    'refund': ['refund','reversal','credit note'],
    'adjustment': ['adjust','adj','write-off','correction','contra']
}

def categorize_texts(texts):
    """
    Returns list/Series of predicted categories. Uses simple keyword match first, else centroid nearest neighbor with SBERT.
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    cats = []
    for t in texts:
        tl = t.lower()
        matched = None
        for label, kws in DEFAULT_LABELS.items():
            for kw in kws:
                if kw in tl:
                    matched = label
                    break
            if matched:
                break
        cats.append(matched if matched else 'unknown')
    # For unknowns, attempt embedding-based similarity to prototype examples (if sbert available)
    if sbert_model:
        # build prototypes by label using small examples
        prototypes = []
        labels = []
        for lab, kws in DEFAULT_LABELS.items():
            prototypes.append(" ".join(kws))
            labels.append(lab)
        proto_emb = sbert_model.encode(prototypes, convert_to_numpy=True)
        txt_emb = sbert_model.encode([t for t in texts], convert_to_numpy=True)
        # assign nearest prototype if unknown
        for i,t in enumerate(texts):
            if cats[i] == 'unknown':
                sims = np.dot(txt_emb[i], proto_emb.T) / (np.linalg.norm(txt_emb[i]) * np.linalg.norm(proto_emb, axis=1) + 1e-9)
                best = np.argmax(sims)
                if sims[best] > 0.45:
                    cats[i] = labels[best]
    return pd.Series(cats)

# Payment terms detection
PAYMENT_TERM_RE = re.compile(r'\b(net\s*(\d{1,3})|due on receipt|due upon receipt|immediate|end of month|eom)\b', re.IGNORECASE)
def detect_payment_terms(text):
    if not text or pd.isna(text):
        return None
    m = PAYMENT_TERM_RE.search(text)
    if not m:
        return None
    tok = m.group(0).lower()
    if 'net' in tok:
        n = re.search(r'\d+', tok)
        if n:
            return f"Net {n.group(0)}"
    return tok.title()

# Summarization
def summarize_texts(texts, max_length=120, use_openai=False, openai_model="gpt-3.5-turbo", openai_api_key_env='OPENAI_API_KEY'):
    """
    If openai is available and OPENAI_API_KEY in env, makes a single prompt to summarize list of texts.
    Else tries local summarizer pipeline if available; otherwise returns a simple concatenated heuristic summary.
    """
    # Flatten small set
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    texts = [str(t) for t in texts if t]
    if not texts:
        return ""
    # OpenAI path (optional)
    if use_openai and OPENAI_AVAILABLE and os.getenv(openai_api_key_env):
        try:
            openai.api_key = os.getenv(openai_api_key_env)
            prompt = "Summarize the following transaction descriptions into a short paragraph highlighting anomalies, repeating patterns, and suggested investigative actions:\n\n"
            for t in texts[:50]:
                prompt += "- " + t + "\n"
            resp = openai.ChatCompletion.create(model=openai_model, messages=[{"role":"user","content":prompt}], temperature=0.2, max_tokens=300)
            return resp['choices'][0]['message']['content'].strip()
        except Exception:
            pass
    # Transformer summarizer
    if summarizer_pipeline:
        joined = " ".join(texts[:50])
        try:
            out = summarizer_pipeline(joined, max_length=max_length, min_length=30, do_sample=False)
            return out[0]['summary_text']
        except Exception:
            pass
    # fallback heuristic: top keywords + counts
    all_text = " ".join(texts[:200]).lower()
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    freq = {}
    for w in words:
        freq[w] = freq.get(w,0) + 1
    common = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    top = ", ".join([f"{w}({c})" for w,c in common])
    return f"Heuristic summary: common tokens -> {top}"
