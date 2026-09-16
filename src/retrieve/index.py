"""
index.py

Retrieval index over (customer_text -> company_text) pairs, used to ground
generated replies in real historical resolutions.

Backend: sentence-transformers embeddings if installed (better semantic
match), falling back to TF-IDF + cosine similarity otherwise -- e.g. this
repo's dev sandbox has no Hugging Face access. Same query() interface
either way, so nothing downstream needs to know which backend is active.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except ImportError:
    _HAS_ST = False


class RetrievalIndex:
    def __init__(self, backend: str = "auto", model_name: str = "all-MiniLM-L6-v2"):
        if backend == "auto":
            backend = "embeddings" if _HAS_ST else "tfidf"
        if backend == "embeddings" and not _HAS_ST:
            raise ImportError("sentence-transformers not installed; pass backend='tfidf' or install it")
        self.backend = backend
        if self.backend == "embeddings":
            self._model = SentenceTransformer(model_name)
        else:
            self._vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2)

    def fit(self, pairs: pd.DataFrame, text_col: str = "customer_text") -> "RetrievalIndex":
        self.pairs = pairs.reset_index(drop=True)
        texts = self.pairs[text_col].astype(str).tolist()
        if self.backend == "embeddings":
            self._matrix = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        else:
            self._matrix = self._vectorizer.fit_transform(texts)
        return self

    def query(self, text: str, k: int = 5) -> pd.DataFrame:
        if self.backend == "embeddings":
            q = self._model.encode([text], normalize_embeddings=True)
            sims = (self._matrix @ q.T).ravel()
        else:
            q = self._vectorizer.transform([text])
            sims = cosine_similarity(self._matrix, q).ravel()
        top_idx = np.argsort(sims)[::-1][:k]
        out = self.pairs.iloc[top_idx].copy()
        out["similarity"] = sims[top_idx]
        return out.reset_index(drop=True)
