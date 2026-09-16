"""
01_taxonomy_exploration.py

Bootstraps intent-taxonomy candidates from real customer messages via
TF-IDF + KMeans clustering. This is a starting point for open-coding, not
a finished taxonomy -- clusters get read, merged, split, and named by hand
in taxonomy.md. Kept as a plain script (not a notebook) so `python3
notebooks/01_taxonomy_exploration.py` reproduces it without a kernel.
"""
import re
import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

DetectorFactory.seed = 0
N_CLUSTERS = 12
LANG_SAMPLE = 6000
MAX_ENGLISH = 4000
RANDOM_STATE = 42

MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")


def clean(text: str) -> str:
    text = MENTION_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    return text.strip()


def safe_detect(t: str) -> str:
    try:
        return detect(t)
    except LangDetectException:
        return "unk"


if __name__ == "__main__":
    pairs = pd.read_csv("data/amazonhelp_pairs.csv")

    sample = pairs.sample(min(LANG_SAMPLE, len(pairs)), random_state=RANDOM_STATE).copy()
    sample["lang"] = sample["customer_text"].astype(str).apply(safe_detect)
    en = sample[sample["lang"] == "en"].head(MAX_ENGLISH).copy()
    print(f"English subset for clustering: {len(en):,} / {len(sample):,} sampled")

    en["clean_text"] = en["customer_text"].astype(str).apply(clean)
    en = en[en["clean_text"].str.len() > 5]  # drop near-empty after stripping mentions/links

    vec = TfidfVectorizer(max_features=3000, stop_words="english", min_df=3)
    X = vec.fit_transform(en["clean_text"])

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    en["cluster"] = km.fit_predict(X)

    terms = vec.get_feature_names_out()
    for c in range(N_CLUSTERS):
        members = en[en["cluster"] == c]
        centroid = km.cluster_centers_[c]
        top_terms = [terms[i] for i in centroid.argsort()[-8:][::-1]]
        print(f"\n=== cluster {c} | n={len(members)} | top terms: {', '.join(top_terms)} ===")
        for t in members["customer_text"].sample(min(4, len(members)), random_state=1):
            print("  -", t[:160].replace("\n", " "))
