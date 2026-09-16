"""
load_data.py

Utilities for loading the Customer Support on Twitter dataset (twcs.csv),
profiling brand (company) accounts, and reconstructing conversation threads
from the response_tweet_id / in_response_to_tweet_id link fields.
"""
import re
import pandas as pd

RAW_PATH = "C:\Users\Nael\Downloads\hiver-agent\hiver-agent\twcs.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "tweet_id": "int64",
            "author_id": "string",
            "inbound": "bool",
            "created_at": "string",
            "text": "string",
            "response_tweet_id": "string",
        },
    )
    # in_response_to_tweet_id has NaNs -> nullable Int64 so it can still
    # join cleanly against tweet_id (int64) when walking threads.
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype("Int64")
    return df


def brand_volume(df: pd.DataFrame) -> pd.Series:
    """Outbound (company) tweet counts per author_id, descending."""
    return df.loc[~df["inbound"], "author_id"].value_counts()


REDIRECT_RE = re.compile(
    r"\b(?:dm|d\.m\.|direct message|private message|pm us"
    r"|reach out|share your (?:details|info)|contact (?:us|our)|fill out|click here)\b",
    re.IGNORECASE,
)


def brand_reply_profile(df: pd.DataFrame, brand: str, sample_n: int = 300, seed: int = 42) -> dict:
    """
    Rough signal for how 'canned' a brand's replies are:
    - redirect_rate: fraction of sampled replies that mention DM/private message
    - median_reply_chars: shorter median often correlates with boilerplate
    This is a heuristic for brand triage, not a real quality measure.
    """
    replies = df.loc[(~df["inbound"]) & (df["author_id"] == brand), "text"]
    n = len(replies)
    if n == 0:
        return {"brand": brand, "n_replies": 0, "redirect_rate": None, "median_reply_chars": None}
    sample = replies.sample(min(sample_n, n), random_state=seed)
    return {
        "brand": brand,
        "n_replies": n,
        "redirect_rate": round(sample.str.contains(REDIRECT_RE).mean(), 3),
        "median_reply_chars": int(sample.str.len().median()),
    }


def build_tweet_index(df: pd.DataFrame) -> pd.DataFrame:
    """Index by tweet_id for O(1) parent lookups when walking threads."""
    return df.set_index("tweet_id", drop=False)


def reconstruct_thread(indexed: pd.DataFrame, tweet_id: int, max_depth: int = 6) -> list:
    """
    Walk in_response_to_tweet_id backward from `tweet_id` to build the
    preceding context as an ordered list [oldest ... tweet_id]. Stops at
    max_depth or when the chain breaks (missing/absent parent id) -- broken
    chains are common in this dataset and are handled by stopping, not
    raising.
    """
    chain = []
    current = tweet_id
    for _ in range(max_depth):
        if current not in indexed.index:
            break
        row = indexed.loc[current]
        chain.append({
            "tweet_id": int(row["tweet_id"]),
            "author_id": row["author_id"],
            "inbound": bool(row["inbound"]),
            "text": row["text"],
        })
        parent = row["in_response_to_tweet_id"]
        if pd.isna(parent):
            break
        current = int(parent)
    return list(reversed(chain))


def build_pairs(df: pd.DataFrame, indexed: pd.DataFrame, brand: str) -> pd.DataFrame:
    """
    One row per brand reply that has an identifiable customer parent tweet:
    the reply text, the customer message it replied to, and metadata. This
    is the corpus the retrieval/grounding step gets built on later.
    """
    brand_replies = df[(~df["inbound"]) & (df["author_id"] == brand)]
    rows = []
    for r in brand_replies.itertuples(index=False):
        parent_id = r.in_response_to_tweet_id
        if pd.isna(parent_id):
            continue
        parent_id = int(parent_id)
        if parent_id not in indexed.index:
            continue
        parent = indexed.loc[parent_id]
        if not parent["inbound"]:
            continue  # brand replying to itself / another company -- skip
        rows.append({
            "company_tweet_id": r.tweet_id,
            "customer_tweet_id": parent_id,
            "customer_text": parent["text"],
            "company_text": r.text,
            "created_at": r.created_at,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {len(df):,} rows | {df['inbound'].sum():,} inbound / {(~df['inbound']).sum():,} outbound")

    vol = brand_volume(df)
    print("\nTop 12 brands by outbound tweet volume:")
    print(vol.head(12).to_string())

    candidates = vol.head(8).index.tolist()
    print("\nReply profile for top candidates (sample_n=300):")
    for b in candidates:
        print(brand_reply_profile(df, b))
